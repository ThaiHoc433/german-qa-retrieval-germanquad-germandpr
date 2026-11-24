import { render, renderHook, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { ApiError } from "./api/client";
import type { AskParams, QAResult } from "./api/types";
import { useReady } from "./hooks/useReady";
import { MAX_QUESTION_CHARS } from "./pages/AskPage";
import { deferred, experiments, experimentsWithHybrid, fakeApi, qa, qaReranked, readyBody, readyHybrid, retrieved } from "./test/fixtures";

const ask = async (user: ReturnType<typeof userEvent.setup>, q: string) => {
  const box = screen.getByLabelText(/Câu hỏi/);
  await user.clear(box);
  await user.type(box, q);
  await user.click(screen.getByRole("button", { name: "Hỏi" }));
};

beforeEach(() => {
  window.location.hash = "";
});

describe("hỏi đáp", () => {
  it("hiện đáp án, tô sáng đúng chữ dù có emoji đứng trước, và thời gian + phiên bản", async () => {
    const user = userEvent.setup();
    render(<App api={fakeApi()} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Hỏi" })).toBeDisabled()); // chưa nhập gì
    await ask(user, "Wer schläft?");
    const mark = await screen.findByTestId("answer-mark");
    expect(mark).toHaveTextContent("Katze");
    expect(screen.getAllByText("Katze").length).toBeGreaterThan(0);
    expect(screen.getByText("Truy hồi", { selector: "th" }).nextSibling).toHaveTextContent("1,1 ms");
    expect(screen.getByText(/reader_x/)).toBeInTheDocument();
  });

  it("response CŨ về sau response MỚI thì không ghi đè (tiêu chí nghiệm thu)", async () => {
    const user = userEvent.setup();
    const first = deferred<QAResult>();
    const second = deferred<QAResult>();
    const queue = [first, second];
    // fake KHÔNG tôn trọng AbortSignal, mô phỏng trường hợp hủy không kịp: chỉ số thứ tự mới cứu được
    const api = fakeApi({ ask: () => queue.shift()!.promise });
    render(<App api={api} />);
    await waitFor(() => expect(screen.queryByText(/Đang kiểm tra/)).not.toBeInTheDocument());

    await ask(user, "Frage eins"); // yêu cầu 1 đang bay...
    expect(screen.getByRole("status")).toHaveTextContent("Frage eins");
    await ask(user, "Frage zwei"); // ...người dùng hỏi tiếp: nút vẫn bật, yêu cầu 1 bị thay thế
    expect(screen.getByRole("status")).toHaveTextContent("Frage zwei");

    const neu = qa({ request_id: "neu", answer: "NEU", passage_text: "NEU", answer_start_utf16: 0, answer_end_utf16: 3, trace: null });
    const alt = qa({ request_id: "alt", answer: "ALT", passage_text: "ALT", answer_start_utf16: 0, answer_end_utf16: 3, trace: null });
    second.resolve(neu);
    expect(await screen.findByTestId("answer-mark")).toHaveTextContent("NEU");
    first.resolve(alt); // response cũ về SAU cùng
    await new Promise((r) => setTimeout(r, 30));
    expect(screen.getByTestId("answer-mark")).toHaveTextContent("NEU");
    expect(screen.queryByText("ALT")).toBeNull();
  });

  it("hook useAsk: hai yêu cầu liên tiếp, response cũ về sau cùng -> state vẫn là của yêu cầu mới", async () => {
    const { useAsk } = await import("./hooks/useAsk");
    const { act } = await import("@testing-library/react");
    const a = deferred<QAResult>();
    const b = deferred<QAResult>();
    const q = [a, b];
    const api = fakeApi({ ask: () => q.shift()!.promise });
    const { result } = renderHook(() => useAsk(api));
    act(() => void result.current.ask({ question: "alt" }));
    act(() => void result.current.ask({ question: "neu" }));
    await act(async () => b.resolve(qa({ request_id: "new", answer: "NEU" })));
    await act(async () => a.resolve(qa({ request_id: "old", answer: "ALT" })));
    expect(result.current.state).toMatchObject({ phase: "done", result: { request_id: "new", answer: "NEU" } });
  });

  it("hook useAsk: response cũ LỖI về sau response mới cũng không hiện lỗi", async () => {
    const { useAsk } = await import("./hooks/useAsk");
    const { act } = await import("@testing-library/react");
    const a = deferred<QAResult>();
    const b = deferred<QAResult>();
    const q = [a, b];
    const { result } = renderHook(() => useAsk(fakeApi({ ask: () => q.shift()!.promise })));
    act(() => void result.current.ask({ question: "alt" }));
    act(() => void result.current.ask({ question: "neu" }));
    await act(async () => b.resolve(qa({ request_id: "new" })));
    await act(async () => a.reject(new ApiError("timeout", "x")));
    expect(result.current.state.phase).toBe("done");
  });

  it("nội dung corpus có HTML KHÔNG được thực thi hay dựng thành phần tử", async () => {
    const user = userEvent.setup();
    const evil = 'Vorher <img src=x onerror="window.__pwned=1"> <script>window.__pwned=2</script> Antwort & <b>fett</b> nachher';
    const start = evil.indexOf("Antwort");
    const api = fakeApi({ ask: async () => qa({ passage_text: evil, answer: "Antwort", answer_start_utf16: start, answer_end_utf16: start + 7 }) });
    const { container } = render(<App api={api} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Hỏi" })).toBeDisabled());
    await ask(user, "Frage");
    await screen.findByTestId("answer-mark");
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector(".passage b")).toBeNull();
    expect(container.querySelector(".passage")!.textContent).toContain("<img src=x");
    expect((window as unknown as Record<string, unknown>).__pwned).toBeUndefined();
  });

  it("câu hỏi quá dài thì khóa nút và báo", async () => {
    const user = userEvent.setup();
    render(<App api={fakeApi()} />);
    await waitFor(() => expect(screen.getByLabelText(/Câu hỏi/)).toBeInTheDocument());
    const box = screen.getByLabelText(/Câu hỏi/);
    await user.click(box);
    await user.paste("x".repeat(MAX_QUESTION_CHARS + 1));
    expect(screen.getByRole("alert")).toHaveTextContent(String(MAX_QUESTION_CHARS));
    expect(screen.getByRole("button", { name: "Hỏi" })).toBeDisabled();
  });

  it.each([
    ["busy", /bận/],
    ["timeout", /quá lâu/],
    ["network", /Không kết nối được/],
    ["empty_question", /nhập một câu hỏi/],
    ["mã_lạ", /Lỗi mã_lạ/],
  ])("lỗi %s hiện thông điệp rõ kèm mã yêu cầu", async (code, pattern) => {
    const user = userEvent.setup();
    render(<App api={fakeApi({ ask: async () => { throw new ApiError(code, "m", 503, "req-77"); } })} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Hỏi" })).toBeDisabled());
    await ask(user, "Frage");
    const alert = await screen.findByText(pattern);
    expect(alert.closest("[role=alert]")).toHaveTextContent("req-77");
  });

  it("không tìm thấy đáp án (no_valid_span) không hiện đoạn nguồn giả", async () => {
    const user = userEvent.setup();
    const api = fakeApi({ ask: async () => qa({ status: "no_valid_span", answer: "", passage_text: "", passage_id: "", trace: null }) });
    const { container } = render(<App api={api} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Hỏi" })).toBeDisabled());
    await ask(user, "Frage");
    expect(await screen.findByText(/Không tìm thấy đáp án/)).toBeInTheDocument();
    expect(container.querySelector(".passage")).toBeNull();
  });

  it("không có chỗ nào tự chấm EM/F1 cho câu hỏi mới", async () => {
    const user = userEvent.setup();
    const { container } = render(<App api={fakeApi()} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Hỏi" })).toBeDisabled());
    await ask(user, "Frage");
    await screen.findByTestId("answer-mark");
    expect(container.textContent).not.toMatch(/\bEM\b|\bF1\b/);
  });
});

describe("bộ chọn truy hồi và trạng thái sẵn sàng", () => {
  it("dense chỉ hiện khi backend báo available", async () => {
    const { unmount } = render(<App api={fakeApi()} />);
    const sel = await screen.findByRole("combobox");
    expect(within(sel).getAllByRole("option").map((o) => o.getAttribute("value"))).toEqual(["bm25", "dense"]);
    unmount();

    const noDense = readyBody({ retrievers: { bm25: { available: true, reason: null }, dense: { available: false, reason: "faiss_not_installed" } } });
    render(<App api={fakeApi({ getReady: async () => noDense })} />);
    await waitFor(() => expect(screen.getByText(/Truy hồi: BM25/)).toBeInTheDocument());
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(screen.queryByText(/Dense/)).toBeNull();
  });

  it("gửi đúng retriever đã chọn", async () => {
    const user = userEvent.setup();
    const calls: unknown[] = [];
    render(<App api={fakeApi({ ask: async (p) => (calls.push(p), qa()) })} />);
    await user.selectOptions(await screen.findByRole("combobox"), "dense");
    await ask(user, "Frage");
    await screen.findByTestId("answer-mark");
    expect(calls[0]).toMatchObject({ question: "Frage", retriever: "dense", trace: true });
  });

  it("đang nạp: khóa nút Hỏi và giải thích; hết nạp thì tự mở", async () => {
    let n = 0;
    const api = fakeApi({ getReady: async () => (++n < 2 ? readyBody({ status: "loading" }) : readyBody()) });
    render(<App api={api} />);
    expect(await screen.findByText(/đang nạp mô hình/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hỏi" })).toBeDisabled();
    await userEvent.setup().type(screen.getByLabelText(/Câu hỏi/), "Frage");
    expect(screen.getByRole("button", { name: "Hỏi" })).toBeDisabled();
  });

  it("useReady: thăm dò đến khi ready rồi dừng; mất kết nối thì thử lại", async () => {
    const seq = ["fail", "loading", "ready"];
    let calls = 0;
    const api = fakeApi({
      getReady: async () => {
        const s = seq[Math.min(calls++, seq.length - 1)];
        if (s === "fail") throw new ApiError("network", "x");
        return readyBody({ status: s as "loading" | "ready" });
      },
    });
    const { result } = renderHook(() => useReady(api, 10));
    await waitFor(() => expect(result.current.phase).toBe("ready"));
    const after = calls;
    await new Promise((r) => setTimeout(r, 60));
    expect(calls).toBe(after); // đã ready thì không thăm dò nữa
  });

  it("backend failed: báo lỗi nạp", async () => {
    render(<App api={fakeApi({ getReady: async () => readyBody({ status: "failed" }) })} />);
    expect(await screen.findByText(/không nạp được mô hình/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hỏi" })).toBeDisabled();
  });
});

describe("các đoạn ứng viên", () => {
  it("tải văn bản theo yêu cầu và tô sáng đáp án của từng đoạn bằng offset đổi sang UTF-16", async () => {
    const user = userEvent.setup();
    const calls: unknown[] = [];
    const api = fakeApi({ retrieve: async (p) => (calls.push(p), retrieved()) });
    render(<App api={api} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Hỏi" })).toBeDisabled());
    await ask(user, "Frage");
    await screen.findByText(/Các đoạn được truy hồi \(3\)/);
    expect(screen.getByText("được chọn")).toBeInTheDocument();
    expect(screen.getByText(/không đưa vào reader/)).toBeInTheDocument(); // đoạn hạng 3
    await user.click(screen.getByRole("button", { name: "Tải văn bản các đoạn" }));
    await screen.findAllByText("Văn bản đoạn");
    expect(calls[0]).toEqual({ question: "Wer?", retriever: "bm25", top_k: 3 });
    const marks = screen.getAllByTestId("answer-mark").map((m) => m.textContent);
    expect(marks).toContain("Hund");
    expect(marks.filter((m) => m === "Katze")).toHaveLength(2); // đáp án chính + đáp án trong danh sách ứng viên (đoạn emoji)
  });

  it("hỏi câu mới thì văn bản đã tải của câu cũ biến mất, và response retrieve cũ không hiện nhầm", async () => {
    const user = userEvent.setup();
    const slow = deferred<ReturnType<typeof retrieved>>();
    let asks = 0;
    const api = fakeApi({
      ask: async () => qa({ request_id: `r${++asks}` }),
      retrieve: () => slow.promise,
    });
    render(<App api={api} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Hỏi" })).toBeDisabled());
    await ask(user, "Frage 1");
    await user.click(await screen.findByRole("button", { name: "Tải văn bản các đoạn" }));
    await ask(user, "Frage 2"); // câu mới trong lúc retrieve của câu 1 còn đang bay
    await screen.findByRole("button", { name: "Tải văn bản các đoạn" });
    slow.resolve(retrieved());
    await new Promise((r) => setTimeout(r, 30));
    expect(screen.queryByText("Văn bản đoạn")).toBeNull();
  });
});

describe("trang số đo", () => {
  it("hiện dev và dev_unseen cạnh nhau, khoảng tin cậy, cảnh báo, phiên bản, và không có 'tốt nhất'", async () => {
    window.location.hash = "#benchmark";
    const { container } = render(<App api={fakeApi()} />);
    const dense = await screen.findByText("eval_dev_dense_20261104T121120Z_57716bd8");
    const card = dense.closest("article")!;
    const heads = within(card).getAllByRole("columnheader").map((h) => h.textContent);
    expect(heads).toEqual(["Số đo", "dev", "dev_unseen"]);
    expect(within(card).getByText(/Tập dev bị nhiễm/)).toBeInTheDocument();
    expect(card).toHaveTextContent("0,338"); // 0,3376 làm tròn
    expect(card).toHaveTextContent("[0,310; 0,360]");
    expect(card).toHaveTextContent("BM25 · EM");
    expect(card).toHaveTextContent("corpus abc123");
    expect(card).toHaveTextContent("2026-11-04 12:11 UTC");
    expect(container.textContent).toMatch(/1 file bị bỏ qua/);
    expect(container.textContent).not.toMatch(/tốt nhất:|best/i);
  });

  it("lỗi tải số đo hiện thông điệp", async () => {
    window.location.hash = "#benchmark";
    render(<App api={fakeApi({ getExperiments: async () => { throw new ApiError("network", "x"); } })} />);
    expect(await screen.findByText(/Không kết nối được/)).toBeInTheDocument();
  });

  it("thư mục kết quả rỗng", async () => {
    window.location.hash = "#benchmark";
    render(<App api={fakeApi({ getExperiments: async () => ({ ...experiments(), count: 0, skipped: 0, items: [] }) })} />);
    expect(await screen.findByText(/Chưa có thí nghiệm nào/)).toBeInTheDocument();
  });
});

describe("trang số đo: ma trận B0/B1/H0/H1 và gộp lần chạy trùng", () => {
  it("hiện ma trận theo từng tập, H1 chỉ ở dev_unseen, kèm cảnh báo và không xếp hạng 'tốt nhất'", async () => {
    window.location.hash = "#benchmark";
    const { container } = render(<App api={fakeApi()} />);
    const title = await screen.findByRole("heading", { name: "Ma trận B0 / B1 / H0 / H1" });
    const sec = title.closest("section")!;
    const unseen = within(sec).getByRole("table", { name: "Ma trận trên tập dev_unseen" });
    const dev = within(sec).getByRole("table", { name: "Ma trận trên tập dev" });
    expect(within(unseen).getAllByRole("row")).toHaveLength(4); // tiêu đề + B0, H0, H1
    expect(within(dev).getAllByRole("row")).toHaveLength(3); // tiêu đề + B0, H0: H1 không có toàn dev
    expect(within(dev).queryByText("H1")).toBeNull();
    expect(unseen).toHaveTextContent("0,889"); // Hit@5 của H0 trên dev_unseen
    expect(unseen).toHaveTextContent("[0,158; 0,261]"); // khoảng tin cậy EM của H0
    expect(sec).toHaveTextContent("234 câu");
    expect(sec).toHaveTextContent("H1 chỉ đo trên dev_unseen");
    expect(container.textContent).not.toMatch(/tốt nhất:|best/i);
  });

  it("lỗi tải ma trận không làm mất danh sách số đo", async () => {
    window.location.hash = "#benchmark";
    render(<App api={fakeApi({ getMatrix: async () => { throw new ApiError("network", "x"); } })} />);
    expect(await screen.findByText("eval_dev_dense_20261104T121120Z_57716bd8")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Ma trận B0 / B1 / H0 / H1" })).toBeNull();
  });

  it("ma trận rỗng thì không hiện khung ma trận", async () => {
    window.location.hash = "#benchmark";
    render(<App api={fakeApi({ getMatrix: async () => ({ count: 0, caveats: [], rows: [] }) })} />);
    expect(await screen.findByText("eval_dev_dense_20261104T121120Z_57716bd8")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Ma trận B0 / B1 / H0 / H1" })).toBeNull();
  });

  it("hybrid có số đo, mã H0 trên bản đại diện, và bản trùng cấu hình nằm trong mục gập", async () => {
    window.location.hash = "#benchmark";
    render(<App api={fakeApi({ getExperiments: async () => experimentsWithHybrid() })} />);
    const ref = (await screen.findAllByText("eval_dev_hybrid_20261107T135423Z_598ddf6f"))[0];
    const card = ref.closest("article")!;
    expect(card.closest("details")).toBeNull(); // bản đại diện luôn hiện
    expect(within(card).getByTitle("Số đo tham chiếu của mốc đã đóng băng")).toHaveTextContent("H0");
    expect(card).toHaveTextContent("Dense · EM"); // nhãn mới cho số đo dense trong tệp hybrid
    expect(within(card).getByText(/Hybrid dùng encoder/)).toBeInTheDocument();
    const dup = screen.getByText("eval_dev_hybrid_20261107T131435Z_598ddf6f").closest("article")!;
    expect(dup.closest("details")).not.toBeNull(); // bản trùng bị gập
    expect(screen.getByText(/1 lần chạy trùng cấu hình/)).toBeInTheDocument();
    expect(dup).toHaveTextContent("Trùng cấu hình với");
  });
});

describe("GĐ18: Hybrid và xếp lại (rerank)", () => {
  const hybridApi = (calls: AskParams[] = [], result: QAResult = qaReranked()) =>
    fakeApi({ getReady: async () => readyHybrid(), ask: async (p) => { calls.push(p); return result; } });
  const waitHybrid = async () => {
    const sel = await screen.findByRole("combobox");
    await waitFor(() => expect(sel).toHaveValue("hybrid"));
    return sel;
  };

  it("mặc định theo backend là Hybrid và liệt kê đủ ba kiểu truy hồi", async () => {
    render(<App api={hybridApi()} />);
    const sel = await waitHybrid();
    expect(within(sel).getAllByRole("option").map((o) => o.textContent)).toEqual(["Hybrid (BM25 + dense)", "BM25 (khớp từ)", "Dense (khớp nghĩa)"]);
  });

  it("backend không báo reranker thì KHÔNG có công tắc xếp lại", async () => {
    render(<App api={fakeApi()} />);
    await screen.findByRole("combobox");
    expect(screen.queryByLabelText(/Xếp lại bằng cross-encoder/)).toBeNull();
    expect(screen.queryByText("Xếp lại chỉ dùng với Hybrid")).toBeNull();
  });

  it("có reranker: hiện công tắc, bật thì gửi rerank: true cùng retriever", async () => {
    const user = userEvent.setup();
    const calls: AskParams[] = [];
    render(<App api={hybridApi(calls)} />);
    await waitHybrid();
    await user.click(await screen.findByLabelText(/Xếp lại bằng cross-encoder/));
    expect(screen.getByText(/Theo số đo GĐ15/)).toBeInTheDocument(); // cảnh báo đánh đổi hiện khi bật
    await ask(user, "Wer?");
    await screen.findByTestId("answer-mark");
    expect(calls[0]).toMatchObject({ retriever: "hybrid", rerank: true });
  });

  it("không bật công tắc thì không gửi cờ rerank", async () => {
    const user = userEvent.setup();
    const calls: AskParams[] = [];
    render(<App api={hybridApi(calls, qa())} />);
    await waitHybrid();
    await ask(user, "Wer?");
    await screen.findByTestId("answer-mark");
    expect(calls[0].rerank).toBeUndefined();
  });

  it("chọn BM25 thì không có công tắc xếp lại (chỉ hybrid -> cross-encoder)", async () => {
    const user = userEvent.setup();
    render(<App api={hybridApi()} />);
    const sel = await waitHybrid();
    await screen.findByLabelText(/Xếp lại bằng cross-encoder/);
    await user.selectOptions(sel, "bm25");
    await waitFor(() => expect(screen.queryByLabelText(/Xếp lại bằng cross-encoder/)).toBeNull());
  });

  it("đổi sang retriever không hỗ trợ rerank thì công tắc biến mất và cờ không được gửi", async () => {
    const user = userEvent.setup();
    const calls: AskParams[] = [];
    render(<App api={hybridApi(calls, qa())} />);
    const sel = await waitHybrid();
    await user.click(await screen.findByLabelText(/Xếp lại bằng cross-encoder/));
    await user.selectOptions(sel, "dense"); // reranker chỉ dùng được sau hybrid
    await waitFor(() => expect(screen.queryByLabelText(/Xếp lại bằng cross-encoder/)).toBeNull());
    expect(screen.getByText("Xếp lại chỉ dùng với Hybrid")).toBeInTheDocument();
    await ask(user, "Wer?");
    await screen.findByTestId("answer-mark");
    expect(calls[0]).toMatchObject({ retriever: "dense" });
    expect(calls[0].rerank).toBeUndefined();
  });

  it("kết quả xếp lại: hạng trước → sau, điểm xếp lại, thời gian xếp lại, hạng từng nhánh và ghi chú không phải xác suất", async () => {
    const user = userEvent.setup();
    render(<App api={hybridApi()} />);
    await waitHybrid();
    await user.click(await screen.findByLabelText(/Xếp lại bằng cross-encoder/));
    await ask(user, "Wer?");
    await screen.findByTestId("answer-mark");
    expect(screen.getByText("Điểm xếp lại (logit)").nextSibling).toHaveTextContent("4,50");
    expect(screen.getByText("Điểm truy hồi (RRF)").nextSibling).toHaveTextContent("0,081");
    expect(screen.getByText("Xếp lại", { selector: "th" }).nextSibling).toHaveTextContent("585");
    const items = screen.getAllByRole("listitem").filter((li) => li.querySelector(".cand-row"));
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent("#1"); // p3 lên đầu
    expect(items[0]).toHaveTextContent("(trước: #3)");
    expect(items[0]).toHaveTextContent("xếp lại 4,50");
    expect(items[0]).toHaveTextContent("0,081"); // điểm RRF giữ 3 chữ số (0,08 sẽ làm mất phân biệt giữa các đoạn)
    expect(items[0]).toHaveTextContent("BM25 #– · dense #1"); // p3 chỉ có ở nhánh dense
    expect(items[2]).toHaveTextContent("(trước: #1)"); // p1 tụt xuống cuối
    expect(screen.getByText(/đã được sắp theo điểm xếp lại/, { selector: "p" })).toHaveTextContent("không phải xác suất"); // câu riêng của trường hợp xếp lại
  });

  it("kết quả KHÔNG xếp lại giữ nguyên như trước: không có hàng/ô xếp lại", async () => {
    const user = userEvent.setup();
    render(<App api={fakeApi({ ask: async () => qa() })} />);
    await screen.findByRole("combobox");
    await ask(user, "Wer?");
    await screen.findByTestId("answer-mark");
    expect(screen.queryByText("Xếp lại", { selector: "th" })).toBeNull();
    expect(screen.queryByText("Điểm xếp lại (logit)")).toBeNull();
    expect(screen.getByText("Điểm truy hồi (BM25)")).toBeInTheDocument();
    expect(screen.queryByText(/trước: #/)).toBeNull();
  });
});
