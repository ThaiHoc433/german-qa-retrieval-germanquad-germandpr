// Chạy với backend THẬT (chỉ khi đặt E2E_API), ví dụ:  E2E_API=http://127.0.0.1:8000 npm test
// Kiểm hợp đồng giữa kiểu TypeScript, giao diện và API Python: mọi test khác dùng API giả.
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { createApi } from "./api/client";

const base = process.env.E2E_API;

describe.runIf(base)("với backend thật", () => {
  const api = createApi(base);

  it("/health/ready đúng dạng và dense khả dụng", async () => {
    const r = await api.getReady();
    expect(r.status).toBe("ready");
    expect(Object.keys(r.retrievers).sort()).toEqual(["bm25", "dense", "hybrid"]); // GĐ18 thêm hybrid
  });

  it("hỏi qua giao diện: đáp án, highlight trong đoạn nguồn, ứng viên và văn bản đoạn", async () => {
    const user = userEvent.setup();
    render(<App api={api} />);
    await waitFor(() => expect(screen.getByRole("combobox")).toBeInTheDocument(), { timeout: 60000 });
    await user.selectOptions(screen.getByRole("combobox"), "dense");
    await user.type(screen.getByLabelText(/Câu hỏi/), "Welcher Architekt entwarf den Connaught Place in Neu-Delhi?");
    await user.click(screen.getByRole("button", { name: "Hỏi" }));
    const mark = await screen.findByTestId("answer-mark", {}, { timeout: 60000 });
    expect(mark).toHaveTextContent("Robert Tor Russell");
    expect(mark.closest(".passage")!.textContent).toContain("Chefarchitekten der indischen Regierung");

    await screen.findByText(/Các đoạn được truy hồi \(20\)/);
    await user.click(screen.getByRole("button", { name: "Tải văn bản các đoạn" }));
    const details = await screen.findAllByText("Văn bản đoạn", {}, { timeout: 60000 });
    expect(details.length).toBe(20);
    const chosen = document.querySelector(".cands li.chosen")!;
    expect(within(chosen as HTMLElement).getByTestId("answer-mark")).toHaveTextContent("Robert Tor Russell");
  }, 120000);

  it("trang số đo đọc /experiments thật, có khoảng tin cậy và cả dev lẫn dev_unseen nếu có file dense", async () => {
    const data = await api.getExperiments();
    expect(data.count).toBeGreaterThan(0);
    expect(data.items.every((i) => typeof i.experiment_id === "string" && typeof i.metrics === "object")).toBe(true);
    expect(JSON.stringify(data)).not.toMatch(/[A-Za-z]:\\/); // không lộ đường dẫn Windows
    expect(data.items.some((i) => Object.keys(i.intervals).length > 0)).toBe(true);
  });

  it("đọc đoạn văn tự nhập: reader thật trích đúng ngày và EM = 1", async () => {
    const r = await api.read({
      question: "Wann wurde die Berliner Mauer gebaut?",
      context: "Die Berliner Mauer wurde am 13. August 1961 errichtet und trennte West-Berlin von Ost-Berlin.",
      expected_answer: "Am 13. August 1961",
    });
    expect(r.status).toBe("ok");
    expect(r.context.slice(r.answer_start, r.answer_end)).toBe(r.answer);
    expect(r.em).toBe(1);
  }, 60000);

  it("lỗi thật: câu hỏi trống -> 422 empty_question", async () => {
    await expect(api.ask({ question: "   " })).rejects.toMatchObject({ code: "empty_question", status: 422 });
  });
});
