import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { ApiError } from "./api/client";
import type { ReadParams } from "./api/types";
import { BERLIN_CTX, fakeApi, readResult } from "./test/fixtures";

beforeEach(() => {
  window.location.hash = "#read";
});

const fill = async (user: ReturnType<typeof userEvent.setup>, q: string, c: string, a = "") => {
  await user.type(screen.getByLabelText(/Câu hỏi/), q);
  await user.type(screen.getByLabelText(/Đoạn văn \(context\)/), c);
  if (a) await user.type(screen.getByLabelText(/Đáp án mong đợi/), a);
};

describe("Đọc đoạn văn (context tự nhập)", () => {
  it("có tab riêng, gửi đúng question/context/expected_answer và hiện đáp án tô sáng + EM/F1", async () => {
    const user = userEvent.setup();
    const calls: ReadParams[] = [];
    render(<App api={fakeApi({ read: async (p) => { calls.push(p); return readResult(); } })} />);
    expect(screen.getByRole("link", { name: "Đọc đoạn văn" })).toHaveAttribute("aria-current", "page");
    const btn = screen.getByRole("button", { name: "Đọc" });
    await waitFor(() => expect(btn).toBeDisabled());
    await fill(user, "Wann wurde die Berliner Mauer gebaut?", BERLIN_CTX, "Am 13. August 1961");
    await user.click(btn);
    expect(await screen.findByTestId("answer-mark")).toHaveTextContent("am 13. August 1961");
    expect(calls).toEqual([{ question: "Wann wurde die Berliner Mauer gebaut?", context: BERLIN_CTX, expected_answer: "Am 13. August 1961" }]);
    const v = screen.getByTestId("verdict");
    expect(v).toHaveTextContent("Khớp đáp án mong đợi");
    expect(v).toHaveTextContent("EM 1");
    expect(v).toHaveTextContent("F1 1,00");
  });

  it("nút ví dụ điền sẵn cả ba ô; không nhập đáp án mong đợi thì không gửi và không hiện EM", async () => {
    const user = userEvent.setup();
    const calls: ReadParams[] = [];
    render(<App api={fakeApi({ read: async (p) => { calls.push(p); return readResult({ expected_answer: null, em: null, f1: null }); } })} />);
    await user.click(await screen.findByRole("button", { name: "Berliner Mauer" }));
    expect(screen.getByLabelText(/Đoạn văn \(context\)/)).toHaveValue(BERLIN_CTX);
    await user.clear(screen.getByLabelText(/Đáp án mong đợi/));
    await waitFor(() => expect(screen.getByRole("button", { name: "Đọc" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Đọc" }));
    await screen.findByTestId("answer-mark");
    expect(calls[0].expected_answer).toBeUndefined();
    expect(screen.queryByTestId("verdict")).toBeNull();
  });

  it("sai đáp án mong đợi thì báo chưa khớp; không có span thì báo không tìm được", async () => {
    const user = userEvent.setup();
    const results = [readResult({ expected_answer: "1962", em: 0, f1: 0 }), readResult({ status: "no_valid_span", answer: "", answer_start: -1, answer_end: -1, answer_start_utf16: -1, answer_end_utf16: -1, reader_score: null, expected_answer: null, em: null, f1: null })];
    render(<App api={fakeApi({ read: async () => results.shift()! })} />);
    await fill(user, "Wann?", BERLIN_CTX, "1962");
    await waitFor(() => expect(screen.getByRole("button", { name: "Đọc" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Đọc" }));
    expect(await screen.findByTestId("verdict")).toHaveTextContent("Chưa khớp");
    await user.click(screen.getByRole("button", { name: "Đọc" }));
    expect(await screen.findByText(/không tìm được khoảng đáp án/)).toBeInTheDocument();
    expect(screen.queryByTestId("answer-mark")).toBeNull();
  });

  it("thiếu đoạn văn thì không cho gửi; lỗi máy chủ hiện thông điệp tiếng Việt", async () => {
    const user = userEvent.setup();
    render(<App api={fakeApi({ read: async () => { throw new ApiError("context_too_long", "x", 422); } })} />);
    await user.type(screen.getByLabelText(/Câu hỏi/), "Wann?");
    expect(screen.getByRole("button", { name: "Đọc" })).toBeDisabled();
    await user.type(screen.getByLabelText(/Đoạn văn \(context\)/), "Text");
    await waitFor(() => expect(screen.getByRole("button", { name: "Đọc" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Đọc" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Đoạn văn quá dài.");
  });
});
