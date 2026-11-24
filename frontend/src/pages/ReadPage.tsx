import { useEffect, useRef, useState } from "react";
import { ApiError, api as defaultApi, isAbort } from "../api/client";
import type { Api, ReadResult } from "../api/types";
import { HighlightedPassage } from "../components/HighlightedPassage";
import type { ReadyState } from "../hooks/useReady";
import { fmt, ms } from "../lib/format";
import { describeError, isRetryable, toApiError } from "../lib/messages";
import { MAX_QUESTION_CHARS } from "./AskPage";

export const MAX_CONTEXT_CHARS = 20000; // khớp max_context_chars trong configs/api.yaml; máy chủ vẫn là nơi quyết định

const EXAMPLES = [
  {
    label: "Berliner Mauer",
    question: "Wann wurde die Berliner Mauer gebaut?",
    context: "Die Berliner Mauer wurde am 13. August 1961 errichtet und trennte West-Berlin von Ost-Berlin.",
    answer: "Am 13. August 1961",
  },
  {
    label: "Faust",
    question: "Wer schrieb das Drama Faust?",
    context: "Johann Wolfgang von Goethe schrieb das Drama Faust, dessen erster Teil 1808 erschien.",
    answer: "Johann Wolfgang von Goethe",
  },
];

type State =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "done"; result: ReadResult }
  | { phase: "error"; error: ApiError };

/** Đọc hiểu: người dùng tự đưa đoạn văn, bỏ qua truy hồi; chỉ reader chạy. Có đáp án mong đợi thì hiện EM/F1. */
export function ReadPage({ ready, api = defaultApi }: { ready: ReadyState; api?: Api }) {
  const [question, setQuestion] = useState("");
  const [context, setContext] = useState("");
  const [expected, setExpected] = useState("");
  const [state, setState] = useState<State>({ phase: "idle" });
  const seq = useRef(0);
  const ctrl = useRef<AbortController | null>(null);
  useEffect(() => () => ctrl.current?.abort(), []);

  const qTooLong = question.length > MAX_QUESTION_CHARS;
  const cTooLong = context.length > MAX_CONTEXT_CHARS;
  const canSubmit = ready.phase === "ready" && question.trim() !== "" && context.trim() !== "" && !qTooLong && !cTooLong;

  async function submit(e?: { preventDefault(): void }) {
    e?.preventDefault();
    if (!canSubmit) return;
    ctrl.current?.abort();
    const mine = ++seq.current;
    const c = new AbortController();
    ctrl.current = c;
    setState({ phase: "loading" });
    try {
      const exp = expected.trim();
      const result = await api.read({ question: question.trim(), context, ...(exp ? { expected_answer: exp } : {}) }, c.signal);
      if (mine === seq.current) setState({ phase: "done", result });
    } catch (err) {
      if (mine !== seq.current || isAbort(err)) return;
      setState({ phase: "error", error: toApiError(err) });
    }
  }

  const ctrlEnter = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(e);
  };

  return (
    <div className="stack">
      <form className="card" onSubmit={submit}>
        <p className="hint">Tự đưa đoạn văn: hệ thống bỏ qua bước truy hồi, chỉ reader đọc đoạn này và trích đáp án.</p>
        <label htmlFor="rq">Câu hỏi (tiếng Đức)</label>
        <textarea id="rq" lang="de" rows={2} value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={ctrlEnter} />
        <span className={qTooLong ? "count over" : "count"}>{question.length}/{MAX_QUESTION_CHARS}</span>

        <label htmlFor="rc">Đoạn văn (context)</label>
        <textarea id="rc" lang="de" rows={6} value={context} onChange={(e) => setContext(e.target.value)} onKeyDown={ctrlEnter} />
        <span className={cTooLong ? "count over" : "count"}>{context.length}/{MAX_CONTEXT_CHARS}</span>

        <label htmlFor="ra">Đáp án mong đợi (không bắt buộc, để tính EM/F1)</label>
        <input id="ra" type="text" lang="de" value={expected} onChange={(e) => setExpected(e.target.value)} />

        <div className="row between">
          <div className="examples">
            <span className="dim">Thử:</span>
            {EXAMPLES.map((ex) => (
              <button key={ex.label} type="button" className="chip" onClick={() => { setQuestion(ex.question); setContext(ex.context); setExpected(ex.answer); }}>
                {ex.label}
              </button>
            ))}
          </div>
          <button type="submit" disabled={!canSubmit}>Đọc</button>
        </div>
        {qTooLong && <p className="error-inline" role="alert">Câu hỏi dài quá {MAX_QUESTION_CHARS} ký tự.</p>}
        {cTooLong && <p className="error-inline" role="alert">Đoạn văn dài quá {MAX_CONTEXT_CHARS} ký tự.</p>}
        {ready.phase !== "ready" && <p className="notice" role="status">Đợi máy chủ sẵn sàng thì nút Đọc mới bật.</p>}
      </form>

      <div aria-live="polite">
        {state.phase === "loading" && <p className="card loading" role="status">Reader đang đọc đoạn văn…</p>}
        {state.phase === "error" && (
          <div className="card error" role="alert">
            <b>{describeError(state.error)}</b>
            {isRetryable(state.error) && <p className="hint">Có thể thử lại ngay.</p>}
          </div>
        )}
      </div>
      {state.phase === "done" && <ReadAnswer r={state.result} />}
    </div>
  );
}

function ReadAnswer({ r }: { r: ReadResult }) {
  const found = r.status !== "no_valid_span" && r.answer !== "";
  return (
    <section className="card answer" aria-labelledby="read-h">
      <h2 id="read-h">Đáp án</h2>
      {found ? (
        <p className="answer-text" lang="de">{r.answer}</p>
      ) : (
        <p className="answer-text none">Reader không tìm được khoảng đáp án hợp lệ trong đoạn này.</p>
      )}
      {r.status === "question_truncated" && <p className="hint">Câu hỏi dài hơn 64 token nên đã bị cắt trước khi đưa vào reader.</p>}
      {found && <HighlightedPassage text={r.context} startUtf16={r.answer_start_utf16} endUtf16={r.answer_end_utf16} />}

      {r.expected_answer && r.em != null && r.f1 != null && (
        <div className={r.em === 1 ? "verdict ok" : "verdict bad"} data-testid="verdict">
          <b>{r.em === 1 ? "Khớp đáp án mong đợi" : "Chưa khớp đáp án mong đợi"}</b>
          <span> · mong đợi <q lang="de">{r.expected_answer}</q> · EM {fmt(r.em, 0)} · F1 {fmt(r.f1, 2)}</span>
        </div>
      )}
      <p className="hint">EM/F1 tính như lúc đánh giá: chữ thường, bỏ dấu câu, gộp khoảng trắng.</p>

      <dl className="meta">
        <div><dt>Điểm reader</dt><dd>{found && r.reader_score != null ? fmt(r.reader_score, 2) : "–"}</dd></div>
        <div><dt>Reader</dt><dd>{ms(r.timings.reader_ms)}</dd></div>
        <div><dt>Tổng (phía máy chủ)</dt><dd>{ms(r.timings.total_ms)}</dd></div>
      </dl>
      <p className="hint">Mô hình <code>{r.model}</code> · mã yêu cầu <code>{r.request_id}</code></p>
    </section>
  );
}
