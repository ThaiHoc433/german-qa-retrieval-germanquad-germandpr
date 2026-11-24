import { useEffect, useState } from "react";
import { api as defaultApi } from "../api/client";
import type { Api, QAResult, RetrieverName } from "../api/types";
import { CandidatesPanel } from "../components/CandidatesPanel";
import { HighlightedPassage } from "../components/HighlightedPassage";
import { availableRetrievers, rerankAvailable, type ReadyState } from "../hooks/useReady";
import { useAsk } from "../hooks/useAsk";
import { fmt, ms } from "../lib/format";
import { describeError, isRetryable } from "../lib/messages";

export const MAX_QUESTION_CHARS = 500; // khớp max_question_chars trong configs/api.yaml; máy chủ vẫn là nơi quyết định

const EXAMPLES = [
  "Welcher Architekt entwarf den Connaught Place in Neu-Delhi?",
  "Wann wurde Hundefett in der Medizin verwendet?",
  "Was ist die höchste Stelle der Lhasa-Bahn-Strecke?",
  "Welche Industrie war im 17. und 18.Jhd. in Cork weit verbreitet?",
];

const RETRIEVER_LABEL: Record<string, string> = { hybrid: "Hybrid (BM25 + dense)", bm25: "BM25 (khớp từ)", dense: "Dense (khớp nghĩa)" };

// Tên đúng của điểm truy hồi theo bộ truy hồi: ba thang khác nhau, không so với nhau và không phải xác suất.
const SCORE_LABEL: Record<string, string> = { bm25: "Điểm truy hồi (BM25)", dense: "Điểm truy hồi (tích vô hướng)", hybrid: "Điểm truy hồi (RRF)" };

export function AskPage({ ready, api = defaultApi }: { ready: ReadyState; api?: Api }) {
  const { state, ask } = useAsk(api);
  const [question, setQuestion] = useState("");
  const [retriever, setRetriever] = useState<string>("");
  const [rerank, setRerank] = useState(false);

  const info = ready.phase === "ready" ? ready.info : null;
  const options = availableRetrievers(info);
  // Chọn mặc định theo backend; nếu lựa chọn hiện tại không còn khả dụng thì quay về mặc định.
  useEffect(() => {
    if (info && !options.includes(retriever)) setRetriever(options.includes(info.default_retriever) ? info.default_retriever : options[0] ?? "");
  }, [info, options.join(","), retriever]);

  const canRerank = rerankAvailable(info, retriever);
  // Công tắc chỉ có nghĩa khi backend có reranker cho đúng retriever đang chọn; đổi sang retriever không hỗ trợ thì tự tắt.
  useEffect(() => {
    if (!canRerank && rerank) setRerank(false);
  }, [canRerank, rerank]);

  const trimmed = question.trim();
  const tooLong = question.length > MAX_QUESTION_CHARS;
  // Đang chờ câu trước vẫn được hỏi câu mới: yêu cầu cũ bị hủy và response cũ (nếu vẫn về) bị bỏ qua (useAsk).
  const canSubmit = ready.phase === "ready" && trimmed.length > 0 && !tooLong;

  function submit(e?: { preventDefault(): void }) {
    e?.preventDefault();
    if (!canSubmit) return;
    ask({ question: trimmed, retriever: (retriever || undefined) as RetrieverName | undefined, rerank: rerank && canRerank ? true : undefined });
  }

  return (
    <div className="stack">
      <form className="card" onSubmit={submit}>
        <label htmlFor="q">Câu hỏi (tiếng Đức)</label>
        <textarea
          id="q"
          lang="de"
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(e);
          }}
          placeholder="Ví dụ: Wer entwarf den Connaught Place?"
          aria-describedby="q-count"
        />
        <div className="row between">
          <span id="q-count" className={tooLong ? "count over" : "count"}>
            {question.length}/{MAX_QUESTION_CHARS}
          </span>
          <div className="row">
            {options.length > 1 ? (
              <label className="inline">
                Truy hồi
                <select value={retriever} onChange={(e) => setRetriever(e.target.value)}>
                  {options.map((o) => (
                    <option key={o} value={o}>{RETRIEVER_LABEL[o] ?? o}</option>
                  ))}
                </select>
              </label>
            ) : (
              options.length === 1 && <span className="dim">Truy hồi: {RETRIEVER_LABEL[options[0]] ?? options[0]}</span>
            )}
            {!canRerank && info?.reranker?.available && (
              <span className="dim">Xếp lại chỉ dùng với Hybrid</span>
            )}
            {canRerank && (
              <label className="inline">
                <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
                Xếp lại bằng cross-encoder
              </label>
            )}
            <button type="submit" disabled={!canSubmit}>
              Hỏi
            </button>
          </div>
        </div>
        {tooLong && <p className="error-inline" role="alert">Câu hỏi dài quá {MAX_QUESTION_CHARS} ký tự.</p>}
        {canRerank && rerank && (
          <p className="hint">
            Theo số đo GĐ15, xếp lại tìm thêm được đoạn đúng nhưng đáp án cuối kém hơn một chút và chậm hơn nhiều lần. Bật khi muốn xem hạng trước/sau.
          </p>
        )}
        <div className="examples">
          <span className="dim">Thử:</span>
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" className="chip" lang="de" onClick={() => setQuestion(ex)}>{ex}</button>
          ))}
        </div>
        {ready.phase !== "ready" && <ReadyNotice ready={ready} />}
      </form>

      <div aria-live="polite">
        {state.phase === "loading" && <p className="card loading" role="status">Đang tìm đáp án cho “{state.question}”…</p>}
        {state.phase === "error" && (
          <div className="card error" role="alert">
            <b>{describeError(state.error)}</b>
            {isRetryable(state.error) && <p className="hint">Có thể thử lại ngay.</p>}
            {state.error.requestId && <p className="hint">Mã yêu cầu: <code>{state.error.requestId}</code> (tìm trong log máy chủ)</p>}
          </div>
        )}
      </div>
      {state.phase === "done" && <Answer result={state.result} api={api} />}
    </div>
  );
}

function ReadyNotice({ ready }: { ready: ReadyState }) {
  const text: Record<string, string> = {
    checking: "Đang kiểm tra máy chủ…",
    loading: "Máy chủ đang nạp mô hình (khoảng 20 giây). Nút Hỏi sẽ bật khi sẵn sàng.",
    failed: "Máy chủ không nạp được mô hình. Xem log máy chủ.",
    unreachable: "Không kết nối được tới máy chủ. Đang thử lại…",
  };
  return <p className="notice" role="status">{text[ready.phase]}</p>;
}

function Answer({ result, api }: { result: QAResult; api: Api }) {
  const found = result.answer !== "" && (result.status === "ok" || result.status === "question_truncated");
  const t = result.timings;
  return (
    <>
      <section className="card answer" aria-labelledby="ans-h">
        <h2 id="ans-h">Đáp án</h2>
        {found ? (
          <p className="answer-text" lang="de">{result.answer}</p>
        ) : (
          <p className="answer-text none">Không tìm thấy đáp án ({STATUS_TEXT[result.status] ?? result.status}).</p>
        )}
        {result.status === "question_truncated" && <p className="hint">Câu hỏi dài hơn 64 token nên đã bị cắt trước khi đưa vào reader.</p>}
        <p className="hint">Đáp án là một đoạn trích từ văn bản nguồn bên dưới. Hệ thống không biết nó đúng hay sai, hãy đối chiếu với đoạn nguồn.</p>
        {found && (
          <>
            <h3>Đoạn nguồn</h3>
            <HighlightedPassage text={result.passage_text} startUtf16={result.answer_start_utf16} endUtf16={result.answer_end_utf16} />
          </>
        )}
        <dl className="meta">
          <div><dt>Truy hồi</dt><dd>{result.retriever}</dd></div>
          {result.reranked && <div><dt>Xếp lại</dt><dd>cross-encoder</dd></div>}
          <div><dt>Điểm reader</dt><dd>{found ? fmt(result.reader_score, 2) : "–"}</dd></div>
          <div><dt>{SCORE_LABEL[result.retriever] ?? "Điểm truy hồi"}</dt><dd>{found ? fmt(result.retrieval_score, result.retriever === "hybrid" ? 3 : 2) : "–"}</dd></div>
          {result.reranked && <div><dt>Điểm xếp lại (logit)</dt><dd>{found && result.rerank_score != null ? fmt(result.rerank_score, 2) : "–"}</dd></div>}
        </dl>
      </section>

      <section className="card" aria-labelledby="time-h">
        <h2 id="time-h">Thời gian</h2>
        <table className="kv">
          <tbody>
            <tr><th>Truy hồi</th><td>{ms(t.retrieval_ms)}</td></tr>
            {result.reranked && <tr><th>Xếp lại</th><td>{ms(t.rerank_ms ?? 0)}</td></tr>}
            <tr><th>Reader</th><td>{ms(t.reader_ms)}</td></tr>
            <tr><th>Chờ đến lượt</th><td>{ms(t.queue_ms)}</td></tr>
            <tr><th>Tổng (phía máy chủ)</th><td>{ms(t.total_ms)}</td></tr>
          </tbody>
        </table>
        <p className="hint">
          Mô hình <code>{result.versions.model}</code> · corpus <code>{result.versions.corpus}</code> · mã yêu cầu <code>{result.request_id}</code>
        </p>
      </section>

      <CandidatesPanel result={result} api={api} />
    </>
  );
}

const STATUS_TEXT: Record<string, string> = {
  no_passages: "truy hồi không trả đoạn nào",
  no_valid_span: "reader không tìm được khoảng hợp lệ trong các đoạn",
  empty_question: "câu hỏi trống",
};
