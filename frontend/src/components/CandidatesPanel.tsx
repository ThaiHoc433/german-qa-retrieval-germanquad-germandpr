import { useEffect, useRef, useState } from "react";
import { api as defaultApi, isAbort } from "../api/client";
import type { Api, Candidate, QAResult } from "../api/types";
import { utf16FromCodePoint } from "../lib/highlight";
import { describeError, toApiError } from "../lib/messages";
import { fmt } from "../lib/format";
import { HighlightedPassage } from "./HighlightedPassage";

type Texts =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "done"; forId: string; byId: Record<string, Candidate> }
  | { phase: "error"; message: string };

/**
 * Các đoạn được truy hồi và đáp án reader tìm được trong từng đoạn (từ `trace` của /qa).
 * `trace` không chứa văn bản đoạn nên văn bản được tải THEO YÊU CẦU qua /retrieve; kết quả gắn với request_id của
 * lần hỏi hiện tại, nên lần hỏi mới không bao giờ hiện nhầm văn bản của lần cũ.
 */
export function CandidatesPanel({ result, api = defaultApi }: { result: QAResult; api?: Api }) {
  const [texts, setTexts] = useState<Texts>({ phase: "idle" });
  const current = useRef(result.request_id);
  const ctrl = useRef<AbortController | null>(null);

  useEffect(() => {
    current.current = result.request_id;
    ctrl.current?.abort();
    setTexts({ phase: "idle" });
  }, [result.request_id]);
  useEffect(() => () => ctrl.current?.abort(), []);

  const trace = result.trace;
  if (!trace || trace.retrieved.length === 0) return null;
  const read = new Map(trace.candidates.map((c) => [c.passage_id, c]));
  const rr = trace.reranked ?? null; // GĐ18: có khi đã xếp lại bằng cross-encoder
  const rrById = new Map((rr ?? []).map((x) => [x.passage_id, x]));
  const byId = new Map(trace.retrieved.map((h) => [h.passage_id, h]));
  // Thứ tự hiển thị là thứ tự CUỐI CÙNG (sau xếp lại nếu có), vì reader đọc các đoạn đầu của thứ tự đó.
  const rows = rr ? rr.map((x) => byId.get(x.passage_id)).filter((h): h is NonNullable<typeof h> => h !== undefined) : trace.retrieved;
  const digits = result.retriever === "hybrid" ? 3 : 2;
  const scoreName = result.retriever === "bm25" ? "BM25" : result.retriever === "hybrid" ? "RRF (tổng 1/(c + hạng) của hai nhánh)" : "dense (tích vô hướng)";

  async function loadTexts() {
    const mine = result.request_id;
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    setTexts({ phase: "loading" });
    try {
      const r = await api.retrieve({ question: result.question, retriever: result.retriever, top_k: trace!.retrieved.length }, c.signal);
      if (current.current !== mine) return;
      setTexts({ phase: "done", forId: mine, byId: Object.fromEntries(r.candidates.map((x) => [x.passage_id, x])) });
    } catch (e) {
      if (isAbort(e) || current.current !== mine) return;
      setTexts({ phase: "error", message: describeError(toApiError(e)) });
    }
  }

  return (
    <section className="card" aria-labelledby="cand-h">
      <h2 id="cand-h">Các đoạn được truy hồi ({trace.retrieved.length})</h2>
      <p className="hint">
        Reader chỉ đọc {trace.candidates.length} đoạn đầu. Điểm truy hồi là điểm {scoreName}; các thang điểm khác nhau không so được với nhau và không phải xác suất.
        {rr && " Danh sách đã được sắp theo điểm xếp lại (logit của cross-encoder, không phải xác suất); reader chỉ nhận các đoạn đầu sau khi xếp lại."}
      </p>
      {texts.phase !== "done" && (
        <button type="button" className="secondary" onClick={loadTexts} disabled={texts.phase === "loading"}>
          {texts.phase === "loading" ? "Đang tải…" : "Tải văn bản các đoạn"}
        </button>
      )}
      {texts.phase === "error" && <p className="error-inline" role="alert">{texts.message}</p>}
      <ol className="cands">
        {rows.map((h) => {
          const c = read.get(h.passage_id);
          const x = rrById.get(h.passage_id);
          const chosen = h.passage_id === result.passage_id;
          const full = texts.phase === "done" && texts.forId === result.request_id ? texts.byId[h.passage_id] : undefined;
          let s = -1;
          let e = -1;
          if (full && c && c.answer) {
            s = utf16FromCodePoint(full.text, c.start);
            e = utf16FromCodePoint(full.text, c.end);
          }
          return (
            <li key={h.passage_id} className={chosen ? "chosen" : undefined}>
              <div className="cand-row">
                <span className="rank">#{x ? x.rank : h.rank}</span>
                {x && <span className="dim" title="hạng truy hồi trước khi xếp lại">(trước: #{x.rank_before})</span>}
                <code className="pid" title={h.passage_id}>{h.passage_id.slice(0, 10)}…</code>
                <span className="num" title="điểm truy hồi">{fmt(h.score, digits)}</span>
                {x && <span className="num dim" title="điểm xếp lại (logit cross-encoder)">xếp lại {fmt(x.rerank_score, 2)}</span>}
                {h.sources && <span className="dim" title="hạng trong từng nhánh">BM25 #{h.sources.bm25 ?? "–"} · dense #{h.sources.dense ?? "–"}</span>}
                {c ? (
                  <span className="cand-ans">
                    {c.answer ? <>“{c.answer}” <span className="num dim">reader {c.reader_score === null ? "–" : fmt(c.reader_score, 2)}</span></> : <span className="dim">không có đáp án hợp lệ</span>}
                  </span>
                ) : (
                  <span className="dim">không đưa vào reader</span>
                )}
                {chosen && <span className="badge">được chọn</span>}
              </div>
              {full && (
                <details>
                  <summary>Văn bản đoạn</summary>
                  <HighlightedPassage text={full.text} startUtf16={s} endUtf16={e} />
                </details>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
