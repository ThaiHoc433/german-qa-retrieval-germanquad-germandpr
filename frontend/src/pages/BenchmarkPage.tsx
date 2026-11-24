import { useEffect, useState } from "react";
import { api as defaultApi, isAbort } from "../api/client";
import type { Api, ExperimentItem, ExperimentList, MatrixResponse } from "../api/types";
import { fmt, fmtTime } from "../lib/format";
import { describeError, toApiError } from "../lib/messages";

const KIND_TITLE: Record<string, string> = {
  eval_dev: "Pipeline end-to-end, dev (BM25)",
  eval_dev_dense: "Pipeline end-to-end, dev (BM25 so với dense)",
  eval_dev_hybrid: "Pipeline end-to-end, dev (BM25, dense, hybrid)",
  eval_test: "Pipeline end-to-end, test (BM25)",
  pipeline: "Pipeline, lần chạy GĐ5",
  bm25: "BM25, retrieval",
  reader: "Huấn luyện reader",
  test: "Lần chạy thử đầu",
};

const METRIC_LABEL: Record<string, string> = {
  em: "EM", f1: "F1", em_all_refs: "EM (mọi đáp án)", f1_all_refs: "F1 (mọi đáp án)", "hit@1": "Hit@1", "hit@5": "Hit@5", "hit@20": "Hit@20", "mrr@10": "MRR@10", total_p50_ms: "Trễ tổng p50 (ms)",
};

function label(metric: string): string {
  if (metric.startsWith("bm25_")) return "BM25 · " + label(metric.slice(5));
  if (metric.startsWith("dense_")) return "Dense · " + label(metric.slice(6));
  return METRIC_LABEL[metric] ?? metric;
}

type State = { phase: "loading" } | { phase: "error"; message: string } | { phase: "done"; data: ExperimentList; matrix: MatrixResponse | null };

const MATRIX_METRICS: [string, string][] = [["hit@5", "Hit@5"], ["hit@20", "Hit@20"], ["em", "EM"], ["f1", "F1"]];
const SPLIT_TITLE: Record<string, string> = {
  dev_unseen: "dev_unseen: các câu hỏi encoder dense chưa thấy",
  dev: "dev: toàn bộ",
};

export function BenchmarkPage({ api = defaultApi }: { api?: Api }) {
  const [state, setState] = useState<State>({ phase: "loading" });
  useEffect(() => {
    const c = new AbortController();
    // Ma trận là phần bổ sung: lỗi tải ma trận không được làm mất danh sách số đo.
    Promise.all([api.getExperiments(c.signal), api.getMatrix(c.signal).catch((e) => { if (isAbort(e)) throw e; return null; })])
      .then(([data, matrix]) => setState({ phase: "done", data, matrix }))
      .catch((e) => {
        if (!isAbort(e)) setState({ phase: "error", message: describeError(toApiError(e)) });
      });
    return () => c.abort();
  }, [api]);

  if (state.phase === "loading") return <p className="card loading" role="status">Đang tải số đo…</p>;
  if (state.phase === "error") return <div className="card error" role="alert"><b>{state.message}</b></div>;

  const groups = new Map<string, ExperimentItem[]>();
  for (const it of state.data.items) groups.set(it.kind, [...(groups.get(it.kind) ?? []), it]);

  return (
    <div className="stack">
      <section className="card">
        <h2>Số đo đã lưu</h2>
        <p className="hint">
          Trang này chỉ <b>đọc</b> các file kết quả đã lưu khi chạy thí nghiệm; không tính lại và không xếp hạng "tốt nhất". Mỗi mục ghi rõ mô hình, corpus và thời điểm.
          Khoảng trong ngoặc vuông là khoảng tin cậy bootstrap 95% (chỉ có với EM/F1 của file <code>eval_*</code>).
          Hit@k, MRR không có khoảng ở đây.
        </p>
        <p className="hint">{state.data.count} thí nghiệm{state.data.skipped > 0 && `, ${state.data.skipped} file bị bỏ qua vì không phải kết quả thí nghiệm`}.</p>
      </section>
      {state.matrix && state.matrix.rows.length > 0 && <MatrixCard m={state.matrix} />}
      {state.data.items.length === 0 && <p className="card">Chưa có thí nghiệm nào trong thư mục kết quả.</p>}
      {[...groups.entries()].map(([kind, items]) => (
        <section key={kind} aria-labelledby={`k-${kind}`}>
          <h2 id={`k-${kind}`} className="group">{KIND_TITLE[kind] ?? kind}</h2>
          <div className="stack">{items.filter((it) => !it.duplicate_of).map((it) => <Experiment key={it.experiment_id} it={it} />)}</div>
          {items.some((it) => it.duplicate_of) && (
            <details className="dups">
              <summary>{items.filter((it) => it.duplicate_of).length} lần chạy trùng cấu hình (đã gộp)</summary>
              <div className="stack">{items.filter((it) => it.duplicate_of).map((it) => <Experiment key={it.experiment_id} it={it} />)}</div>
            </details>
          )}
        </section>
      ))}
    </div>
  );
}

function Experiment({ it }: { it: ExperimentItem }) {
  const splits = Object.keys(it.metrics);
  const metrics = [...new Set(splits.flatMap((s) => Object.keys(it.metrics[s])))];
  return (
    <article className="card exp">
      <header>
        <code className="eid">{it.experiment_id}</code>
        {it.code && <span className="code-badge" title="Số đo tham chiếu của mốc đã đóng băng">{it.code}</span>}
        {it.duplicate_of && <p className="hint">Trùng cấu hình với <code>{it.duplicate_of}</code> (có thể là lần chạy thử).</p>}
        <div className="hint">
          {fmtTime(it.created_at)}
          {it.retriever && <> · truy hồi {it.retriever}</>}
          {it.checkpoint && <> · mô hình <code>{it.checkpoint}</code></>}
          {it.corpus_version && <> · corpus <code>{it.corpus_version}</code></>}
        </div>
      </header>
      {it.caveats.map((c) => <p key={c} className="caveat" role="note">{c}</p>)}
      {metrics.length === 0 ? (
        <p className="dim">Không có số đo hiển thị được.</p>
      ) : (
        <div className="tw">
          <table>
            <thead>
              <tr><th>Số đo</th>{splits.map((s) => <th key={s}>{s}</th>)}</tr>
            </thead>
            <tbody>
              {metrics.map((m) => (
                <tr key={m}>
                  <th scope="row">{label(m)}</th>
                  {splits.map((s) => {
                    const v = it.metrics[s][m];
                    const ci = it.intervals[s]?.[m];
                    return (
                      <td key={s} className="num">
                        {v === undefined ? "–" : m.endsWith("_ms") ? fmt(v, 1) : fmt(v)}
                        {ci && <span className="ci"> [{fmt(ci[0])}; {fmt(ci[1])}]</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </article>
  );
}

function MatrixCard({ m }: { m: MatrixResponse }) {
  const splits = ["dev_unseen", "dev"];
  return (
    <section className="card" aria-labelledby="matrix-title">
      <h2 id="matrix-title">Ma trận B0 / B1 / H0 / H1</h2>
      <p className="hint">
        Mỗi hàng là một mốc đã đóng băng (manifest trong <code>baselines/</code>). Chỉ so các hàng trong cùng một bảng: cùng tập, cùng reader.
      </p>
      {m.caveats.map((c) => <p key={c} className="caveat" role="note">{c}</p>)}
      {splits.map((s) => {
        const rows = m.rows.filter((r) => r.splits[s]);
        if (rows.length === 0) return null;
        const n = rows[0].queries[s];
        return (
          <div key={s}>
            <h3 className="group">{SPLIT_TITLE[s] ?? s}{n !== undefined && <> · {n} câu</>}</h3>
            <div className="tw">
              <table aria-label={`Ma trận trên tập ${s}`}>
                <thead>
                  <tr><th>Mã</th><th>Cấu hình</th>{MATRIX_METRICS.map(([k, l]) => <th key={k}>{l}</th>)}</tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.code}>
                      <th scope="row"><code>{r.code}</code></th>
                      <td>{r.description}</td>
                      {MATRIX_METRICS.map(([k]) => {
                        const v = r.splits[s][k];
                        const ci = k === "em" ? r.intervals[s]?.em : undefined;
                        return (
                          <td key={k} className="num">
                            {v === undefined ? "–" : fmt(v)}
                            {ci && <span className="ci"> [{fmt(ci[0])}; {fmt(ci[1])}]</span>}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </section>
  );
}
