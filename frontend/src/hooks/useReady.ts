import { useEffect, useState } from "react";
import { api as defaultApi, isAbort } from "../api/client";
import type { Api, Ready } from "../api/types";

export type ReadyState =
  | { phase: "checking" }
  | { phase: "loading"; info: Ready }
  | { phase: "ready"; info: Ready }
  | { phase: "failed"; info: Ready }
  | { phase: "unreachable" };

/** Hỏi /health/ready lúc mở trang và cứ mỗi `intervalMs` cho đến khi ready hoặc failed. Đang nạp (~20 s) là bình thường. */
export function useReady(api: Api = defaultApi, intervalMs = 2000): ReadyState {
  const [state, setState] = useState<ReadyState>({ phase: "checking" });

  useEffect(() => {
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      try {
        const info = await api.getReady(ctrl.signal);
        setState({ phase: info.status, info });
        if (info.status === "loading") timer = setTimeout(tick, intervalMs);
      } catch (e) {
        if (isAbort(e)) return;
        setState({ phase: "unreachable" });
        timer = setTimeout(tick, intervalMs);
      }
    };
    tick();
    return () => {
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [api, intervalMs]);

  return state;
}

/** Có dùng được xếp lại (rerank) với retriever này không: chỉ khi backend báo reranker khả dụng cho đúng retriever đó. */
export function rerankAvailable(info: Ready | null, retriever: string): boolean {
  return !!info?.reranker?.available && info.reranker.retrievers.includes(retriever);
}

/** Kiểu truy hồi nào được phép chọn: chỉ những cái backend báo available. BM25 luôn có mặt như lựa chọn mặc định nếu ready. */
export function availableRetrievers(info: Ready | null): string[] {
  if (!info) return [];
  return Object.entries(info.retrievers).filter(([, r]) => r.available).map(([name]) => name);
}
