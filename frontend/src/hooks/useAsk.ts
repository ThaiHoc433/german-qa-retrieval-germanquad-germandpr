import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api as defaultApi, isAbort } from "../api/client";
import type { Api, AskParams, QAResult } from "../api/types";
import { toApiError } from "../lib/messages";

export type AskState =
  | { phase: "idle" }
  | { phase: "loading"; question: string }
  | { phase: "done"; result: QAResult }
  | { phase: "error"; error: ApiError; question: string };

/**
 * Hỏi đáp với hai lớp chống "response cũ ghi đè response mới":
 * 1) AbortController hủy yêu cầu đang bay khi có yêu cầu mới (tiết kiệm mạng, nhưng máy chủ vẫn chạy nốt);
 * 2) số thứ tự yêu cầu: dù response cũ vẫn tới được (hủy không kịp, hoặc fetch giả không tôn trọng signal),
 *    chỉ response của yêu cầu MỚI NHẤT mới được ghi vào state.
 */
export function useAsk(api: Api = defaultApi) {
  const [state, setState] = useState<AskState>({ phase: "idle" });
  const seq = useRef(0);
  const ctrl = useRef<AbortController | null>(null);

  const ask = useCallback(
    async (params: AskParams) => {
      ctrl.current?.abort();
      const mine = ++seq.current;
      const c = new AbortController();
      ctrl.current = c;
      setState({ phase: "loading", question: params.question });
      try {
        const result = await api.ask({ ...params, trace: true }, c.signal);
        if (mine === seq.current) setState({ phase: "done", result });
      } catch (e) {
        if (mine !== seq.current || isAbort(e)) return;
        setState({ phase: "error", error: toApiError(e), question: params.question });
      }
    },
    [api],
  );

  const reset = useCallback(() => {
    ctrl.current?.abort();
    seq.current++;
    setState({ phase: "idle" });
  }, []);

  useEffect(() => () => ctrl.current?.abort(), []);
  return { state, ask, reset };
}
