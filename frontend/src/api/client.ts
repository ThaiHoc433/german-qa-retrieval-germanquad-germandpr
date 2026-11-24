import type { Api, ExperimentList, MatrixResponse, QAResult, ReadResult, Ready, RetrieveResult } from "./types";

/** Mọi lỗi từ API hoặc từ mạng đều thành một kiểu này, để UI chỉ có một đường xử lý lỗi. */
export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number | null = null,
    public requestId: string | null = null,
    public details: unknown = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export const isAbort = (e: unknown): boolean => e instanceof DOMException && e.name === "AbortError";

async function parseError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    const e = body?.error;
    if (e && typeof e.code === "string") {
      return new ApiError(e.code, String(e.message ?? ""), res.status, e.request_id ?? null, e.details ?? null);
    }
  } catch {
    /* thân không phải JSON: rơi xuống lỗi chung */
  }
  return new ApiError("http_error", `HTTP ${res.status}`, res.status);
}

async function call(base: string, path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(base + path, init);
  } catch (e) {
    if (isAbort(e)) throw e; // hủy chủ động không phải lỗi mạng
    throw new ApiError("network", "Không kết nối được tới máy chủ.");
  }
}

async function json<T>(base: string, path: string, init?: RequestInit): Promise<T> {
  const res = await call(base, path, init);
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

const post = (body: unknown, signal?: AbortSignal): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
  signal,
});

export function createApi(base = ""): Api {
  return {
    ask: (params, signal) => json<QAResult>(base, "/api/v1/qa", post(params, signal)),
    read: (params, signal) => json<ReadResult>(base, "/api/v1/read", post(params, signal)),
    retrieve: (params, signal) => json<RetrieveResult>(base, "/api/v1/retrieve", post(params, signal)),
    getExperiments: (signal) => json<ExperimentList>(base, "/api/v1/experiments", { signal }),
    getMatrix: (signal) => json<MatrixResponse>(base, "/api/v1/matrix", { signal }),
    // /health/ready trả 503 KÈM thân hợp lệ khi đang nạp hoặc thất bại: đó là dữ liệu, không phải lỗi.
    async getReady(signal) {
      const res = await call(base, "/health/ready", { signal });
      if (res.ok || res.status === 503) {
        const body = (await res.json()) as Ready;
        if (body && typeof body.status === "string") return body;
      }
      throw await parseError(res);
    },
  };
}

export const api: Api = createApi();
