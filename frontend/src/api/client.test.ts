import { ApiError, createApi, isAbort } from "./client";

const res = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

afterEach(() => vi.unstubAllGlobals());

describe("createApi", () => {
  it("lỗi API theo dạng {error:{code,...}} thành ApiError có code và request_id", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(res(422, { error: { code: "empty_question", message: "Câu hỏi trống.", request_id: "abc" } })));
    const err = await createApi().ask({ question: " " }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ code: "empty_question", status: 422, requestId: "abc" });
  });

  it("thân lỗi không phải JSON -> http_error kèm mã HTTP", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>502</html>", { status: 502 })));
    const err = await createApi().getExperiments().catch((e) => e);
    expect(err).toMatchObject({ code: "http_error", status: 502 });
  });

  it("mất mạng -> ApiError network", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    expect(await createApi().ask({ question: "x" }).catch((e) => e)).toMatchObject({ code: "network" });
  });

  it("hủy chủ động KHÔNG bị coi là lỗi mạng", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new DOMException("aborted", "AbortError")));
    const err = await createApi().ask({ question: "x" }).catch((e) => e);
    expect(isAbort(err)).toBe(true);
    expect(err).not.toBeInstanceOf(ApiError);
  });

  it("/health/ready trả 503 kèm thân hợp lệ là DỮ LIỆU (đang nạp), không phải lỗi", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(res(503, { status: "loading", default_retriever: "bm25", retrievers: {} })));
    expect((await createApi().getReady()).status).toBe("loading");
  });

  it("/health/ready 503 mà thân không đúng dạng -> lỗi", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(res(503, { whatever: 1 })));
    await expect(createApi().getReady()).rejects.toBeInstanceOf(ApiError);
  });

  it("gửi JSON đúng đường dẫn tương đối, với trace và retriever", async () => {
    const f = vi.fn().mockResolvedValue(res(200, {}));
    vi.stubGlobal("fetch", f);
    await createApi().ask({ question: "Wer?", retriever: "dense", trace: true });
    const [url, init] = f.mock.calls[0];
    expect(url).toBe("/api/v1/qa");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ question: "Wer?", retriever: "dense", trace: true });
  });
});
