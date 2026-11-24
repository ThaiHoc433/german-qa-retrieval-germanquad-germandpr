"""FastAPI backend (GĐ8). create_app(settings, loader) dựng ứng dụng; loader(settings) -> (pipelines, unavailable).

Tài nguyên nặng nạp MỘT lần trong lifespan, ở nền: /health/live trả lời ngay, /health/ready trả 503 cho đến khi
nạp xong và chạy thử được một câu. Mọi lần gọi model đi qua Runner (một luồng model tại một thời điểm)."""
import asyncio
import contextlib
import hashlib
import json
import logging
import math
import re
import time
import uuid
from contextlib import asynccontextmanager

from anyio import to_thread
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from germanqa.api.errors import ApiError, error_body
from germanqa.api.experiments import MATRIX_CAVEATS, list_experiments, list_matrix
from germanqa.api.loader import RERANK_RETRIEVERS, RERANK_SUFFIX
from germanqa.api.runner import Runner
from germanqa.api.schemas import (ErrorResponse, ExperimentList, MatrixResponse, QARequest, QAResult, ReadRequest, ReadResult,
                                  Ready, RetrieveRequest, RetrieveResult)
from germanqa.api.settings import ApiSettings, load_settings
from germanqa.reader.qa_metrics import exact_match, f1_score
from germanqa.schemas import to_utf16_offset

log = logging.getLogger("germanqa.api")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
OK_STATUSES = ("ok", "question_truncated")


class Service:
    """Trạng thái dùng chung của một ứng dụng: tài nguyên đã nạp + bộ chạy model."""

    def __init__(self, settings):
        self.settings = settings
        self.runner = Runner(settings.max_inflight, settings.request_timeout_s)
        self.status = "loading"                 # loading | ready | failed
        self.pipelines, self.unavailable = {}, {}
        self.load_seconds = self.detail = None

    def ready_body(self):
        s = self.settings
        info = {n: {"available": n in self.pipelines, "reason": self.unavailable.get(n)} for n in s.retrievers}
        p = self.pipelines.get(s.default_retriever) or next(iter(self.pipelines.values()), None)
        rr_names = sorted(n[:-len(RERANK_SUFFIX)] for n in self.pipelines if n.endswith(RERANK_SUFFIX))
        rerank = {"configured": s.reranker, "available": bool(rr_names), "retrievers": rr_names,
                  "reason": None if rr_names else (self.unavailable.get("rerank") or ("not_configured" if not s.reranker else "load_failed"))}
        return {"status": self.status, "default_retriever": s.default_retriever, "retrievers": info, "reranker": rerank,
                "model": p.model_version if p else None, "corpus": p.corpus_version if p else None,
                "load_seconds": self.load_seconds, "detail": self.detail}


def _finite(obj):
    """JSON không có -inf/NaN (reader_score = -inf cho đoạn không có span hợp lệ): thay bằng None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _ms(seconds):
    return round(seconds * 1000, 3)


def check_question(question, s):
    if not question.strip():
        raise ApiError(422, "empty_question", "Câu hỏi trống.")
    if len(question) > s.max_question_chars:
        raise ApiError(422, "question_too_long", f"Câu hỏi dài {len(question)} ký tự, tối đa {s.max_question_chars}.")


def check_top_k(name, value, upper):
    if value is not None and not 1 <= value <= upper:
        raise ApiError(422, "invalid_top_k", f"{name} phải trong khoảng 1..{upper}, nhận được {value}.")


def create_app(settings=None, loader=None):
    settings = settings or load_settings()
    if loader is None:
        from germanqa.api.loader import load_pipelines
        loader = load_pipelines
    svc = Service(settings)

    async def load_in_background():
        t0 = time.perf_counter()
        try:
            def work():
                pipelines, unavailable = loader(settings)
                for name in list(pipelines):                         # chạy thử một câu: nạp luôn phần nạp lười (encoder câu hỏi)
                    try:
                        r = pipelines[name].answer(settings.warmup_question)
                        if r.status not in OK_STATUSES:
                            raise RuntimeError(f"warmup status={r.status}")
                    except Exception:                                # noqa: BLE001
                        log.exception("warmup %s thất bại", name)
                        unavailable[name] = "warmup_failed"
                        del pipelines[name]
                return pipelines, unavailable
            pipelines, unavailable = await to_thread.run_sync(work, abandon_on_cancel=True)
            svc.pipelines, svc.unavailable = pipelines, unavailable
            if settings.default_retriever not in pipelines:
                raise RuntimeError(f"retriever mặc định '{settings.default_retriever}' không khả dụng")
            svc.status = "ready"
        except asyncio.CancelledError:
            raise
        except Exception as exc:                                     # noqa: BLE001
            log.exception("nạp tài nguyên thất bại")
            svc.status, svc.detail = "failed", type(exc).__name__
        finally:
            svc.load_seconds = round(time.perf_counter() - t0, 2)
            log.info(json.dumps({"event": "startup", "status": svc.status, "load_seconds": svc.load_seconds,
                                 "retrievers": sorted(svc.pipelines), "unavailable": svc.unavailable}))

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(load_in_background())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="GermanQA", version="1.0", lifespan=lifespan,
                  description="Hỏi đáp trích xuất tiếng Đức: truy hồi (BM25 / dense / hybrid RRF), tùy chọn xếp lại bằng cross-encoder, + reader GBERT.")
    app.state.svc = svc

    # ---------- middleware: request_id + log mỗi request (đăng ký trước CORS để CORS nằm ngoài cùng) ----------
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        rid = rid if REQUEST_ID_RE.match(rid) else uuid.uuid4().hex[:16]
        request.state.request_id, request.state.log = rid, {}
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:                                            # noqa: BLE001 - lỗi ngoài ý muốn: log đầy đủ, client chỉ thấy mã
            log.exception("request %s lỗi nội bộ", rid)
            response = JSONResponse(error_body("internal_error", "Lỗi nội bộ.", rid), status_code=500)
        response.headers["X-Request-ID"] = rid
        log.info(json.dumps({"event": "request", "request_id": rid, "method": request.method,
                             "path": request.url.path, "status": response.status_code,
                             "ms": _ms(time.perf_counter() - t0), **request.state.log}))
        return response

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "X-Request-ID"], expose_headers=["X-Request-ID"])

    # ---------- lỗi: một dạng JSON ----------
    def rid_of(request):
        return getattr(request.state, "request_id", None)

    @app.exception_handler(ApiError)
    async def on_api_error(request, exc):
        return JSONResponse(error_body(exc.code, exc.message, rid_of(request), exc.details), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def on_validation(request, exc):
        details = [{"field": ".".join(str(x) for x in e["loc"][1:]) or str(e["loc"][0]), "message": e["msg"]}
                   for e in exc.errors()]
        return JSONResponse(error_body("invalid_request", "Request không hợp lệ.", rid_of(request), details),
                            status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request, exc):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse(error_body(code, str(exc.detail), rid_of(request)), status_code=exc.status_code)

    # ---------- helpers cho route ----------
    def need_ready():
        if svc.status != "ready":
            raise ApiError(503, "not_ready" if svc.status == "loading" else "load_failed",
                           "Máy chủ đang nạp tài nguyên." if svc.status == "loading" else "Nạp tài nguyên thất bại.")

    def pick(name, rerank=False):
        name = name or settings.default_retriever
        if name not in svc.pipelines:
            raise ApiError(422, "retriever_unavailable", f"Retriever '{name}' không khả dụng.",
                           [{"retriever": name, "reason": svc.unavailable.get(name, "not_configured")}])
        if rerank:
            key = name + RERANK_SUFFIX
            if key not in svc.pipelines:
                if name not in RERANK_RETRIEVERS:
                    raise ApiError(422, "rerank_unavailable", "Chỉ xếp lại được khi truy hồi bằng hybrid (pipeline H1).",
                                   [{"retriever": name, "reason": "hybrid_only"}])
                raise ApiError(422, "rerank_unavailable", "Máy chủ không có reranker cho retriever này.",
                               [{"retriever": name, "reason": svc.unavailable.get(key) or svc.unavailable.get("rerank") or ("not_configured" if not settings.reranker else "load_failed")}])
            return name, svc.pipelines[key]
        return name, svc.pipelines[name]

    def note(request, **kv):
        request.state.log.update(kv)

    def note_question(request, question, retriever):
        note(request, question_chars=len(question), question_sha=hashlib.sha1(question.encode("utf-8")).hexdigest()[:8],
             retriever=retriever)

    # ---------- health ----------
    @app.get("/health/live", tags=["health"])
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready", response_model=Ready, tags=["health"], responses={503: {"model": Ready}})
    async def ready():
        return JSONResponse(svc.ready_body(), status_code=200 if svc.status == "ready" else 503)

    # ---------- hỏi đáp ----------
    err = {422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}}

    @app.post("/api/v1/qa", response_model=QAResult, tags=["qa"], responses=err)
    async def qa(req: QARequest, request: Request):
        t0 = time.perf_counter()
        need_ready()
        check_question(req.question, settings)
        check_top_k("retrieval_top_k", req.retrieval_top_k, settings.max_retrieval_top_k)
        check_top_k("reader_top_k", req.reader_top_k, settings.max_reader_top_k)
        name, pipe = pick(req.retriever, req.rerank)
        k_ret = req.retrieval_top_k or pipe.retrieval_top_k
        k_read = req.reader_top_k or pipe.reader_top_k
        if req.rerank and k_ret > settings.max_rerank_pool:
            raise ApiError(422, "invalid_top_k", f"Khi rerank, retrieval_top_k ({k_ret}) không được lớn hơn {settings.max_rerank_pool} (max_rerank_pool).")
        if k_read > k_ret:
            raise ApiError(422, "invalid_top_k", f"reader_top_k ({k_read}) không được lớn hơn retrieval_top_k ({k_ret}).")
        note_question(request, req.question, name)
        note(request, rerank=bool(req.rerank))
        r, queued = await svc.runner.run(pipe.answer, req.question, k_ret, k_read, req.trace)
        d = r.to_dict()
        d.update(request_id=request.state.request_id, retriever=name, reranked=bool(req.rerank),
                 answer_start_utf16=to_utf16_offset(r.passage_text, r.answer_start),
                 answer_end_utf16=to_utf16_offset(r.passage_text, r.answer_end),
                 timings={"queue_ms": _ms(queued), "retrieval_ms": _ms(r.retrieval_time), "reader_ms": _ms(r.reader_time),
                          "rerank_ms": _ms(r.rerank_time), "pipeline_ms": _ms(r.total_time), "total_ms": _ms(time.perf_counter() - t0)},
                 versions={"model": r.model_version, "corpus": r.corpus_version, "experiment_id": r.experiment_id})
        for k in ("retrieval_time", "reader_time", "rerank_time", "total_time", "model_version", "corpus_version", "experiment_id"):
            d.pop(k)
        d["trace"] = _finite(d["trace"])
        note(request, status_pipeline=r.status, **d["timings"])
        return d

    @app.post("/api/v1/retrieve", response_model=RetrieveResult, tags=["qa"], responses=err)
    async def retrieve(req: RetrieveRequest, request: Request):
        t0 = time.perf_counter()
        need_ready()
        check_question(req.question, settings)
        check_top_k("top_k", req.top_k, settings.max_retrieval_top_k)
        name, pipe = pick(req.retriever)
        k = req.top_k or pipe.retrieval_top_k
        question = req.question.strip()
        note_question(request, req.question, name)
        def timed():
            t = time.perf_counter()
            return pipe.retriever.retrieve(question, k), time.perf_counter() - t
        (hits, retrieval), queued = await svc.runner.run(timed)
        note(request, queue_ms=_ms(queued), retrieval_ms=_ms(retrieval))
        return {"request_id": request.state.request_id, "question": question, "retriever": name, "top_k": k,
                "candidates": [{"passage_id": h["passage_id"], "rank": h["rank"], "score": h["score"],
                                "text": pipe.passages[h["passage_id"]],
                                **{k: h[k] for k in ("sources", "bm25_score", "dense_score") if k in h}} for h in hits],
                "timings": {"queue_ms": _ms(queued), "retrieval_ms": _ms(retrieval),
                            "total_ms": _ms(time.perf_counter() - t0)},
                "versions": {"model": pipe.model_version, "corpus": pipe.corpus_version,
                             "experiment_id": pipe.experiment_id}}

    @app.post("/api/v1/read", response_model=ReadResult, tags=["qa"], responses=err)
    async def read(req: ReadRequest, request: Request):
        """Đọc hiểu với đoạn văn tự nhập: bỏ qua truy hồi, đưa thẳng (câu hỏi, đoạn) cho CÙNG reader của pipeline."""
        t0 = time.perf_counter()
        need_ready()
        check_question(req.question, settings)
        if not req.context.strip():
            raise ApiError(422, "empty_context", "Đoạn văn trống.")
        if len(req.context) > settings.max_context_chars:
            raise ApiError(422, "context_too_long", f"Đoạn văn dài {len(req.context)} ký tự, tối đa {settings.max_context_chars}.")
        pipe = svc.pipelines.get(settings.default_retriever) or next(iter(svc.pipelines.values()))
        question, context = req.question.strip(), req.context
        note(request, question_chars=len(question), context_chars=len(context))

        def work():
            q, cut = pipe._cut_question(question)                    # cùng quy tắc cắt 64 token như /qa
            t = time.perf_counter()
            span = pipe.reader.read(q, [{"passage_id": "user", "text": context}])[0]
            return span, cut, time.perf_counter() - t
        (span, cut, reader_s), queued = await svc.runner.run(work)
        ok = span["start"] >= 0
        answer = span["text"] if ok else ""
        a, b = (span["start"], span["end"]) if ok else (-1, -1)
        if ok and context[a:b] != answer:                            # bất biến như ở pipeline
            raise ValueError("đáp án không khớp văn bản gốc tại offset")
        exp = req.expected_answer.strip() if req.expected_answer and req.expected_answer.strip() else None
        timings = {"queue_ms": _ms(queued), "reader_ms": _ms(reader_s), "total_ms": _ms(time.perf_counter() - t0)}
        note(request, status_pipeline="ok" if ok else "no_valid_span", **timings)
        return {"request_id": request.state.request_id,
                "status": ("question_truncated" if cut else "ok") if ok else "no_valid_span",
                "question": question, "context": context, "answer": answer, "answer_start": a, "answer_end": b,
                "answer_start_utf16": to_utf16_offset(context, a) if ok else -1,
                "answer_end_utf16": to_utf16_offset(context, b) if ok else -1,
                "reader_score": _finite(span["score"]) if ok else None, "expected_answer": exp,
                "em": exact_match(answer, exp) if exp else None, "f1": f1_score(answer, exp) if exp else None,
                "timings": timings, "model": pipe.model_version}

    # ---------- thí nghiệm đã lưu (chỉ đọc) ----------
    @app.get("/api/v1/experiments", response_model=ExperimentList, tags=["experiments"])
    async def experiments(kind: str = None):
        items, skipped = await to_thread.run_sync(list_experiments, settings.results_path(), kind, settings.baselines_path())
        return {"count": len(items), "skipped": skipped, "items": items}

    @app.get("/api/v1/matrix", response_model=MatrixResponse, tags=["experiments"])
    async def matrix():
        rows = await to_thread.run_sync(list_matrix, settings.baselines_path(), settings.results_path())
        return {"count": len(rows), "caveats": MATRIX_CAVEATS, "rows": rows}

    return app
