"""Chạy model một request tại một thời điểm (GĐ8).

Vì sao không để FastAPI chạy thẳng: pipeline đồng bộ (torch, FAISS) sẽ chặn event loop, và hai request cùng dùng GPU 4 GB
có thể tranh bộ nhớ. Nên: (1) mỗi lần gọi model chạy trong một luồng riêng, (2) một KHÓA luồng cho đúng một lần chạy
model tại một thời điểm, (3) số request đang chờ + đang chạy bị chặn trên (quá thì 503), (4) có hạn chót tổng.

Vì sao khóa luồng mà không dùng anyio.CapacityLimiter(1): khi request hết hạn chót (abandon_on_cancel), anyio TRẢ chỗ
trong limiter ngay dù luồng vẫn đang chạy model, nên request kế tiếp chạy CHỒNG lên nó (đã gặp trong test). Khóa nằm
trong luồng thì giữ đến khi model thật sự xong. Request đã hết hạn mà còn đang chờ khóa thì bỏ qua, không chạy model
cho một khách đã nhận 504."""
import threading
import time
from functools import partial

import anyio
from anyio import to_thread

from germanqa.api.errors import ApiError


class Runner:
    def __init__(self, max_inflight, timeout_s):
        self.max_inflight, self.timeout_s = max_inflight, timeout_s
        self._model_lock = threading.Lock()         # giữ suốt lúc model chạy
        self._lock = threading.Lock()               # bảo vệ hai bộ đếm
        self._waiting = 0
        self._running = 0

    @property
    def load(self):
        with self._lock:
            return {"waiting": self._waiting, "running": self._running}

    async def run(self, fn, *args, **kwargs):
        """Chạy fn(*args, **kwargs) trong luồng model. Trả về (kết quả, số giây đã chờ đến lượt)."""
        with self._lock:
            if self._waiting + self._running >= self.max_inflight:
                raise ApiError(503, "busy", "Máy chủ đang bận, hãy thử lại sau.")
            self._waiting += 1
        t_admit = time.perf_counter()
        released = False
        cancelled = threading.Event()

        def release_wait():                      # idempotent: luồng giành được khóa và nhánh kết thúc đều có thể gọi
            nonlocal released
            with self._lock:
                if not released:
                    released = True
                    self._waiting -= 1

        def job():
            with self._model_lock:
                if cancelled.is_set():
                    return None, 0.0
                queued = time.perf_counter() - t_admit
                release_wait()
                with self._lock:
                    self._running += 1
                try:
                    return partial(fn, *args, **kwargs)(), queued
                finally:
                    with self._lock:
                        self._running -= 1

        try:
            with anyio.fail_after(self.timeout_s):
                return await to_thread.run_sync(job, abandon_on_cancel=True)
        except TimeoutError:
            cancelled.set()
            raise ApiError(504, "timeout", f"Quá {self.timeout_s:g} giây, request bị hủy.") from None
        finally:
            release_wait()
