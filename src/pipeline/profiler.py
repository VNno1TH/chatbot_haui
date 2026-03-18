"""
profiler.py
Đo latency từng bước trong pipeline — thêm vào src/pipeline/

Cách dùng:
    from src.pipeline.profiler import LatencyProfiler, profiler_enabled

    # Bật/tắt qua biến môi trường: HAUI_PROFILER=1
    p = LatencyProfiler()
    p.mark("start")
    ...
    p.mark("normalize")
    ...
    p.report(query="câu hỏi gốc")
"""

import os
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger("haui.latency")

# Bật/tắt bằng env var: HAUI_PROFILER=1 python ...
profiler_enabled: bool = os.environ.get("HAUI_PROFILER", "0") == "1"


class LatencyProfiler:
    def __init__(self, enabled: bool | None = None):
        # None → dùng global setting
        self.enabled = profiler_enabled if enabled is None else enabled
        self._marks : dict[str, float] = {}
        self._order : list[str]        = []

    def mark(self, step: str) -> None:
        if not self.enabled:
            return
        self._marks[step] = time.perf_counter()
        self._order.append(step)

    @contextmanager
    def step(self, name: str):
        """
        Context manager — tự động đánh dấu start/end.

        Ví dụ:
            with p.step("rerank"):
                results = reranker.rerank(query, chunks)
        """
        self.mark(f"{name}_start")
        try:
            yield
        finally:
            self.mark(f"{name}_end")

    def report(self, query: str = "", log_level: int = logging.INFO) -> dict[str, float]:
        """
        In báo cáo latency ra logger và trả về dict.

        Returns:
            {"clean→normalize": 820.3, "normalize→classify": 45.1, ..., "TOTAL": 1234.5}
        """
        if not self.enabled or len(self._order) < 2:
            return {}

        result: dict[str, float] = {}
        lines: list[str]         = []

        for i in range(1, len(self._order)):
            a, b = self._order[i - 1], self._order[i]
            ms   = (self._marks[b] - self._marks[a]) * 1000
            key  = f"{a}→{b}"
            result[key] = round(ms, 1)
            # Đánh dấu bước chậm (> 500ms) bằng ❌
            flag = " ❌" if ms > 500 else (" ⚠️" if ms > 200 else "")
            lines.append(f"  {key:<35} {ms:>7.0f}ms{flag}")

        total = (self._marks[self._order[-1]] - self._marks[self._order[0]]) * 1000
        result["TOTAL"] = round(total, 1)
        lines.append(f"  {'TOTAL (trước streaming)':<35} {total:>7.0f}ms")

        header = f"[Latency] query='{query[:50]}'"
        logger.log(log_level, "\n".join([header] + lines))

        # In ra stderr khi chạy tay để dễ thấy
        if os.environ.get("HAUI_PROFILER") == "1":
            print(f"\n{'─'*55}")
            print(f"\n".join([header] + lines))
            print(f"{'─'*55}\n")

        return result

    def reset(self) -> None:
        self._marks.clear()
        self._order.clear()