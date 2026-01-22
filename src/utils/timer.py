import time
from typing import Dict, Any
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self._start_times: Dict[str, float] = {}
        self._measurements: Dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        """Контекстный менеджер для измерения времени выполнения блока кода."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._measurements[name] = round(elapsed, 3)

    def get_metrics(self) -> Dict[str, Any]:
        """Возвращает все замеренные метрики + общее время."""
        total = sum(self._measurements.values())
        return {
            'total_time_sec': round(total, 3),
            **self._measurements,
            'num_phases': len(self._measurements)
        }