from time import perf_counter, sleep
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from typing import Dict, Callable, Optional


class CallSectionsTimer:
    """
    Per-call segmented timer.
    Use as: with t("section_name"):
    Aggregates durations for the same section within one call.
    """
    def __init__(self, sync: Optional[Callable[[], None]] = None):
        self.durations: Dict[str, float] = defaultdict(float)  # name -> seconds (current call)
        self._sync = sync  # Optional sync (e.g., torch.cuda.synchronize)

    @contextmanager
    def section(self, name: str):
        if self._sync:
            self._sync()
        t0 = perf_counter()
        try:
            yield
        finally:
            if self._sync:
                self._sync()
            self.durations[name] += perf_counter() - t0

    def __call__(self, name: str):
        return self.section(name)


class SimpleAverager:
    """
    Minimal cumulative averager (per-section).
    """
    def __init__(self, unit: str = "ms", warmup: int = 0,
                 printer: Callable[[str], None] = print):
        self.scale = 1e3 if unit == "ms" else 1.0
        self.unit = unit
        self.warmup = warmup
        self.printer = printer
        self.call_idx = 0
        self.totals: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)

    def update_and_print(self, last: Dict[str, float], tag: str):
        self.call_idx += 1
        c = self.call_idx
        if c <= self.warmup:
            return
        lines = [f"[{tag}] call #{c}"]
        for name, dt in sorted(last.items(), key=lambda kv: kv[1], reverse=True):
            self.totals[name] += dt
            self.counts[name] += 1
            avg = self.totals[name] / self.counts[name]
            lines.append(
                f"- {name}: last={dt*self.scale:.3f} {self.unit}, "
                f"avg({self.counts[name]})={avg*self.scale:.3f} {self.unit}"
            )
        self.printer("\n".join(lines))


def avg_sections(name: Optional[str] = None, unit: str = "ms", warmup: int = 0,
                        printer: Callable[[str], None] = print,
                        sync: Optional[Callable[[], None]] = None):
    """
    Minimal decorator for instance methods.
    It sets self._current_timer during the call so you can write: with self.t("stage"):
    """
    def deco(func):
        tag = name or func.__qualname__
        stats = SimpleAverager(unit=unit, warmup=warmup, printer=printer)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            prev = getattr(self, "_current_timer", None)  # support simple nesting
            t = CallSectionsTimer(sync=sync)
            setattr(self, "_current_timer", t)
            try:
                return func(self, *args, **kwargs)
            finally:
                setattr(self, "_current_timer", prev)
                stats.update_and_print(t.durations, tag)
        return wrapper
    return deco


# -------- Example --------
class Model:
    def __init__(self):
        # Define self.t in __init__, no globals involved.
        self._current_timer: Optional[CallSectionsTimer] = None

        @contextmanager
        def _noop():
            # No-op context when no timer is active
            yield

        def section(name: str):
            # If a timer is active, delegate; otherwise, no-op
            return self._current_timer(name) if self._current_timer else _noop()

        self.t = section  # now you can: with self.t("stage"):

    @avg_sections(name="Model.step", unit="ms", warmup=1)
    def step(self):
        with self.t("load"):
            sleep(0.020)
        for _ in range(2):
            with self.t("compute"):
                sleep(0.010)
        with self.t("postprocess"):
            sleep(0.005)
            

if __name__ == '__main__':
    m = Model()
    for _ in range(5):
        m.step()