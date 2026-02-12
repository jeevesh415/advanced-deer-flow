from typing import Dict, Optional, Callable, Any
from functools import wraps
import time
import threading
from pydantic import BaseModel

class ResourceUsage(BaseModel):
    """Tracks resource consumption."""
    tokens_input: int = 0
    tokens_output: int = 0
    time_ms: float = 0.0
    api_calls: int = 0
    cost_usd: float = 0.0

class ResourceMonitor:
    """Global resource tracker."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceMonitor, cls).__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self._usage = ResourceUsage()
        self._budget = ResourceUsage() # To enforce limits
        self._active_start_time = None

    def start_tracking(self):
        self._active_start_time = time.time()

    def stop_tracking(self):
        if self._active_start_time:
            self._usage.time_ms += (time.time() - self._active_start_time) * 1000
            self._active_start_time = None

    def record_usage(self, usage: Dict[str, Any]):
        """Update usage stats."""
        for key, value in usage.items():
            if hasattr(self._usage, key):
                current_val = getattr(self._usage, key)
                setattr(self._usage, key, current_val + value)
            else:
                # Log warning: untracked resource type
                pass

    def check_limits(self) -> bool:
        """Return True if within limits, False if exceeded."""
        if self._budget.tokens_input > 0 and self._usage.tokens_input > self._budget.tokens_input:
            return False
        # Add more checks as needed
        return True

    def set_budget(self, budget: ResourceUsage):
        self._budget = budget

    def get_status(self) -> str:
        return f"Usage: {self._usage}, Budget: {self._budget}"


def track_time(func: Callable):
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        monitor = ResourceMonitor()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = (time.time() - start) * 1000
            monitor.record_usage({"time_ms": elapsed})
    return wrapper
