"""Memory monitoring for ANE-GPU.

Tracks unified memory usage across MLX (Metal GPU) and Python process.
Provides warnings when memory pressure is high to prevent OOM.
"""

from __future__ import annotations

import logging
import os
import resource
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Point-in-time memory usage snapshot."""

    # Metal GPU memory (MLX)
    metal_active_mb: float = 0.0
    metal_peak_mb: float = 0.0
    metal_cache_mb: float = 0.0

    # Python process memory
    process_rss_mb: float = 0.0

    # System-level (if available)
    system_pressure: Optional[str] = None  # "normal", "warn", "critical"

    @property
    def total_estimated_mb(self) -> float:
        """Estimated total memory used by this process."""
        return self.metal_active_mb + self.process_rss_mb

    def __str__(self) -> str:
        parts = [
            f"Metal: {self.metal_active_mb:.0f}MB active / {self.metal_peak_mb:.0f}MB peak",
            f"Process RSS: {self.process_rss_mb:.0f}MB",
            f"Est. total: {self.total_estimated_mb:.0f}MB",
        ]
        if self.system_pressure:
            parts.append(f"System pressure: {self.system_pressure}")
        return " | ".join(parts)


def get_memory_snapshot() -> MemorySnapshot:
    """Capture current memory usage across all subsystems."""
    snap = MemorySnapshot()

    # Metal GPU memory via MLX
    try:
        import mlx.core as mx

        snap.metal_active_mb = mx.get_active_memory() / (1024 * 1024)
        snap.metal_peak_mb = mx.get_peak_memory() / (1024 * 1024)
        snap.metal_cache_mb = mx.get_cache_memory() / (1024 * 1024)
    except (ImportError, AttributeError):
        pass

    # Python process RSS
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in bytes on macOS
        snap.process_rss_mb = usage.ru_maxrss / (1024 * 1024)
    except (ValueError, AttributeError):
        pass

    # System memory pressure
    try:
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout.lower()
        if "critical" in output:
            snap.system_pressure = "critical"
        elif "warn" in output:
            snap.system_pressure = "warn"
        else:
            snap.system_pressure = "normal"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return snap


class MemoryMonitor:
    """Continuous memory monitoring with configurable warnings.

    Usage:
        monitor = MemoryMonitor(warning_threshold_gb=20.0)
        monitor.check()  # Logs warning if threshold exceeded

        # In a generation loop:
        for token in generate():
            if monitor.check():
                # Memory is critical — take action
                break
    """

    def __init__(
        self,
        warning_threshold_gb: float = 20.0,
        critical_threshold_gb: float = 22.0,
    ):
        self.warning_threshold_mb = warning_threshold_gb * 1024
        self.critical_threshold_mb = critical_threshold_gb * 1024
        self._warned = False

    def check(self) -> bool:
        """Check memory and log warnings. Returns True if critical."""
        snap = get_memory_snapshot()

        if snap.total_estimated_mb > self.critical_threshold_mb:
            logger.critical(f"CRITICAL memory usage: {snap}")
            return True
        elif snap.total_estimated_mb > self.warning_threshold_mb and not self._warned:
            logger.warning(f"High memory usage: {snap}")
            self._warned = True

        if snap.system_pressure == "critical":
            logger.critical(f"System memory pressure is critical: {snap}")
            return True

        return False

    def log_snapshot(self, label: str = "") -> MemorySnapshot:
        """Log current memory state with an optional label."""
        snap = get_memory_snapshot()
        prefix = f"[{label}] " if label else ""
        logger.info(f"{prefix}Memory: {snap}")
        return snap
