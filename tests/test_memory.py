"""Tests for memory monitoring."""

from src.utils.memory import MemoryMonitor, MemorySnapshot, get_memory_snapshot


class TestMemorySnapshot:
    def test_snapshot_captures_data(self):
        """Should capture at least process RSS."""
        snap = get_memory_snapshot()
        # Process RSS should always be available
        assert snap.process_rss_mb > 0

    def test_snapshot_string_format(self):
        """String representation should be readable."""
        snap = MemorySnapshot(
            metal_active_mb=100,
            metal_peak_mb=200,
            process_rss_mb=300,
            system_pressure="normal",
        )
        s = str(snap)
        assert "100" in s
        assert "300" in s
        assert "normal" in s

    def test_total_estimated(self):
        snap = MemorySnapshot(metal_active_mb=100, process_rss_mb=300)
        assert snap.total_estimated_mb == 400


class TestMemoryMonitor:
    def test_check_normal(self):
        """Normal memory usage should not trigger critical."""
        monitor = MemoryMonitor(warning_threshold_gb=100, critical_threshold_gb=200)
        assert not monitor.check()  # Should not be critical

    def test_log_snapshot(self):
        """Should return a valid snapshot."""
        monitor = MemoryMonitor()
        snap = monitor.log_snapshot("test")
        assert isinstance(snap, MemorySnapshot)
