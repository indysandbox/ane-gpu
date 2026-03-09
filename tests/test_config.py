"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest

from src.utils.config import (
    Config,
    DraftConfig,
    HardwareInfo,
    SpeculativeConfig,
    TargetConfig,
)


class TestHardwareInfo:
    def test_auto_detection(self):
        """Hardware info should auto-detect on Apple Silicon."""
        hw = HardwareInfo()
        assert hw.chip  # Should detect something
        assert hw.total_memory_gb > 0
        assert hw.macos_version
        assert hw.python_version

    def test_chip_generation(self):
        """Should extract chip generation from brand string."""
        hw = HardwareInfo(chip="Apple M4")
        assert hw.chip_generation == "M4"

        hw = HardwareInfo(chip="Apple M3 Pro")
        assert hw.chip_generation == "M3"

    def test_recommended_quantization(self):
        hw = HardwareInfo(chip="Apple M4", total_memory_gb=24)
        assert hw.recommended_target_quantization() == "Q4"

        hw = HardwareInfo(chip="Apple M4", total_memory_gb=16)
        assert hw.recommended_target_quantization() == "Q3"

        hw = HardwareInfo(chip="Apple M4 Max", total_memory_gb=48)
        assert hw.recommended_target_quantization() == "Q8"


class TestConfig:
    def test_default_config(self):
        """Default config should create valid settings."""
        config = Config.default()
        assert config.draft.model_id
        assert config.target.model_id
        assert config.speculative.k == 5
        assert config.hardware.total_memory_gb > 0

    def test_yaml_roundtrip(self):
        """Config should survive YAML save/load."""
        config = Config(
            draft=DraftConfig(model_id="test/draft-model"),
            target=TargetConfig(model_id="test/target-model"),
            speculative=SpeculativeConfig(k=7, temperature=0.8),
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            config.to_yaml(f.name)
            loaded = Config.from_yaml(f.name)

        assert loaded.draft.model_id == "test/draft-model"
        assert loaded.target.model_id == "test/target-model"
        assert loaded.speculative.k == 7
        assert loaded.speculative.temperature == 0.8

    def test_validate_warnings(self):
        """Validation should catch suspicious configs."""
        config = Config(
            speculative=SpeculativeConfig(k=50),
            hardware=HardwareInfo(chip="Apple M4", total_memory_gb=8),
        )
        warnings = config.validate()
        assert len(warnings) >= 2  # Low memory + unusual K


class TestModelConfig:
    def test_resolve_local_path(self):
        """Should resolve local paths directly."""
        config = DraftConfig(model_id="/tmp/test-model", local_path="/tmp/test-model")
        assert config.resolve_path() == Path("/tmp/test-model")

    def test_resolve_hf_id(self):
        """HuggingFace IDs resolve as relative paths (downloaded on use)."""
        config = DraftConfig(model_id="Qwen/Qwen3.5-0.8B")
        path = config.resolve_path()
        assert "Qwen" in str(path)
