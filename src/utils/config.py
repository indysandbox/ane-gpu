"""Configuration management for ANE-GPU speculative decoding.

Supports loading from YAML files, CLI args, or direct construction.
Hardware-agnostic defaults that adapt to the running system.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Detected hardware capabilities. Auto-populated on creation."""

    chip: str = ""
    total_memory_gb: float = 0.0
    macos_version: str = ""
    python_version: str = ""

    def __post_init__(self) -> None:
        if not self.chip:
            self.chip = self._detect_chip()
        if self.total_memory_gb == 0.0:
            self.total_memory_gb = self._detect_memory()
        if not self.macos_version:
            self.macos_version = platform.mac_ver()[0]
        if not self.python_version:
            self.python_version = platform.python_version()

    @staticmethod
    def _detect_chip() -> str:
        """Detect Apple Silicon chip model."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Unknown"

    @staticmethod
    def _detect_memory() -> float:
        """Detect total unified memory in GB."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return int(result.stdout.strip()) / (1024**3)
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            return 0.0

    @property
    def chip_generation(self) -> str:
        """Extract chip generation (e.g., 'M4' from 'Apple M4')."""
        chip = self.chip.lower()
        for gen in ("m5", "m4", "m3", "m2", "m1"):
            if gen in chip:
                return gen.upper()
        return "Unknown"

    def recommended_target_quantization(self) -> str:
        """Suggest quantization level based on available memory."""
        if self.total_memory_gb >= 48:
            return "Q8"
        elif self.total_memory_gb >= 32:
            return "Q5"
        elif self.total_memory_gb >= 24:
            return "Q4"
        elif self.total_memory_gb >= 16:
            return "Q3"
        else:
            return "Q3"  # Might not fit at all

    def max_target_params_b(self) -> float:
        """Estimate max target model size (billions of params) at Q4."""
        # Rough rule: Q4 uses ~0.6GB per billion params
        # Reserve ~6GB for draft model + KV cache + system overhead
        available = self.total_memory_gb - 6.0
        return available / 0.6


@dataclass
class ModelConfig:
    """Configuration for a single model (draft or target).

    Models can be specified as:
    - HuggingFace model ID: "Qwen/Qwen3.5-0.8B"
    - MLX community model: "mlx-community/Qwen3.5-27B-4bit"
    - Local path: "/path/to/model" or "models/draft.mlpackage"
    """

    model_id: str  # HF model ID or local path
    local_path: Optional[str] = None  # Resolved local path (set during loading)
    quantization: Optional[str] = None  # e.g., "Q4", "Q8", "FP16"
    max_seq_len: int = 2048  # Maximum sequence length

    def resolve_path(self) -> Path:
        """Resolve model to a local filesystem path."""
        if self.local_path:
            return Path(self.local_path)
        # Check if model_id is already a local path
        p = Path(self.model_id)
        if p.exists():
            return p
        # Otherwise it's a HuggingFace ID — will be downloaded on first use
        return p


@dataclass
class DraftConfig(ModelConfig):
    """Configuration specific to the draft (ANE) model."""

    compute_unit: str = "CPU_AND_NE"  # Core ML compute unit preference
    ane_seq_len: int = 128  # Fixed sequence length for ANE compilation
    use_kv_cache: bool = False  # Start without KV caching (Phase 3 optimization)


@dataclass
class TargetConfig(ModelConfig):
    """Configuration specific to the target (GPU) model."""

    pass  # Target uses MLX defaults; extend as needed


@dataclass
class SpeculativeConfig:
    """Parameters for the speculative decoding algorithm."""

    k: int = 5  # Number of draft tokens per speculation round
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 200  # Max tokens to generate
    seed: Optional[int] = None  # For reproducible sampling


@dataclass
class Config:
    """Top-level configuration for ANE-GPU speculative decoding.

    Example usage:
        # From defaults (auto-detects hardware)
        config = Config.default()

        # From YAML file
        config = Config.from_yaml("config.yaml")

        # Direct construction
        config = Config(
            draft=DraftConfig(model_id="Qwen/Qwen3.5-0.8B"),
            target=TargetConfig(model_id="mlx-community/Qwen3.5-27B-4bit"),
        )
    """

    draft: DraftConfig = field(
        default_factory=lambda: DraftConfig(model_id="Qwen/Qwen3.5-0.8B")
    )
    target: TargetConfig = field(
        default_factory=lambda: TargetConfig(model_id="mlx-community/Qwen3.5-27B-4bit")
    )
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    hardware: HardwareInfo = field(default_factory=HardwareInfo)

    # Paths
    models_dir: str = "models"  # Where converted models are stored
    log_level: str = "INFO"

    @classmethod
    def default(cls) -> Config:
        """Create config with auto-detected hardware and sensible defaults."""
        hw = HardwareInfo()
        logger.info(
            f"Detected: {hw.chip}, {hw.total_memory_gb:.0f}GB RAM, "
            f"macOS {hw.macos_version}"
        )

        quant = hw.recommended_target_quantization()
        logger.info(f"Recommended target quantization: {quant}")

        return cls(
            target=TargetConfig(
                model_id=f"mlx-community/Qwen3.5-27B-4bit",
                quantization=quant,
            ),
            hardware=hw,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        draft_data = data.get("draft", {})
        target_data = data.get("target", {})
        spec_data = data.get("speculative", {})

        return cls(
            draft=DraftConfig(**draft_data),
            target=TargetConfig(**target_data),
            speculative=SpeculativeConfig(**spec_data),
            models_dir=data.get("models_dir", "models"),
            log_level=data.get("log_level", "INFO"),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        from dataclasses import asdict

        data = {
            "draft": asdict(self.draft),
            "target": asdict(self.target),
            "speculative": asdict(self.speculative),
            "models_dir": self.models_dir,
            "log_level": self.log_level,
        }
        # Don't save hardware info — it's auto-detected
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> list[str]:
        """Check config for potential issues. Returns list of warnings."""
        warnings = []

        if self.hardware.total_memory_gb < 16:
            warnings.append(
                f"Only {self.hardware.total_memory_gb:.0f}GB RAM — "
                f"speculative decoding with a 27B target may not fit. "
                f"Consider a smaller target model."
            )

        if self.speculative.k < 1 or self.speculative.k > 20:
            warnings.append(f"Speculation length K={self.speculative.k} is unusual (typical: 3-8)")

        if self.draft.ane_seq_len < 32:
            warnings.append(f"ANE sequence length {self.draft.ane_seq_len} is very short")

        return warnings
