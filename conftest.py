"""Pytest configuration and custom markers."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: marks tests that require model downloads or heavy computation",
    )
