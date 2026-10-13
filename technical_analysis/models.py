"""Data-layer contracts shared by the adapter and analysis service."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class RawDataset:
    records: list[dict[str, Any]] = field(repr=False)
    retrieved_at: datetime
    endpoint: str
    api_seconds: float = 0.0
    cache_hit: bool = False


@dataclass
class NormalizedBars:
    frame: pd.DataFrame
    issues: list[str] = field(default_factory=list)


class DataError(Exception):
    """Sanitized, user-safe message. Never attach raw provider exceptions."""

    def __init__(self, message: str, code: str = "request_failed", *,
                 api_seconds: float = 0.0, cache_hit: bool = False):
        super().__init__(message)
        self.code = code
        self.api_seconds = api_seconds
        self.cache_hit = cache_hit
