"""A registry version created when the tournament was seeded."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeededVersion:
    family: str
    role: str
    registered_model: str
    version: str
    source_run: str
    mae: float
