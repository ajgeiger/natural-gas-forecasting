"""Settings for the model tournament, read from the tournament section of config/tuning.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from backtesting.BacktestReport import HORIZON_BUCKETS
from optimization.config.TuningConfig import TuningConfig

ALL_DAYS = "all days"
# The three events: short, medium, and long range
EVENTS = [(first, last, f"days {first} to {last}") for first, last in HORIZON_BUCKETS]
EVENT_BUCKETS = [bucket for _, _, bucket in EVENTS]
SCORED_BUCKETS = [*EVENT_BUCKETS, ALL_DAYS]


@dataclass(frozen=True)
class TournamentConfig:
    """Which model families compete, and the rules of the trials and the events.

    families         model families with their own registry, champion, and challenger
    baseline         the model every champion has to beat in an event
    registry_prefix  registry name is the prefix plus the family, natgas-gradient-boosting
    trial_days       matured production days a challenger needs before it can be promoted
    window_days      production days in the rolling scorecard
    min_improvement  share of the baseline error a champion must beat to qualify
    """

    families: list[str]
    baseline: str = "naive"
    registry_prefix: str = "natgas-"
    trial_days: int = 60
    window_days: int = 30
    min_improvement: float = 0.01
    tracking_uri: str = "sqlite:///mlflow.db"

    def __post_init__(self) -> None:
        if not self.families:
            raise ValueError("the tournament needs at least one model family")
        if self.baseline in self.families:
            raise ValueError(f"{self.baseline} is the baseline and cannot be a family")
        if self.trial_days < 1 or self.window_days < 1:
            raise ValueError("trial_days and window_days must be at least 1")

    @classmethod
    def from_tuning(cls, tuning: TuningConfig, section: dict | None = None) -> TournamentConfig:
        """Families default to every model in tuning.yaml except the baseline."""
        section = section or {}
        baseline = section.get("baseline", "naive")
        families = section.get("families") or [
            name for name in tuning.model_settings if name != baseline
        ]
        return cls(
            families=list(families),
            baseline=baseline,
            registry_prefix=section.get("registry_prefix", "natgas-"),
            trial_days=int(section.get("trial_days", 60)),
            window_days=int(section.get("window_days", 30)),
            min_improvement=float(section.get("min_improvement", tuning.min_improvement)),
            tracking_uri=tuning.tracking_uri,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TournamentConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_tuning(TuningConfig.from_yaml(path), raw.get("tournament"))

    def registered_model(self, family: str) -> str:
        return f"{self.registry_prefix}{family.replace('_', '-')}"

    @staticmethod
    def metric_name(bucket: str, metric: str = "mae") -> str:
        """MLflow metric for a bucket: mae for all days, mae_days_1_to_5 for an event."""
        if bucket == ALL_DAYS:
            return metric
        return f"{metric}_{bucket.replace(' ', '_')}"
