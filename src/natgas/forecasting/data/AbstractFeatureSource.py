"""Base class for any extra data a model can use, such as weather or storage."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class AbstractFeatureSource(ABC):
    """One source of feature columns.

    The key rule: features_for() may only return values that were known on the
    as-of date. Each source decides what "known" means for its own data, for
    example a weekly report that is published a few days after the week ends.
    """

    @property
    @abstractmethod
    def columns(self) -> list[str]:
        """Names of the columns this source adds."""

    @abstractmethod
    def features_for(self, dates: pd.DatetimeIndex, as_of: pd.Timestamp) -> pd.DataFrame:
        """Feature values for each date, using only what was known on as_of.

        Returns a frame indexed by dates with one column per name in columns.
        Values that were not known yet should be NaN.
        """
