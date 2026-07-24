"""Detector abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from platrix.core.types import Frame, PlateDetection


class PlateDetector(ABC):
    """Base class for every plate-detection backend.

    A detector receives a :class:`~platrix.core.types.Frame` and returns zero
    or more :class:`~platrix.core.types.PlateDetection` regions.
    """

    name: str = "base"

    @abstractmethod
    def detect(self, frame: Frame) -> list[PlateDetection]:  # pragma: no cover
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional hook to load weights / run a dummy inference."""

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} name={self.name!r}>"
