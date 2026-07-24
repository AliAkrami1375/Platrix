"""OCR abstraction and a no-op backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

from platrix.core.types import PlateDetection


class PlateOCR(ABC):
    """Base class for backends that turn a plate crop into text."""

    name: str = "base"

    @abstractmethod
    def read(self, detection: PlateDetection) -> tuple[str, float]:  # pragma: no cover
        """Return ``(normalized_text, confidence)`` for a plate crop."""
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional hook to load weights."""


class NullOCR(PlateOCR):
    """A backend that performs no recognition.

    Useful when you only need detection + snapshot logging, or before an OCR
    model has been trained. It always returns an empty string.
    """

    name = "none"

    def read(self, detection: PlateDetection) -> tuple[str, float]:
        return "", 0.0
