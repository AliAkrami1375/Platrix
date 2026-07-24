"""Frame source abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from platrix.core.types import Frame


class FrameSource(ABC):
    """A source that yields :class:`~platrix.core.types.Frame` objects."""

    name: str = "base"

    @abstractmethod
    def frames(self) -> Iterator[Frame]:  # pragma: no cover - interface
        """Yield frames until the source is exhausted or closed."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any underlying resources."""

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
