"""Single-image frame source."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from platrix.core.types import Frame
from platrix.sources.base import FrameSource


class ImageSource(FrameSource):
    """Yields exactly one frame decoded from an image file or ndarray."""

    name = "image"

    def __init__(self, path_or_array: "Path | str | np.ndarray") -> None:
        if isinstance(path_or_array, np.ndarray):
            self._image = path_or_array
            self._label = "image:memory"
        else:
            path = Path(path_or_array)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {path}")
            self._image = image
            self._label = f"image:{path.name}"

    def frames(self) -> Iterator[Frame]:
        yield Frame(
            image=self._image,
            index=0,
            timestamp=datetime.now(timezone.utc),
            source=self._label,
        )
