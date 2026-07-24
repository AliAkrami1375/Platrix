"""Structure-aware decoding for the standard Iranian plate layout.

A standard Iranian civilian plate reads left → right as::

    D D  L  D D D   D D
    │ │  │  │ │ │   └─┴─ province / region code (2 digits)
    │ │  │  └─┴─┴─────── serial (3 digits)
    │ │  └────────────── letter (1 Persian letter)
    └─┴───────────────── prefix (2 digits)

Knowing this grammar lets us constrain the OCR: at digit positions we pick the
best **digit** class and at the letter position the best **letter** class, which
eliminates digit↔letter confusions and improves accuracy.
"""

from __future__ import annotations

import numpy as np


def _split_classes(labels: list[str]) -> tuple[list[int], list[int]]:
    digits = [i for i, c in enumerate(labels) if c.isdigit()]
    letters = [i for i, c in enumerate(labels) if not c.isdigit()]
    return digits, letters


def decode_plate(
    probs: np.ndarray, labels: list[str]
) -> tuple[list[str], list[float]]:
    """Decode per-glyph probabilities into characters using the plate grammar.

    ``probs`` is ``(n_glyphs, n_classes)`` of softmax probabilities.
    Returns ``(chars, confidences)``.
    """
    digit_idx, letter_idx = _split_classes(labels)
    n = len(probs)
    if n == 0 or not digit_idx or not letter_idx:
        chars = [labels[int(np.argmax(p))] for p in probs]
        return chars, [float(p.max()) for p in probs]

    # A plate has exactly one letter. Pick the glyph that most looks like a
    # letter *relative* to its best digit — this is robust to a segmentation
    # that is off by a glyph, unlike assuming a fixed index.
    letter_margin = [
        max(p[i] for i in letter_idx) - max(p[i] for i in digit_idx) for p in probs
    ]
    letter_pos = int(np.argmax(letter_margin))

    chars: list[str] = []
    confs: list[float] = []
    for j, p in enumerate(probs):
        pool = letter_idx if j == letter_pos else digit_idx
        best = pool[int(np.argmax([p[k] for k in pool]))]
        chars.append(labels[best])
        confs.append(float(p[best]))
    return chars, confs
