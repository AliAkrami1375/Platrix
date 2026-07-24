import numpy as np

from platrix.ocr.persian import format_iranian_plate
from platrix.ocr.plate_grammar import decode_plate

LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "ب", "و", "ص"]
D = {c: i for i, c in enumerate(LABELS)}


def _onehot(seq):
    probs = np.full((len(seq), len(LABELS)), 0.01, dtype="float32")
    for r, ch in enumerate(seq):
        probs[r, D[ch]] = 0.9
    return probs


def test_grammar_keeps_single_letter_and_digits():
    # A clean plate: 2 digits, letter, 3 digits, 2 digits.
    probs = _onehot("11و11427")
    chars, confs = decode_plate(probs, LABELS)
    assert "".join(chars) == "11و11427"
    assert sum(1 for c in chars if not c.isdigit()) == 1  # exactly one letter


def test_grammar_forces_digits_at_digit_positions():
    # Glyph 0 is ambiguous but leans letter; grammar must still emit a digit
    # there because the real letter (و, index 2) is more letter-like.
    probs = _onehot("11و11427")
    probs[0, D["ص"]] = 0.4  # nudge glyph 0 toward a letter
    chars, _ = decode_plate(probs, LABELS)
    assert chars[0].isdigit()
    assert chars[2] == "و"


def test_format_standard_layout():
    assert format_iranian_plate("11و11427") == "۱۱ و ۱۱۴ ۲۷"


def test_format_non_standard_passthrough():
    # Not 8 chars → no grouping, only digit conversion.
    assert format_iranian_plate("11و114") == "۱۱و۱۱۴"
