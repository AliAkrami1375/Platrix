"""Helpers for the Iranian license-plate format.

Standard Iranian civilian plates read (left → right):

    ``DD  L  DDD  |  DD``

i.e. two digits, one Persian letter, three digits, and a two-digit provincial
code. This module maps model class indices to that alphabet and renders both an
ASCII-normalized and a Persian-friendly string.
"""

from __future__ import annotations

# Persian (Eastern Arabic) digits used on plates and in the UI.
FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
EN_DIGITS = "0123456789"

# Letters that legally appear on Iranian plates.
PLATE_LETTERS = [
    "الف", "ب", "پ", "ت", "ث", "ج", "د", "ز", "ژ", "س",
    "ش", "ص", "ط", "ع", "ف", "ق", "ک", "گ", "ل", "م",
    "ن", "و", "ه", "ی", "D", "S",  # D/S = diplomatic / service
]

# Romanized aliases, handy for logs and search.
LETTER_LATIN = {
    "الف": "A", "ب": "B", "پ": "P", "ت": "T", "ث": "Th", "ج": "J",
    "د": "D", "ز": "Z", "ژ": "Zh", "س": "S", "ش": "Sh", "ص": "Sad",
    "ط": "Ta", "ع": "Ein", "ف": "F", "ق": "Gh", "ک": "K", "گ": "G",
    "ل": "L", "م": "M", "ن": "N", "و": "V", "ه": "H", "ی": "Y",
}

_FA_TO_EN = {fa: en for fa, en in zip(FA_DIGITS, EN_DIGITS)}


def to_english_digits(text: str) -> str:
    """Convert any Persian digits in *text* to ASCII digits."""
    return "".join(_FA_TO_EN.get(ch, ch) for ch in text)


def to_persian_digits(text: str) -> str:
    """Convert ASCII digits in *text* to Persian digits."""
    table = {en: fa for fa, en in zip(FA_DIGITS, EN_DIGITS)}
    return "".join(table.get(ch, ch) for ch in text)


def format_iranian_plate(text: str) -> str:
    """Render a plate string in the standard grouped layout with Persian digits.

    An 8-character plate ``"11و11427"`` becomes ``"۱۱ و ۱۱۴ ۲۷"`` (two digits,
    letter, three digits, two-digit region). Other lengths are returned as-is
    (only digit conversion applied).
    """
    chars = list(text)
    if len(chars) == 8 and not chars[2].isdigit():
        grouped = (
            f"{chars[0]}{chars[1]} {chars[2]} "
            f"{chars[3]}{chars[4]}{chars[5]} {chars[6]}{chars[7]}"
        )
    else:
        grouped = text
    return to_persian_digits(grouped)


def format_plate(chars: list[str]) -> tuple[str, str]:
    """Assemble recognized characters into ``(ascii, persian)`` strings.

    ``chars`` is the ordered list of recognized tokens (digits as ASCII, letters
    as Persian). The function is tolerant: it never raises on unexpected lengths,
    it simply renders what it has.
    """
    if not chars:
        return "", ""

    ascii_parts = [LETTER_LATIN.get(c, c) for c in chars]
    ascii_text = "".join(to_english_digits(p) for p in ascii_parts)

    # Persian representation keeps the letter glyph and uses Persian digits.
    fa_text = " ".join(
        to_persian_digits(c) if c in EN_DIGITS else c for c in chars
    )
    return ascii_text, fa_text
