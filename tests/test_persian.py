from platrix.ocr.persian import (
    format_plate,
    to_english_digits,
    to_persian_digits,
)


def test_digit_roundtrip():
    assert to_english_digits("۱۲۳") == "123"
    assert to_persian_digits("123") == "۱۲۳"
    assert to_english_digits(to_persian_digits("905")) == "905"


def test_format_plate_digits_and_letter():
    chars = ["1", "2", "ب", "3", "4", "5", "6", "7"]
    ascii_text, fa_text = format_plate(chars)
    assert ascii_text == "12B34567"
    assert "ب" in fa_text
    assert "۱" in fa_text


def test_format_plate_empty():
    assert format_plate([]) == ("", "")
