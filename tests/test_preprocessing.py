"""
Unit Tests for Text Preprocessing Module
"""

import pytest
from src.preprocessing import clean_text, prepare_text_pair, calculate_text_statistics


def test_clean_text_null_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   ") == ""


def test_clean_text_whitespace_normalization():
    raw = "  Senior    Data   Scientist \n\n\n  with Python \t skills.   "
    cleaned = clean_text(raw)
    assert cleaned == "Senior Data Scientist\nwith Python skills."


def test_clean_text_preserves_technical_symbols():
    text = "Proficient in C++, C#, .NET, CI/CD, A/B Testing, and REST APIs."
    cleaned = clean_text(text)
    assert "C++" in cleaned
    assert "C#" in cleaned
    assert ".NET" in cleaned
    assert "CI/CD" in cleaned
    assert "A/B Testing" in cleaned


def test_prepare_text_pair():
    resume = "  Experienced Python developer. \n\n"
    jd = "  Looking for Python engineer. \t"
    clean_res, clean_jd = prepare_text_pair(resume, jd)
    assert clean_res == "Experienced Python developer."
    assert clean_jd == "Looking for Python engineer."


def test_calculate_text_statistics():
    sample = "Python machine learning models deployed on AWS."
    stats = calculate_text_statistics(sample)
    assert stats["word_count"] == 7
    assert stats["char_count"] == len(sample)
    assert stats["line_count"] == 1
