from pathlib import Path
import pytest
from src.config import SAMPLE_DATA_DIR
from src.resume_parser import (
    extract_resume_text,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_txt,
)


def test_extract_text_from_txt():
    txt_path = SAMPLE_DATA_DIR / "resume_data_scientist.txt"
    if txt_path.exists():
        text = extract_text_from_txt(txt_path)
        assert len(text) > 50
        assert "Alex Morgan" in text
        assert "Data Scientist" in text
        assert "Python" in text


def test_extract_text_from_docx():
    docx_path = SAMPLE_DATA_DIR / "resume_software_engineer.docx"
    if docx_path.exists():
        text = extract_text_from_docx(docx_path)
        assert len(text) > 50
        assert "Taylor Reed" in text
        assert "Software Engineer" in text
        assert "Docker" in text


def test_extract_text_from_pdf():
    pdf_path = SAMPLE_DATA_DIR / "resume_marketing_manager.pdf"
    if pdf_path.exists():
        text = extract_text_from_pdf(pdf_path)
        assert len(text) > 50
        assert "Jordan Lee" in text
        assert "Marketing" in text


def test_extract_resume_text_auto_detection():
    txt_path = SAMPLE_DATA_DIR / "resume_data_scientist.txt"
    if txt_path.exists():
        text = extract_resume_text(txt_path)
        assert "Data Scientist" in text


def test_extract_nonexistent_file_raises_error():
    with pytest.raises(Exception):
        extract_resume_text("non_existent_file_xyz.pdf")
