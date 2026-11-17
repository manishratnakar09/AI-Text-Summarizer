"""Document text extraction + validation (TXT, PDF, DOCX)."""
import io
import os
import re
import zipfile

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "300"))


class ExtractionError(ValueError):
    """Raised with a user-friendly message when a file cannot be read."""


def clean_text(text: str) -> str:
    """Normalise whitespace/control chars while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_file(filename: str, data: bytes, max_bytes: int) -> str:
    """Validate extension, size and file signature. Returns the extension."""
    if not filename:
        raise ExtractionError("No file selected.")
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise ExtractionError("Unsupported file type. Please upload a .txt, .pdf or .docx file.")
    if not data:
        raise ExtractionError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ExtractionError(f"File is too large. Maximum allowed size is {max_bytes // (1024 * 1024)} MB.")
    # Check magic bytes so a renamed file is not blindly parsed.
    if ext == ".pdf" and not data.lstrip()[:5] == b"%PDF-":
        raise ExtractionError("This file is not a valid PDF.")
    if ext == ".docx" and not data.startswith(b"PK"):
        raise ExtractionError("This file is not a valid DOCX document.")
    if ext == ".txt" and b"\x00" in data[:4096] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise ExtractionError("This file does not look like a plain-text file.")
    return ext


def _read_txt(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ExtractionError("Could not decode the text file.")


def _read_pdf(data: bytes) -> str:
    import pdfplumber
    pages = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > MAX_PDF_PAGES:
                raise ExtractionError(f"PDF has too many pages (limit {MAX_PDF_PAGES}).")
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
    except ExtractionError:
        raise
    except Exception:
        raise ExtractionError("Could not read this PDF. It may be corrupted or password-protected.")
    text = "\n\n".join(pages)
    if not text.strip():
        raise ExtractionError("No selectable text found in this PDF (it may be a scanned image).")
    return text


def _read_docx(data: bytes) -> str:
    import docx
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception:
        raise ExtractionError("Could not read this DOCX file. It may be corrupted.")
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n\n".join(parts)
    if not text.strip():
        raise ExtractionError("This DOCX file contains no text.")
    return text


def extract_text(filename: str, data: bytes, max_bytes: int = 10 * 1024 * 1024) -> str:
    ext = validate_file(filename, data, max_bytes)
    raw = {".txt": _read_txt, ".pdf": _read_pdf, ".docx": _read_docx}[ext](data)
    text = clean_text(raw)
    if not text:
        raise ExtractionError("No readable text found in the file.")
    return text
