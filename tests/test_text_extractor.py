import io
import pytest

from app.utils.text_extractor import (
    extract_pdf,
    extract_txt,
    extract_docx,
    extract_csv,
    extract_text,
    _EXTRACTORS,
)



def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    """Build a .docx in memory with python-docx."""
    from docx import Document
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# Minimal single-page PDF containing the text "Hello, World!".
# pdfplumber and pypdf are lenient about the xref offsets so this parses.
MINIMAL_PDF = b"""%PDF-1.1
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 24 Tf 100 700 Td (Hello, World!) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000274 00000 n 
0000000375 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
438
%%EOF
"""



class TestExtractTxt:
    def test_utf8_roundtrip(self):
        raw = "Hello world. This is a test.".encode("utf-8")
        result = extract_txt(raw)
        assert result["text"] == "Hello world. This is a test."
        assert result["metadata"]["encoding"] == "utf-8"
        assert result["metadata"]["page_count"] == 1

    def test_latin1_fallback(self):
        # 0xE9 is not valid standalone utf-8, but is 'é' in latin-1
        raw = b"caf\xe9 au lait"
        result = extract_txt(raw)
        assert result["text"] == "café au lait"
        assert result["metadata"]["encoding"] == "latin-1"

    def test_empty_bytes(self):
        result = extract_txt(b"")
        assert result["text"] == ""

    def test_metadata_word_count_is_char_count(self):
        """Documents a known bug: word_count is set to len(text), i.e. chars."""
        raw = b"one two three"
        result = extract_txt(raw)
        # Current behaviour: 13 (chars), not 3 (words).
        assert result["metadata"]["word_count"] == len("one two three")

class TestExtractCsv:
    def test_basic_csv(self):
        raw = b"name,age\nalice,30\nbob,25\n"
        result = extract_csv(raw)
        # Rows are joined with "|"
        assert "name|age" in result["text"]
        assert "alice|30" in result["text"]
        assert "bob|25" in result["text"]

    def test_metadata(self):
        result = extract_csv(b"a,b\n1,2\n")
        assert result["metadata"]["page_count"] == 1

    def test_empty_csv(self):
        result = extract_csv(b"")
        assert result["text"] == ""

class TestExtractDocx:
    def test_paragraphs_extracted(self):
        raw = _make_docx_bytes(["First paragraph.", "Second paragraph."])
        result = extract_docx(raw)
        assert "First paragraph." in result["text"]
        assert "Second paragraph." in result["text"]

    def test_blank_paragraphs_skipped(self):
        raw = _make_docx_bytes(["Keep me.", "", "   ", "Keep me too."])
        result = extract_docx(raw)
        # Blank and whitespace-only paragraphs are dropped
        assert result["text"] == "Keep me.\nKeep me too."



class TestExtractPdf:
    def test_pdf_text_extracted(self):
        result = extract_pdf(MINIMAL_PDF)
        assert "Hello, World!" in result["text"]
        assert result["metadata"]["page_count"] == 1
        assert len(result["metadata"]["pages"]) == 1

    def test_page_metadata_has_number(self):
        result = extract_pdf(MINIMAL_PDF)
        page = result["metadata"]["pages"][0]
        assert page["page_number"] == 1
        assert "char_count" in page

    def test_corrupt_pdf_raises(self):
        # Both pdfplumber and the pypdf fallback should fail
        with pytest.raises(Exception):
            extract_pdf(b"this is definitely not a pdf")



class TestExtractTextDispatch:
    @pytest.mark.asyncio
    async def test_txt_dispatch(self):
        result = await extract_text(b"hello from txt", "txt")
        assert result["text"] == "hello from txt"

    @pytest.mark.asyncio
    async def test_csv_dispatch(self):
        result = await extract_text(b"x,y\n1,2\n", "csv")
        assert "x|y" in result["text"]

    @pytest.mark.asyncio
    async def test_md_dispatches_to_txt(self):
        result = await extract_text(b"# heading", "md")
        assert result["text"] == "# heading"

    @pytest.mark.asyncio
    async def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            await extract_text(b"anything", "exe")

    def test_extractor_map_covers_all_declared_types(self):
        for ext in ("pdf", "txt", "docx", "md", "csv"):
            assert ext in _EXTRACTORS, f"missing extractor for .{ext}"
        # 'md' is registered as an alias of the txt extractor
        assert _EXTRACTORS["md"] is _EXTRACTORS["txt"]