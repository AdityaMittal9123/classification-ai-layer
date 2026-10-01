import io
import os
import logging
import tempfile
from typing import List, Optional

from schemas import EmailAttachment
from llm_client import LLMClientWrapper

# All extraction libraries below are free and open source
try:
    from langchain_core.documents import Document  # MIT
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader  # MIT
except ImportError:
    Document = None
    PyPDFLoader = None
    Docx2txtLoader = None

try:
    import pandas as pd  # BSD-3-Clause
except ImportError:
    pd = None

try:
    from markitdown import MarkItDown, StreamInfo  # MIT
except ImportError:
    MarkItDown = None
    StreamInfo = None

try:
    import pypdfium2 as pdfium  # Apache-2.0 / BSD-3-Clause
except ImportError:
    pdfium = None

try:
    from PIL import Image  # Pillow, MIT-CMU
except ImportError:
    Image = None

logger = logging.getLogger(__name__)

PNG_MIME = "image/png"

# Formats read with LangChain document loaders
LANGCHAIN_TYPES = {".pdf", ".docx", ".xlsx", ".xls"}
# Images the vision model accepts directly
VISION_IMAGE_TYPES = {
    ".png": PNG_MIME,
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
# Images that are converted to PNG before being sent to the vision model
CONVERTIBLE_IMAGE_TYPES = {".tif", ".tiff", ".bmp"}
# Other rich formats converted to text by MarkItDown
MARKITDOWN_TYPES = {".pptx", ".msg", ".html", ".htm", ".epub"}
# Plain text formats that are decoded as-is
TEXT_TYPES = {".txt", ".csv", ".md", ".json", ".xml", ".log"}

# Resolution used when rendering scanned PDF pages for the vision model
PDF_RENDER_DPI = 150


class AttachmentExtractor:
    """
    Reads the full text of attachment files (no summarization, no truncation):
    - PDF (LangChain PyPDFLoader): text per page; scanned pages (no text) are rendered and read by the vision model.
    - Word .docx (LangChain Docx2txtLoader).
    - Excel .xlsx / .xls (pandas, one LangChain Document per sheet).
    - PowerPoint, Outlook .msg, HTML, EPUB (MarkItDown).
    - Images (.png, .jpg, .jpeg, .gif, .webp, .tif, .tiff, .bmp): transcribed/described by the vision model.
    - Plain text (.txt, .csv, .md, .json, .xml, .log): decoded as UTF-8.
    """

    def __init__(self, llm_client: LLMClientWrapper):
        self.llm_client = llm_client
        self.markitdown = MarkItDown(enable_plugins=False) if MarkItDown else None

    def extract_text(self, attachment: EmailAttachment) -> Optional[str]:
        """Returns the attachment's full text, or None if it has no file data or the type is unsupported."""
        data = self._load_bytes(attachment)
        if data is None:
            logger.warning(f"Attachment '{attachment.filename}' has no content or file_path; skipping extraction.")
            return None

        ext = os.path.splitext(attachment.filename)[1].lower()
        try:
            if ext in LANGCHAIN_TYPES:
                documents = self._load_with_langchain(data, ext)
                return "\n".join(doc.page_content for doc in documents)
            if ext in MARKITDOWN_TYPES:
                return self._extract_with_markitdown(data, ext)
            if ext in VISION_IMAGE_TYPES:
                return self.llm_client.describe_image(data, VISION_IMAGE_TYPES[ext])
            if ext in CONVERTIBLE_IMAGE_TYPES:
                return self.llm_client.describe_image(self._image_to_png(data), PNG_MIME)
            if ext in TEXT_TYPES:
                return data.decode("utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to extract text from attachment '{attachment.filename}': {e}")
            return None

        logger.warning(f"Unsupported attachment type '{ext}' for '{attachment.filename}'; skipping extraction.")
        return None

    def _load_bytes(self, attachment: EmailAttachment) -> Optional[bytes]:
        if attachment.content is not None:
            return attachment.content
        if attachment.file_path:
            with open(attachment.file_path, "rb") as f:
                return f.read()
        return None

    def _load_with_langchain(self, data: bytes, ext: str) -> List["Document"]:
        """Loads PDF / Word / Excel data into LangChain Documents."""
        if Document is None:
            raise ImportError("LangChain is required: pip install langchain-community pypdf docx2txt")

        if ext in (".xlsx", ".xls"):
            return self._load_excel(data)

        # LangChain file loaders read from disk, so the bytes are written to a temporary file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            if ext == ".pdf":
                return self._load_pdf(tmp_path, data)
            return Docx2txtLoader(tmp_path).load()
        finally:
            os.remove(tmp_path)

    def _load_pdf(self, path: str, data: bytes) -> List["Document"]:
        """One Document per page; pages without a text layer (scanned) are read by the vision model."""
        pages = PyPDFLoader(path, mode="page").load()
        for page_index, page in enumerate(pages):
            if not page.page_content.strip():
                page.page_content = self.llm_client.describe_image(self._render_pdf_page(data, page_index), PNG_MIME)
            page.page_content = f"[Page {page_index + 1}]\n{page.page_content.strip()}"
        return pages

    def _load_excel(self, data: bytes) -> List["Document"]:
        """One Document per sheet, rows written as 'a | b | c'."""
        if pd is None:
            raise ImportError("pandas is required for Excel extraction: pip install pandas openpyxl xlrd")

        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=str)
        documents = []
        for sheet_name, frame in sheets.items():
            frame = frame.dropna(how="all").fillna("")
            rows = [" | ".join(str(value) for value in row) for row in frame.itertuples(index=False)]
            documents.append(
                Document(page_content=f"[Sheet: {sheet_name}]\n" + "\n".join(rows), metadata={"sheet": sheet_name})
            )
        return documents

    def _render_pdf_page(self, data: bytes, page_index: int) -> bytes:
        if pdfium is None:
            raise ImportError("pypdfium2 is required to read scanned PDF pages: pip install pypdfium2")

        pdf = pdfium.PdfDocument(data)
        try:
            pil_image = pdf[page_index].render(scale=PDF_RENDER_DPI / 72).to_pil()
        finally:
            pdf.close()
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _extract_with_markitdown(self, data: bytes, ext: str) -> str:
        if self.markitdown is None:
            raise ImportError("MarkItDown is required for PowerPoint/Outlook/HTML: pip install 'markitdown[pptx,outlook]'")

        result = self.markitdown.convert_stream(io.BytesIO(data), stream_info=StreamInfo(extension=ext))
        return result.text_content

    def _image_to_png(self, data: bytes) -> bytes:
        if Image is None:
            raise ImportError("Pillow is required for TIFF/BMP conversion: pip install pillow")

        with Image.open(io.BytesIO(data)) as img:
            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue()
