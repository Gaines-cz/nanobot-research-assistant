"""Document parser for various file formats.

This module provides document parsing functionality extracted from the
original parser.py file.
"""

import re
from pathlib import Path
from typing import Dict


class DocumentParser:
    """Parse various document formats (PDF, Markdown, Word, text)."""

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Remove invalid Unicode surrogate characters."""
        # Remove unpaired surrogates (D800-DFFF)
        return re.sub(r'[\ud800-\udfff]', '', text)

    @staticmethod
    def parse(path: Path) -> tuple[str, str]:
        """
        Parse a document and return (content, file_type).

        Returns:
            Tuple of (content_text, file_type) where file_type is
            'pdf', 'markdown', 'word', or 'text'.
        """
        ext = path.suffix.lower()
        if ext == ".pdf":
            content = DocumentParser.parse_pdf(path)
        elif ext in (".md", ".markdown"):
            content = DocumentParser.parse_markdown(path)
        elif ext in (".docx", ".doc"):
            content = DocumentParser.parse_word(path)
        else:
            content = DocumentParser.parse_text(path)

        # Sanitize: remove invalid Unicode surrogates
        content = DocumentParser._sanitize_text(content)

        file_type = ext.lstrip(".")
        return content, file_type

    @staticmethod
    def parse_pdf(path: Path) -> str:
        """Parse a PDF file using PyMuPDF with pypdf fallback."""
        # Try PyMuPDF first (better quality)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            parts: list[str] = []

            for page in doc:
                # Use blocks to maintain reading order
                blocks = page.get_text("blocks", sort=True)
                for block in blocks:
                    text = block[4].strip()
                    if text:
                        parts.append(text)

            doc.close()
            if parts:
                return "\n\n".join(parts)
        except ImportError:
            pass

        # Fallback to pypdf
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "PyMuPDF or pypdf is required for PDF parsing. "
                "Install with: pip install 'nanobot-ai[rag]'"
            )

        reader = PdfReader(path)
        parts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
        return "\n\n".join(parts)

    @staticmethod
    def parse_markdown(path: Path) -> str:
        """Parse a Markdown file."""
        return path.read_text(encoding="utf-8")

    @staticmethod
    def parse_word(path: Path) -> str:
        """Parse a Word document (.docx) using python-docx."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required for Word parsing. "
                "Install with: pip install 'nanobot-ai[rag]'"
            )

        doc = Document(path)
        parts: list[str] = []
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)
        return "\n\n".join(parts)

    @staticmethod
    def parse_text(path: Path) -> str:
        """Parse a plain text file."""
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def extract_metadata(path: Path, content: str) -> Dict:
        """
        Extract document metadata: {title, doc_type, abstract}.

        Args:
            path: Document path
            content: Document content

        Returns:
            Dict with title, doc_type, and abstract (all optional)
        """
        filename = path.name.lower()
        title = path.stem

        # Try to guess doc_type from filename/content
        doc_type = "other"
        if any(keyword in filename or keyword in content[:1000].lower()
               for keyword in ["paper", "arxiv", "neurips", "icml", "cvpr", "acl"]):
            doc_type = "paper"
        elif any(keyword in filename for keyword in ["note", "notes", "lab", "experiment"]):
            doc_type = "lab_note"
        elif any(keyword in filename for keyword in ["concept", "definition", "define"]):
            doc_type = "concept"

        # Try to extract abstract (first paragraph after Abstract heading)
        abstract = None
        abstract_match = re.search(
            r"(?:Abstract|摘要)[\s:]*([^\n]{0,1000}?)(?=\n\s*\n|\n\s*1\.\s|\n\s*Introduction|\Z)",
            content[:3000],
            re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            abstract_group = abstract_match.group(1)
            abstract = abstract_group.strip() if abstract_group else None

        return {
            "title": title,
            "doc_type": doc_type,
            "abstract": abstract
        }
