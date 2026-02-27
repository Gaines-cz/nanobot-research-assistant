"""Document parser for various file formats."""

import re
from pathlib import Path


class DocumentParser:
    """Parse various document formats (PDF, Markdown, Word, text)."""

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
            return DocumentParser.parse_pdf(path), "pdf"
        elif ext in (".md", ".markdown"):
            return DocumentParser.parse_markdown(path), "markdown"
        elif ext in (".docx", ".doc"):
            return DocumentParser.parse_word(path), "word"
        else:
            return DocumentParser.parse_text(path), "text"

    @staticmethod
    def parse_pdf(path: Path) -> str:
        """Parse a PDF file using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF parsing. "
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
    def chunk_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[tuple[str, int, int]]:
        """
        Split text into chunks at sentence boundaries.

        Args:
            text: Input text to split
            chunk_size: Target size of each chunk (characters)
            chunk_overlap: Overlap between chunks (characters)

        Returns:
            List of tuples (content, start_pos, end_pos)
        """
        if not text.strip():
            return []

        # Split into sentences (simple heuristic)
        sentences = re.split(r"(?<=[。！？.!?])\s+", text)
        chunks: list[tuple[str, int, int]] = []

        current_chunk: list[str] = []
        current_length = 0
        current_start = 0
        text_position = 0

        for sentence in sentences:
            if not sentence:
                continue

            sentence_len = len(sentence)

            # If adding this sentence would exceed chunk size and we already have content
            if current_chunk and current_length + sentence_len > chunk_size:
                # Join the current chunk
                chunk_content = "".join(current_chunk)
                chunks.append((chunk_content, current_start, text_position))

                # Calculate overlap for next chunk
                overlap_content = chunk_content[-chunk_overlap:] if chunk_overlap > 0 else ""
                current_chunk = [overlap_content] if overlap_content else []
                current_length = len(overlap_content)
                current_start = text_position - len(overlap_content) if overlap_content else text_position

            current_chunk.append(sentence)
            current_length += sentence_len
            text_position += sentence_len

        # Add the last chunk
        if current_chunk:
            chunk_content = "".join(current_chunk)
            chunks.append((chunk_content, current_start, text_position))

        return chunks
