"""Document parser for various file formats."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

ChunkType = Literal[
    "abstract", "introduction", "method", "experiment",
    "result", "conclusion", "lab_record", "concept_def", "other"
]


@dataclass
class SemanticChunk:
    """A chunk with semantic metadata."""
    content: str
    chunk_type: Optional[ChunkType] = None
    section_title: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0


class DocumentParser:
    """Parse various document formats (PDF, Markdown, Word, text)."""

    # Section heading patterns for Phase 2b - ordered by specificity (more specific first)
    SECTION_PATTERNS = [
        # English numbered sections - generic, no hardcoded numbers
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(Introduction)\s*$", "introduction"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(Related\s+Work|Background|Prior\s+Work)\s*$", "introduction"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(Method|Methods|Methodology|Approach)\s*$", "method"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(Experiment|Experiments|Experimental\s+Setup)\s*$", "experiment"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(Result|Results|Evaluation)\s*$", "result"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(Conclusion|Conclusions|Summary)\s*$", "conclusion"),
        # Chinese numbered sections - generic, no hardcoded numbers
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(引言)\s*$", "introduction"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(相关工作|背景|文献综述)\s*$", "introduction"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(方法|算法|方法论)\s*$", "method"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(实验|实验设置)\s*$", "experiment"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(结果|评估)\s*$", "result"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(结论|总结)\s*$", "conclusion"),
        # English section headings without numbers
        (r"^\s*(Abstract)\s*$", "abstract"),
        (r"^\s*(Introduction)\s*$", "introduction"),
        (r"^\s*(Related\s+Work)\s*$", "introduction"),
        (r"^\s*(Method|Methods|Methodology)\s*$", "method"),
        (r"^\s*(Experiment|Experiments|Experimental\s+Setup)\s*$", "experiment"),
        (r"^\s*(Result|Results)\s*$", "result"),
        (r"^\s*(Conclusion|Conclusions)\s*$", "conclusion"),
        (r"^\s*(References|Bibliography)\s*$", "other"),
        # Chinese section headings without numbers
        (r"^\s*(摘要)\s*$", "abstract"),
        (r"^\s*(引言)\s*$", "introduction"),
        (r"^\s*(相关工作|背景)\s*$", "introduction"),
        (r"^\s*(方法)\s*$", "method"),
        (r"^\s*(实验)\s*$", "experiment"),
        (r"^\s*(结果)\s*$", "result"),
        (r"^\s*(结论)\s*$", "conclusion"),
        (r"^\s*(参考文献)\s*$", "other"),
        # Markdown headings
        (r"^#\s+(.*)$", "other"),
        (r"^##\s+(.*)$", "other"),
        (r"^###\s+(.*)$", "other"),
        # Numbered sections (generic) - keep these last
        (r"^\s*(\d+\.\d*\s+[A-Z][A-Za-z\s]+)\s*$", "other"),
        (r"^\s*(\d+\.\d*\s+[\u4e00-\u9fff]+)\s*$", "other"),
    ]

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
    def extract_metadata(path: Path, content: str) -> dict:
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
            abstract = abstract_match.group(1).strip()

        return {
            "title": title,
            "doc_type": doc_type,
            "abstract": abstract
        }

    @staticmethod
    def detect_section_headings(text: str) -> list[tuple[str, int, int, ChunkType]]:
        """
        Phase 2b: Detect obvious section headings.

        Supports formats:
        - "Abstract", "摘要"
        - "1. Introduction", "1.1 相关工作"
        - "References", "参考文献"
        - "Method", "Methods", "方法"
        - "Experiment", "Experiments", "实验"
        - "Results", "结果"
        - "Conclusion", "Conclusions", "结论"

        Args:
            text: Input text

        Returns:
            List of tuples (section_title, start_pos, end_pos, chunk_type)
        """
        headings = []
        lines = text.splitlines()
        current_pos = 0

        for line in lines:
            line_start = current_pos
            line_end = current_pos + len(line)

            for pattern, chunk_type in DocumentParser.SECTION_PATTERNS:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # If there are capture groups, find the last non-numeric one as title
                    if match.groups():
                        title = None
                        for group in match.groups():
                            if group and not re.match(r"^\d+(?:\.\d+)*$", group.strip()):
                                title = group.strip()
                                break
                        if not title:
                            title = line.strip()
                    else:
                        title = line.strip()

                    headings.append((title, line_start, line_end, chunk_type))
                    break

            current_pos = line_end + 1  # +1 for newline

        return headings

    @staticmethod
    def chunk_by_paragraph_sentence(
        text: str,
        min_chunk_size: int = 500,
        max_chunk_size: int = 800,
        overlap_ratio: float = 0.12,
    ) -> list[SemanticChunk]:
        """
        Phase 2a: Split text by paragraph/sentence boundaries with overlap.

        - Priority on paragraph boundaries
        - Falls back to sentence boundaries for large paragraphs
        - Preserves overlap_ratio between chunks

        Args:
            text: Input text
            min_chunk_size: Minimum chunk size (characters)
            max_chunk_size: Maximum chunk size (characters)
            overlap_ratio: Overlap ratio between chunks (0.0-1.0)

        Returns:
            List of SemanticChunk objects
        """
        if not text.strip():
            return []

        overlap_size = int(max_chunk_size * overlap_ratio)
        chunks: list[SemanticChunk] = []

        # Split into paragraphs first
        paragraphs = re.split(r"\n\s*\n", text)
        para_boundaries: list[tuple[str, int, int]] = []
        current_pos = 0

        for para in paragraphs:
            if not para.strip():
                current_pos += len(para) + 2  # +2 for newlines
                continue
            para_start = current_pos
            para_end = current_pos + len(para)
            para_boundaries.append((para, para_start, para_end))
            current_pos = para_end + 2  # +2 for newlines

        current_chunk_text = ""
        current_start = 0

        for para_text, para_start, para_end in para_boundaries:
            para_length = len(para_text)

            # Case 1: Paragraph is way too big - split by sentences
            if para_length > max_chunk_size * 1.5:
                # If we have accumulated content, finalize it first
                if current_chunk_text:
                    chunks.append(SemanticChunk(
                        content=current_chunk_text,
                        start_pos=current_start,
                        end_pos=para_start
                    ))
                    # Prepare overlap for next chunk
                    overlap_text = current_chunk_text[-overlap_size:] if overlap_size > 0 else ""
                    current_chunk_text = overlap_text
                    current_start = para_start - len(overlap_text) if overlap_text else para_start

                # Split this large paragraph by sentences
                sentences = re.split(r"(?<=[。！？.!?])\s+", para_text)
                sentence_pos = para_start

                for sentence in sentences:
                    if not sentence.strip():
                        sentence_pos += len(sentence)
                        continue

                    sentence_len = len(sentence)

                    # Check if adding this sentence would exceed max_chunk_size
                    if len(current_chunk_text) + sentence_len > max_chunk_size and len(current_chunk_text) >= min_chunk_size:
                        chunks.append(SemanticChunk(
                            content=current_chunk_text,
                            start_pos=current_start,
                            end_pos=sentence_pos
                        ))
                        # Prepare overlap
                        overlap_text = current_chunk_text[-overlap_size:] if overlap_size > 0 else ""
                        current_chunk_text = overlap_text + sentence
                        current_start = sentence_pos - len(overlap_text) if overlap_text else sentence_pos
                    else:
                        current_chunk_text += (" " if current_chunk_text and not current_chunk_text.endswith(" ") else "") + sentence

                    sentence_pos += len(sentence)

            # Case 2: Paragraph fits reasonably
            else:
                # Check if adding this paragraph would exceed max_chunk_size
                if len(current_chunk_text) + para_length > max_chunk_size and len(current_chunk_text) >= min_chunk_size:
                    chunks.append(SemanticChunk(
                        content=current_chunk_text,
                        start_pos=current_start,
                        end_pos=para_start
                    ))
                    # Prepare overlap
                    overlap_text = current_chunk_text[-overlap_size:] if overlap_size > 0 else ""
                    current_chunk_text = overlap_text + para_text
                    current_start = para_start - len(overlap_text) if overlap_text else para_start
                else:
                    if current_chunk_text:
                        current_chunk_text += "\n\n" + para_text
                    else:
                        current_chunk_text = para_text
                        current_start = para_start

        # Add the final chunk
        if current_chunk_text.strip():
            chunks.append(SemanticChunk(
                content=current_chunk_text,
                start_pos=current_start,
                end_pos=len(text)
            ))

        return chunks

    @staticmethod
    def chunk_with_section_awareness(
        text: str,
        min_chunk_size: int = 500,
        max_chunk_size: int = 800,
        overlap_ratio: float = 0.12,
    ) -> list[SemanticChunk]:
        """
        Phase 2b main method: Chunk with section awareness.

        1. First detect section headings
        2. If detected, prefer section boundaries for chunking
        3. If not detected, fall back to paragraph/sentence chunking
        4. All strategies preserve overlap

        Args:
            text: Input text
            min_chunk_size: Minimum chunk size (characters)
            max_chunk_size: Maximum chunk size (characters)
            overlap_ratio: Overlap ratio between chunks (0.0-1.0)

        Returns:
            List of SemanticChunk objects with section metadata
        """
        # First detect section headings
        headings = DocumentParser.detect_section_headings(text)

        # If no headings detected, fall back to paragraph/sentence chunking
        if not headings:
            return DocumentParser.chunk_by_paragraph_sentence(
                text, min_chunk_size, max_chunk_size, overlap_ratio
            )

        # With headings: Create sections based on headings
        overlap_size = int(max_chunk_size * overlap_ratio)
        chunks: list[SemanticChunk] = []

        # Create sections between headings
        sections: list[tuple[str, int, int, Optional[str], ChunkType]] = []

        # First section (before first heading)
        if headings[0][1] > 0:
            sections.append((text[0:headings[0][1]], 0, headings[0][1], None, "other"))

        # Sections from headings
        for i, (title, start, end, chunk_type) in enumerate(headings):
            section_start = start
            if i < len(headings) - 1:
                section_end = headings[i + 1][1]
            else:
                section_end = len(text)
            section_content = text[section_start:section_end]
            sections.append((section_content, section_start, section_end, title, chunk_type))

        # Process each section
        for section_content, section_start, section_end, section_title, chunk_type in sections:
            if not section_content.strip():
                continue

            # If section is small enough, keep as single chunk
            if len(section_content) <= max_chunk_size:
                chunks.append(SemanticChunk(
                    content=section_content,
                    chunk_type=chunk_type,
                    section_title=section_title,
                    start_pos=section_start,
                    end_pos=section_end
                ))
            else:
                # Section is too big - split internally using paragraph/sentence
                sub_chunks = DocumentParser.chunk_by_paragraph_sentence(
                    section_content, min_chunk_size, max_chunk_size, overlap_ratio
                )
                # Adjust positions and add section metadata
                for sub_chunk in sub_chunks:
                    sub_chunk.start_pos += section_start
                    sub_chunk.end_pos += section_start
                    sub_chunk.chunk_type = chunk_type
                    sub_chunk.section_title = section_title
                    chunks.append(sub_chunk)

        return chunks

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[tuple[str, int, int]]:
        """
        Legacy chunking method - maintained for backward compatibility.
        Delegates to chunk_with_section_awareness with backward-compatible parameters.

        Args:
            text: Input text to split
            chunk_size: Target size of each chunk (characters)
            chunk_overlap: Overlap between chunks (characters)

        Returns:
            List of tuples (content, start_pos, end_pos)
        """
        # Convert to new parameter format
        min_chunk_size = max(300, chunk_size // 2)
        max_chunk_size = chunk_size
        overlap_ratio = chunk_overlap / chunk_size if chunk_size > 0 else 0.12

        # Use the new semantic chunking
        semantic_chunks = DocumentParser.chunk_with_section_awareness(
            text, min_chunk_size, max_chunk_size, overlap_ratio
        )

        # Convert back to tuple format for backward compatibility
        return [(chunk.content, chunk.start_pos, chunk.end_pos) for chunk in semantic_chunks]
