"""Chunker module for splitting text into semantic chunks.

This module provides text chunking functionality extracted from parser.py.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from loguru import logger

ChunkType = Literal[
    "abstract", "introduction", "method", "experiment",
    "result", "conclusion", "lab_record", "concept_def", "other"
]


# ============================================================================
# Smart Chunking Configuration
# ============================================================================

# 章节类型配置 - 定义每个章节的分块规则
PAPER_STRUCTURE_RULES = {
    "abstract": {
        "keep_together": True,      # 摘要必须完整
        "max_size": 500,
        "priority": 1,
        "exclude": False,
    },
    "introduction": {
        "keep_together": False,     # 引言可以分割
        "max_size": 1000,
        "priority": 2,
        "exclude": False,
    },
    "method": {
        "keep_together": True,       # 方法论必须完整
        "max_size": 1200,
        "priority": 3,
        "exclude": False,
    },
    "experiment": {
        "keep_together": False,     # 实验可以分割
        "max_size": 1000,
        "priority": 4,
        "exclude": False,
    },
    "result": {
        "keep_together": True,      # 实验结果必须完整
        "max_size": 1000,
        "priority": 5,
        "exclude": False,
    },
    "conclusion": {
        "keep_together": True,      # 结论必须完整
        "max_size": 500,
        "priority": 6,
        "exclude": False,
    },
    # 特殊区域 - 排除
    "references": {
        "keep_together": True,
        "max_size": 99999,
        "priority": 99,
        "exclude": True,            # 排除，不索引
    },
    "appendix": {
        "keep_together": True,
        "max_size": 99999,
        "priority": 98,
        "exclude": True,            # 排除，不索引
    },
    "other": {
        "keep_together": False,
        "max_size": 800,
        "priority": 50,
        "exclude": False,
    },
}


# 避免断开的模式 - 用于边界检测
UNBREAKABLE_PATTERNS = [
    r'\$[^\$]+\$',                  # LaTeX 公式 $...$
    r'\\\[.+?\\\]',                 # LaTeX 公式 \[...\]
    r'\\begin\{[^}]+\}.+?\\end\{[^}]+\}',  # LaTeX 环境
    r'```[\s\S]+?```',             # 代码块
    r'\|.+\|.+\|',                 # 表格行
    r'Figure\s+\d+',              # 图片引用
    r'Table\s+\d+',               # 表格引用
    r'Algorithm\s+\d+',           # 算法引用
    r'Equation\s*\(\d+\)',        # 公式引用
    r'Eq\.\s*\(\d+\)',            # 公式引用 (简写)
    r'Ref\.\s*\d+',               # 引用
    r'\[\d+(?:,\s*\d+)*\]',           # 参考文献编号 [1], [1,2,3]
]


@dataclass
class SemanticChunk:
    """A chunk with semantic metadata."""
    content: str
    chunk_type: Optional[ChunkType] = None
    section_title: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class HierarchicalChunks:
    """Hierarchical chunks with large and small granularities."""
    large_chunks: List[SemanticChunk]
    small_chunks: List[SemanticChunk]
    # small_to_large: small_chunk_list_index -> large_chunk_list_index
    small_to_large: Dict[int, int]


class Chunker:
    """Abstract base class for text chunkers."""

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Split text into chunks."""
        raise NotImplementedError()


class SemanticChunker(Chunker):
    """
    Semantic-aware text chunker with section awareness.

    Detects section headings and applies intelligent chunking based on
    section type and content boundaries.
    """

    # Section heading patterns - ordered by specificity (more specific first)
    SECTION_PATTERNS = [
        # English numbered sections - generic
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(Introduction)\s*$", "introduction"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(Related\s+Work|Background|Prior\s+Work)\s*$", "introduction"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(Method|Methods|Methodology|Approach)\s*$", "method"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(Experiment|Experiments|Experimental\s+Setup)\s*$", "experiment"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(Result|Results|Evaluation)\s*$", "result"),
        (r"^\s*(\d+(?:\.\d+)*)\.?\s+(Conclusion|Conclusions|Summary)\s*$", "conclusion"),
        # Chinese numbered sections - generic
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
        (r"^\s*(References|Bibliography)\s*$", "references"),
        (r"^\s*(Appendix|Appendices)\s*$", "appendix"),
        # Chinese section headings without numbers
        (r"^\s*(摘要)\s*$", "abstract"),
        (r"^\s*(引言)\s*$", "introduction"),
        (r"^\s*(相关工作|背景)\s*$", "introduction"),
        (r"^\s*(方法)\s*$", "method"),
        (r"^\s*(实验)\s*$", "experiment"),
        (r"^\s*(结果)\s*$", "result"),
        (r"^\s*(结论)\s*$", "conclusion"),
        (r"^\s*(参考文献)\s*$", "references"),
        (r"^\s*(附录|附錄)\s*$", "appendix"),
        # Markdown headings
        (r"^#\s+(.*)$", "other"),
        (r"^##\s+(.*)$", "other"),
        (r"^###\s+(.*)$", "other"),
        # Numbered sections (generic) - keep these last
        (r"^\s*(\d+\.\d*\s+[A-Z][A-Za-z\s]+)\s*$", "other"),
        (r"^\s*(\d+\.\d*\s+[\u4e00-\u9fff]+)\s*$", "other"),
    ]

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Remove invalid Unicode surrogate characters."""
        return re.sub(r'[\ud800-\udfff]', '', text)

    @staticmethod
    def _remove_section_heading(section_text: str) -> str:
        """Remove section heading from section content."""
        lines = section_text.split('\n')
        if lines:
            first_line = lines[0].strip()
            # Check if first line matches any section heading pattern
            for pattern, _ in SemanticChunker.SECTION_PATTERNS:
                if re.match(pattern, first_line, re.IGNORECASE):
                    return '\n'.join(lines[1:])
        return section_text

    @staticmethod
    def detect_section_headings(text: str) -> List[tuple[str, int, int, ChunkType]]:
        """
        Detect obvious section headings.

        Returns:
            List of tuples (section_title, start_pos, end_pos, chunk_type)
        """
        headings = []
        lines = text.splitlines()
        current_pos = 0

        for line in lines:
            line_start = current_pos
            line_end = current_pos + len(line)

            for pattern, chunk_type in SemanticChunker.SECTION_PATTERNS:
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
    def _find_forbidden_ranges(text: str) -> List[tuple[int, int]]:
        """Find all positions where we shouldn't break (formulas, tables, code blocks)."""
        forbidden_ranges = []
        for pattern in UNBREAKABLE_PATTERNS:
            try:
                for match in re.finditer(pattern, text, re.DOTALL):
                    forbidden_ranges.append((match.start(), match.end()))
            except re.error:
                continue
        return forbidden_ranges

    @staticmethod
    def _is_position_safe(pos: int, forbidden_ranges: List[tuple[int, int]], min_gap: int = 10) -> bool:
        """Check if position is safe (not in forbidden range)."""
        for start, end in forbidden_ranges:
            # 如果位置在禁止范围内，或者距离太近
            if start - min_gap <= pos <= end + min_gap:
                return False
        return True

    @staticmethod
    def _find_safe_break_points(text: str, min_size: int = 100, max_size: int = 800) -> List[int]:
        """Find safe break point positions."""
        forbidden_ranges = SemanticChunker._find_forbidden_ranges(text)

        # 优先找段落边界
        safe_points = []
        para_pattern = r'\n\s*\n'

        last_end = 0
        for match in re.finditer(para_pattern, text):
            pos = match.end()
            if SemanticChunker._is_position_safe(pos, forbidden_ranges):
                # 检查这个位置是否在合理的大小范围内
                chunk_size = pos - last_end
                if min_size <= chunk_size <= max_size * 1.3:
                    safe_points.append(pos)
            last_end = match.start()

        # 如果段落边界不够，找句子边界
        if len(safe_points) < 3:
            sentence_pattern = r'(?<=[。！？.!?])\s+'
            safe_points = []
            last_end = 0

            for match in re.finditer(sentence_pattern, text):
                pos = match.end()
                if SemanticChunker._is_position_safe(pos, forbidden_ranges):
                    chunk_size = pos - last_end
                    if min_size <= chunk_size <= max_size * 1.3:
                        safe_points.append(pos)
                last_end = match.start()

        return safe_points

    @staticmethod
    def smart_chunk_section(
        section_text: str,
        section_type: str,
        min_size: int = 400,
        max_size: int = 800,
        overlap_ratio: float = 0.12,
    ) -> List[SemanticChunk]:
        """
        Smart chunking based on section type.
        """
        rule = PAPER_STRUCTURE_RULES.get(section_type, PAPER_STRUCTURE_RULES["other"])

        # 特殊区域排除
        if rule.get("exclude", False):
            return []

        section_max_size = rule.get("max_size", max_size)
        keep_together = rule.get("keep_together", False)

        # 清理section_text，去除章节标题本身
        section_text = SemanticChunker._remove_section_heading(section_text)

        # 特殊章节保持完整
        if keep_together and len(section_text) <= section_max_size:
            return [SemanticChunk(
                content=section_text.strip(),
                chunk_type=section_type,
            )]

        # 需要分割时，使用智能分块
        return SemanticChunker._smart_split_with_boundaries(
            section_text, section_type, min_size, section_max_size, overlap_ratio
        )

    @staticmethod
    def _smart_split_with_boundaries(
        text: str,
        section_type: str,
        min_size: int,
        max_size: int,
        overlap_ratio: float,
    ) -> List[SemanticChunk]:
        """Smart split text avoiding formulas, tables, etc."""
        if not text.strip():
            return []

        rule = PAPER_STRUCTURE_RULES.get(section_type, PAPER_STRUCTURE_RULES["other"])
        keep_together = rule.get("keep_together", False)

        # 对于 method/result 等需要保持完整性的章节，先尝试按段落
        if keep_together:
            # 尝试按大段落分割
            paragraphs = re.split(r'\n\s*\n', text)
            chunks = []
            current_chunk = ""

            for i, para in enumerate(paragraphs):
                if not para.strip():
                    continue

                if len(current_chunk) + len(para) <= max_size * 1.1:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    if current_chunk.strip():
                        chunks.append(SemanticChunk(
                            content=current_chunk.strip(),
                            chunk_type=section_type,
                        ))
                    # 如果段落本身太大，使用句子分割
                    if len(para) > max_size * 1.3:
                        sub_chunks = SemanticChunker._split_by_sentences(
                            para, min_size, max_size, overlap_ratio
                        )
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = para

            if current_chunk.strip():
                chunks.append(SemanticChunk(
                    content=current_chunk.strip(),
                    chunk_type=section_type,
                ))

            return chunks if chunks else [SemanticChunk(content=text.strip(), chunk_type=section_type)]
        else:
            # 对于可以分割的章节，使用标准的段落/句子分割
            return SemanticChunker.chunk_by_paragraph_sentence(
                text, min_size, max_size, overlap_ratio
            )

    @staticmethod
    def _split_by_sentences(
        text: str,
        min_size: int,
        max_size: int,
        overlap_ratio: float,
    ) -> List[SemanticChunk]:
        """Split large paragraphs by sentences."""
        # 找到安全的断点
        safe_points = SemanticChunker._find_safe_break_points(text, min_size, max_size)

        if not safe_points:
            return [SemanticChunk(content=text.strip())]

        chunks = []
        overlap_size = int(max_size * overlap_ratio)

        last_end = 0
        for point in safe_points:
            chunk_text = text[last_end:point]
            if len(chunk_text) >= min_size:
                chunks.append(SemanticChunk(content=chunk_text.strip(), chunk_type="other"))
            last_end = point - overlap_size if point - overlap_size > last_end else last_end

        # 处理剩余部分
        if last_end < len(text):
            remaining = text[last_end:]
            if len(remaining) >= min_size // 2:
                chunks.append(SemanticChunk(content=remaining.strip(), chunk_type="other"))

        return chunks

    @staticmethod
    def chunk_by_paragraph_sentence(
        text: str,
        min_chunk_size: int = 500,
        max_chunk_size: int = 800,
        overlap_ratio: float = 0.12,
    ) -> List[SemanticChunk]:
        """
        Split text by paragraph/sentence boundaries with overlap.
        """
        if not text.strip():
            return []

        overlap_size = int(max_chunk_size * overlap_ratio)
        chunks: List[SemanticChunk] = []

        # Split into paragraphs first
        paragraphs = re.split(r"\n\s*\n", text)
        para_boundaries: List[tuple[str, int, int]] = []
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
    ) -> List[SemanticChunk]:
        """
        Main method: Chunk with section awareness.
        """
        # First detect section headings
        headings = SemanticChunker.detect_section_headings(text)

        # If no headings detected, fall back to paragraph/sentence chunking
        if not headings:
            return SemanticChunker.chunk_by_paragraph_sentence(
                text, min_chunk_size, max_chunk_size, overlap_ratio
            )

        # With headings: Create sections based on headings
        chunks: List[SemanticChunk] = []

        # Create sections between headings
        sections: List[tuple[str, int, int, Optional[str], ChunkType]] = []

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

        # Process each section - filter out excluded sections (references, appendix)
        for section_content, section_start, section_end, section_title, chunk_type in sections:
            if not section_content.strip():
                continue

            # Check if this section should be excluded (references, appendix)
            section_lower = (section_title or "").lower()
            if any(kw in section_lower for kw in ["reference", "bibliography", "appendix"]):
                logger.debug(f"[SmartChunk] Excluding section: {section_title}")
                continue

            # Get the rule for this section type
            rule = PAPER_STRUCTURE_RULES.get(chunk_type, PAPER_STRUCTURE_RULES["other"])

            # If excluded by rule, skip
            if rule.get("exclude", False):
                logger.debug(f"[SmartChunk] Excluding section (rule): {section_title} ({chunk_type})")
                continue

            # Use smart_chunk_section for intelligent splitting
            section_max_size = rule.get("max_size", max_chunk_size)

            # If section is small enough, keep as single chunk
            if len(section_content) <= section_max_size:
                # Clean section content - remove heading
                cleaned_content = SemanticChunker._remove_section_heading(section_content)
                # Skip empty chunks (e.g., section with only a heading)
                if cleaned_content.strip():
                    chunks.append(SemanticChunk(
                        content=cleaned_content,
                        chunk_type=chunk_type,
                        section_title=section_title,
                        start_pos=section_start,
                        end_pos=section_end
                    ))
            else:
                # Section is too big - use smart chunking based on section type
                sub_chunks = SemanticChunker.smart_chunk_section(
                    section_content, chunk_type, min_chunk_size, section_max_size, overlap_ratio
                )
                # Adjust positions and add section metadata
                for sub_chunk in sub_chunks:
                    sub_chunk.start_pos += section_start
                    sub_chunk.end_pos += section_start
                    if not sub_chunk.chunk_type:
                        sub_chunk.chunk_type = chunk_type
                    if not sub_chunk.section_title:
                        sub_chunk.section_title = section_title
                    chunks.append(sub_chunk)

        return chunks

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Split text into semantic chunks."""
        return self.chunk_with_section_awareness(text)


class HierarchicalChunker(Chunker):
    """
    Hierarchical chunker that creates both large and small chunks.

    First creates large chunks with section awareness,
    then splits each large chunk into smaller chunks.
    """

    def __init__(
        self,
        small_min: int = 400,
        small_max: int = 700,
        small_overlap: float = 0.15,
        large_min: int = 1200,
        large_max: int = 1800,
        large_overlap: float = 0.1,
    ):
        self.small_min = small_min
        self.small_max = small_max
        self.small_overlap = small_overlap
        self.large_min = large_min
        self.large_max = large_max
        self.large_overlap = large_overlap

    def chunk_hierarchical(self, text: str) -> HierarchicalChunks:
        """
        Create hierarchical chunks:
        1. First create large chunks with section awareness
        2. Then split each large chunk into smaller chunks
        """
        logger.info("[DualGranularity] Chunking: {} chars, large={}-{} ({}%), small={}-{} ({}%)",
                    len(text), self.large_min, self.large_max, int(self.large_overlap * 100),
                    self.small_min, self.small_max, int(self.small_overlap * 100))

        # Step 1: Create large chunks first (section-aware)
        large_chunks = SemanticChunker.chunk_with_section_awareness(
            text, self.large_min, self.large_max, self.large_overlap
        )

        # Step 2: Split each large chunk into small chunks
        small_chunks = []
        small_to_large = {}

        for large_idx, large_chunk in enumerate(large_chunks):
            # Extract the large chunk content
            sub_text = large_chunk.content

            # Split into small chunks within this large chunk
            sub_chunks = SemanticChunker.chunk_with_section_awareness(
                sub_text, self.small_min, self.small_max, self.small_overlap
            )

            # Adjust positions to be relative to the full text
            for sub_chunk in sub_chunks:
                sub_chunk.start_pos += large_chunk.start_pos
                sub_chunk.end_pos += large_chunk.start_pos
                sub_chunk.section_title = large_chunk.section_title
                sub_chunk.chunk_type = large_chunk.chunk_type

                # Record the mapping
                small_to_large[len(small_chunks)] = large_idx
                small_chunks.append(sub_chunk)

        logger.info("[DualGranularity] Done: {} large, {} small chunks", len(large_chunks), len(small_chunks))

        return HierarchicalChunks(
            large_chunks=large_chunks,
            small_chunks=small_chunks,
            small_to_large=small_to_large,
        )

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Split text into chunks (returns large chunks)."""
        return self.chunk_hierarchical(text).large_chunks
