import json
import logging
import re
from typing import Dict, List, Optional, Union

from src.schemas.indexing.models import ChunkMetadata, TextChunk

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits paper text into overlapping chunks for hybrid search.

    Two strategies:
    1. Section-based: uses Docling-extracted sections as boundaries (preferred)
    2. Word-based: sliding window fallback when sections unavailable

    Like a Strategy pattern — picks the best approach automatically.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        overlap_size: int = 100,
        min_chunk_size: int = 100,
    ):
        """
        chunk_size:    target words per chunk (600 = ~2400 chars, ~3 paragraphs)
        overlap_size:  words shared between adjacent chunks (prevents boundary splits)
        min_chunk_size: skip chunks smaller than this (avoids indexing headers alone)
        """
        if overlap_size >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

        logger.info(f"TextChunker ready: size={chunk_size}, overlap={overlap_size}, min={min_chunk_size}")

    # ── Public API ─────────────────────────────────────────────────

    def chunk_paper(
        self,
        title: str,
        abstract: str,
        full_text: str,
        arxiv_id: str,
        paper_id: str,
        sections: Optional[Union[Dict, str, list]] = None,
    ) -> List[TextChunk]:
        """
        Main entry point — chunk a full paper.

        Tries section-based first, falls back to word-based.
        Always prepends title + abstract to each chunk for context.

        Returns List[TextChunk] ready for embedding and indexing.
        """
        # Try section-based chunking first
        if sections:
            try:
                section_chunks = self._chunk_by_sections(title, abstract, arxiv_id, paper_id, sections)
                if section_chunks:
                    logger.info(f"Section-based: {len(section_chunks)} chunks for {arxiv_id}")
                    return section_chunks
            except Exception as e:
                logger.warning(f"Section chunking failed for {arxiv_id}: {e} — falling back to word-based")

        # Fallback to word-based
        logger.info(f"Word-based chunking for {arxiv_id}")
        return self.chunk_text(full_text, arxiv_id, paper_id)

    def chunk_text(
        self,
        text: str,
        arxiv_id: str,
        paper_id: str,
    ) -> List[TextChunk]:
        """
        Word-based sliding window chunking.
        Used as fallback when no sections available.
        """
        if not text or not text.strip():
            logger.warning(f"Empty text for paper {arxiv_id}")
            return []

        words = self._split_into_words(text)

        # Too small for chunking — return as single chunk
        if len(words) < self.min_chunk_size:
            logger.warning(f"Paper {arxiv_id} has only {len(words)} words — returning as single chunk")
            if words:
                return [
                    TextChunk(
                        text=self._reconstruct_text(words),
                        metadata=ChunkMetadata(
                            chunk_index=0,
                            start_char=0,
                            end_char=len(text),
                            word_count=len(words),
                            overlap_with_previous=0,
                            overlap_with_next=0,
                        ),
                        arxiv_id=arxiv_id,
                        paper_id=paper_id,
                    )
                ]
            return []

        chunks = []
        chunk_index = 0
        current_position = 0

        while current_position < len(words):
            chunk_start = current_position
            chunk_end = min(current_position + self.chunk_size, len(words))

            chunk_words = words[chunk_start:chunk_end]
            chunk_text = self._reconstruct_text(chunk_words)

            # Approximate character offsets
            start_char = len(" ".join(words[:chunk_start])) if chunk_start > 0 else 0
            end_char = len(" ".join(words[:chunk_end]))

            overlap_with_previous = min(self.overlap_size, chunk_start) if chunk_start > 0 else 0
            overlap_with_next = self.overlap_size if chunk_end < len(words) else 0

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=end_char,
                        word_count=len(chunk_words),
                        overlap_with_previous=overlap_with_previous,
                        overlap_with_next=overlap_with_next,
                        section_title=None,
                    ),
                    arxiv_id=arxiv_id,
                    paper_id=paper_id,
                )
            )

            # Slide forward by chunk_size - overlap
            # This is what creates the overlap
            current_position += self.chunk_size - self.overlap_size
            chunk_index += 1

            if chunk_end >= len(words):
                break

        logger.info(f"Word-based: {len(words)} words → {len(chunks)} chunks for {arxiv_id}")
        return chunks

    # ── Private methods ────────────────────────────────────────────

    def _split_into_words(self, text: str) -> List[str]:
        r"""Split text into words using regex.
        \S+ matches any non-whitespace sequence.
        Handles multiple spaces, tabs, newlines correctly.
        """
        return re.findall(r"\S+", text)

    def _reconstruct_text(self, words: List[str]) -> str:
        """Join words back to text with single spaces."""
        return " ".join(words)

    def _chunk_by_sections(
        self,
        title: str,
        abstract: str,
        arxiv_id: str,
        paper_id: str,
        sections: Union[Dict, str, list],
    ) -> List[TextChunk]:
        """
        Section-based hybrid chunking.

        Three cases based on section size:
        < 100 words  → combine with adjacent sections
        100-800      → perfect, one chunk per section
        > 800        → split with word-based chunking
        """
        sections_dict = self._parse_sections(sections)
        if not sections_dict:
            return []

        sections_dict = self._filter_sections(sections_dict, abstract)
        if not sections_dict:
            logger.warning(f"No usable sections after filtering for {arxiv_id}")
            return []

        # Header prepended to every chunk for context
        # Even if the user gets chunk 15, they still see the paper title
        header = f"{title}\n\nAbstract: {abstract}\n\n"

        chunks = []
        small_sections = []  # buffer for combining tiny sections
        section_items = list(sections_dict.items())

        for i, (section_title, section_content) in enumerate(section_items):
            content_str = str(section_content) if section_content else ""
            section_words = len(content_str.split())

            if section_words < 100:
                # Accumulate small sections
                small_sections.append((section_title, content_str, section_words))

                # Flush buffer if last section or next section is large
                is_last = i == len(section_items) - 1
                next_is_large = not is_last and len(str(section_items[i + 1][1]).split()) >= 100

                if is_last or next_is_large:
                    new_chunks = self._create_combined_chunk(header, small_sections, chunks, arxiv_id, paper_id)
                    chunks.extend(new_chunks)
                    small_sections = []

            elif section_words <= 800:
                # Perfect size — one chunk
                chunk_text = f"{header}Section: {section_title}\n\n{content_str}"
                chunks.append(self._create_section_chunk(chunk_text, section_title, len(chunks), arxiv_id, paper_id))

            else:
                # Too large — split with word-based chunking
                section_text = f"Section: {section_title}\n\n{content_str}"
                full_section = f"{header}{section_text}"

                split_chunks = self._split_large_section(full_section, header, section_title, len(chunks), arxiv_id, paper_id)
                chunks.extend(split_chunks)

        return chunks

    def _parse_sections(self, sections: Union[Dict, str, list]) -> Dict[str, str]:
        """
        Parse sections into a dict regardless of input format.
        Docling returns sections as a list of dicts.
        Sometimes stored as JSON string in PostgreSQL.
        """
        if isinstance(sections, dict):
            return sections

        elif isinstance(sections, list):
            result = {}
            for i, section in enumerate(sections):
                if isinstance(section, dict):
                    title = section.get("title", section.get("heading", f"Section {i + 1}"))
                    content = section.get("content", section.get("text", ""))
                    result[title] = content
                else:
                    result[f"Section {i + 1}"] = str(section)
            return result

        elif isinstance(sections, str):
            try:
                parsed = json.loads(sections)
                # Recurse with the parsed object
                return self._parse_sections(parsed)
            except json.JSONDecodeError:
                logger.warning("Failed to parse sections as JSON")

        return {}

    def _filter_sections(self, sections_dict: Dict[str, str], abstract: str) -> Dict[str, str]:
        """
        Remove sections that would pollute the search index:
        - Empty sections
        - Metadata sections (author affiliations, emails)
        - Sections that duplicate the abstract
        - Very short sections that are just headers
        """
        filtered = {}
        abstract_words = set(abstract.lower().split())

        for title, content in sections_dict.items():
            content_str = str(content).strip()

            if not content_str:
                continue

            if self._is_metadata_section(title):
                continue

            if self._is_duplicate_abstract(content_str, abstract, abstract_words):
                logger.debug(f"Skipping duplicate abstract section: {title}")
                continue

            if len(content_str.split()) < 20 and self._is_metadata_content(content_str):
                logger.debug(f"Skipping metadata content: {title}")
                continue

            filtered[title] = content_str

        return filtered

    def _is_metadata_section(self, title: str) -> bool:
        """Detect header/metadata sections by title."""
        title_lower = title.lower().strip()

        metadata_indicators = [
            "content",
            "header",
            "authors",
            "author",
            "affiliation",
            "email",
            "arxiv",
            "preprint",
            "submitted",
            "received",
            "accepted",
        ]

        if title_lower in metadata_indicators or len(title_lower) < 5:
            return True

        for indicator in metadata_indicators:
            if indicator in title_lower and len(title_lower) < 20:
                return True

        return False

    def _is_duplicate_abstract(self, content: str, abstract: str, abstract_words: set) -> bool:
        """
        Check if section content is basically the abstract.
        Prevents indexing the abstract twice.
        """
        content_lower = content.lower().strip()
        abstract_lower = abstract.lower().strip()

        # Direct substring match
        if abstract_lower in content_lower or content_lower in abstract_lower:
            return True

        # Word overlap > 80% → likely duplicate
        if len(abstract_words) > 10:
            content_words = set(content_lower.split())
            overlap = len(abstract_words.intersection(content_words))
            if overlap / len(abstract_words) > 0.8:
                return True

        return False

    def _is_metadata_content(self, content: str) -> bool:
        """Detect content that's just metadata (emails, affiliations etc)."""
        content_lower = content.lower()
        metadata_patterns = [
            "@",
            "arxiv:",
            "university",
            "institute",
            "department",
            "college",
            "gmail.com",
            "edu",
            "ac.uk",
            "preprint",
        ]

        word_count = len(content.split())
        if word_count < 30:
            matches = sum(1 for p in metadata_patterns if p in content_lower)
            if matches >= 2:
                return True

        return False

    def _create_combined_chunk(
        self,
        header: str,
        small_sections: List,
        existing_chunks: List,
        arxiv_id: str,
        paper_id: str,
    ) -> List[TextChunk]:
        """
        Combine multiple small sections into one chunk.
        If the combined result is still tiny, merge into the previous chunk.
        """
        if not small_sections:
            return []

        combined_parts = []
        total_words = 0

        for section_title, content, word_count in small_sections:
            combined_parts.append(f"Section: {section_title}\n\n{content}")
            total_words += word_count

        combined_text = f"{header}" + "\n\n".join(combined_parts)

        # Still tiny AND there's a previous chunk → merge into it
        if total_words + len(header.split()) < 200 and existing_chunks:
            prev = existing_chunks[-1]
            merged_text = prev.text + "\n\n" + "\n\n".join(combined_parts)
            existing_chunks[-1] = TextChunk(
                text=merged_text,
                metadata=ChunkMetadata(
                    chunk_index=prev.metadata.chunk_index,
                    start_char=0,
                    end_char=len(merged_text),
                    word_count=len(merged_text.split()),
                    overlap_with_previous=0,
                    overlap_with_next=0,
                    section_title=f"{prev.metadata.section_title} + Combined",
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )
            return []  # already merged into previous, no new chunk

        # Build a title from the combined section names
        titles = [t for t, _, _ in small_sections]
        combined_title = " + ".join(titles[:3])
        if len(titles) > 3:
            combined_title += f" + {len(titles) - 3} more"

        return [self._create_section_chunk(combined_text, combined_title, len(existing_chunks), arxiv_id, paper_id)]

    def _create_section_chunk(
        self,
        chunk_text: str,
        section_title: str,
        chunk_index: int,
        arxiv_id: str,
        paper_id: str,
    ) -> TextChunk:
        """Create one section-based chunk."""
        return TextChunk(
            text=chunk_text,
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(chunk_text),
                word_count=len(chunk_text.split()),
                overlap_with_previous=0,
                overlap_with_next=0,
                section_title=section_title,
            ),
            arxiv_id=arxiv_id,
            paper_id=paper_id,
        )

    def _split_large_section(
        self,
        full_section_text: str,
        header: str,
        section_title: str,
        base_chunk_index: int,
        arxiv_id: str,
        paper_id: str,
    ) -> List[TextChunk]:
        """
        Split a large section using word-based chunking.
        Adds the header back to each sub-chunk for context.
        """
        # Strip header before word-based chunking
        section_only = full_section_text[len(header) :]
        sub_chunks = self.chunk_text(section_only, arxiv_id, paper_id)

        enhanced = []
        for i, chunk in enumerate(sub_chunks):
            enhanced_text = f"{header}{chunk.text}"
            enhanced.append(
                TextChunk(
                    text=enhanced_text,
                    metadata=ChunkMetadata(
                        chunk_index=base_chunk_index + i,
                        start_char=chunk.metadata.start_char,
                        end_char=chunk.metadata.end_char + len(header),
                        word_count=len(enhanced_text.split()),
                        overlap_with_previous=chunk.metadata.overlap_with_previous,
                        overlap_with_next=chunk.metadata.overlap_with_next,
                        section_title=f"{section_title} (Part {i + 1})",
                    ),
                    arxiv_id=arxiv_id,
                    paper_id=paper_id,
                )
            )

        return enhanced
