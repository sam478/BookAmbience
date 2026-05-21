#pdf_splitter.py
"""PDF text extraction with semantic segmentation."""
import os
import re
import pathlib
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF

CHAPTER_PATTERN = re.compile(r'^\s*(?:CHAPTER|PART)\s+[0-9IVXLC]+', re.IGNORECASE)


def _clean_text(text: str) -> str:
    """
    Fix character-by-character line splits that some PDFs produce.
    A run of 3+ consecutive single-character lines is almost certainly
    a word that was split glyph-by-glyph; rejoin them.
    """
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # Detect start of a single-char run (need at least 3 to be safe)
        if (len(stripped) == 1
                and i + 1 < len(lines) and len(lines[i + 1].strip()) == 1
                and i + 2 < len(lines) and len(lines[i + 2].strip()) == 1):
            chars = []
            while i < len(lines) and len(lines[i].strip()) <= 1:
                c = lines[i].strip()
                if c:
                    chars.append(c)
                i += 1
            result.append(''.join(chars))
        else:
            result.append(lines[i])
            i += 1
    return '\n'.join(result)


def _page_text(page: fitz.Page) -> str:
    return _clean_text(page.get_text("text") or "")


# ============================================================
# Clean block-mode extraction (used by semantic path)
# Ported from the old repair script so the normal pipeline
# produces clean paragraphs without running headers, page
# numbers, or front-matter contamination.
# ============================================================

_SENT_END_RE = re.compile(r"[.!?][\"”’')\]]*\s*$")
_ROMAN_MARKER = re.compile(r"[ivxlcIVXLC]+\.?")

FRONT_MATTER_MARKERS = (
    "library of america",
    "storyoftheweek",
    "story of the week",
    "first published in",
    "first collected",
    "tales of soldiers and civilians",
    "san francisco examiner",
    "miss a single story",
)


def _flatten_block(text: str) -> str:
    """Collapse a block's internal line wraps into a single paragraph string."""
    text = re.sub(r"\xad\s*\n\s*", "", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _ends_sentence(text: str) -> bool:
    s = text.rstrip()
    if not s:
        return True
    if _ROMAN_MARKER.fullmatch(s):
        return True
    return bool(_SENT_END_RE.search(s))


def _header_key(b: str) -> str:
    s = re.sub(r"^\d+\s+", "", b)
    s = re.sub(r"\s+\d+$", "", s)
    return s


def _is_front_matter(b: str) -> bool:
    low = b.lower()
    return any(m in low for m in FRONT_MATTER_MARKERS)


def _collect_repeat_keys(doc: fitz.Document, p_start: int, p_end: int) -> set:
    """Scan the page range and return header/footer keys that repeat ≥3 times."""
    counts: dict = {}
    for p in range(p_start, p_end + 1):
        page = doc.load_page(p)
        for block in page.get_text("blocks"):
            text = block[4]
            if not text or not text.strip():
                continue
            if len(block) >= 7 and block[6] != 0:
                continue
            cleaned = _flatten_block(text)
            if cleaned and len(cleaned) < 80:
                k = _header_key(cleaned)
                counts[k] = counts.get(k, 0) + 1
    return {k for k, n in counts.items() if n >= 3}


def _get_chapter_text_clean(doc: fitz.Document, start_page: int, end_page: int,
                            repeat_keys: set) -> str:
    """Extract clean paragraphs from a page range using block mode + cleanup.

    Strips running headers, page numbers, front-matter; merges paragraphs that
    continue across page/block breaks (so sentences don't get cut in half).
    Returns paragraphs joined by "\\n\\n".
    """
    raw_blocks = []
    for p in range(start_page, end_page + 1):
        page = doc.load_page(p)
        for block in page.get_text("blocks"):
            text = block[4]
            if not text or not text.strip():
                continue
            if len(block) >= 7 and block[6] != 0:
                continue
            cleaned = _flatten_block(text)
            if cleaned:
                raw_blocks.append(cleaned)

    blocks = [
        b for b in raw_blocks
        if _header_key(b) not in repeat_keys and not _is_front_matter(b)
    ]

    merged: list = []
    for b in blocks:
        if merged and not _ends_sentence(merged[-1]):
            prev = merged[-1]
            if prev.endswith("\xad"):
                merged[-1] = prev.rstrip("\xad") + b
            elif prev.endswith("-") and b and b[0].islower():
                merged[-1] = prev[:-1] + b
            else:
                merged[-1] = prev + " " + b
        else:
            merged.append(b)

    return "\n\n".join(merged)


def _extract_chapter_ranges(doc: fitz.Document) -> List[Tuple[int, int]]:
    """Find chapter boundaries based on CHAPTER/PART headings."""
    starts = []
    for i, page in enumerate(doc):
        text = _page_text(page)
        for line in text.splitlines()[:12]:
            if CHAPTER_PATTERN.match(line.strip()):
                starts.append(i)
                break
    if not starts:
        return []
    ranges = []
    for j in range(len(starts)):
        a = starts[j]
        b = starts[j+1]-1 if j+1 < len(starts) else len(doc)-1
        ranges.append((a, b))
    return ranges


def _get_chapter_text(doc: fitz.Document, start_page: int, end_page: int) -> str:
    """Extract all text from a page range."""
    text_parts = []
    for p in range(start_page, end_page + 1):
        text_parts.append(_page_text(doc.load_page(p)))
    return "\n".join(text_parts).strip()


def split_pdf_semantic(
    pdf_path: str,
    output_root: str = "segments",
    book_title: Optional[str] = None,
    progress_callback=None,
    chapter_limit: Optional[int] = None,
    output_dir: Optional[str] = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
) -> Dict:
    """
    Semantic segmentation using AI to detect natural story boundaries.
    """
    from semantic_segmenter import segment_chapter_semantically

    pdf_path = pathlib.Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    book_slug = re.sub(r'\W+', '_', pdf_path.stem).lower()
    book_title = book_title or pdf_path.stem.replace('_', ' ').title()

    # Resolve 1-indexed page range to 0-indexed
    p_start = max(0, (page_start - 1) if page_start is not None else 0)
    p_end   = min(total_pages - 1, (page_end - 1) if page_end is not None else total_pages - 1)

    if output_dir:
        out_dir = pathlib.Path(output_dir)
    else:
        out_dir = pathlib.Path(output_root) / book_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    chapter_ranges = _extract_chapter_ranges(doc)
    if not chapter_ranges:
        chapter_ranges = [(i, min(i + 29, total_pages - 1))
                          for i in range(0, total_pages, 30)]

    # Clip to requested page range
    if page_start is not None or page_end is not None:
        chapter_ranges = [
            (max(s, p_start), min(e, p_end))
            for s, e in chapter_ranges
            if s <= p_end and e >= p_start
        ]

    segments_info = []
    seg_idx = 1
    total_chapters = len(chapter_ranges)

    if chapter_limit is not None:
        chapter_ranges = chapter_ranges[:chapter_limit]
        total_chapters = len(chapter_ranges)
        print(f"  (Limited to first {chapter_limit} chapters)")

    # Precompute repeating header/footer keys across the requested page range
    # so dedup is accurate (a header that appears 3× in the whole doc won't
    # appear 3× within a single chapter).
    scan_start = min((s for s, _ in chapter_ranges), default=p_start)
    scan_end   = max((e for _, e in chapter_ranges), default=p_end)
    repeat_keys = _collect_repeat_keys(doc, scan_start, scan_end)
    if repeat_keys:
        print(f"  Stripping {len(repeat_keys)} recurring header/footer phrase(s): {sorted(repeat_keys)[:3]}{'…' if len(repeat_keys) > 3 else ''}")

    for chap_num, (start, end) in enumerate(chapter_ranges, 1):
        if progress_callback:
            progress = int((chap_num / total_chapters) * 50)
            progress_callback(progress, f"Analyzing chapter {chap_num}/{total_chapters}...")

        chapter_text = _get_chapter_text_clean(doc, start, end, repeat_keys)
        pages_in_chapter = end - start + 1

        print(f"\nAnalyzing Chapter {chap_num} (pages {start+1}-{end+1})...")

        semantic_segments = segment_chapter_semantically(
            chapter_text=chapter_text,
            chapter_num=chap_num,
            book_title=book_title,
            page_offset=start,
            pages_in_chapter=pages_in_chapter,
            provider_name=provider_name,
            model=model,
        )

        if not semantic_segments:
            semantic_segments = [{
                "text": chapter_text,
                "boundary_reason": f"Chapter {chap_num}",
                "page_range": [start + 1, end + 1],
            }]

        for seg_data in semantic_segments:
            fname = f"segment{seg_idx:02d}.txt"
            fpath = out_dir / fname
            fpath.write_text(seg_data["text"], encoding="utf-8")

            segments_info.append({
                "segment_number": seg_idx,
                "chapter": chap_num,
                "txt_path": str(fpath),
                "page_range": seg_data["page_range"],
                "boundary_reason": seg_data.get("boundary_reason"),
            })

            print(f"  Segment {seg_idx}: {seg_data.get('boundary_reason', '')[:60]}...")
            seg_idx += 1

    print(f"\n[Semantic] {len(segments_info)} segments saved in '{out_dir}/'")
    return {
        "book_slug": book_slug,
        "total_pages": len(doc),
        "segments": segments_info,
        "mode": "semantic",
    }


def split_pdf_to_segments(
    pdf_path: str,
    output_root: str = "segments",
    book_title: Optional[str] = None,
    progress_callback=None,
    chapter_limit: Optional[int] = None,
    output_dir: Optional[str] = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict:
    """Main entry point for PDF segmentation."""
    return split_pdf_semantic(
        pdf_path,
        output_root,
        book_title,
        progress_callback,
        chapter_limit,
        output_dir,
        provider_name=provider_name,
        model=model,
        page_start=page_start,
        page_end=page_end,
    )
