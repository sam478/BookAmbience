# semantic_segmenter.py
"""
AI-powered semantic segmentation that detects natural story boundaries.
Instead of blindly splitting every 4 pages, this analyzes the narrative
to find scene changes, mood shifts, and natural break points.
"""

import re
import time
from typing import List, Dict
from pydantic import BaseModel, Field
from providers import get_provider


class StorySegment(BaseModel):
    """A semantically meaningful story segment."""
    start_paragraph: int = Field(description="1-based index of first paragraph in this segment")
    end_paragraph: int = Field(description="1-based index of last paragraph in this segment")
    boundary_reason: str = Field(description="Why does a new segment start here? What changes — mood, location, time, narrative focus, emotional register? For the first segment write 'Opening of the text'. 1-2 sentences.")


class SegmentationPlan(BaseModel):
    """Complete segmentation plan for a text chunk."""
    segments: List[StorySegment] = Field(description="List of semantically meaningful segments")


_ROMAN_ONLY = re.compile(r'^[ivxlcIVXLC]+\.?$')


def extract_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs. Keeps short paragraphs only if they look
    like chapter/section markers (roman numerals); otherwise drops short
    fragments that are likely extraction noise."""
    paragraphs = re.split(r'\n\s*\n', text)
    kept = []
    for p in paragraphs:
        s = p.strip()
        if not s:
            continue
        if len(s) > 20 or _ROMAN_ONLY.match(s):
            kept.append(s)
    return kept


def analyze_text_for_segments(text: str, book_title: str = "Unknown",
                              provider_name: str = None, model: str = None) -> SegmentationPlan:
    """
    Use AI to analyze text and identify natural story boundaries.
    Returns segments based on scene changes, mood shifts, and narrative beats.
    """
    paragraphs = extract_paragraphs(text)

    if not paragraphs:
        return SegmentationPlan(segments=[])

    numbered_text = "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(paragraphs)])

    prompt = f"""You are an expert literary analyst identifying natural story segments for ambient music synchronization.

Analyze the following text from "{book_title}" and identify SEMANTICALLY MEANINGFUL segments.
Each segment should represent a cohesive scene or moment that warrants its own musical accompaniment.

IMPORTANT GUIDELINES:
- Create segments at NATURAL STORY BOUNDARIES: scene changes, mood shifts, location changes, time jumps, major revelations
- A segment must be SUBSTANTIAL enough to sustain 2-3 minutes of its own musical identity. If a dramatic beat is too short to warrant that (a single line or short paragraph), fold it into a neighboring segment rather than isolating it.
- Fewer, bigger segments are better than many small ones — every transition is costly to the listener's immersion. When in doubt, MERGE rather than split.
- DON'T split mid-scene or mid-conversation.
- DO split when the emotional tone significantly and durably shifts (not on minor wobbles within a scene).
- DO split when action transitions to contemplation (or vice versa) for a sustained stretch.
- DO split at chapter boundaries or major narrative beats.

The text has {len(paragraphs)} paragraphs numbered [1] through [{len(paragraphs)}].

TEXT:
{numbered_text}

Identify the natural segments in this text."""

    for attempt in range(3):
        try:
            provider = get_provider(provider_name, model)
            result = provider.generate_structured(prompt, SegmentationPlan)
            return SegmentationPlan(**result)
        except Exception as e:
            if attempt < 2:
                wait = 10 * (attempt + 1)
                print(f"  Segmentation attempt {attempt + 1} failed: {e}. Retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"Warning: Failed to parse segmentation response: {e}")
                return SegmentationPlan(segments=[
                    StorySegment(
                        start_paragraph=1,
                        end_paragraph=len(paragraphs),
                        boundary_reason="Opening of the text.",
                    )
                ])


def segment_chapter_semantically(
    chapter_text: str,
    chapter_num: int,
    book_title: str,
    page_offset: int = 0,
    pages_in_chapter: int = 1,
    provider_name: str = None,
    model: str = None,
) -> List[Dict]:
    """
    Semantically segment a chapter into meaningful story units.

    Returns list of segment dicts with:
    - text, boundary_reason
    - page_range: estimated pages based on text position in the chapter
    - paragraph_range: 1-based [start, end] paragraph indices
    """
    paragraphs = extract_paragraphs(chapter_text)

    if not paragraphs:
        return []

    plan = analyze_text_for_segments(chapter_text, book_title, provider_name, model)

    if not plan.segments:
        return []

    segments = []
    total_chars = sum(len(p) for p in paragraphs)

    for seg in plan.segments:
        start_idx = max(0, seg.start_paragraph - 1)
        end_idx = min(len(paragraphs), seg.end_paragraph)

        segment_paragraphs = paragraphs[start_idx:end_idx]
        if not segment_paragraphs:
            continue

        segment_text = "\n\n".join(segment_paragraphs)

        chars_before = sum(len(p) for p in paragraphs[:start_idx])
        chars_in_segment = sum(len(p) for p in segment_paragraphs)

        if total_chars > 0:
            start_ratio = chars_before / total_chars
            end_ratio = (chars_before + chars_in_segment) / total_chars

            start_page = page_offset + int(start_ratio * pages_in_chapter) + 1
            end_page = page_offset + int(end_ratio * pages_in_chapter) + 1
        else:
            start_page = page_offset + 1
            end_page = page_offset + pages_in_chapter

        segments.append({
            "text": segment_text,
            "boundary_reason": seg.boundary_reason,
            "page_range": [start_page, max(start_page, end_page)],
            "paragraph_range": [seg.start_paragraph, seg.end_paragraph],
        })

    return segments
