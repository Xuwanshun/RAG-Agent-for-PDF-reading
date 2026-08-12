"""
Unit tests for document_process/services.py pipeline functions.

Covers:
  BoundingBox   — area, intersection, validity, merge, round-trip
  ReadingOrderService — heuristic sort correctness and attribute mutation
  _region_type_for_label — label-to-type mapping (tables, figures, text)
  _best_region_match — spatial overlap matching
  _dedupe_regions — duplicate region elimination
  _compute_crop_box — padding rules and image-bounds clamping
  build_visual_summaries — text aggregation for table/figure regions
  build_chunks — character-budget chunking with overlap carry-forward
  AssociationService — OCR-item-to-region matching and block grouping

Nothing here touches Paddle, OpenAI, or the filesystem.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from document_process.models import (
    BoundingBox,
    CroppedRegionAsset,
    LayoutRegion,
    OCRPageResult,
    OCRTextItem,
    OrderedTextBlock,
    ProcessedChunk,
)
from document_process.services import (
    AssociationService,
    OCRService,
    PageContext,
    ReadingOrderService,
    _best_region_match,
    _compute_crop_box,
    _dedupe_regions,
    _reading_order_key,
    _region_type_for_label,
    build_chunks,
    build_visual_summaries,
)

# ── Lightweight factory helpers ───────────────────────────────────────────────


def _bbox(x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _item(item_id: str, text: str, bbox: BoundingBox, *, page_number: int = 1) -> OCRTextItem:
    return OCRTextItem(item_id=item_id, page_number=page_number, text=text, bbox=bbox)


def _region(region_id: str, region_type: str, bbox: BoundingBox, *, page_number: int = 1) -> LayoutRegion:
    return LayoutRegion(region_id=region_id, region_type=region_type, page_number=page_number, bbox=bbox)


def _ocr_page(page_number: int, items: list[OCRTextItem]) -> OCRPageResult:
    return OCRPageResult(page_number=page_number, items=items)


def _block(
    block_id: str,
    text: str,
    bbox: BoundingBox,
    *,
    page_number: int = 1,
    region_ids: list[str] | None = None,
    reading_order: int = 1,
) -> OrderedTextBlock:
    return OrderedTextBlock(
        block_id=block_id,
        page_number=page_number,
        text=text,
        bbox=bbox,
        region_ids=region_ids or [],
        reading_order=reading_order,
    )


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source_region_ids: list[str] | None = None,
    page_number: int = 1,
) -> ProcessedChunk:
    return ProcessedChunk(
        chunk_id=chunk_id,
        text=text,
        page_content=text,
        page_number=page_number,
        source_region_ids=source_region_ids or [],
    )


def _crop_asset(
    region_id: str,
    *,
    page_number: int = 1,
    region_type: str = "table",
    crop_path: str = "/tmp/crop.png",
) -> CroppedRegionAsset:
    return CroppedRegionAsset(
        asset_id=f"asset_{region_id}",
        region_id=region_id,
        page_number=page_number,
        region_type=region_type,
        crop_path=crop_path,
        bbox=_bbox(0, 0, 100, 100),
    )


def _text_block(
    block_id: str,
    text: str,
    page_number: int = 1,
    region_ids: list[str] | None = None,
) -> OrderedTextBlock:
    return OrderedTextBlock(
        block_id=block_id,
        page_number=page_number,
        text=text,
        region_ids=region_ids or [],
        bbox=_bbox(0, 0, 100, 20),
        reading_order=1,
    )


# ── BoundingBox tests ─────────────────────────────────────────────────────────


class TestBoundingBox:
    def test_area_of_normal_box(self):
        assert _bbox(0, 0, 10, 20).area() == 200.0

    def test_area_is_zero_when_width_is_zero(self):
        assert _bbox(5, 0, 5, 20).area() == 0.0

    def test_area_is_zero_when_height_is_zero(self):
        assert _bbox(0, 5, 10, 5).area() == 0.0

    def test_intersection_of_overlapping_boxes(self):
        a = _bbox(0, 0, 10, 10)
        b = _bbox(5, 5, 15, 15)
        assert a.intersection_area(b) == 25.0

    def test_intersection_is_zero_for_non_overlapping(self):
        assert _bbox(0, 0, 5, 5).intersection_area(_bbox(10, 10, 20, 20)) == 0.0

    def test_intersection_is_zero_for_touching_edges(self):
        # Boxes share an edge but do not overlap — area must be 0
        assert _bbox(0, 0, 5, 5).intersection_area(_bbox(5, 0, 10, 5)) == 0.0

    def test_inner_box_fully_contained(self):
        outer = _bbox(0, 0, 100, 100)
        inner = _bbox(10, 10, 20, 20)
        assert outer.intersection_area(inner) == inner.area()

    def test_is_valid_for_normal_box(self):
        assert _bbox(0, 0, 10, 10).is_valid() is True

    def test_is_invalid_when_x_inverted(self):
        assert _bbox(10, 0, 5, 10).is_valid() is False

    def test_is_invalid_when_y1_equals_y0(self):
        assert _bbox(0, 5, 10, 5).is_valid() is False

    def test_from_list_as_list_round_trip(self):
        values = [1.0, 2.0, 3.0, 4.0]
        assert BoundingBox.from_list(values).as_list() == values

    def test_merge_takes_outer_bounds(self):
        merged = BoundingBox.merge([_bbox(0, 0, 5, 5), _bbox(3, 3, 10, 10)])
        assert merged is not None
        assert merged.as_list() == [0.0, 0.0, 10.0, 10.0]

    def test_merge_single_box(self):
        box = _bbox(5, 6, 15, 16)
        assert BoundingBox.merge([box]) == box

    def test_merge_empty_list_returns_none(self):
        assert BoundingBox.merge([]) is None


# ── ReadingOrderService tests ─────────────────────────────────────────────────


class TestReadingOrderKey:
    def test_top_item_sorts_before_bottom_item(self):
        top = _item("t", "T", _bbox(50, 50, 200, 70))
        bottom = _item("b", "B", _bbox(50, 500, 200, 520))
        assert _reading_order_key(top) < _reading_order_key(bottom)

    def test_items_on_same_line_sorted_left_to_right(self):
        left = _item("l", "L", _bbox(10, 100, 100, 118))
        right = _item("r", "R", _bbox(400, 100, 500, 118))
        assert _reading_order_key(left) < _reading_order_key(right)

    def test_same_bucket_different_x_sorted_correctly(self):
        # y0=100 → bucket round(100/18)=6; y0=105 → bucket round(105/18)=6 — same
        left_col = _item("l", "L", _bbox(50, 100, 200, 118))
        right_col = _item("r", "R", _bbox(300, 105, 500, 123))
        assert _reading_order_key(left_col) < _reading_order_key(right_col)


class TestReadingOrderService:
    def _resolve_page(self, items: list[OCRTextItem], page: int = 1) -> list[str]:
        result, issues = ReadingOrderService().resolve([_ocr_page(page, items)])
        assert issues == []
        return result["pages"][0]["ordered_item_ids"]

    def test_single_item(self):
        assert self._resolve_page([_item("i1", "hi", _bbox(10, 10, 100, 30))]) == ["i1"]

    def test_top_before_bottom(self):
        top = _item("top", "a", _bbox(10, 50, 200, 70))
        bottom = _item("bot", "b", _bbox(10, 300, 200, 320))
        assert self._resolve_page([bottom, top]) == ["top", "bot"]  # reversed input

    def test_left_before_right_on_same_line(self):
        left = _item("l", "L", _bbox(10, 100, 100, 120))
        right = _item("r", "R", _bbox(400, 100, 500, 120))
        assert self._resolve_page([right, left]) == ["l", "r"]  # reversed input

    def test_reading_order_attribute_set_on_items(self):
        i1 = _item("i1", "A", _bbox(10, 10, 100, 30))
        i2 = _item("i2", "B", _bbox(10, 200, 100, 220))
        ReadingOrderService().resolve([_ocr_page(1, [i1, i2])])
        assert i1.reading_order == 1
        assert i2.reading_order == 2

    def test_empty_page_gives_empty_order(self):
        result, issues = ReadingOrderService().resolve([_ocr_page(1, [])])
        assert issues == []
        assert result["pages"][0]["ordered_item_ids"] == []

    def test_multi_page_order_kept_per_page(self):
        p1 = _item("p1", "page1", _bbox(10, 10, 200, 30), page_number=1)
        p2 = _item("p2", "page2", _bbox(10, 10, 200, 30), page_number=2)
        result, _ = ReadingOrderService().resolve([_ocr_page(1, [p1]), _ocr_page(2, [p2])])
        assert result["document_order_item_ids"] == ["p1", "p2"]


# ── Region type label mapping ─────────────────────────────────────────────────


class TestRegionTypeForLabel:
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("table", "table"),
            ("borderless_table", "table"),
            ("table_with_merged_cells", "table"),
            ("figure", "figure"),
            ("chart", "figure"),
            ("image", "figure"),
            ("graph", "figure"),
            ("text", "text_block"),
            ("title", "text_block"),
            ("header", "text_block"),
            ("footer", "text_block"),
            ("caption", "text_block"),
            ("reference", "text_block"),
            ("aside_text", "text_block"),  # listed explicitly
            ("custom_text", "text_block"),  # ends with _text
            ("body_text", "text_block"),  # ends with _text
        ],
    )
    def test_known_labels(self, label: str, expected: str):
        assert _region_type_for_label(label) == expected

    @pytest.mark.parametrize("label", ["logo", "watermark", "qrcode", "unknown", ""])
    def test_unknown_labels_return_none(self, label: str):
        assert _region_type_for_label(label) is None


# ── _best_region_match tests ──────────────────────────────────────────────────


class TestBestRegionMatch:
    def test_item_fully_inside_region_matched_with_ratio_1(self):
        item = _item("i", "hello", _bbox(20, 20, 80, 40))
        region = _region("r1", "text_block", _bbox(0, 0, 100, 100))
        matched, ratio = _best_region_match(item, [region])
        assert matched is region
        assert ratio == pytest.approx(1.0)

    def test_item_outside_all_regions_returns_none(self):
        item = _item("i", "hello", _bbox(500, 500, 600, 520))
        region = _region("r1", "text_block", _bbox(0, 0, 100, 100))
        matched, ratio = _best_region_match(item, [region])
        assert matched is None
        assert ratio == 0.0

    def test_picks_region_with_larger_overlap(self):
        # Item x=0..100; region1 covers 60%, region2 covers 20%
        item = _item("i", "text", _bbox(0, 0, 100, 20))
        r1 = _region("r1", "text_block", _bbox(0, 0, 60, 20))
        r2 = _region("r2", "text_block", _bbox(80, 0, 100, 20))
        matched, _ = _best_region_match(item, [r1, r2])
        assert matched is r1

    def test_empty_region_list_returns_none(self):
        item = _item("i", "text", _bbox(10, 10, 100, 30))
        assert _best_region_match(item, []) == (None, 0.0)

    def test_touching_edge_not_counted_as_overlap(self):
        item = _item("i", "text", _bbox(100, 0, 200, 20))
        region = _region("r1", "text_block", _bbox(0, 0, 100, 20))
        matched, _ = _best_region_match(item, [region])
        assert matched is None


# ── _dedupe_regions tests ─────────────────────────────────────────────────────


class TestDedupeRegions:
    def test_unique_regions_all_kept(self):
        r1 = _region("r1", "table", _bbox(0, 0, 100, 100))
        r2 = _region("r2", "figure", _bbox(200, 0, 300, 100))
        assert len(_dedupe_regions([r1, r2])) == 2

    def test_exact_duplicate_keeps_first(self):
        r1 = _region("r1", "table", _bbox(0, 0, 100, 100))
        r2 = _region("r2", "table", _bbox(0, 0, 100, 100))
        result = _dedupe_regions([r1, r2])
        assert len(result) == 1
        assert result[0].region_id == "r1"

    def test_same_bbox_different_type_both_kept(self):
        r1 = _region("r1", "table", _bbox(0, 0, 100, 100))
        r2 = _region("r2", "figure", _bbox(0, 0, 100, 100))
        assert len(_dedupe_regions([r1, r2])) == 2

    def test_near_duplicate_within_rounding_removed(self):
        # 0.04 rounds to 0.0 at 1 decimal place → treated as the same key
        r1 = _region("r1", "table", _bbox(0.0, 0.0, 100.0, 100.0))
        r2 = _region("r2", "table", _bbox(0.04, 0.04, 100.04, 100.04))
        assert len(_dedupe_regions([r1, r2])) == 1

    def test_same_bbox_different_page_both_kept(self):
        r1 = _region("r1", "table", _bbox(0, 0, 100, 100), page_number=1)
        r2 = _region("r2", "table", _bbox(0, 0, 100, 100), page_number=2)
        assert len(_dedupe_regions([r1, r2])) == 2


# ── _compute_crop_box tests ───────────────────────────────────────────────────


class TestComputeCropBox:
    def test_table_crop_is_padded_outward(self):
        region = _region("r1", "table", _bbox(100, 100, 400, 300))
        left, top, right, bottom = _compute_crop_box(region, image_width=2000, image_height=2000)
        assert left < 100 and top < 100 and right > 400 and bottom > 300

    def test_figure_crop_is_larger_than_table_crop_for_same_bbox(self):
        table_r = _region("r1", "table", _bbox(100, 100, 400, 300))
        figure_r = _region("r2", "figure", _bbox(100, 100, 400, 300))
        t = _compute_crop_box(table_r, image_width=2000, image_height=2000)
        f = _compute_crop_box(figure_r, image_width=2000, image_height=2000)
        t_area = (t[2] - t[0]) * (t[3] - t[1])
        f_area = (f[2] - f[0]) * (f[3] - f[1])
        assert f_area > t_area

    def test_zero_height_returns_none(self):
        region = _region("r1", "table", _bbox(100, 100, 300, 100))  # height = 0
        assert _compute_crop_box(region, image_width=2000, image_height=2000) is None

    def test_zero_width_returns_none(self):
        region = _region("r1", "table", _bbox(100, 100, 100, 300))  # width = 0
        assert _compute_crop_box(region, image_width=2000, image_height=2000) is None

    def test_left_top_clamped_to_zero(self):
        # Region at origin; padding would push coordinates negative
        region = _region("r1", "table", _bbox(0, 0, 100, 100))
        left, top, _, _ = _compute_crop_box(region, image_width=2000, image_height=2000)
        assert left >= 0 and top >= 0

    def test_right_bottom_clamped_to_image_bounds(self):
        # Region at image edge; padding would exceed image size
        region = _region("r1", "table", _bbox(1900, 1900, 2000, 2000))
        _, _, right, bottom = _compute_crop_box(region, image_width=2000, image_height=2000)
        assert right <= 2000 and bottom <= 2000

    def test_normal_table_crop_meets_minimum_size(self):
        region = _region("r1", "table", _bbox(100, 100, 400, 300))
        left, top, right, bottom = _compute_crop_box(region, image_width=2000, image_height=2000)
        assert (right - left) >= 48 and (bottom - top) >= 48

    def test_tiny_region_where_padded_crop_is_under_minimum_returns_none(self):
        # 2×2 region: passes the ≤1 check, but with 28px min padding on a 50px
        # canvas the result is only 40×40 — below the 48px minimum
        region = _region("r1", "table", _bbox(10, 10, 12, 12))
        assert _compute_crop_box(region, image_width=50, image_height=50) is None


# ── build_visual_summaries tests ──────────────────────────────────────────────


class TestBuildVisualSummaries:
    def test_table_region_produces_summary(self):
        region = _region("r1", "table", _bbox(0, 0, 100, 100))
        summaries = build_visual_summaries(regions=[region], ordered_blocks=[], chunks=[], cropped_assets=[])
        assert len(summaries) == 1
        s = summaries[0]
        assert s.region_id == "r1" and s.region_type == "table"

    def test_figure_region_produces_summary(self):
        region = _region("r1", "figure", _bbox(0, 0, 200, 150))
        summaries = build_visual_summaries(regions=[region], ordered_blocks=[], chunks=[], cropped_assets=[])
        assert len(summaries) == 1 and summaries[0].region_type == "figure"

    def test_text_block_regions_excluded(self):
        text_r = _region("r1", "text_block", _bbox(0, 0, 100, 100))
        assert build_visual_summaries(regions=[text_r], ordered_blocks=[], chunks=[], cropped_assets=[]) == []

    def test_mixed_regions_only_table_and_figure_summarised(self):
        table = _region("r1", "table", _bbox(0, 0, 100, 100))
        figure = _region("r2", "figure", _bbox(200, 0, 300, 100))
        text = _region("r3", "text_block", _bbox(400, 0, 500, 100))
        summaries = build_visual_summaries(
            regions=[table, figure, text], ordered_blocks=[], chunks=[], cropped_assets=[]
        )
        assert len(summaries) == 2
        assert {s.region_id for s in summaries} == {"r1", "r2"}

    def test_summary_text_from_overlapping_block(self):
        region = _region("r1", "table", _bbox(0, 0, 100, 100))
        blk = _block("b1", "Revenue Q1 2024", _bbox(5, 5, 95, 50), region_ids=["r1"])
        summaries = build_visual_summaries(regions=[region], ordered_blocks=[blk], chunks=[], cropped_assets=[])
        assert "Revenue Q1 2024" in summaries[0].summary_text

    def test_fallback_to_nearest_block_when_no_overlap(self):
        # Figure y=200..400; caption block just below at y=405..425 — no spatial overlap
        region = _region("r1", "figure", _bbox(0, 200, 100, 400))
        caption = _block("b1", "Figure 1: Sales trend", _bbox(0, 405, 100, 425))
        summaries = build_visual_summaries(regions=[region], ordered_blocks=[caption], chunks=[], cropped_assets=[])
        assert "Figure 1: Sales trend" in summaries[0].summary_text

    def test_placeholder_text_when_no_nearby_blocks(self):
        # VLM GAP: without a VLM, isolated figures yield a useless placeholder
        region = _region("r1", "figure", _bbox(0, 0, 100, 100), page_number=5)
        summaries = build_visual_summaries(regions=[region], ordered_blocks=[], chunks=[], cropped_assets=[])
        assert summaries[0].summary_text == "Detected figure region on page 5."

    def test_summary_text_capped_at_1200_chars(self):
        region = _region("r1", "table", _bbox(0, 0, 100, 100))
        blk = _block("b1", "X" * 2000, _bbox(5, 5, 95, 50), region_ids=["r1"])
        summaries = build_visual_summaries(regions=[region], ordered_blocks=[blk], chunks=[], cropped_assets=[])
        assert len(summaries[0].summary_text) == 1200

    def test_linked_chunk_ids_populated(self):
        region = _region("r1", "table", _bbox(0, 0, 100, 100))
        c = _chunk("doc:chunk:1", "table data", source_region_ids=["r1"])
        summaries = build_visual_summaries(regions=[region], ordered_blocks=[], chunks=[c], cropped_assets=[])
        assert "doc:chunk:1" in summaries[0].linked_chunk_ids

    def test_asset_metadata_populated_when_crop_present(self):
        region = _region("r1", "table", _bbox(0, 0, 100, 100))
        asset = _crop_asset("r1", crop_path="/crops/tables/r1.png")
        summaries = build_visual_summaries(regions=[region], ordered_blocks=[], chunks=[], cropped_assets=[asset])
        assert summaries[0].asset_id == "asset_r1"
        assert summaries[0].crop_path == "/crops/tables/r1.png"

    def test_asset_id_none_when_no_crop(self):
        region = _region("r1", "table", _bbox(0, 0, 100, 100))
        summaries = build_visual_summaries(regions=[region], ordered_blocks=[], chunks=[], cropped_assets=[])
        assert summaries[0].asset_id is None


# ── build_chunks tests ────────────────────────────────────────────────────────


class TestBuildChunks:
    def test_single_block_yields_one_chunk(self):
        chunks = build_chunks(
            document_id="doc",
            source_file="test.pdf",
            ordered_blocks=[_text_block("b1", "Hello world")],
            regions=[],
        )
        assert len(chunks) == 1

    def test_chunk_text_contains_block_content(self):
        chunks = build_chunks(
            document_id="doc",
            source_file="test.pdf",
            ordered_blocks=[_text_block("b1", "The quick brown fox")],
            regions=[],
        )
        assert "The quick brown fox" in chunks[0].text

    def test_chunk_id_prefixed_with_document_id(self):
        chunks = build_chunks(
            document_id="mydoc",
            source_file="f.pdf",
            ordered_blocks=[_text_block("b1", "text")],
            regions=[],
        )
        assert chunks[0].chunk_id.startswith("mydoc:chunk:")

    def test_short_blocks_stay_in_one_chunk(self):
        blocks = [_text_block(f"b{i}", f"word {i}") for i in range(5)]
        chunks = build_chunks(
            document_id="doc", source_file="f.pdf", ordered_blocks=blocks, regions=[], target_chars=1800
        )
        assert len(chunks) == 1

    def test_long_blocks_split_into_multiple_chunks(self):
        blocks = [_text_block(f"b{i}", "A" * 600) for i in range(5)]
        chunks = build_chunks(
            document_id="doc", source_file="f.pdf", ordered_blocks=blocks, regions=[], target_chars=1000
        )
        assert len(chunks) > 1

    def test_chunk_page_number_set_from_block(self):
        chunks = build_chunks(
            document_id="doc",
            source_file="f.pdf",
            ordered_blocks=[_text_block("b1", "text", page_number=3)],
            regions=[],
        )
        assert chunks[0].page_number == 3

    def test_blocks_on_different_pages_in_separate_chunks(self):
        b1 = _text_block("b1", "A" * 100, page_number=1)
        b2 = _text_block("b2", "B" * 100, page_number=2)
        chunks = build_chunks(document_id="doc", source_file="f.pdf", ordered_blocks=[b1, b2], regions=[])
        assert {c.page_number for c in chunks} == {1, 2}

    def test_whitespace_only_blocks_omitted_from_text(self):
        blocks = [_text_block("b1", "real"), _text_block("b2", "   "), _text_block("b3", "content")]
        chunks = build_chunks(document_id="doc", source_file="f.pdf", ordered_blocks=blocks, regions=[])
        assert len(chunks) == 1
        assert "   " not in chunks[0].text
        assert "real" in chunks[0].text and "content" in chunks[0].text

    def test_region_ids_tracked_in_chunk(self):
        region = LayoutRegion(region_id="r1", region_type="table", page_number=1, bbox=_bbox(0, 0, 100, 100))
        block = _text_block("b1", "table data", region_ids=["r1"])
        chunks = build_chunks(document_id="doc", source_file="f.pdf", ordered_blocks=[block], regions=[region])
        assert "r1" in chunks[0].source_region_ids

    def test_empty_blocks_returns_no_chunks(self):
        assert build_chunks(document_id="doc", source_file="f.pdf", ordered_blocks=[], regions=[]) == []

    def test_ordered_block_ids_in_chunk_metadata(self):
        b1 = _text_block("b1", "first")
        b2 = _text_block("b2", "second")
        chunks = build_chunks(document_id="doc", source_file="f.pdf", ordered_blocks=[b1, b2], regions=[])
        assert "b1" in chunks[0].ordered_block_ids
        assert "b2" in chunks[0].ordered_block_ids

    def test_overlap_blocks_carried_into_next_chunk(self):
        # First two 600-char blocks → chunk emitted; third block should appear in chunk 2
        sentinel = "SENTINEL_TEXT_FOR_OVERLAP"
        blocks = [
            _text_block("b1", "A" * 600),
            _text_block("b2", sentinel),  # this is the last block of chunk 1
            _text_block("b3", "B" * 600),  # triggers chunk split; b2 carried forward
        ]
        chunks = build_chunks(
            document_id="doc",
            source_file="f.pdf",
            ordered_blocks=blocks,
            regions=[],
            target_chars=700,
            overlap_chars=len(sentinel) + 1,
        )
        assert len(chunks) >= 2
        # The sentinel block should appear in the second chunk via overlap
        second_chunk_text = " ".join(c.text for c in chunks[1:])
        assert sentinel in second_chunk_text


# ── AssociationService tests ──────────────────────────────────────────────────


class TestAssociationService:
    def _associate(self, items: list[OCRTextItem], regions: list[LayoutRegion], page: int = 1):
        ocr_pages = [_ocr_page(page, items)]
        reading_order, _ = ReadingOrderService().resolve(ocr_pages)
        return AssociationService().associate(ocr_pages, reading_order, regions)

    def test_item_inside_region_assigned_that_region(self):
        item = _item("i1", "hello", _bbox(10, 10, 90, 30))
        region = _region("r1", "text_block", _bbox(0, 0, 100, 100))
        associations, _, _ = self._associate([item], [region])
        assoc = next(a for a in associations if a.item_id == "i1")
        assert assoc.region_id == "r1"
        assert assoc.region_type == "text_block"

    def test_item_outside_regions_has_no_region_id(self):
        item = _item("i1", "hello", _bbox(500, 500, 600, 520))
        region = _region("r1", "text_block", _bbox(0, 0, 100, 100))
        associations, _, _ = self._associate([item], [region])
        assoc = next(a for a in associations if a.item_id == "i1")
        assert assoc.region_id is None

    def test_unmatched_item_has_zero_overlap_ratio(self):
        item = _item("i1", "text", _bbox(500, 500, 600, 520))
        region = _region("r1", "text_block", _bbox(0, 0, 100, 100))
        associations, _, _ = self._associate([item], [region])
        assert associations[0].overlap_ratio == 0.0

    def test_items_in_same_region_and_line_grouped_into_one_block(self):
        i1 = _item("i1", "word one", _bbox(10, 10, 90, 30))
        i2 = _item("i2", "word two", _bbox(10, 10, 90, 30))  # same bbox → same bucket
        region = _region("r1", "text_block", _bbox(0, 0, 100, 100))
        _, blocks, _ = self._associate([i1, i2], [region])
        assert len(blocks) == 1
        assert set(blocks[0].item_ids) == {"i1", "i2"}

    def test_items_on_different_line_buckets_form_separate_blocks(self):
        # y0=10 → bucket 0; y0=300 → bucket 15
        i1 = _item("i1", "line one", _bbox(10, 10, 200, 30))
        i2 = _item("i2", "line two", _bbox(10, 300, 200, 320))
        _, blocks, _ = self._associate([i1, i2], [])
        assert len(blocks) == 2

    def test_full_text_contains_all_item_text(self):
        i1 = _item("i1", "hello", _bbox(10, 10, 200, 30))
        i2 = _item("i2", "world", _bbox(10, 300, 200, 320))
        _, _, ordered_text = self._associate([i1, i2], [])
        assert "hello" in ordered_text["full_text"]
        assert "world" in ordered_text["full_text"]

    def test_empty_page_produces_no_blocks(self):
        _, blocks, ordered_text = self._associate([], [])
        assert blocks == []
        assert ordered_text["full_text"] == ""

    def test_items_in_different_regions_form_different_blocks(self):
        # Item 1 in region r1, item 2 in region r2 → must not be merged
        i1 = _item("i1", "table text", _bbox(10, 10, 90, 30))
        i2 = _item("i2", "figure caption", _bbox(10, 200, 90, 220))
        r1 = _region("r1", "table", _bbox(0, 0, 100, 100))
        r2 = _region("r2", "figure", _bbox(0, 150, 100, 250))
        _, blocks, _ = self._associate([i1, i2], [r1, r2])
        assert len(blocks) >= 2
        block_for_i1 = next(b for b in blocks if "i1" in b.item_ids)
        block_for_i2 = next(b for b in blocks if "i2" in b.item_ids)
        assert block_for_i1.block_id != block_for_i2.block_id

    def test_associate_start_index_offsets_block_ids(self):
        """When start_index=5 the first block id must be 'p1_block_5' not 'p1_block_1'."""
        item = _item("i1", "hello", _bbox(0, 0, 100, 20))
        page = _ocr_page(1, [item])
        reading_order = {
            "resolver": "ocr_bbox_sort_v1",
            "document_order_item_ids": ["i1"],
            "pages": [{"page_number": 1, "ordered_item_ids": ["i1"]}],
        }
        region = _region("r1", "text_block", _bbox(0, 0, 100, 100))
        svc = AssociationService()
        _, blocks, _ = svc.associate([page], reading_order, [region], start_index=5)
        assert blocks[0].block_id == "p1_block_5"


def test_ocr_extract_calls_on_page_done_once_per_page():
    """on_page_done must be called exactly once per page processed."""
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(2):
            p = Path(d) / f"page_{i + 1}.png"
            p.write_bytes(b"fake")
            paths.append(p)

        pages = [PageContext(page_number=i + 1, width=100.0, height=200.0, page_image_path=paths[i]) for i in range(2)]

        svc = OCRService()
        fake_result = MagicMock()
        fake_result.json = {"res": {"rec_texts": [], "rec_scores": [], "rec_boxes": [], "dt_polys": []}}
        fake_predictor = MagicMock()
        fake_predictor.predict.return_value = [fake_result]

        callback = MagicMock()

        with patch("document_process.services._get_paddle_ocr", return_value=fake_predictor):
            svc.extract(pages, on_page_done=callback)

        assert callback.call_count == 2


class TestLayoutDetectionRegionIds:
    """Region IDs must be unique across the whole document, not per detect() call.

    The pipeline calls detect() once per page batch, so an ID counter that is
    local to detect() restarts every batch and mints the same name several
    times. Colliding IDs silently overwrite each other in the region lookups
    and in the crop filenames.
    """

    @staticmethod
    def _pages(page_numbers: list[int], directory: Path) -> list[PageContext]:
        pages = []
        for number in page_numbers:
            image = directory / f"page_{number}.png"
            image.write_bytes(b"fake")
            pages.append(PageContext(page_number=number, width=100.0, height=200.0, page_image_path=image))
        return pages

    @staticmethod
    def _predictor(boxes_per_page: int) -> MagicMock:
        result = MagicMock()
        result.json = {
            "res": {
                "boxes": [
                    {"label": "table", "coordinate": [0, 20 * i, 100, 20 * i + 15], "score": 0.9}
                    for i in range(boxes_per_page)
                ]
            }
        }
        predictor = MagicMock()
        predictor.predict.return_value = [result]
        return predictor

    def test_region_ids_unique_across_batches(self):
        from document_process.services import LayoutDetectionService

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            service = LayoutDetectionService()
            predictor = self._predictor(boxes_per_page=3)

            with patch("document_process.services._get_paddle_layout_detector", return_value=predictor):
                first, _, _ = service.detect(self._pages([1, 2], path), [])
                second, _, _ = service.detect(self._pages([3, 4], path), [])

            all_ids = [region.region_id for region in first + second]
            assert len(all_ids) == len(set(all_ids)), f"duplicate region ids across batches: {all_ids}"

    def test_region_ids_unique_within_a_batch(self):
        from document_process.services import LayoutDetectionService

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            service = LayoutDetectionService()

            with patch(
                "document_process.services._get_paddle_layout_detector",
                return_value=self._predictor(boxes_per_page=3),
            ):
                regions, _, _ = service.detect(self._pages([1, 2, 3], path), [])

            ids = [region.region_id for region in regions]
            assert len(ids) == len(set(ids))


class TestPaddleOcrPredictorCache:
    """The predictor cache must key on something hashable.

    `_get_paddle_ocr` is lru_cached, so its argument has to be hashable.
    Settings is a pydantic model and is not, which means every call raises
    TypeError before the predictor is ever built. Passing None (as the unit
    tests did) hides it, because None is hashable — but the pipeline always
    constructs OCRService with a real Settings.
    """

    def test_predictor_is_built_from_real_settings(self):
        from config import Settings
        from document_process.services import _build_paddle_ocr, _get_paddle_ocr

        _build_paddle_ocr.cache_clear()
        with patch("paddleocr.PaddleOCR", return_value=MagicMock()) as paddle_ocr:
            _get_paddle_ocr(Settings())

        assert paddle_ocr.call_count == 1
        kwargs = paddle_ocr.call_args.kwargs
        assert kwargs["text_detection_model_name"] == Settings().ocr_detection_model
        assert kwargs["text_recognition_model_name"] == Settings().ocr_recognition_model

    def test_predictor_is_reused_for_the_same_models(self):
        from config import Settings
        from document_process.services import _build_paddle_ocr, _get_paddle_ocr

        _build_paddle_ocr.cache_clear()
        with patch("paddleocr.PaddleOCR", return_value=MagicMock()) as paddle_ocr:
            first = _get_paddle_ocr(Settings())
            second = _get_paddle_ocr(Settings())

        assert first is second
        assert paddle_ocr.call_count == 1, "predictor rebuilt despite identical model names"

    def test_changing_models_builds_a_new_predictor(self):
        from config import Settings
        from document_process.services import _build_paddle_ocr, _get_paddle_ocr

        _build_paddle_ocr.cache_clear()
        with patch("paddleocr.PaddleOCR", side_effect=lambda **_: MagicMock()) as paddle_ocr:
            _get_paddle_ocr(Settings(ocr_detection_model="PP-OCRv4_mobile_det"))
            _get_paddle_ocr(Settings(ocr_detection_model="PP-OCRv5_mobile_det"))

        assert paddle_ocr.call_count == 2


class TestTextLayerReadingOrder:
    """Text-layer pages arrive in reading order and must not be re-sorted.

    _group_rects_into_lines already returns line boxes top-to-bottom and each
    item is a whole line. The bbox sort's 18px bucket collapses adjacent table
    rows into one band and the x tiebreak then swaps them.
    """

    @staticmethod
    def _page(text_source: str) -> OCRPageResult:
        # Two stacked table rows 16px apart that both round into 18px bucket 11
        # (190/18 -> 11, 206/18 -> 11). The lower row starts further left, so a
        # bbox sort swaps them.
        upper = _item("i1", "Intl 944 778", _bbox(120, 190, 700, 204))
        lower = _item("i2", "WW 2,083 1,645", _bbox(100, 206, 700, 220))
        return OCRPageResult(page_number=1, items=[upper, lower], text_source=text_source)

    def test_text_layer_order_is_preserved(self):
        page = self._page("pdf_text_layer")
        result, _ = ReadingOrderService().resolve([page])
        assert result["pages"][0]["ordered_item_ids"] == ["i1", "i2"]
        assert result["resolver"] == "pdf_text_layer_order_v1"

    def test_ocr_pages_are_still_sorted(self):
        page = self._page("paddleocr")
        result, _ = ReadingOrderService().resolve([page])
        # The bbox sort swaps them — that behaviour is unchanged for OCR pages.
        assert result["pages"][0]["ordered_item_ids"] == ["i2", "i1"]
        assert result["resolver"] == "ocr_bbox_sort_v1"

    def test_reading_order_index_is_assigned_on_text_layer_pages(self):
        page = self._page("pdf_text_layer")
        ReadingOrderService().resolve([page])
        assert [i.reading_order for i in page.items] == [1, 2]

    def test_mixed_document_reports_both_resolvers(self):
        digital = self._page("pdf_text_layer")
        scanned = self._page("paddleocr")
        scanned.page_number = 2
        result, _ = ReadingOrderService().resolve([digital, scanned])
        assert result["resolver"] == "mixed:ocr_bbox_sort_v1+pdf_text_layer_order_v1"


class TestBlockGroupingFromBlockBoxes:
    """PP-DocBlockLayout boxes are the grouping unit for text blocks.

    The 20px line bucket it replaces put every text-layer line in its own block,
    so each line became its own paragraph and chunk text was lines joined by
    blank lines.
    """

    @staticmethod
    def _three_lines() -> list[OCRTextItem]:
        return [
            _item("i1", "This report includes estimates, projections,", _bbox(90, 100, 700, 118)),
            _item("i2", "statements relating to our business plans", _bbox(90, 122, 700, 140)),
            _item("i3", "and expected operating results.", _bbox(90, 144, 700, 162)),
        ]

    @staticmethod
    def _resolve(items: list[OCRTextItem]) -> tuple[OCRPageResult, dict]:
        page = OCRPageResult(page_number=1, items=items, text_source="pdf_text_layer")
        order, _ = ReadingOrderService().resolve([page])
        return page, order

    def test_lines_in_one_block_box_merge_into_one_block(self):
        items = self._three_lines()
        page, order = self._resolve(items)
        _, blocks, _ = AssociationService().associate([page], order, [], block_boxes={1: [_bbox(80, 90, 720, 175)]})
        assert len(blocks) == 1
        assert blocks[0].text == (
            "This report includes estimates, projections, "
            "statements relating to our business plans "
            "and expected operating results."
        )

    def test_separate_block_boxes_stay_separate(self):
        items = self._three_lines()
        page, order = self._resolve(items)
        _, blocks, _ = AssociationService().associate(
            [page],
            order,
            [],
            block_boxes={1: [_bbox(80, 90, 720, 141), _bbox(80, 142, 720, 175)]},
        )
        assert len(blocks) == 2
        assert blocks[1].text == "and expected operating results."

    def test_item_outside_every_block_box_stays_its_own_block(self):
        items = self._three_lines()
        items.insert(0, _item("hdr", "PART I", _bbox(500, 20, 560, 34)))
        page, order = self._resolve(items)
        _, blocks, _ = AssociationService().associate([page], order, [], block_boxes={1: [_bbox(80, 90, 720, 175)]})
        assert len(blocks) == 2
        assert blocks[0].text == "PART I"

    def test_falls_back_to_line_bucket_without_block_boxes(self):
        items = self._three_lines()
        page, order = self._resolve(items)
        _, blocks, _ = AssociationService().associate([page], order, [], block_boxes=None)
        # Previous behaviour: one block per line.
        assert len(blocks) == 3

    def test_region_change_still_splits_within_one_block_box(self):
        items = self._three_lines()
        page, order = self._resolve(items)
        regions = [
            _region("r_table", "table", _bbox(80, 90, 720, 141)),
            _region("r_text", "text_block", _bbox(80, 142, 720, 175)),
        ]
        _, blocks, _ = AssociationService().associate(
            [page], order, regions, block_boxes={1: [_bbox(80, 90, 720, 175)]}
        )
        assert len(blocks) == 2
        assert blocks[0].region_ids == ["r_table"]
        assert blocks[1].region_ids == ["r_text"]


class TestTableStructureService:
    """Cell structure is recovered from the table crops already on disk.

    Layout detection only yields a table's bounding box, so without this a
    balance sheet flattens to row strings and the column a figure belongs to is
    lost.
    """

    HTML = "<table><tr><td>Cash</td><td>24,296</td><td>30,242</td></tr></table>"

    @staticmethod
    def _predictor(table_res_list):
        result = MagicMock()
        result.json = {"res": {"table_res_list": table_res_list}}
        predictor = MagicMock()
        predictor.predict.return_value = [result]
        return predictor

    @staticmethod
    def _page(tmp_path, page_number=1, size=(600, 800)):
        from PIL import Image

        path = tmp_path / f"page_{page_number}.png"
        Image.new("RGB", size, "white").save(path)
        return PageContext(page_number=page_number, width=float(size[0]), height=float(size[1]), page_image_path=path)

    def _run(self, pages, regions, table_res_list, assets=None):
        from document_process.services import TableStructureService

        with patch(
            "document_process.services._get_paddle_table_recognizer",
            return_value=self._predictor(table_res_list),
        ):
            return TableStructureService().recognize(pages=pages, regions=regions, assets=assets)

    def test_recovers_html_for_a_table_region(self, tmp_path):
        page = self._page(tmp_path)
        region = _region("r1", "table", _bbox(50, 60, 550, 700))
        structures, issues = self._run(
            [page], [region], [{"pred_html": self.HTML, "cell_box_list": [[0, 0, 1, 1]] * 3}]
        )
        assert issues == []
        assert len(structures) == 1
        assert structures[0].html == self.HTML
        assert structures[0].cell_count == 3
        assert structures[0].region_id == "r1"

    def test_crops_tight_to_the_region_not_the_padded_asset(self, tmp_path):
        """The saved crop is padded for the VLM; that padding would read as rows."""
        from document_process.services import TableStructureService

        page = self._page(tmp_path)
        region = _region("r1", "table", _bbox(50, 60, 550, 700))
        predictor = self._predictor([{"pred_html": self.HTML}])
        with patch("document_process.services._get_paddle_table_recognizer", return_value=predictor):
            TableStructureService().recognize(pages=[page], regions=[region], assets=[_crop_asset("r1")])
        sent = predictor.predict.call_args.args[0]
        assert sent.shape[0] == 700 - 60  # height, tight to the region
        assert sent.shape[1] == 550 - 50  # width

    def test_figures_are_skipped(self, tmp_path):
        page = self._page(tmp_path)
        region = _region("r1", "figure", _bbox(50, 60, 550, 700))
        structures, issues = self._run([page], [region], [{"pred_html": self.HTML}])
        assert structures == [] and issues == []

    def test_missing_page_image_warns_without_failing(self, tmp_path):
        page = PageContext(page_number=1, width=600.0, height=800.0, page_image_path=tmp_path / "gone.png")
        region = _region("r1", "table", _bbox(50, 60, 550, 700))
        structures, issues = self._run([page], [region], [{"pred_html": self.HTML}])
        assert structures == []
        assert [i.code for i in issues] == ["table_page_image_missing"]

    def test_recognition_failure_is_isolated_to_one_table(self, tmp_path):
        from document_process.services import TableStructureService

        page = self._page(tmp_path)
        regions = [
            _region("r1", "table", _bbox(50, 60, 550, 300)),
            _region("r2", "table", _bbox(50, 320, 550, 700)),
        ]
        predictor = MagicMock()
        good = MagicMock()
        good.json = {"res": {"table_res_list": [{"pred_html": self.HTML}]}}
        predictor.predict.side_effect = [RuntimeError("boom"), [good]]

        with patch("document_process.services._get_paddle_table_recognizer", return_value=predictor):
            structures, issues = TableStructureService().recognize(pages=[page], regions=regions)

        assert [s.region_id for s in structures] == ["r2"]
        assert [i.code for i in issues] == ["table_structure_failed"]

    def test_html_becomes_the_visual_summary_text(self, tmp_path):
        from document_process.models import TableStructure

        region = _region("r1", "table", _bbox(0, 0, 100, 100))
        block = _text_block("b1", "Cash and cash equivalents $ 24,296 $ 30,242", region_ids=["r1"])
        structure = TableStructure(
            table_id="table_r1",
            region_id="r1",
            page_number=1,
            crop_path="/tmp/c.png",
            html=self.HTML,
            cell_count=3,
        )
        summaries = build_visual_summaries(
            regions=[region],
            ordered_blocks=[block],
            chunks=[],
            cropped_assets=[],
            table_structures=[structure],
        )
        assert summaries[0].summary_text == self.HTML
        assert summaries[0].metadata["cell_count"] == 3

    def test_summary_falls_back_to_block_text_without_structure(self):
        region = _region("r1", "table", _bbox(0, 0, 100, 100))
        block = _text_block("b1", "Cash and cash equivalents", region_ids=["r1"])
        summaries = build_visual_summaries(regions=[region], ordered_blocks=[block], chunks=[], cropped_assets=[])
        assert summaries[0].summary_text == "Cash and cash equivalents"
        assert "table_html" not in summaries[0].metadata
