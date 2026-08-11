"""
Tests for preferring a PDF's embedded text layer over OCR.

Born-digital PDFs (Word/web exports) carry exact text. Rendering them to images
and re-reading them with OCR loses word spacing -- PaddleOCR emitted
"Microsoft Cloudrevenuewas$51.5billion" for text the file already stored as
"Microsoft Cloud revenue was $51.5 billion". These tests pin the extraction and
the coordinate transform that let the pipeline use the real text instead.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from document_process.services import (
    _merge_fragments_into_lines,
    _pdf_rect_to_image_bbox,
)

# ── coordinate transform ──────────────────────────────────────────────────────
# PDF text lives in point space with a bottom-left origin and y pointing up.
# Layout regions and OCR boxes live in rendered-image pixel space with a
# top-left origin and y pointing down. Getting this wrong silently misassociates
# every line with the wrong layout region.


def test_pdf_rect_converts_to_image_pixel_space():
    # Letter page (612x792pt) rendered at scale 2.0 -> 1224x1584px.
    bbox = _pdf_rect_to_image_bbox(
        (73.0, 705.7, 174.0, 715.6),  # (left, bottom, right, top) in points
        page_width_pt=612.0,
        page_height_pt=792.0,
        image_width=1224.0,
        image_height=1584.0,
    )
    assert bbox.x0 == 146.0
    assert bbox.x1 == 348.0
    # top of the rect in PDF space becomes the SMALL y in image space
    assert round(bbox.y0, 1) == 152.8
    assert round(bbox.y1, 1) == 172.6


def test_converted_bbox_is_valid_for_downstream_association():
    """y must increase downward, or BoundingBox.is_valid() rejects the item."""
    bbox = _pdf_rect_to_image_bbox(
        (10.0, 100.0, 200.0, 120.0),
        page_width_pt=612.0,
        page_height_pt=792.0,
        image_width=1224.0,
        image_height=1584.0,
    )
    assert bbox.is_valid()


# ── fragment merging ──────────────────────────────────────────────────────────
# pypdfium2 returns style runs, not lines: 'Microsoft Cloud ', 'and AI ',
# 'Strength '. Concatenating them verbatim reproduces the original line exactly,
# including its spaces.


def _frag(x0, y0, x1, y1, text):
    from document_process.models import BoundingBox

    return (BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1), text)


def test_fragments_on_one_line_merge_preserving_exact_spacing():
    """This is the bug: OCR lost these spaces, the text layer has them."""
    fragments = [
        _frag(100, 200, 180, 215, "Microsoft Cloud "),
        _frag(180, 200, 230, 215, "revenue "),
        _frag(230, 200, 260, 215, "was "),
        _frag(260, 200, 300, 215, "$51.5 "),
        _frag(300, 200, 340, 215, "billion"),
    ]

    lines = _merge_fragments_into_lines(fragments, y_tolerance=4.0)

    assert len(lines) == 1
    assert lines[0][1] == "Microsoft Cloud revenue was $51.5 billion"


def test_merged_line_bbox_spans_all_its_fragments():
    fragments = [
        _frag(100, 200, 180, 215, "left "),
        _frag(180, 202, 340, 214, "right"),
    ]

    ((bbox, _text),) = _merge_fragments_into_lines(fragments, y_tolerance=4.0)

    assert (bbox.x0, bbox.y0, bbox.x1, bbox.y1) == (100, 200, 340, 215)


def test_fragments_on_separate_lines_stay_separate():
    fragments = [
        _frag(100, 200, 200, 215, "first line"),
        _frag(100, 240, 200, 255, "second line"),
    ]

    lines = _merge_fragments_into_lines(fragments, y_tolerance=4.0)

    assert [text for _bbox, text in lines] == ["first line", "second line"]


def test_fragments_are_ordered_left_to_right_within_a_line():
    """pypdfium2 does not guarantee reading order within a line."""
    fragments = [
        _frag(300, 200, 340, 215, "billion"),
        _frag(100, 200, 180, 215, "Microsoft Cloud "),
        _frag(180, 200, 300, 215, "revenue was $51.5 "),
    ]

    lines = _merge_fragments_into_lines(fragments, y_tolerance=4.0)

    assert lines[0][1] == "Microsoft Cloud revenue was $51.5 billion"


def test_lines_are_ordered_top_to_bottom():
    fragments = [
        _frag(100, 240, 200, 255, "second"),
        _frag(100, 200, 200, 215, "first"),
    ]

    lines = _merge_fragments_into_lines(fragments, y_tolerance=4.0)

    assert [text for _bbox, text in lines] == ["first", "second"]


def test_blank_fragments_are_dropped():
    fragments = [
        _frag(100, 200, 180, 215, "kept"),
        _frag(180, 200, 190, 215, "   "),
    ]

    lines = _merge_fragments_into_lines(fragments, y_tolerance=4.0)

    assert [text for _bbox, text in lines] == ["kept"]


def test_no_fragments_yields_no_lines():
    """A scanned page has an empty text layer, so the caller falls back to OCR."""
    assert _merge_fragments_into_lines([], y_tolerance=4.0) == []


# ── table column gaps ─────────────────────────────────────────────────────────
# Fragments in different table cells sit on the same visual line but are far
# apart horizontally, and a cell's text carries no trailing space. Concatenating
# verbatim glued adjacent column headers together: "Revenue" + "Income" became
# "RevenueIncome". A wide gap means a column boundary, so a space belongs there.


def test_wide_horizontal_gap_between_table_cells_becomes_a_space():
    fragments = [
        _frag(100, 200, 200, 215, "Revenue"),
        _frag(320, 200, 420, 215, "Income"),  # next column, big gap
    ]

    lines = _merge_fragments_into_lines(fragments, y_tolerance=4.0)

    assert lines[0][1] == "Revenue Income"


def test_adjacent_style_runs_are_not_given_an_extra_space():
    """Style changes mid-sentence have no gap and must join verbatim."""
    fragments = [
        _frag(100, 200, 180, 215, "Microsoft Cloud "),
        _frag(180, 200, 260, 215, "revenue"),
    ]

    lines = _merge_fragments_into_lines(fragments, y_tolerance=4.0)

    assert lines[0][1] == "Microsoft Cloud revenue"


def test_gap_after_text_that_already_ends_in_a_space_stays_single_spaced():
    fragments = [
        _frag(100, 200, 200, 215, "Revenue "),
        _frag(320, 200, 420, 215, "Income"),
    ]

    lines = _merge_fragments_into_lines(fragments, y_tolerance=4.0)

    assert lines[0][1] == "Revenue Income"


# ── OCRService routing ────────────────────────────────────────────────────────


def _page(number=1, path="/tmp/page_1.png"):
    from pathlib import Path

    from document_process.services import PageContext

    return PageContext(page_number=number, width=1224.0, height=1584.0, page_image_path=Path(path))


def _text_item(page_number=1, text="Microsoft Cloud revenue was $51.5 billion"):
    from document_process.models import BoundingBox, OCRTextItem

    return OCRTextItem(
        item_id=f"p{page_number}_pdftext_1",
        page_number=page_number,
        text=text,
        bbox=BoundingBox(x0=1, y0=1, x1=100, y1=20),
        confidence=1.0,
        source="pdf_text_layer",
    )


def test_born_digital_pdf_uses_text_layer_and_never_loads_paddle():
    """The whole point: don't OCR a picture of text the file already contains."""
    from document_process.services import OCRService

    with (
        patch("document_process.services._extract_pdf_text_layer", return_value={1: [_text_item()]}),
        patch("document_process.services._get_paddle_ocr") as mock_paddle,
    ):
        results, issues = OCRService().extract([_page()], pdf_path=Path("doc.pdf"))

    mock_paddle.assert_not_called()
    assert results[0].text_source == "pdf_text_layer"
    assert results[0].items[0].text == "Microsoft Cloud revenue was $51.5 billion"
    assert issues == []


def test_scanned_pdf_with_no_text_layer_falls_back_to_paddle():
    from document_process.services import OCRService

    with (
        patch("document_process.services._extract_pdf_text_layer", return_value={}),
        patch("document_process.services._get_paddle_ocr") as mock_paddle,
    ):
        mock_paddle.return_value.predict.return_value = [
            MagicMock(json={"res": {"rec_texts": [], "rec_scores": [], "rec_boxes": [], "dt_polys": []}})
        ]
        results, _issues = OCRService().extract([_page()], pdf_path=Path("scan.pdf"))

    mock_paddle.assert_called_once()
    assert results[0].text_source != "pdf_text_layer"


def test_hybrid_pdf_routes_each_page_independently():
    """Digital page uses the text layer; scanned page still gets OCR."""
    from document_process.services import OCRService

    with (
        patch("document_process.services._extract_pdf_text_layer", return_value={1: [_text_item(1)]}),
        patch("document_process.services._get_paddle_ocr") as mock_paddle,
    ):
        mock_paddle.return_value.predict.return_value = [
            MagicMock(json={"res": {"rec_texts": [], "rec_scores": [], "rec_boxes": [], "dt_polys": []}})
        ]
        results, _issues = OCRService().extract(
            [_page(1, "/tmp/page_1.png"), _page(2, "/tmp/page_2.png")],
            pdf_path=Path("hybrid.pdf"),
        )

    assert results[0].text_source == "pdf_text_layer"
    assert results[1].text_source != "pdf_text_layer"


def test_images_skip_the_text_layer_entirely():
    """A .png has no text layer to read; pdf_path is None."""
    from document_process.services import OCRService

    with (
        patch("document_process.services._extract_pdf_text_layer") as mock_layer,
        patch("document_process.services._get_paddle_ocr") as mock_paddle,
    ):
        mock_paddle.return_value.predict.return_value = [
            MagicMock(json={"res": {"rec_texts": [], "rec_scores": [], "rec_boxes": [], "dt_polys": []}})
        ]
        OCRService().extract([_page()], pdf_path=None)

    mock_layer.assert_not_called()
    mock_paddle.assert_called_once()
