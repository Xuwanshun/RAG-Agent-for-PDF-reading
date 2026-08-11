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
    _group_rects_into_lines,
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


# ── grouping rects into line boxes ────────────────────────────────────────────
# We no longer rebuild line text ourselves. Concatenating fragment strings and
# guessing where spaces belong produced three successive defects on the same
# corpus: missing spaces ("Cloudrevenuewas$51.5billion"), reordered words
# ("e was on a non Net incom"), then dropped hyphens ("nonGAAP") and numbers
# split across fragments ("$25,04 ... 7").
#
# pypdfium2 already knows the correct order and spacing from the PDF content
# stream. So rects are used only to locate lines; the text of each line is read
# back out of the document by box. These tests pin the grouping geometry.


def test_rects_are_grouped_by_vertical_overlap_not_edge_proximity():
    """
    Neither edge of a rect is stable across a line: tops rise with ascenders and
    bottoms drop with descenders. 'Strength' (descender 'g') sits 3.7px below its
    own line's other rects, and edge-based grouping split it into a second,
    overlapping line — so its text was extracted twice.
    """
    rects = [
        (100.0, 700.0, 200.0, 712.0),  # no descender
        (210.0, 697.0, 300.0, 712.0),  # descender drops the bottom
    ]

    lines = _group_rects_into_lines(rects)

    assert len(lines) == 1
    assert lines[0] == (100.0, 697.0, 300.0, 712.0)


def test_rects_on_genuinely_different_lines_stay_separate():
    rects = [
        (100.0, 700.0, 200.0, 712.0),
        (100.0, 680.0, 200.0, 692.0),
    ]

    assert len(_group_rects_into_lines(rects)) == 2


def test_lines_are_returned_top_to_bottom():
    """PDF y increases upward, so the highest line comes first."""
    rects = [
        (100.0, 680.0, 200.0, 692.0),
        (100.0, 700.0, 200.0, 712.0),
    ]

    lines = _group_rects_into_lines(rects)

    assert [line[3] for line in lines] == [712.0, 692.0]


def test_a_merged_line_box_spans_every_member():
    rects = [
        (100.0, 700.0, 150.0, 712.0),
        (400.0, 700.0, 500.0, 715.0),
    ]

    (line,) = _group_rects_into_lines(rects)

    assert line == (100.0, 700.0, 500.0, 715.0)


def test_a_superscript_does_not_start_its_own_line():
    """Footnote markers sit high but overlap their line substantially."""
    rects = [
        (100.0, 700.0, 200.0, 712.0),
        (201.0, 706.0, 206.0, 714.0),
    ]

    assert len(_group_rects_into_lines(rects)) == 1


def test_no_rects_yields_no_lines():
    assert _group_rects_into_lines([]) == []


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
