"""
Tests that DocumentPreprocessingPipeline produces identical results
regardless of batch size, using mocked services.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from document_process.models import (
    BoundingBox,
    LayoutRegion,
    OCRPageResult,
    OCRTextItem,
    OrderedTextBlock,
    RegionAssociation,
)
from document_process.services import (
    LoadedDocument,
    PageContext,
)


def _bbox(x0, y0, x1, y1):
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _make_page(n: int, tmp_path: Path) -> PageContext:
    img = tmp_path / f"page_{n}.png"
    img.write_bytes(b"fake")
    return PageContext(page_number=n, width=100.0, height=200.0, page_image_path=img)


def _make_ocr_page(n: int) -> OCRPageResult:
    item = OCRTextItem(item_id=f"p{n}_ocr_1", page_number=n, text=f"text{n}", bbox=_bbox(0, 0, 50, 10))
    return OCRPageResult(page_number=n, items=[item])


def _make_region(n: int) -> LayoutRegion:
    return LayoutRegion(region_id=f"region_{n}", region_type="text_block", page_number=n, bbox=_bbox(0, 0, 100, 200))


def _make_block(n: int, item_id: str) -> OrderedTextBlock:
    return OrderedTextBlock(
        block_id=f"block_{n}",
        page_number=n,
        text=f"text{n}",
        item_ids=[item_id],
        region_ids=[],
        reading_order=n,
        bbox=_bbox(0, 0, 100, 200),
    )


def _reading_order_for_pages(page_numbers: list[int]) -> dict:
    return {
        "resolver": "ocr_bbox_sort_v1",
        "document_order_item_ids": [f"p{n}_ocr_1" for n in page_numbers],
        "pages": [{"page_number": n, "ordered_item_ids": [f"p{n}_ocr_1"]} for n in page_numbers],
    }


def _build_pipeline(settings, tmp_path, num_pages: int):
    """Build a pipeline with fully mocked services for `num_pages` pages."""
    from document_process.pipeline import DocumentPreprocessingPipeline

    pages = [_make_page(n, tmp_path) for n in range(1, num_pages + 1)]
    ocr_pages = [_make_ocr_page(n) for n in range(1, num_pages + 1)]
    regions = [_make_region(n) for n in range(1, num_pages + 1)]

    loaded = LoadedDocument(
        document_id="test-doc",
        source_path=tmp_path / "test.pdf",
        working_dir=tmp_path / "processed",
        original_copy_path=tmp_path / "test.pdf",
        pages=pages,
    )
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "test.pdf").write_bytes(b"fake pdf")

    loader_mock = MagicMock()
    loader_mock.load.return_value = loaded

    ocr_mock = MagicMock()

    def ocr_extract(batch_pages, *, pdf_path=None, on_page_done=None):
        nums = [p.page_number for p in batch_pages]
        if on_page_done is not None:
            for _ in batch_pages:
                on_page_done()
        return [o for o in ocr_pages if o.page_number in nums], []

    ocr_mock.extract.side_effect = ocr_extract

    reading_order_mock = MagicMock()

    def ro_resolve(batch_ocr):
        nums = [o.page_number for o in batch_ocr]
        return _reading_order_for_pages(nums), []

    reading_order_mock.resolve.side_effect = ro_resolve

    layout_mock = MagicMock()

    def layout_detect(batch_pages, batch_ocr):
        nums = [p.page_number for p in batch_pages]
        return [r for r in regions if r.page_number in nums], [], "PP-DocLayout_plus-L"

    layout_mock.detect.side_effect = layout_detect

    assoc_mock = MagicMock()

    def association_associate(batch_ocr, batch_ro, batch_regions, *, start_index=1):
        nums = [o.page_number for o in batch_ocr]
        blocks = [_make_block(start_index + i, f"p{n}_ocr_1") for i, n in enumerate(nums)]
        assocs = [
            RegionAssociation(
                association_id=f"assoc_{start_index + i}",
                page_number=n,
                item_id=f"p{n}_ocr_1",
                region_id=f"region_{n}",
                region_type="text_block",
                overlap_ratio=1.0,
            )
            for i, n in enumerate(nums)
        ]
        ordered_text = {
            "pages": [{"page_number": n, "blocks": [], "text": f"text{n}", "text_region_count": 1} for n in nums],
            "full_text": " ".join(f"text{n}" for n in nums),
        }
        return assocs, blocks, ordered_text

    assoc_mock.associate.side_effect = association_associate

    crop_mock = MagicMock()
    crop_mock.crop_visual_regions.return_value = ([], [])

    pipeline = DocumentPreprocessingPipeline(
        settings,
        loader=loader_mock,
        ocr=ocr_mock,
        reading_order=reading_order_mock,
        layout=layout_mock,
        association=assoc_mock,
        cropping=crop_mock,
    )
    return pipeline


@patch("document_process.pipeline.build_chunks", return_value=[])
@patch("document_process.pipeline.build_visual_summaries", return_value=[])
@patch("document_process.pipeline.build_document_artifacts")
@patch("document_process.pipeline.export_artifacts")
def test_batching_calls_ocr_per_batch(mock_export, mock_build_doc, mock_vis, mock_chunks, tmp_settings, tmp_path):
    """OCR must be called once per batch, not once for all pages."""
    mock_export.return_value = tmp_path / "processed" / "document.json"
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    mock_build_doc.return_value = (MagicMock(), MagicMock())

    settings = tmp_settings(preprocess_page_batch_size=2)
    pipeline = _build_pipeline(settings, tmp_path, num_pages=5)
    pipeline.run(tmp_path / "test.pdf", document_id="test-doc")

    # 5 pages with batch_size=2 → 3 batches (2+2+1)
    assert pipeline.ocr.extract.call_count == 3


@patch("document_process.pipeline.build_chunks", return_value=[])
@patch("document_process.pipeline.build_visual_summaries", return_value=[])
@patch("document_process.pipeline.build_document_artifacts")
@patch("document_process.pipeline.export_artifacts")
def test_batching_accumulates_all_regions(mock_export, mock_build_doc, mock_vis, mock_chunks, tmp_settings, tmp_path):
    """All regions from all batches must reach build_document_artifacts."""
    mock_export.return_value = tmp_path / "processed" / "document.json"
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)

    captured = {}

    def capture_build_doc(**kwargs):
        captured["regions"] = kwargs.get("regions", [])
        return MagicMock(), MagicMock()

    mock_build_doc.side_effect = capture_build_doc

    settings = tmp_settings(preprocess_page_batch_size=2)
    pipeline = _build_pipeline(settings, tmp_path, num_pages=5)
    pipeline.run(tmp_path / "test.pdf", document_id="test-doc")

    assert len(captured["regions"]) == 5


@patch("document_process.pipeline.build_chunks", return_value=[])
@patch("document_process.pipeline.build_visual_summaries", return_value=[])
@patch("document_process.pipeline.build_document_artifacts")
@patch("document_process.pipeline.export_artifacts")
def test_reading_order_merges_all_pages(mock_export, mock_build_doc, mock_vis, mock_chunks, tmp_settings, tmp_path):
    """Merged reading order must contain item IDs from all pages."""
    mock_export.return_value = tmp_path / "processed" / "document.json"
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)

    captured = {}

    def capture_export(**kwargs):
        captured["reading_order"] = kwargs.get("reading_order", {})
        return tmp_path / "processed" / "document.json"

    mock_export.side_effect = capture_export
    mock_build_doc.return_value = (MagicMock(), MagicMock())

    settings = tmp_settings(preprocess_page_batch_size=2)
    pipeline = _build_pipeline(settings, tmp_path, num_pages=4)
    pipeline.run(tmp_path / "test.pdf", document_id="test-doc")

    ro = captured["reading_order"]
    assert len(ro["document_order_item_ids"]) == 4
    assert len(ro["pages"]) == 4


@patch("document_process.pipeline.build_chunks", return_value=[])
@patch("document_process.pipeline.build_visual_summaries", return_value=[])
@patch("document_process.pipeline.build_document_artifacts")
@patch("document_process.pipeline.export_artifacts")
def test_association_start_index_increments_across_batches(
    mock_export, mock_build_doc, mock_vis, mock_chunks, tmp_settings, tmp_path
):
    """start_index passed to associate() must increment by batch size each batch."""
    mock_export.return_value = tmp_path / "processed" / "document.json"
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    mock_build_doc.return_value = (MagicMock(), MagicMock())

    settings = tmp_settings(preprocess_page_batch_size=2)
    pipeline = _build_pipeline(settings, tmp_path, num_pages=5)
    pipeline.run(tmp_path / "test.pdf", document_id="test-doc")

    calls = pipeline.association.associate.call_args_list
    assert calls[0].kwargs["start_index"] == 1
    assert calls[1].kwargs["start_index"] == 3  # 1 + 2 pages in first batch
    assert calls[2].kwargs["start_index"] == 5  # 3 + 2 pages in second batch


@patch("document_process.pipeline.build_chunks", return_value=[])
@patch("document_process.pipeline.build_visual_summaries", return_value=[])
@patch("document_process.pipeline.build_document_artifacts")
@patch("document_process.pipeline.export_artifacts")
def test_batch_size_larger_than_pages_runs_single_batch(
    mock_export, mock_build_doc, mock_vis, mock_chunks, tmp_settings, tmp_path
):
    """batch_size > num_pages should behave identically to no batching."""
    mock_export.return_value = tmp_path / "processed" / "document.json"
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    mock_build_doc.return_value = (MagicMock(), MagicMock())

    settings = tmp_settings(preprocess_page_batch_size=100)
    pipeline = _build_pipeline(settings, tmp_path, num_pages=3)
    pipeline.run(tmp_path / "test.pdf", document_id="test-doc")

    assert pipeline.ocr.extract.call_count == 1


@patch("document_process.pipeline.build_chunks", return_value=[])
@patch("document_process.pipeline.build_visual_summaries", return_value=[])
@patch("document_process.pipeline.build_document_artifacts")
@patch("document_process.pipeline.export_artifacts")
def test_batch_progress_logged(mock_export, mock_build_doc, mock_vis, mock_chunks, tmp_settings, tmp_path, caplog):
    """A progress log line must be emitted at INFO for each batch."""
    mock_export.return_value = tmp_path / "processed" / "document.json"
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    mock_build_doc.return_value = (MagicMock(), MagicMock())

    settings = tmp_settings(preprocess_page_batch_size=2)
    pipeline = _build_pipeline(settings, tmp_path, num_pages=4)

    with caplog.at_level(logging.INFO, logger="document_process.pipeline"):
        pipeline.run(tmp_path / "test.pdf", document_id="test-doc")

    batch_logs = [r for r in caplog.records if "Batch" in r.message and "pages" in r.message]
    assert len(batch_logs) == 2  # 4 pages / 2 per batch = 2 batches


@patch("document_process.pipeline.build_chunks", return_value=[])
@patch("document_process.pipeline.build_visual_summaries", return_value=[])
@patch("document_process.pipeline.build_document_artifacts")
@patch("document_process.pipeline.export_artifacts")
def test_on_progress_receives_cumulative_page_counts(
    mock_export, mock_build_doc, mock_vis, mock_chunks, tmp_settings, tmp_path
):
    """on_progress must be called with (pages_done, total_pages) for each page,
    accumulating across batches."""
    mock_export.return_value = tmp_path / "processed" / "document.json"
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    mock_build_doc.return_value = (MagicMock(), MagicMock())

    settings = tmp_settings(preprocess_page_batch_size=2)
    pipeline = _build_pipeline(settings, tmp_path, num_pages=4)

    progress_calls = []
    pipeline.run(
        tmp_path / "test.pdf",
        document_id="test-doc",
        on_progress=lambda done, total: progress_calls.append((done, total)),
    )

    # 4 pages → on_progress called 4 times
    assert len(progress_calls) == 4
    # pages_done increments from 1 to 4
    assert [c[0] for c in progress_calls] == [1, 2, 3, 4]
    # total_pages is always 4
    assert all(c[1] == 4 for c in progress_calls)
