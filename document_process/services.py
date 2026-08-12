from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

# ── PaddlePaddle 3.3.x stability patches ─────────────────────────────────────
# PaddlePaddle 3.3.x on x86 CPU crashes during inference via its PIR executor.
# Setting FLAGS_* as environment variables is insufficient — the inference API
# ignores them. These patches must run at import time, before any predictor
# is created.
try:
    import paddle as _paddle
    import paddle.inference as _pi

    # Prevent oneDNN/mkldnn from being enabled — causes NotImplementedError in
    # the PIR+oneDNN code path on x86 (PaddlePaddle issue #77340).
    _pi.Config.enable_mkldnn = lambda self: None  # type: ignore[method-assign]

    # Switch from the PIR executor to the legacy executor. The PIR executor in
    # 3.3.x crashes non-deterministically on x86 CPU (AddFunctor, DepthwiseConv,
    # etc.) — the exact op varies per run, making it a runtime bug in PIR itself.
    _paddle.set_flags({"FLAGS_enable_pir_in_executor": False})

except Exception as _e:
    # Paddle not installed (e.g. unit tests) — patches are skipped silently.
    logging.getLogger(__name__).debug("Paddle patches skipped: %s", _e)
# ─────────────────────────────────────────────────────────────────────────────

from config import Settings
from document_process.models import (
    BoundingBox,
    CroppedRegionAsset,
    LayoutRegion,
    OCRPageResult,
    OCRTextItem,
    OrderedTextBlock,
    ProcessedChunk,
    ProcessedDocument,
    ProcessedManifest,
    ProcessingIssue,
    ProcessingMetadata,
    RegionAssociation,
    TableStructure,
    VisualRegionSummary,
)

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TEXT_BLOCK_LABELS = {
    "text",
    "title",
    "doc_title",
    "figure_title",
    "paragraph_title",
    "header",
    "footer",
    "reference",
    "caption",
    "list",
    "number",
    "formula_caption",
    "table_caption",
    "figure_caption",
    "aside_text",
}
FIGURE_LABELS = {"image", "figure", "chart", "graph"}
logger = logging.getLogger(__name__)


def _configure_paddle_env(cache_dir: Path) -> None:
    """
    Set environment variables controlling where PaddleOCR/PaddleX store downloaded
    models and temp files.

    Without these, Paddle defaults to writing into the working directory, which
    pollutes the project root locally and fails in read-only Docker containers.
    """
    cache_home = cache_dir.resolve()
    cache_home.mkdir(parents=True, exist_ok=True)
    (cache_home / "temp").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_home))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    # Disable oneDNN and the PIR executor — the PIR + oneDNN combination causes
    # NotImplementedError on ECS Fargate Intel CPUs with PaddlePaddle 3.x.
    # FLAGS_use_mkldnn disables oneDNN; FLAGS_enable_pir_in_executor forces the
    # old executor which does not have the PIR oneDNN instruction bug.
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_pir_in_executor"] = "0"


@dataclass(frozen=True)
class PageContext:
    page_number: int
    width: float | None
    height: float | None
    page_image_path: Path


@dataclass(frozen=True)
class LoadedDocument:
    document_id: str
    source_path: Path
    working_dir: Path
    original_copy_path: Path
    pages: list[PageContext]


class DocumentLoaderService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Configure Paddle cache now that we have the settings object.
        # This is the earliest point we know the correct cache directory.
        _configure_paddle_env(settings.paddle_cache_dir)

    def load(self, source_path: Path, *, document_id: str | None = None) -> LoadedDocument:
        logger.info("Loading document for preprocessing: %s", source_path)
        if source_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported document type: {source_path.suffix or 'no extension'}")

        resolved_id = document_id or self._build_document_id(source_path)
        working_dir = self.settings.processed_documents_dir / resolved_id
        if working_dir.exists():
            shutil.rmtree(working_dir)
        source_dir = working_dir / "source"
        pages_dir = source_dir / "pages"
        source_dir.mkdir(parents=True, exist_ok=True)
        pages_dir.mkdir(parents=True, exist_ok=True)
        original_copy_path = source_dir / source_path.name
        if source_path.resolve() != original_copy_path.resolve():
            shutil.copy2(source_path, original_copy_path)

        if original_copy_path.suffix.lower() == ".pdf":
            pages = _load_pdf_pages(original_copy_path, pages_dir, render_scale=self.settings.pdf_render_scale)
        else:
            pages = [_load_image_page(original_copy_path, page_number=1)]

        return LoadedDocument(
            document_id=resolved_id,
            source_path=source_path,
            working_dir=working_dir,
            original_copy_path=original_copy_path,
            pages=pages,
        )

    def _build_document_id(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def _pdf_rect_to_image_bbox(
    rect: tuple[float, float, float, float],
    *,
    page_width_pt: float,
    page_height_pt: float,
    image_width: float,
    image_height: float,
) -> BoundingBox:
    """
    Convert a pypdfium2 text rect into rendered-image pixel space.

    PDF text rects are (left, bottom, right, top) in points, measured from a
    bottom-left origin with y pointing up. Layout regions and OCR boxes are in
    image pixels from a top-left origin with y pointing down, so the y axis has
    to be flipped as well as scaled. Getting this wrong does not crash — it
    silently associates every line with the wrong layout region.
    """
    left, bottom, right, top = rect
    scale_x = image_width / page_width_pt if page_width_pt else 1.0
    scale_y = image_height / page_height_pt if page_height_pt else 1.0
    return BoundingBox(
        x0=left * scale_x,
        y0=(page_height_pt - top) * scale_y,
        x1=right * scale_x,
        y1=(page_height_pt - bottom) * scale_y,
    )


def _group_rects_into_lines(
    rects: list[tuple[float, float, float, float]],
    *,
    overlap_ratio: float = 0.5,
) -> list[tuple[float, float, float, float]]:
    """
    Group PDF text rects belonging to the same visual line into one box.

    Rects are ``(left, bottom, right, top)`` in PDF points, y increasing upward.

    Grouping is by vertical OVERLAP rather than by edge proximity, because
    neither edge is stable across a line: tops rise with ascenders and bottoms
    drop with descenders. Keying on an edge split a descender-bearing word onto
    its own overlapping line, and querying both boxes then returned that text
    twice. Two rects join the same line when they overlap vertically by at least
    ``overlap_ratio`` of the shorter one.

    Returns merged boxes ordered top-to-bottom.
    """
    if not rects:
        return []

    # Tallest first, so a line is anchored by its full-height rect rather than by
    # a superscript that happens to come first in reading order.
    ordered = sorted(rects, key=lambda r: (-(r[3] - r[1]), -r[3], r[0]))

    groups: list[list[float]] = []  # [left, bottom, right, top] accumulators
    members: list[list[tuple[float, float, float, float]]] = []
    for rect in ordered:
        left, bottom, right, top = rect
        height = max(top - bottom, 1e-6)
        placed = False
        for index, group in enumerate(groups):
            overlap = min(top, group[3]) - max(bottom, group[1])
            shorter = min(height, max(group[3] - group[1], 1e-6))
            if overlap >= overlap_ratio * shorter:
                group[0] = min(group[0], left)
                group[1] = min(group[1], bottom)
                group[2] = max(group[2], right)
                group[3] = max(group[3], top)
                members[index].append(rect)
                placed = True
                break
        if not placed:
            groups.append([left, bottom, right, top])
            members.append([rect])

    boxes = [(g[0], g[1], g[2], g[3]) for g in groups]
    boxes.sort(key=lambda b: -b[3])
    return boxes


def _extract_pdf_text_layer(
    pdf_path: Path,
    pages: list[PageContext],
    *,
    baseline_tolerance: float = 2.0,
) -> dict[int, list[OCRTextItem]]:
    """
    Read the PDF's embedded text layer, one entry per page that has usable text.

    Pages absent from the returned mapping have no text layer (a scan) and must
    fall back to OCR. Decided per page so hybrid PDFs — digital pages plus
    scanned inserts — take the right path for each.
    """
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception:
        logger.warning("pypdfium2 unavailable; cannot read PDF text layer, falling back to OCR")
        return {}

    by_page: dict[int, list[OCRTextItem]] = {}
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for page in pages:
            index = page.page_number - 1
            if index < 0 or index >= len(pdf):
                continue
            try:
                pdf_page = pdf[index]
                text_page = pdf_page.get_textpage()
                page_width_pt = float(pdf_page.get_width())
                page_height_pt = float(pdf_page.get_height())

                rects = [text_page.get_rect(i) for i in range(text_page.count_rects())]
                line_boxes = _group_rects_into_lines(rects)

                items: list[OCRTextItem] = []
                for order, box in enumerate(line_boxes, start=1):
                    # Ask the document for this line's text rather than gluing
                    # fragment strings together. pypdfium2 knows the reading
                    # order and spacing from the content stream; reassembling it
                    # by hand dropped hyphens ("non-GAAP" -> "nonGAAP") and split
                    # numbers across fragments.
                    text = text_page.get_text_bounded(*box)
                    if not text or not text.strip():
                        continue
                    bbox = _pdf_rect_to_image_bbox(
                        box,
                        page_width_pt=page_width_pt,
                        page_height_pt=page_height_pt,
                        image_width=float(page.width or page_width_pt),
                        image_height=float(page.height or page_height_pt),
                    )
                    if not bbox.is_valid():
                        continue
                    items.append(
                        OCRTextItem(
                            item_id=f"p{page.page_number}_pdftext_{order}",
                            page_number=page.page_number,
                            text=" ".join(text.split()),
                            bbox=bbox,
                            # The text layer is ground truth, not a prediction.
                            confidence=1.0,
                            source="pdf_text_layer",
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "Text layer read failed on page %s (%s); falling back to OCR for that page",
                    page.page_number,
                    exc,
                )
                continue

            if items:
                by_page[page.page_number] = items
    finally:
        pdf.close()
    return by_page


class OCRService:
    def __init__(self, settings: Settings | None = None) -> None:
        # Only needed to pick the OCR models and to label their output; a scan
        # is the sole case where any of it runs.
        self.settings = settings

    def extract(
        self,
        pages: list[PageContext],
        *,
        pdf_path: Path | None = None,
        on_page_done: Callable[[], None] | None = None,
    ) -> tuple[list[OCRPageResult], list[ProcessingIssue]]:
        # Prefer the PDF's own text layer where it exists. OCR reconstructs text
        # from pixels and loses word spacing on tightly-set type; the text layer
        # is what the document actually says. OCR stays the path for scans.
        text_layer = _extract_pdf_text_layer(pdf_path, pages) if pdf_path is not None else {}
        if text_layer:
            logger.info(
                "Using embedded PDF text layer for %s of %s page(s)",
                len(text_layer),
                len(pages),
            )

        # Loaded lazily so a fully born-digital PDF never pays to start Paddle.
        ocr = None
        results: list[OCRPageResult] = []
        issues: list[ProcessingIssue] = []
        for page in pages:
            layer_items = text_layer.get(page.page_number)
            if layer_items:
                results.append(
                    OCRPageResult(
                        page_number=page.page_number,
                        width=page.width,
                        height=page.height,
                        items=layer_items,
                        text_source="pdf_text_layer",
                        page_image_path=str(page.page_image_path),
                    )
                )
                try:
                    if on_page_done is not None:
                        on_page_done()
                except Exception:
                    pass
                continue

            if ocr is None:
                logger.info("Running PaddleOCR text extraction (no usable text layer)")
                ocr = _get_paddle_ocr(self.settings)
            try:
                payload = ocr.predict(str(page.page_image_path))[0].json["res"]
            except Exception as exc:
                raise RuntimeError(
                    f"PaddleOCR text extraction failed on page {page.page_number}: {type(exc).__name__}: {exc}"
                ) from exc

            items: list[OCRTextItem] = []
            rec_texts = payload.get("rec_texts") or []
            rec_scores = payload.get("rec_scores") or []
            rec_boxes = payload.get("rec_boxes") or []
            dt_polys = payload.get("dt_polys") or []
            for index, text in enumerate(rec_texts, start=1):
                cleaned = str(text).strip()
                if not cleaned:
                    continue
                bbox = _bbox_from_ocr_payload(rec_boxes, dt_polys, index - 1)
                if bbox is None or not bbox.is_valid():
                    continue
                score = rec_scores[index - 1] if index - 1 < len(rec_scores) else None
                items.append(
                    OCRTextItem(
                        item_id=f"p{page.page_number}_ocr_{index}",
                        page_number=page.page_number,
                        text=cleaned,
                        bbox=bbox,
                        confidence=float(score) if score is not None else None,
                        source="paddleocr",
                    )
                )

            if not items:
                issues.append(
                    ProcessingIssue(
                        code="ocr_no_text",
                        message="PaddleOCR did not return any text for this page.",
                        level="warning",
                        page_number=page.page_number,
                    )
                )

            results.append(
                OCRPageResult(
                    page_number=page.page_number,
                    width=page.width,
                    height=page.height,
                    items=items,
                    text_source=(ocr_text_source_label(self.settings) if self.settings else "paddleocr"),
                    page_image_path=str(page.page_image_path),
                )
            )
            try:
                if on_page_done is not None:
                    on_page_done()
            except Exception:
                pass
        return results, issues


class ReadingOrderService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings
        self._layout_model: Any = None  # loaded lazily

    def resolve(self, pages: list[OCRPageResult]) -> tuple[dict[str, Any], list[ProcessingIssue]]:
        logger.info("Resolving reading order for %s OCR page(s)", len(pages))
        use_lr = self._settings is not None and self._settings.use_layout_reader
        if use_lr:
            return self._resolve_layout_reader(pages)
        return self._resolve_bbox_sort(pages)

    # ── geometric fallback ────────────────────────────────────────────────────

    def _resolve_bbox_sort(self, pages: list[OCRPageResult]) -> tuple[dict[str, Any], list[ProcessingIssue]]:
        ordered_text: list[dict[str, Any]] = []
        all_ids: list[str] = []
        resolvers: set[str] = set()
        for page in pages:
            ordered_items = _order_page_items(page)
            resolvers.add(_page_order_source(page))
            for index, item in enumerate(ordered_items, start=1):
                item.reading_order = index
            ids = [item.item_id for item in ordered_items]
            all_ids.extend(ids)
            ordered_text.append({"page_number": page.page_number, "ordered_item_ids": ids})
        return {
            "resolver": _resolver_label(resolvers),
            "document_order_item_ids": all_ids,
            "pages": ordered_text,
        }, []

    # ── LayoutReader (LayoutLMv3) ─────────────────────────────────────────────

    def _load_layout_model(self) -> Any:
        if self._layout_model is None:
            try:
                from transformers import LayoutLMv3ForTokenClassification

                model_id = self._settings.layout_reader_model if self._settings else "hantian/layoutreader"
                logger.info("Loading LayoutReader model %s", model_id)
                self._layout_model = LayoutLMv3ForTokenClassification.from_pretrained(model_id)
                self._layout_model.eval()
                logger.info("LayoutReader model loaded")
            except Exception as exc:
                logger.warning("Failed to load LayoutReader model, falling back to bbox sort: %s", exc)
        return self._layout_model

    def _resolve_layout_reader(self, pages: list[OCRPageResult]) -> tuple[dict[str, Any], list[ProcessingIssue]]:
        model = self._load_layout_model()
        if model is None:
            logger.warning("LayoutReader unavailable — falling back to bbox sort")
            return self._resolve_bbox_sort(pages)

        ordered_text: list[dict[str, Any]] = []
        all_ids: list[str] = []
        issues: list[ProcessingIssue] = []

        for page in pages:
            items = page.items
            if not items:
                ordered_text.append({"page_number": page.page_number, "ordered_item_ids": []})
                continue

            if _page_is_already_ordered(page):
                ordered_items = list(items)
                for index, item in enumerate(ordered_items, start=1):
                    item.reading_order = index
                ids = [item.item_id for item in ordered_items]
                all_ids.extend(ids)
                ordered_text.append({"page_number": page.page_number, "ordered_item_ids": ids})
                continue

            try:
                order_positions = _layout_reader_order(items, model)
                ordered_items = [items[i] for i in order_positions]
            except Exception as exc:
                logger.warning(
                    "LayoutReader failed on page %s, falling back to bbox sort: %s",
                    page.page_number,
                    exc,
                )
                issues.append(
                    ProcessingIssue(
                        code="layout_reader_fallback",
                        message=f"LayoutReader failed on page {page.page_number}: {exc}",
                        level="warning",
                        page_number=page.page_number,
                    )
                )
                ordered_items = sorted(items, key=_reading_order_key)

            for index, item in enumerate(ordered_items, start=1):
                item.reading_order = index
            ids = [item.item_id for item in ordered_items]
            all_ids.extend(ids)
            ordered_text.append({"page_number": page.page_number, "ordered_item_ids": ids})

        return {
            "resolver": "layout_reader_v3",
            "document_order_item_ids": all_ids,
            "pages": ordered_text,
        }, issues


class LayoutDetectionService:
    def detect(
        self, pages: list[PageContext], ocr_pages: list[OCRPageResult]
    ) -> tuple[list[LayoutRegion], list[ProcessingIssue], str]:
        del ocr_pages
        logger.info("Running Paddle layout detection on %s page(s)", len(pages))
        layout_detector = _get_paddle_layout_detector()
        regions: list[LayoutRegion] = []
        issues: list[ProcessingIssue] = []
        type_counts = {"text_block": 0, "table": 0, "figure": 0}

        for page in pages:
            try:
                payload = layout_detector.predict(str(page.page_image_path))[0].json["res"]
            except Exception as exc:
                raise RuntimeError(
                    "Paddle layout detection failed. Make sure paddlepaddle, paddleocr, and paddlex[ocr] are installed."
                ) from exc

            page_regions: list[LayoutRegion] = []
            skipped_labels: dict[str, int] = {}
            # Numbered within the page and prefixed with it, the way item and
            # block IDs already are. A counter spanning the whole call restarts
            # on every batch — the pipeline calls detect() once per batch — so
            # the same name gets minted several times per document, and the
            # duplicates then overwrite each other in the region lookups and in
            # the crop filenames.
            next_index = 1
            for box in payload.get("boxes") or []:
                label = str(box.get("label") or "").strip().lower()
                region_type = _region_type_for_label(label)
                if region_type is None:
                    skipped_labels[label or "unknown"] = skipped_labels.get(label or "unknown", 0) + 1
                    continue
                bbox = _bbox_from_layout_box(box.get("coordinate"))
                if bbox is None or not bbox.is_valid():
                    continue
                region = LayoutRegion(
                    region_id=f"p{page.page_number}_region_{next_index}",
                    region_type=region_type,
                    page_number=page.page_number,
                    bbox=bbox,
                    confidence=float(box.get("score")) if box.get("score") is not None else None,
                    source="paddle_layout_detection",
                    metadata={
                        "detector": "PP-DocLayout_plus-L",
                        "label": label,
                    },
                )
                page_regions.append(region)
                type_counts[region_type] += 1
                next_index += 1

            logger.info(
                "Page %s layout regions: text_blocks=%s tables=%s figures=%s skipped=%s",
                page.page_number,
                sum(1 for region in page_regions if region.region_type == "text_block"),
                sum(1 for region in page_regions if region.region_type == "table"),
                sum(1 for region in page_regions if region.region_type == "figure"),
                skipped_labels,
            )
            regions.extend(page_regions)

        if not regions:
            issues.append(
                ProcessingIssue(
                    code="layout_no_regions",
                    message="Paddle layout detection did not return any supported regions.",
                    level="warning",
                )
            )

        logger.info(
            "Paddle layout detection created %s text block(s), %s table(s), and %s figure(s)",
            type_counts["text_block"],
            type_counts["table"],
            type_counts["figure"],
        )
        return _dedupe_regions(regions), issues, "PP-DocLayout_plus-L"


class BlockLayoutService:
    """
    Paragraph-level block boxes from PP-DocBlockLayout.

    PP-DocLayout_plus-L answers "what kind of region is this" (table, header,
    body text). PP-DocBlockLayout answers "where does one block of running text
    begin and end", which is what block grouping actually needs. The heuristic
    it replaces bucketed y0 into 20px bands, which on the text-layer path put
    every line in its own block (10,587 blocks from 10,975 items across a
    12-document corpus) — so each line became its own paragraph and chunk text
    ended up as lines joined by blank lines.
    """

    def detect(self, pages: list[PageContext]) -> tuple[dict[int, list[BoundingBox]], list[ProcessingIssue], str]:
        logger.info("Running PP-DocBlockLayout block detection on %s page(s)", len(pages))
        detector = _get_paddle_block_layout_detector()
        by_page: dict[int, list[BoundingBox]] = {}
        issues: list[ProcessingIssue] = []

        for page in pages:
            try:
                payload = detector.predict(str(page.page_image_path))[0].json["res"]
            except Exception as exc:
                # A page without block boxes falls back to the line heuristic
                # rather than failing the document.
                issues.append(
                    ProcessingIssue(
                        code="block_layout_failed",
                        message=f"PP-DocBlockLayout failed on page {page.page_number}: {exc}",
                        level="warning",
                        page_number=page.page_number,
                    )
                )
                logger.warning("Block layout failed on page %s: %s", page.page_number, exc)
                continue

            boxes: list[BoundingBox] = []
            for box in payload.get("boxes") or []:
                bbox = _bbox_from_layout_box(box.get("coordinate"))
                if bbox is not None and bbox.is_valid():
                    boxes.append(bbox)
            boxes.sort(key=lambda b: (b.y0, b.x0))
            by_page[page.page_number] = boxes
            logger.info("Page %s block regions: %s", page.page_number, len(boxes))

        return by_page, issues, "PP-DocBlockLayout"


class AssociationService:
    def associate(
        self,
        ocr_pages: list[OCRPageResult],
        reading_order: dict[str, Any],
        regions: list[LayoutRegion],
        *,
        start_index: int = 1,
        block_boxes: dict[int, list[BoundingBox]] | None = None,
    ) -> tuple[list[RegionAssociation], list[OrderedTextBlock], dict[str, Any]]:
        item_lookup = {item.item_id: item for page in ocr_pages for item in page.items}
        regions_by_page: dict[int, list[LayoutRegion]] = {}
        text_regions_by_page: dict[int, list[LayoutRegion]] = {}
        for region in regions:
            regions_by_page.setdefault(region.page_number, []).append(region)
            if region.region_type == "text_block":
                text_regions_by_page.setdefault(region.page_number, []).append(region)

        associations: list[RegionAssociation] = []
        ordered_blocks: list[OrderedTextBlock] = []
        page_payloads: list[dict[str, Any]] = []
        global_index = start_index

        for page_entry in reading_order.get("pages", []):
            page_number = int(page_entry["page_number"])
            ordered_items = [
                item_lookup[item_id] for item_id in page_entry.get("ordered_item_ids", []) if item_id in item_lookup
            ]
            page_regions = regions_by_page.get(page_number, [])
            page_text_regions = text_regions_by_page.get(page_number, [])
            page_block_boxes = (block_boxes or {}).get(page_number) or []
            page_blocks: list[OrderedTextBlock] = []
            current_items: list[OCRTextItem] = []
            current_key: tuple[Any, Any] | None = None

            for item in ordered_items:
                matched_region, overlap_ratio = _best_region_match(item, page_regions)
                item.region_id = matched_region.region_id if matched_region else None
                associations.append(
                    RegionAssociation(
                        association_id=f"assoc_{len(associations) + 1}",
                        page_number=page_number,
                        item_id=item.item_id,
                        region_id=item.region_id,
                        region_type=matched_region.region_type if matched_region else None,
                        overlap_ratio=round(overlap_ratio, 4),
                    )
                )

                key = _block_grouping_key(item, page_block_boxes)
                if current_items and key != current_key:
                    global_index = _flush_block(page_number, current_items, global_index, page_blocks, ordered_blocks)
                    current_items = []

                current_items.append(item)
                current_key = key

            if current_items:
                global_index = _flush_block(page_number, current_items, global_index, page_blocks, ordered_blocks)

            if not page_blocks and ordered_items:
                logger.warning(
                    "Page %s had OCR text but no Paddle text blocks; falling back to OCR line grouping", page_number
                )
                page_blocks = _build_fallback_blocks(page_number, ordered_items, global_index)
                ordered_blocks.extend(page_blocks)
                global_index += len(page_blocks)
                for block in page_blocks:
                    for item_id in block.item_ids:
                        item_lookup[item_id].block_id = block.block_id
            else:
                association_lookup = {
                    assoc.item_id: assoc for assoc in associations if assoc.page_number == page_number
                }
                for block in page_blocks:
                    for item_id in block.item_ids:
                        association_lookup[item_id].block_id = block.block_id

            page_payloads.append(
                {
                    "page_number": page_number,
                    "blocks": [block.model_dump(mode="json") for block in page_blocks],
                    "text": "\n".join(block.text for block in page_blocks if block.text.strip()).strip(),
                    "text_region_count": len(page_text_regions),
                }
            )

        return (
            associations,
            ordered_blocks,
            {
                "pages": page_payloads,
                "full_text": "\n\n".join(page["text"] for page in page_payloads if page["text"]).strip(),
            },
        )


class CroppingService:
    def crop_visual_regions(
        self,
        *,
        pages: list[PageContext],
        regions: list[LayoutRegion],
        output_dir: Path,
    ) -> tuple[list[CroppedRegionAsset], list[ProcessingIssue]]:
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            return [], [
                ProcessingIssue(
                    code="crop_unavailable",
                    message="Pillow is required for region cropping.",
                    level="warning",
                    details={"error": str(exc)},
                )
            ]

        page_lookup = {page.page_number: page for page in pages}
        for folder in ("tables", "figures"):
            (output_dir / folder).mkdir(parents=True, exist_ok=True)

        assets: list[CroppedRegionAsset] = []
        issues: list[ProcessingIssue] = []
        for region in regions:
            if region.region_type not in {"table", "figure"}:
                continue
            page = page_lookup.get(region.page_number)
            if page is None or not page.page_image_path.exists():
                issues.append(
                    ProcessingIssue(
                        code="missing_page_image",
                        message="Skipping crop because the rendered page image is missing.",
                        level="warning",
                        page_number=region.page_number,
                        details={"region_id": region.region_id},
                    )
                )
                logger.info("Skipping crop for %s because page image is missing", region.region_id)
                continue

            folder = "tables" if region.region_type == "table" else "figures"
            crop_path = output_dir / folder / f"{region.region_id}.png"
            try:
                with Image.open(page.page_image_path) as image:
                    crop_box = _compute_crop_box(region, image.width, image.height)
                    if crop_box is None:
                        issues.append(
                            ProcessingIssue(
                                code="invalid_crop_bounds",
                                message="Skipping crop because the padded crop bounds are invalid.",
                                level="warning",
                                page_number=region.page_number,
                                details={"region_id": region.region_id},
                            )
                        )
                        logger.info("Skipping crop for %s because crop bounds were invalid", region.region_id)
                        continue
                    image.crop(crop_box).save(crop_path)
            except Exception as exc:
                issues.append(
                    ProcessingIssue(
                        code="crop_open_failed",
                        message="Skipping crop because the page image could not be opened.",
                        level="warning",
                        page_number=region.page_number,
                        details={"region_id": region.region_id, "error": str(exc)},
                    )
                )
                logger.info("Skipping crop for %s because saving failed: %s", region.region_id, exc)
                continue

            region.crop_path = str(crop_path)
            assets.append(
                CroppedRegionAsset(
                    asset_id=f"asset_{region.region_id}",
                    region_id=region.region_id,
                    page_number=region.page_number,
                    region_type=region.region_type,
                    crop_path=str(crop_path),
                    bbox=region.bbox,
                )
            )
            logger.info("Saved %s crop for %s to %s", region.region_type, region.region_id, crop_path)
        return assets, issues


class TableStructureService:
    """
    Cell-level table structure for cropped table regions.

    Layout detection gives a table's bounding box and nothing more, so a table
    flattens into a sequence of row strings and the column a figure belongs to
    is lost — a balance sheet becomes "Cash and cash equivalents $ 24,296
    $ 30,242" with no record of which number is which period. Structure
    recognition recovers the grid as HTML.

    It re-crops tight to the region bbox rather than reusing the files
    CroppingService writes. Those are padded for the vision model — 8% of height
    per side on a table, which on a full-page balance sheet is ~46pt, close to
    four lines of surrounding body text, and 21% of the saved crop's area. With
    layout detection disabled the recogniser treats whatever it is given as one
    table, so that padding would be read as extra rows. Padding helps a VLM see
    context; it corrupts a cell grid.

    Cost stays bounded to the table regions in a document (~50 for a 10-Q)
    rather than every page.
    """

    def recognize(
        self,
        *,
        pages: list[PageContext],
        regions: list[LayoutRegion],
        assets: list[CroppedRegionAsset] | None = None,
    ) -> tuple[list[TableStructure], list[ProcessingIssue]]:
        table_regions = [region for region in regions if region.region_type == "table"]
        if not table_regions:
            return [], []

        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            return [], [
                ProcessingIssue(
                    code="table_structure_unavailable",
                    message="Pillow is required to crop tables for structure recognition.",
                    level="warning",
                    details={"error": str(exc)},
                )
            ]

        logger.info("Running table structure recognition on %s table region(s)", len(table_regions))
        recognizer = _get_paddle_table_recognizer()
        page_lookup = {page.page_number: page for page in pages}
        asset_by_region = {asset.region_id: asset for asset in (assets or [])}
        structures: list[TableStructure] = []
        issues: list[ProcessingIssue] = []

        for region in table_regions:
            asset = asset_by_region.get(region.region_id)
            page = page_lookup.get(region.page_number)
            if page is None or not page.page_image_path.exists():
                issues.append(
                    ProcessingIssue(
                        code="table_page_image_missing",
                        message="Skipping table structure because the rendered page image is missing.",
                        level="warning",
                        page_number=region.page_number,
                        details={"region_id": region.region_id},
                    )
                )
                continue
            try:
                with Image.open(page.page_image_path) as image:
                    box = (
                        max(0, int(region.bbox.x0)),
                        max(0, int(region.bbox.y0)),
                        min(image.width, int(region.bbox.x1)),
                        min(image.height, int(region.bbox.y1)),
                    )
                    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
                        continue
                    payload = recognizer.predict(_to_predict_input(image.crop(box)))[0].json["res"]
            except Exception as exc:
                # One unreadable table must not fail the document.
                issues.append(
                    ProcessingIssue(
                        code="table_structure_failed",
                        message=f"Table structure recognition failed for {region.region_id}: {exc}",
                        level="warning",
                        page_number=region.page_number,
                        details={"region_id": region.region_id},
                    )
                )
                logger.warning("Table structure failed for %s: %s", region.region_id, exc)
                continue

            entries = payload.get("table_res_list") or []
            if not entries:
                issues.append(
                    ProcessingIssue(
                        code="table_structure_empty",
                        message=f"No table structure returned for {region.region_id}.",
                        level="warning",
                        page_number=region.page_number,
                        details={"region_id": region.region_id},
                    )
                )
                continue

            # The crop is one table by construction, so take the first result and
            # record the count when the model disagrees.
            entry = entries[0]
            html = str(entry.get("pred_html") or "").strip()
            if not html:
                continue
            cells = entry.get("cell_box_list") or []
            structures.append(
                TableStructure(
                    table_id=f"table_{region.region_id}",
                    region_id=region.region_id,
                    asset_id=asset.asset_id if asset else None,
                    page_number=region.page_number,
                    # The padded crop is what a reader/VLM should look at; the
                    # structure below was read from a tight re-crop.
                    crop_path=asset.crop_path if asset else str(page.page_image_path),
                    html=html,
                    cell_count=len(cells),
                    metadata={
                        "tables_detected_in_crop": len(entries),
                        "cropped_tight_to_region": True,
                    },
                )
            )
            logger.info("Table structure for %s: %s cell(s)", region.region_id, len(cells))

        return structures, issues


def build_chunks(
    *,
    document_id: str,
    source_file: str,
    ordered_blocks: list[OrderedTextBlock],
    regions: list[LayoutRegion],
    target_chars: int = 1800,
    overlap_chars: int = 200,
) -> list[ProcessedChunk]:
    blocks_by_page: dict[int, list[OrderedTextBlock]] = {}
    regions_by_id = {region.region_id: region for region in regions}
    for block in ordered_blocks:
        blocks_by_page.setdefault(block.page_number, []).append(block)

    chunks: list[ProcessedChunk] = []
    next_index = 1
    for page_number, blocks in sorted(blocks_by_page.items()):
        current: list[OrderedTextBlock] = []
        for block in blocks:
            if current and len("\n\n".join(item.text for item in current + [block])) > target_chars:
                chunks.append(_build_chunk(document_id, source_file, page_number, next_index, current, regions_by_id))
                next_index += 1
                current = _overlap_blocks(current, overlap_chars)
            current.append(block)
        if current:
            chunks.append(_build_chunk(document_id, source_file, page_number, next_index, current, regions_by_id))
            next_index += 1
    return chunks


def build_document_artifacts(
    *,
    loaded: LoadedDocument,
    ocr_pages: list[OCRPageResult],
    ordered_text: dict[str, Any],
    regions: list[LayoutRegion],
    cropped_assets: list[CroppedRegionAsset],
    chunks: list[ProcessedChunk],
    reading_order_model: str,
    layout_detection_model: str,
    issues: list[ProcessingIssue],
    block_layout_model: str | None = None,
) -> tuple[ProcessedDocument, ProcessingMetadata]:
    warnings = [issue for issue in issues if issue.level == "warning"]
    errors = [issue for issue in issues if issue.level == "error"]
    document = ProcessedDocument(
        document_id=loaded.document_id,
        source_filename=loaded.original_copy_path.name,
        source_path=str(loaded.original_copy_path),
        page_count=len(loaded.pages),
        full_ordered_text=str(ordered_text.get("full_text", "")),
        region_summaries=[
            {
                "region_id": region.region_id,
                "region_type": region.region_type,
                "page_number": region.page_number,
                "bbox": region.bbox.as_list(),
                "crop_path": region.crop_path,
                "detector": region.metadata.get("detector"),
                "label": region.metadata.get("label"),
                "confidence": region.confidence,
            }
            for region in regions
        ],
        cropped_assets=[asset.model_dump(mode="json") for asset in cropped_assets],
        crop_references=[asset.crop_path for asset in cropped_assets],
        processing_summary={
            "page_count": len(loaded.pages),
            "region_count": len(regions),
            "cropped_asset_count": len(cropped_assets),
            "chunk_count": len(chunks),
        },
        agent_input={},
        agent_output={},
    )
    metadata = ProcessingMetadata(
        processing_timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        schema_version="6.0.0",
        ocr_engine="PaddleOCR",
        reading_order_model=reading_order_model,
        layout_detection_model=layout_detection_model,
        block_layout_model=block_layout_model,
        agent_model=None,
        confidence_summary=_confidence_summary(ocr_pages=ocr_pages, regions=regions, chunks=chunks),
        warnings=warnings,
        errors=errors,
    )
    return document, metadata


def build_visual_summaries(
    *,
    regions: list[LayoutRegion],
    ordered_blocks: list[OrderedTextBlock],
    chunks: list[ProcessedChunk],
    cropped_assets: list[CroppedRegionAsset],
    table_structures: list[TableStructure] | None = None,
) -> list[VisualRegionSummary]:
    asset_by_region = {asset.region_id: asset for asset in cropped_assets}
    structure_by_region = {structure.region_id: structure for structure in (table_structures or [])}
    chunks_by_region: dict[str, list[ProcessedChunk]] = {}
    for chunk in chunks:
        for region_id in chunk.source_region_ids:
            chunks_by_region.setdefault(region_id, []).append(chunk)

    summaries: list[VisualRegionSummary] = []
    for region in regions:
        if region.region_type not in {"table", "figure"}:
            continue
        page_blocks = [
            block for block in ordered_blocks if block.page_number == region.page_number and block.bbox is not None
        ]
        overlapping_blocks = [
            block for block in page_blocks if block.bbox is not None and block.bbox.intersection_area(region.bbox) > 0
        ]
        if not overlapping_blocks:
            overlapping_blocks = sorted(
                page_blocks,
                key=lambda block: min(
                    abs(block.bbox.y0 - region.bbox.y1),  # type: ignore[union-attr]
                    abs(block.bbox.y1 - region.bbox.y0),  # type: ignore[union-attr]
                ),
            )[:3]
        region_chunks = chunks_by_region.get(region.region_id, [])
        block_text = " ".join(block.text for block in overlapping_blocks if block.text.strip()).strip()
        chunk_text = " ".join(chunk.text for chunk in region_chunks if chunk.text.strip()).strip()
        # Recovered cell structure beats the flattened row strings: it is the
        # only form that records which column a figure belongs to. Markup needs
        # a wider cap than prose — a balance sheet's HTML runs past 1200 chars
        # and truncating mid-tag would leave it unparseable.
        structure = structure_by_region.get(region.region_id)
        if structure and structure.html:
            summary_text = structure.html[:4000]
        else:
            summary_text = (
                block_text or chunk_text or f"Detected {region.region_type} region on page {region.page_number}."
            )[:1200]
        asset = asset_by_region.get(region.region_id)
        summaries.append(
            VisualRegionSummary(
                summary_id=f"summary_{region.region_id}",
                region_id=region.region_id,
                asset_id=asset.asset_id if asset else None,
                page_number=region.page_number,
                region_type=region.region_type,
                crop_path=asset.crop_path if asset else region.crop_path,
                linked_block_ids=[block.block_id for block in overlapping_blocks],
                linked_chunk_ids=[chunk.chunk_id for chunk in region_chunks],
                summary_text=summary_text,
                metadata={
                    "label": region.metadata.get("label"),
                    "detector": region.metadata.get("detector"),
                    **(
                        {
                            "table_html": structure.html,
                            "cell_count": structure.cell_count,
                            "structure_source": structure.source,
                        }
                        if structure
                        else {}
                    ),
                },
            )
        )
    return summaries


def export_artifacts(
    *,
    working_dir: Path,
    loaded: LoadedDocument,
    raw_ocr: list[OCRPageResult],
    reading_order: dict[str, Any],
    ordered_text: dict[str, Any],
    regions: list[LayoutRegion],
    region_associations: list[RegionAssociation],
    cropped_assets: list[CroppedRegionAsset],
    visual_summaries: list[VisualRegionSummary],
    chunks: list[ProcessedChunk],
    table_structures: list[TableStructure] | None = None,
    document: ProcessedDocument,
    metadata: ProcessingMetadata,
    descriptor: dict[str, Any] | None = None,
    summary_embedding: list[float] | None = None,
) -> Path:
    crops_dir = working_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest = ProcessedManifest(
        schema_version=metadata.schema_version,
        pipeline_stage="preprocessing",
        processing_status="completed",
        document_id=loaded.document_id,
        source_filename=loaded.original_copy_path.name,
        source_path=str(loaded.original_copy_path),
        working_dir=str(working_dir),
        page_count=len(loaded.pages),
        chunk_count=len(chunks),
        processing_timestamp=metadata.processing_timestamp,
        artifacts={
            "document": "document.json",
            "ocr": "ocr.json",
            "layout": "layout.json",
            "reading_order": "reading_order.json",
            "cropped_assets": "cropped_assets.json",
            "visual_summaries": "visual_summaries.json",
            "table_structures": "table_structures.json",
            "chunks": "chunks.json",
            "metadata": "metadata.json",
        },
    )
    _write_json(working_dir / "manifest.json", manifest.model_dump(mode="json"))
    _write_json(working_dir / "ocr.json", [page.model_dump(mode="json") for page in raw_ocr])
    _write_json(
        working_dir / "reading_order.json",
        {
            "reading_order": reading_order,
            "ordered_text": ordered_text,
        },
    )
    _write_json(
        working_dir / "layout.json",
        {
            "regions": [region.model_dump(mode="json") for region in regions],
            "associations": [assoc.model_dump(mode="json") for assoc in region_associations],
        },
    )
    _write_json(working_dir / "cropped_assets.json", [asset.model_dump(mode="json") for asset in cropped_assets])
    _write_json(
        working_dir / "visual_summaries.json", [summary.model_dump(mode="json") for summary in visual_summaries]
    )
    _write_json(
        working_dir / "table_structures.json",
        [structure.model_dump(mode="json") for structure in (table_structures or [])],
    )
    doc_payload = document.model_dump(mode="json")
    if descriptor:
        doc_payload["descriptor"] = descriptor
    if summary_embedding:
        doc_payload["summary_embedding"] = summary_embedding
    _write_json(working_dir / "document.json", doc_payload)
    _write_json(working_dir / "chunks.json", [chunk.model_dump(mode="json") for chunk in chunks])
    _write_json(working_dir / "metadata.json", metadata.model_dump(mode="json"))
    return working_dir / "document.json"


def _load_pdf_pages(path: Path, pages_dir: Path, *, render_scale: float) -> list[PageContext]:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF rendering requires pypdfium2.") from exc

    pdf = pdfium.PdfDocument(str(path))
    pages: list[PageContext] = []
    try:
        for page_index in range(len(pdf)):
            page_number = page_index + 1
            pdfium_page = pdf[page_index]
            bitmap = pdfium_page.render(scale=render_scale)
            image = bitmap.to_pil()
            image_path = pages_dir / f"page_{page_number}.png"
            image.save(image_path)
            width, height = image.size
            pages.append(
                PageContext(
                    page_number=page_number,
                    width=float(width),
                    height=float(height),
                    page_image_path=image_path,
                )
            )
    finally:
        pdf.close()
    return pages


def _load_image_page(path: Path, *, page_number: int) -> PageContext:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Image input requires Pillow.") from exc

    with Image.open(path) as image:
        width, height = image.size
    return PageContext(page_number=page_number, width=float(width), height=float(height), page_image_path=path)


def ocr_text_source_label(settings: Settings) -> str:
    """
    Provenance label recording which OCR models produced a page's text.

    Derived from the configured model names rather than hardcoded: the previous
    literal claimed "ppocrv5_mobile" while PP-OCRv4_mobile actually ran, so every
    stored artifact misreported its own model version.
    """

    def _version(name: str) -> str:
        match = re.search(r"v(\d+)[_-]?(\w+?)(?:_(?:det|rec))?$", name)
        return f"v{match.group(1)}_{match.group(2)}" if match else name.lower()

    detection = _version(settings.ocr_detection_model)
    recognition = _version(settings.ocr_recognition_model)
    if detection == recognition:
        return f"paddleocr_{detection}"
    return f"paddleocr_det-{detection}_rec-{recognition}"


def _get_paddle_ocr(settings: Settings | None = None) -> Any:
    """
    Build (or reuse) the PaddleOCR predictor for the configured models.

    The cache is keyed on the model names rather than on Settings: Settings is a
    pydantic model and is not hashable, so caching on it raised TypeError on
    every call and no page ever reached the predictor. Keying on the names is
    also more accurate — changing a model should yield a new predictor.
    """
    return _build_paddle_ocr(
        settings.ocr_detection_model if settings else "PP-OCRv6_medium_det",
        settings.ocr_recognition_model if settings else "PP-OCRv6_medium_rec",
    )


@lru_cache(maxsize=1)
def _build_paddle_ocr(detection_model: str, recognition_model: str) -> Any:
    from paddleocr import PaddleOCR

    # enable_mkldnn=False: workaround for PaddlePaddle 3.3.x regression (issue #77340).
    # PaddleX defaults to run_mode="mkldnn" on CPU, which crashes with NotImplementedError
    # in the PIR executor on x86. This kwarg flows through PaddleX's parse_common_args()
    # and switches the predictor to run_mode="paddle", bypassing the broken code path.
    # See: https://github.com/PaddlePaddle/Paddle/issues/77340
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name=detection_model,
        text_recognition_model_name=recognition_model,
        enable_mkldnn=False,
        # Pin the detection input size. Without this, unusual page dimensions
        # after PaddleOCR's internal resize can trigger a crash in PaddlePaddle's
        # DepthwiseConvKernel (Im2ColFunctor) on certain input shapes.
        text_det_limit_side_len=960,
        text_det_limit_type="max",
    )


@lru_cache(maxsize=1)
def _get_paddle_layout_detector() -> Any:
    from paddleocr import LayoutDetection

    # Same enable_mkldnn=False workaround as _get_paddle_ocr above.
    return LayoutDetection(enable_mkldnn=False)


@lru_cache(maxsize=1)
def _get_paddle_table_recognizer() -> Any:
    from paddleocr import TableRecognitionPipelineV2

    # The crop is already a single table region, so layout detection inside the
    # pipeline is redundant work; orientation and unwarping are for photographed
    # pages and never apply to a crop of our own render.
    return TableRecognitionPipelineV2(
        use_layout_detection=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )


@lru_cache(maxsize=1)
def _get_paddle_block_layout_detector() -> Any:
    from paddleocr import LayoutDetection

    # Same LayoutDetection wrapper, different weights: PP-DocBlockLayout emits a
    # single "Region" class marking block extents, where the default
    # PP-DocLayout_plus-L emits typed regions (table, header, text).
    return LayoutDetection(model_name="PP-DocBlockLayout", enable_mkldnn=False)


def _bbox_from_ocr_payload(rec_boxes: list[Any], dt_polys: list[Any], index: int) -> BoundingBox | None:
    if index < len(rec_boxes):
        value = rec_boxes[index]
        if isinstance(value, list) and len(value) == 4:
            return BoundingBox.from_list([float(item) for item in value])
    if index < len(dt_polys):
        points = dt_polys[index]
        if isinstance(points, list) and points:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            return BoundingBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
    return None


def _bbox_from_layout_box(value: Any) -> BoundingBox | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    return BoundingBox.from_list([float(item) for item in value])


def _layout_reader_order(items: list[OCRTextItem], model: Any) -> list[int]:
    """Return item indices sorted by LayoutReader predicted reading order.

    Normalises bounding boxes to the 0-1000 range LayoutLMv3 expects, runs the
    model, then maps the per-item order positions back to the original indices.
    Handles the MAX_LEN=510 cap by falling back to bbox sort for the overflow
    items and appending them at the end.
    """
    import torch

    MAX_LEN = 510
    CLS_TOKEN_ID = 0
    UNK_TOKEN_ID = 3
    EOS_TOKEN_ID = 2

    # Estimate page dimensions from the union of all bboxes (+ 10% padding).
    max_x = max(item.bbox.x1 for item in items) * 1.1 or 1.0
    max_y = max(item.bbox.y1 for item in items) * 1.1 or 1.0

    def _norm(item: OCRTextItem) -> list[int]:
        return [
            int((item.bbox.x0 / max_x) * 1000),
            int((item.bbox.y0 / max_y) * 1000),
            int((item.bbox.x1 / max_x) * 1000),
            int((item.bbox.y1 / max_y) * 1000),
        ]

    active = items[:MAX_LEN]
    boxes = [_norm(it) for it in active]
    n = len(boxes)

    # Build model inputs (CLS + boxes + EOS).
    bbox_tensor = torch.tensor([[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]])
    input_ids = torch.tensor([CLS_TOKEN_ID] + [UNK_TOKEN_ID] * n + [EOS_TOKEN_ID])
    attention_mask = torch.ones(n + 2, dtype=torch.long)

    with torch.no_grad():
        logits = model(
            bbox=bbox_tensor.unsqueeze(0),
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        ).logits.squeeze(0)

    # Parse logits → reading order position for each active item.
    order_positions = _parse_layout_logits(logits, n)

    # Map position → original index and sort.
    idx_by_position = sorted(range(n), key=lambda i: order_positions[i])

    # Overflow items appended in geometric order.
    overflow_indices = list(range(MAX_LEN, len(items)))
    overflow_indices.sort(key=lambda i: _reading_order_key(items[i]))

    return idx_by_position + overflow_indices


def _parse_layout_logits(logits: Any, length: int) -> list[int]:
    """Convert LayoutLMv3 logits to a reading-order position per item."""
    from collections import defaultdict

    logits = logits[1 : length + 1, :length]
    orders = logits.argsort(descending=False).tolist()
    ret = [o.pop() for o in orders]

    while True:
        order_to_idxes: dict[int, list[int]] = defaultdict(list)
        for idx, order in enumerate(ret):
            order_to_idxes[order].append(idx)
        order_to_idxes = {k: v for k, v in order_to_idxes.items() if len(v) > 1}
        if not order_to_idxes:
            break
        for order, idxes in order_to_idxes.items():
            idxes_by_logit = sorted(idxes, key=lambda i: logits[i, order], reverse=True)
            for idx in idxes_by_logit[1:]:
                ret[idx] = orders[idx].pop()

    return ret


def _best_block_index(item: OCRTextItem, block_boxes: list[BoundingBox]) -> int | None:
    """Index of the block box covering most of this item, or None if none does."""
    best_index = None
    best_ratio = 0.0
    item_area = item.bbox.area() or 1.0
    for index, box in enumerate(block_boxes):
        ratio = item.bbox.intersection_area(box) / item_area
        if ratio > best_ratio:
            best_index = index
            best_ratio = ratio
    return best_index


def _block_grouping_key(item: OCRTextItem, block_boxes: list[BoundingBox]) -> tuple[Any, Any]:
    """
    Key that consecutive items must share to land in the same text block.

    With PP-DocBlockLayout boxes available, the block box is the grouping unit,
    so a paragraph's lines merge into one block. Items covered by no box (page
    headers, stray marginalia) each get a unique key so they stay separate
    rather than silently merging into a neighbour.

    Without block boxes — no Paddle, or the model failed on this page — this
    falls back to the previous 20px line bucket, which yields one block per line.
    """
    if not block_boxes:
        return (item.region_id, int(item.bbox.y0 // 20))
    block_index = _best_block_index(item, block_boxes)
    if block_index is None:
        return (item.region_id, f"unblocked:{item.item_id}")
    return (item.region_id, block_index)


def _page_is_already_ordered(page: OCRPageResult) -> bool:
    """
    True when the page's items arrive in reading order and must not be re-sorted.

    Text-layer pages do: _group_rects_into_lines returns line boxes sorted
    top-to-bottom and each item is a whole line, so the sequence is already
    correct. Re-sorting them measurably *degrades* it — the 18px bucket in
    _reading_order_key collapses vertically-adjacent table rows into one band
    and the x tiebreak then swaps them, which on a 12-document corpus produced
    45 inversions of genuinely stacked lines against 2 correct merges.

    OCR pages are different: PaddleOCR emits sub-line fragments in detector
    order, roughly 3x as many items per page, and those do need sorting.
    """
    return page.text_source == "pdf_text_layer"


def _page_order_source(page: OCRPageResult) -> str:
    return "pdf_text_layer_order_v1" if _page_is_already_ordered(page) else "ocr_bbox_sort_v1"


def _resolver_label(resolvers: set[str]) -> str:
    if not resolvers:
        return "ocr_bbox_sort_v1"
    if len(resolvers) == 1:
        return next(iter(resolvers))
    return "mixed:" + "+".join(sorted(resolvers))


def _order_page_items(page: OCRPageResult) -> list[OCRTextItem]:
    if _page_is_already_ordered(page):
        return list(page.items)
    return sorted(page.items, key=_reading_order_key)


def _reading_order_key(item: OCRTextItem) -> tuple[int, float, float]:
    line_bucket = round(item.bbox.y0 / 18.0)
    return (line_bucket, item.bbox.x0, item.bbox.y0)


def _region_type_for_label(label: str) -> str | None:
    if label == "table" or "table" in label:
        return "table"
    if label in TEXT_BLOCK_LABELS or label.endswith("_text"):
        return "text_block"
    if label in FIGURE_LABELS:
        return "figure"
    return None


def _dedupe_regions(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    seen: set[tuple[int, str, tuple[float, float, float, float]]] = set()
    deduped: list[LayoutRegion] = []
    for region in regions:
        key = (region.page_number, region.region_type, tuple(round(value, 1) for value in region.bbox.as_list()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(region)
    return deduped


def _best_region_match(item: OCRTextItem, regions: list[LayoutRegion]) -> tuple[LayoutRegion | None, float]:
    best = None
    best_ratio = 0.0
    item_area = item.bbox.area() or 1.0
    for region in regions:
        overlap = item.bbox.intersection_area(region.bbox)
        if overlap <= 0:
            continue
        ratio = overlap / item_area
        if ratio > best_ratio:
            best = region
            best_ratio = ratio
    return best, best_ratio


def _flush_block(
    page_number: int,
    current_items: list[OCRTextItem],
    global_index: int,
    page_blocks: list[OrderedTextBlock],
    ordered_blocks: list[OrderedTextBlock],
) -> int:
    block = _build_block(page_number, current_items, global_index)
    for item in current_items:
        item.block_id = block.block_id
    page_blocks.append(block)
    ordered_blocks.append(block)
    return global_index + 1


def _build_fallback_blocks(
    page_number: int, ordered_items: list[OCRTextItem], start_index: int
) -> list[OrderedTextBlock]:
    blocks: list[OrderedTextBlock] = []
    current_items: list[OCRTextItem] = []
    current_line_bucket: int | None = None
    next_index = start_index
    for item in ordered_items:
        line_bucket = int(item.bbox.y0 // 20)
        if current_items and line_bucket != current_line_bucket:
            block = _build_block(page_number, current_items, next_index)
            for grouped_item in current_items:
                grouped_item.block_id = block.block_id
            blocks.append(block)
            next_index += 1
            current_items = []
        current_items.append(item)
        current_line_bucket = line_bucket
    if current_items:
        block = _build_block(page_number, current_items, next_index)
        for grouped_item in current_items:
            grouped_item.block_id = block.block_id
        blocks.append(block)
    return blocks


def _build_block(page_number: int, items: list[OCRTextItem], reading_order: int) -> OrderedTextBlock:
    return OrderedTextBlock(
        block_id=f"p{page_number}_block_{reading_order}",
        page_number=page_number,
        text=" ".join(item.text.strip() for item in items if item.text.strip()).strip(),
        item_ids=[item.item_id for item in items],
        region_ids=sorted({item.region_id for item in items if item.region_id}),
        bbox=BoundingBox.merge([item.bbox for item in items]),
        reading_order=reading_order,
    )


def _build_chunk(
    document_id: str,
    source_file: str,
    page_number: int,
    index: int,
    blocks: list[OrderedTextBlock],
    regions_by_id: dict[str, LayoutRegion],
) -> ProcessedChunk:
    chunk_id = f"{document_id}:chunk:{index}"
    text = "\n\n".join(block.text for block in blocks if block.text.strip()).strip()
    region_ids = sorted({region_id for block in blocks for region_id in block.region_ids})
    crop_refs = [
        regions_by_id[region_id].crop_path
        for region_id in region_ids
        if region_id in regions_by_id and regions_by_id[region_id].crop_path
    ]
    crop_asset_ids = [
        f"asset_{region_id}"
        for region_id in region_ids
        if region_id in regions_by_id and regions_by_id[region_id].crop_path
    ]
    region_types = sorted(
        {regions_by_id[region_id].region_type for region_id in region_ids if region_id in regions_by_id}
    )
    bbox_refs = [block.bbox.as_list() for block in blocks if block.bbox is not None]
    item_ids = [item_id for block in blocks for item_id in block.item_ids]
    ordered_block_ids = [block.block_id for block in blocks]
    metadata = {
        "document_id": document_id,
        "source_file": source_file,
        "page_number": page_number,
        "chunk_id": chunk_id,
        "ordered_block_ids": ordered_block_ids,
        "item_ids": item_ids,
        "region_ids": region_ids,
        "region_types": region_types,
        "bbox_references": bbox_refs,
        "crop_references": crop_refs,
        "crop_asset_ids": crop_asset_ids,
    }
    return ProcessedChunk(
        chunk_id=chunk_id,
        text=text,
        page_content=text,
        page_number=page_number,
        ordered_block_ids=ordered_block_ids,
        item_ids=item_ids,
        source_region_ids=region_ids,
        region_types=region_types,
        bbox_references=bbox_refs,
        crop_references=crop_refs,
        metadata=metadata,
    )


def _overlap_blocks(blocks: list[OrderedTextBlock], overlap_chars: int) -> list[OrderedTextBlock]:
    if overlap_chars <= 0:
        return []
    kept: list[OrderedTextBlock] = []
    total = 0
    for block in reversed(blocks):
        kept.insert(0, block)
        total += len(block.text)
        if total >= overlap_chars:
            break
    return kept


def _compute_crop_box(region: LayoutRegion, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    width = region.bbox.x1 - region.bbox.x0
    height = region.bbox.y1 - region.bbox.y0
    if width <= 1 or height <= 1:
        logger.info("Skipping crop for %s because bbox was empty", region.region_id)
        return None

    if region.region_type == "table":
        pad_x = max(28, int(width * 0.05))
        pad_y = max(28, int(height * 0.08))
    else:
        pad_x = max(36, int(width * 0.08))
        pad_y = max(36, int(height * 0.12))

    left = max(0, int(region.bbox.x0 - pad_x))
    top = max(0, int(region.bbox.y0 - pad_y))
    right = min(image_width, int(region.bbox.x1 + pad_x))
    bottom = min(image_height, int(region.bbox.y1 + pad_y))
    if right - left < 48 or bottom - top < 48:
        logger.info("Skipping crop for %s because padded crop was too small", region.region_id)
        return None
    return (left, top, right, bottom)


def _to_predict_input(image: Any) -> Any:
    """
    Hand a PIL crop to a PaddleX predictor without a disk round-trip.

    Predictors take a BGR ndarray; PIL is RGB, so the channels are reversed.
    """
    import numpy as np

    return np.asarray(image.convert("RGB"))[:, :, ::-1]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _confidence_summary(
    *, ocr_pages: list[OCRPageResult], regions: list[LayoutRegion], chunks: list[ProcessedChunk]
) -> dict[str, Any]:
    ocr_confidences = [item.confidence for page in ocr_pages for item in page.items if item.confidence is not None]
    region_confidences = [region.confidence for region in regions if region.confidence is not None]
    return {
        "ocr_item_count": len(ocr_confidences),
        "ocr_average_confidence": round(sum(ocr_confidences) / len(ocr_confidences), 4) if ocr_confidences else None,
        "region_count": len(regions),
        "region_average_confidence": round(sum(region_confidences) / len(region_confidences), 4)
        if region_confidences
        else None,
        "chunk_count": len(chunks),
    }
