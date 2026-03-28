from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import Settings
from document_Process.models import (
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


def _configure_paddle_env() -> None:
    cache_home = Path(".paddlex").resolve()
    cache_home.mkdir(parents=True, exist_ok=True)
    (cache_home / "temp").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_home))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


_configure_paddle_env()


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


class OCRService:
    def extract(self, pages: list[PageContext]) -> tuple[list[OCRPageResult], list[ProcessingIssue]]:
        logger.info("Running PaddleOCR text extraction on %s page(s)", len(pages))
        ocr = _get_paddle_ocr()
        results: list[OCRPageResult] = []
        issues: list[ProcessingIssue] = []
        for page in pages:
            try:
                payload = ocr.predict(str(page.page_image_path))[0].json["res"]
            except Exception as exc:
                raise RuntimeError(
                    "PaddleOCR text extraction failed. Make sure paddlepaddle, paddleocr, and paddlex[ocr] are installed."
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
                    text_source="paddleocr_ppocrv5_mobile",
                    page_image_path=str(page.page_image_path),
                )
            )
        return results, issues


class ReadingOrderService:
    def resolve(self, pages: list[OCRPageResult]) -> tuple[dict[str, Any], list[ProcessingIssue]]:
        logger.info("Resolving reading order for %s OCR page(s)", len(pages))
        ordered_text: list[dict[str, Any]] = []
        all_ids: list[str] = []
        for page in pages:
            ordered_items = sorted(page.items, key=_reading_order_key)
            for index, item in enumerate(ordered_items, start=1):
                item.reading_order = index
            ids = [item.item_id for item in ordered_items]
            all_ids.extend(ids)
            ordered_text.append({"page_number": page.page_number, "ordered_item_ids": ids})
        return {"resolver": "ocr_bbox_sort_v1", "document_order_item_ids": all_ids, "pages": ordered_text}, []


class LayoutDetectionService:
    def detect(self, pages: list[PageContext], ocr_pages: list[OCRPageResult]) -> tuple[list[LayoutRegion], list[ProcessingIssue], str]:
        del ocr_pages
        logger.info("Running Paddle layout detection on %s page(s)", len(pages))
        layout_detector = _get_paddle_layout_detector()
        regions: list[LayoutRegion] = []
        issues: list[ProcessingIssue] = []
        next_id = 1
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
                    region_id=f"region_{next_id}",
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
                next_id += 1

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


class AssociationService:
    def associate(
        self,
        ocr_pages: list[OCRPageResult],
        reading_order: dict[str, Any],
        regions: list[LayoutRegion],
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
        global_index = 1

        for page_entry in reading_order.get("pages", []):
            page_number = int(page_entry["page_number"])
            ordered_items = [item_lookup[item_id] for item_id in page_entry.get("ordered_item_ids", []) if item_id in item_lookup]
            page_regions = regions_by_page.get(page_number, [])
            page_text_regions = text_regions_by_page.get(page_number, [])
            page_blocks: list[OrderedTextBlock] = []
            current_items: list[OCRTextItem] = []
            current_region_id: str | None = None
            current_line_bucket: int | None = None

            for item in ordered_items:
                matched_region, overlap_ratio = _best_region_match(item, page_regions)
                item.region_id = matched_region.region_id if matched_region else None
                line_bucket = int(item.bbox.y0 // 20)
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

                group_region_id = item.region_id
                if current_items and (group_region_id != current_region_id or line_bucket != current_line_bucket):
                    global_index = _flush_block(page_number, current_items, global_index, page_blocks, ordered_blocks)
                    current_items = []

                current_items.append(item)
                current_region_id = group_region_id
                current_line_bucket = line_bucket

            if current_items:
                global_index = _flush_block(page_number, current_items, global_index, page_blocks, ordered_blocks)

            if not page_blocks and ordered_items:
                logger.warning("Page %s had OCR text but no Paddle text blocks; falling back to OCR line grouping", page_number)
                page_blocks = _build_fallback_blocks(page_number, ordered_items, global_index)
                ordered_blocks.extend(page_blocks)
                global_index += len(page_blocks)
                for block in page_blocks:
                    for item_id in block.item_ids:
                        item_lookup[item_id].block_id = block.block_id
            else:
                association_lookup = {assoc.item_id: assoc for assoc in associations if assoc.page_number == page_number}
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

        return associations, ordered_blocks, {
            "pages": page_payloads,
            "full_text": "\n\n".join(page["text"] for page in page_payloads if page["text"]).strip(),
        }


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
        processing_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        schema_version="6.0.0",
        ocr_engine="PaddleOCR",
        reading_order_model=reading_order_model,
        layout_detection_model=layout_detection_model,
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
) -> list[VisualRegionSummary]:
    asset_by_region = {asset.region_id: asset for asset in cropped_assets}
    chunks_by_region: dict[str, list[ProcessedChunk]] = {}
    for chunk in chunks:
        for region_id in chunk.source_region_ids:
            chunks_by_region.setdefault(region_id, []).append(chunk)

    summaries: list[VisualRegionSummary] = []
    for region in regions:
        if region.region_type not in {"table", "figure"}:
            continue
        page_blocks = [block for block in ordered_blocks if block.page_number == region.page_number and block.bbox is not None]
        overlapping_blocks = [
            block
            for block in page_blocks
            if block.bbox is not None and block.bbox.intersection_area(region.bbox) > 0
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
        summary_text = (block_text or chunk_text or f"Detected {region.region_type} region on page {region.page_number}.")[:1200]
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
    document: ProcessedDocument,
    metadata: ProcessingMetadata,
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
    _write_json(working_dir / "visual_summaries.json", [summary.model_dump(mode="json") for summary in visual_summaries])
    _write_json(working_dir / "document.json", document.model_dump(mode="json"))
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


@lru_cache(maxsize=1)
def _get_paddle_ocr() -> Any:
    _configure_paddle_env()
    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
    )


@lru_cache(maxsize=1)
def _get_paddle_layout_detector() -> Any:
    _configure_paddle_env()
    from paddleocr import LayoutDetection

    return LayoutDetection()


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


def _build_fallback_blocks(page_number: int, ordered_items: list[OCRTextItem], start_index: int) -> list[OrderedTextBlock]:
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
    crop_refs = [regions_by_id[region_id].crop_path for region_id in region_ids if region_id in regions_by_id and regions_by_id[region_id].crop_path]
    crop_asset_ids = [f"asset_{region_id}" for region_id in region_ids if region_id in regions_by_id and regions_by_id[region_id].crop_path]
    region_types = sorted({regions_by_id[region_id].region_type for region_id in region_ids if region_id in regions_by_id})
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _confidence_summary(*, ocr_pages: list[OCRPageResult], regions: list[LayoutRegion], chunks: list[ProcessedChunk]) -> dict[str, Any]:
    ocr_confidences = [item.confidence for page in ocr_pages for item in page.items if item.confidence is not None]
    region_confidences = [region.confidence for region in regions if region.confidence is not None]
    return {
        "ocr_item_count": len(ocr_confidences),
        "ocr_average_confidence": round(sum(ocr_confidences) / len(ocr_confidences), 4) if ocr_confidences else None,
        "region_count": len(regions),
        "region_average_confidence": round(sum(region_confidences) / len(region_confidences), 4) if region_confidences else None,
        "chunk_count": len(chunks),
    }

