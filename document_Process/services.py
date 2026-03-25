from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings
from document_Process.clients import build_agent_repair_client
from document_Process.models import (
    AgentAnalysisResult,
    AgentInput,
    AgentToolCall,
    BoundingBox,
    ChartAnalysisOutput,
    CroppedRegionAsset,
    LayoutRegion,
    OCRPageResult,
    OCRTextItem,
    OrderedTextBlock,
    ProcessedChunk,
    ProcessedDocument,
    ProcessingIssue,
    ProcessingMetadata,
    RegionAssociation,
    TableAnalysisOutput,
)
from document_Process.prompts import SYSTEM_PROMPT
from document_Process.tools import TOOL_DESCRIPTORS


SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageContext:
    page_number: int
    width: float | None
    height: float | None
    page_image_path: Path
    native_text_blocks: list[dict[str, Any]]
    native_text: str


@dataclass(frozen=True)
class LoadedDocument:
    document_id: str
    source_path: Path
    working_dir: Path
    original_copy_path: Path
    pages: list[PageContext]


@dataclass(frozen=True)
class AgentExecutionResult:
    tool_results: list[dict[str, Any]]
    summary: AgentAnalysisResult
    model_name: str | None
    issues: list[ProcessingIssue]


class DocumentLoaderService:
    def __init__(self, settings: Settings, *, native_pdf_tool: Any | None = None) -> None:
        self.settings = settings
        self.native_pdf_tool = native_pdf_tool or _build_native_pdf_tool(settings)

    def load(self, source_path: Path, *, document_id: str | None = None) -> LoadedDocument:
        logger.info("Loading document for preprocessing: %s", source_path)
        if source_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported document type: {source_path.suffix or 'no extension'}")

        resolved_id = document_id or self._build_document_id(source_path)
        working_dir = self.settings.processed_documents_dir / resolved_id
        source_dir = working_dir / "source"
        pages_dir = source_dir / "pages"
        source_dir.mkdir(parents=True, exist_ok=True)
        pages_dir.mkdir(parents=True, exist_ok=True)
        original_copy_path = source_dir / source_path.name
        if source_path.resolve() != original_copy_path.resolve():
            shutil.copy2(source_path, original_copy_path)

        if original_copy_path.suffix.lower() == ".pdf":
            if self.native_pdf_tool is not None:
                inspected_pages = self.native_pdf_tool.inspect(original_copy_path, artifact_dir=pages_dir)
                pages: list[PageContext] = []
                for page in inspected_pages:
                    image_path = page.page_image_path
                    if image_path is None:
                        image_path = pages_dir / f"page_{page.page_no}.png"
                        self.native_pdf_tool.render_page(original_copy_path, page.page_no, image_path)
                    pages.append(
                        PageContext(
                            page_number=page.page_no,
                            width=page.width,
                            height=page.height,
                            page_image_path=image_path,
                            native_text_blocks=[{"text": block.text, "bbox": block.bbox} for block in page.text_blocks],
                            native_text=page.native_text,
                        )
                    )
            else:
                pages = _load_pdf_with_system_pdfkit(original_copy_path, pages_dir)
                if not pages:
                    native_text = _extract_pdf_text_fallback(original_copy_path)
                    if not native_text.strip():
                        raise RuntimeError(
                            "No PDF extraction backend is available. Install the project's PDF tools or provide a text-extractable PDF."
                        )
                    pages = [
                        PageContext(
                            page_number=1,
                            width=None,
                            height=None,
                            page_image_path=original_copy_path,
                            native_text_blocks=[],
                            native_text=native_text,
                        )
                    ]
        else:
            pages = [
                PageContext(
                    page_number=1,
                    width=None,
                    height=None,
                    page_image_path=original_copy_path,
                    native_text_blocks=[],
                    native_text="",
                )
            ]

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
    def __init__(self, *, paddleocr_tool: Any | None = None, language: str = "en") -> None:
        self.paddleocr_tool = paddleocr_tool or _build_paddleocr_tool(language=language)

    def extract(self, pages: list[PageContext]) -> tuple[list[OCRPageResult], list[ProcessingIssue]]:
        logger.info("Running PaddleOCR on %s page(s)", len(pages))
        results: list[OCRPageResult] = []
        issues: list[ProcessingIssue] = []
        for page in pages:
            if self.paddleocr_tool is None:
                native_items = _build_native_text_items(page)
                issues.append(
                    ProcessingIssue(
                        code="ocr_unavailable",
                        message="PaddleOCR is unavailable; using native text fallback when possible.",
                        level="warning",
                        page_number=page.page_number,
                    )
                )
                results.append(
                    OCRPageResult(
                        page_number=page.page_number,
                        width=page.width,
                        height=page.height,
                        items=native_items,
                        text_source="native_text_fallback" if native_items else "unavailable",
                        page_image_path=str(page.page_image_path),
                    )
                )
                continue
            try:
                blocks = self.paddleocr_tool.extract(page.page_image_path)
            except RuntimeError as exc:
                issues.append(
                    ProcessingIssue(
                        code="ocr_unavailable",
                        message=str(exc),
                        level="warning",
                        page_number=page.page_number,
                    )
                )
                blocks = []
            items = [
                OCRTextItem(
                    item_id=f"p{page.page_number}_ocr_{index}",
                    page_number=page.page_number,
                    text=block.text,
                    bbox=BoundingBox.from_list(block.bbox),
                    confidence=block.confidence,
                    source="paddleocr",
                )
                for index, block in enumerate(blocks, start=1)
            ]
            results.append(
                OCRPageResult(
                    page_number=page.page_number,
                    width=page.width,
                    height=page.height,
                    items=items,
                    text_source="paddleocr",
                    page_image_path=str(page.page_image_path),
                )
            )
        return results, issues


class ReadingOrderService:
    def __init__(self) -> None:
        self._layoutreader_module = None
        self._layoutreader_error: str | None = None

    def resolve(self, pages: list[OCRPageResult]) -> tuple[dict[str, Any], list[ProcessingIssue]]:
        logger.info("Resolving reading order for %s OCR page(s)", len(pages))
        issues: list[ProcessingIssue] = []
        resolver = "geometric_fallback"
        ordered_text: list[dict[str, Any]] = []
        all_ids: list[str] = []
        module = self._load_layoutreader()

        for page in pages:
            ordered_items = None
            if module is not None:
                ordered_items = self._try_layoutreader(page.items, module)
                if ordered_items is not None:
                    resolver = "layoutreader"
            if ordered_items is None:
                ordered_items = sorted(page.items, key=lambda item: (round(item.bbox.y0 / 14.0), item.bbox.x0, item.bbox.y0))
            for index, item in enumerate(ordered_items, start=1):
                item.reading_order = index
            ids = [item.item_id for item in ordered_items]
            all_ids.extend(ids)
            ordered_text.append({"page_number": page.page_number, "ordered_item_ids": ids})

        if module is None and self._layoutreader_error:
            issues.append(
                ProcessingIssue(
                    code="reading_order_fallback",
                    message="LayoutReader is unavailable; using deterministic geometric fallback.",
                    level="warning",
                    details={"import_error": self._layoutreader_error},
                )
            )

        return {"resolver": resolver, "document_order_item_ids": all_ids, "pages": ordered_text}, issues

    def _load_layoutreader(self):
        if self._layoutreader_module is not None or self._layoutreader_error is not None:
            return self._layoutreader_module
        try:
            import layoutreader  # type: ignore
        except Exception as exc:
            self._layoutreader_error = str(exc)
            return None
        self._layoutreader_module = layoutreader
        return self._layoutreader_module

    def _try_layoutreader(self, items: list[OCRTextItem], module) -> list[OCRTextItem] | None:
        if not items:
            return []
        texts = [item.text for item in items]
        boxes = [item.bbox.as_list() for item in items]
        for attr in ("LayoutReader", "Reader", "Predictor"):
            candidate = getattr(module, attr, None)
            if candidate is None:
                continue
            try:
                instance = candidate() if callable(candidate) else candidate
            except Exception:
                continue
            for method_name in ("predict", "infer", "__call__", "sort_boxes"):
                method = getattr(instance, method_name, None)
                if method is None:
                    continue
                for kwargs in (
                    {"texts": texts, "boxes": boxes},
                    {"ocr_tokens": texts, "bboxes": boxes},
                    {"tokens": texts, "boxes": boxes},
                ):
                    try:
                        result = method(**kwargs)
                    except Exception:
                        continue
                    ordered = _coerce_layoutreader_result(result, items)
                    if ordered is not None:
                        return ordered
        return None


class LayoutDetectionService:
    def __init__(self) -> None:
        self._ppstructure = None
        self._ppstructure_error: str | None = None

    def detect(self, pages: list[PageContext], ocr_pages: list[OCRPageResult]) -> tuple[list[LayoutRegion], list[ProcessingIssue], str]:
        logger.info("Running layout detection on %s page(s)", len(pages))
        client = self._get_client()
        if client is not None:
            regions, issues = self._detect_with_paddle_layout(client, pages)
            return regions, issues, "paddleocr_ppstructure"
        issues = []
        if self._ppstructure_error:
            issues.append(
                ProcessingIssue(
                    code="layout_fallback",
                    message="PaddleOCR layout detection unavailable; using heuristic fallback.",
                    level="warning",
                    details={"import_error": self._ppstructure_error},
                )
            )
        return self._detect_with_fallback(ocr_pages), issues, "heuristic_layout"

    def _get_client(self):
        if self._ppstructure is not None or self._ppstructure_error is not None:
            return self._ppstructure
        try:
            from paddleocr import PPStructure  # type: ignore
        except Exception as exc:
            self._ppstructure_error = str(exc)
            return None
        self._ppstructure = PPStructure(show_log=False)
        return self._ppstructure

    def _detect_with_paddle_layout(self, client, pages: list[PageContext]) -> tuple[list[LayoutRegion], list[ProcessingIssue]]:
        issues: list[ProcessingIssue] = []
        regions: list[LayoutRegion] = []
        next_id = 1
        for page in pages:
            try:
                results = client(str(page.page_image_path)) or []
            except Exception as exc:
                issues.append(
                    ProcessingIssue(
                        code="layout_detection_failed",
                        message=f"Layout detection failed on page {page.page_number}.",
                        level="warning",
                        page_number=page.page_number,
                        details={"error": str(exc)},
                    )
                )
                results = []
            for item in results:
                raw_type = str(item.get("type", "")).lower()
                normalized = _normalize_region_type(raw_type)
                bbox = item.get("bbox") or item.get("box")
                if normalized is None or not bbox or len(bbox) != 4:
                    continue
                regions.append(
                    LayoutRegion(
                        region_id=f"region_{next_id}",
                        region_type=normalized,
                        page_number=page.page_number,
                        bbox=BoundingBox.from_list([float(value) for value in bbox]),
                        confidence=float(item.get("score", 0.0) or 0.0),
                        source="paddleocr_layout",
                        metadata={"raw_type": raw_type, "normalized_from": raw_type},
                    )
                )
                next_id += 1
        return _dedupe_regions(regions), issues

    def _detect_with_fallback(self, ocr_pages: list[OCRPageResult]) -> list[LayoutRegion]:
        regions: list[LayoutRegion] = []
        next_id = 1
        for page in ocr_pages:
            for item in page.items:
                # Without a real layout detector, OCR/native text items are not reliable visual regions.
                # Keep the fallback conservative so we do not create bogus table/chart crops.
                region_type = "text_block"
                regions.append(
                    LayoutRegion(
                        region_id=f"region_{next_id}",
                        region_type=region_type,
                        page_number=page.page_number,
                        bbox=item.bbox,
                        confidence=item.confidence,
                        source="heuristic_layout",
                        metadata={"ocr_item_id": item.item_id},
                    )
                )
                next_id += 1
        return _dedupe_regions(regions)


class AssociationService:
    def associate(
        self,
        ocr_pages: list[OCRPageResult],
        reading_order: dict[str, Any],
        regions: list[LayoutRegion],
    ) -> tuple[list[RegionAssociation], list[OrderedTextBlock], dict[str, Any]]:
        item_lookup = {item.item_id: item for page in ocr_pages for item in page.items}
        regions_by_page: dict[int, list[LayoutRegion]] = {}
        for region in regions:
            regions_by_page.setdefault(region.page_number, []).append(region)

        associations: list[RegionAssociation] = []
        ordered_blocks: list[OrderedTextBlock] = []
        page_payloads: list[dict[str, Any]] = []
        global_index = 1

        for page_entry in reading_order.get("pages", []):
            page_number = int(page_entry["page_number"])
            ordered_items = [item_lookup[item_id] for item_id in page_entry.get("ordered_item_ids", []) if item_id in item_lookup]
            page_regions = regions_by_page.get(page_number, [])
            page_blocks: list[OrderedTextBlock] = []
            current_items: list[OCRTextItem] = []
            current_region_id: str | None = None
            current_line_bucket: int | None = None

            for item in ordered_items:
                matched, overlap_ratio = _best_region_match(item, page_regions)
                item.region_id = matched.region_id if matched else None
                line_bucket = int(item.bbox.y0 // 18)
                if current_items and (item.region_id != current_region_id or line_bucket != current_line_bucket):
                    block = _build_block(page_number, current_items, global_index)
                    page_blocks.append(block)
                    ordered_blocks.append(block)
                    for grouped_item in current_items:
                        grouped_item.block_id = block.block_id
                    global_index += 1
                    current_items = []
                current_items.append(item)
                current_region_id = item.region_id
                current_line_bucket = line_bucket
                associations.append(
                    RegionAssociation(
                        association_id=f"assoc_{len(associations) + 1}",
                        page_number=page_number,
                        item_id=item.item_id,
                        region_id=item.region_id,
                        region_type=matched.region_type if matched else None,
                        overlap_ratio=round(overlap_ratio, 4),
                    )
                )
            if current_items:
                block = _build_block(page_number, current_items, global_index)
                page_blocks.append(block)
                ordered_blocks.append(block)
                for grouped_item in current_items:
                    grouped_item.block_id = block.block_id
                global_index += 1

            association_lookup = {association.item_id: association for association in associations if association.page_number == page_number}
            for block in page_blocks:
                for item_id in block.item_ids:
                    association_lookup[item_id].block_id = block.block_id

            page_payloads.append(
                {
                    "page_number": page_number,
                    "blocks": [block.model_dump(mode="json") for block in page_blocks],
                    "text": "\n".join(block.text for block in page_blocks if block.text.strip()).strip(),
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
        for folder in ("tables", "charts", "text_blocks"):
            (output_dir / folder).mkdir(parents=True, exist_ok=True)

        assets: list[CroppedRegionAsset] = []
        issues: list[ProcessingIssue] = []
        for region in regions:
            if region.region_type not in {"table", "chart"}:
                continue
            if not region.bbox.is_valid():
                issues.append(
                    ProcessingIssue(
                        code="invalid_crop_bbox",
                        message="Skipping crop because the region bounding box is invalid.",
                        level="warning",
                        page_number=region.page_number,
                        details={"region_id": region.region_id, "bbox": region.bbox.as_list()},
                    )
                )
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
                continue
            if page.page_image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
                issues.append(
                    ProcessingIssue(
                        code="missing_page_raster",
                        message="Skipping crop because no rendered page image is available for this region.",
                        level="warning",
                        page_number=region.page_number,
                        details={"region_id": region.region_id, "page_image_path": str(page.page_image_path)},
                    )
                )
                continue
            folder = "charts" if region.region_type == "chart" else "tables"
            crop_path = output_dir / folder / f"{region.region_id}.png"
            try:
                with Image.open(page.page_image_path) as image:
                    x0, y0, x1, y1 = [int(value) for value in region.bbox.as_list()]
                    crop_box = (max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1))
                    if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
                        issues.append(
                            ProcessingIssue(
                                code="invalid_crop_bounds",
                                message="Skipping crop because the scaled crop bounds are empty.",
                                level="warning",
                                page_number=region.page_number,
                                details={"region_id": region.region_id, "crop_box": list(crop_box)},
                            )
                        )
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


def build_agent_input(*, ordered_text: dict[str, Any], regions: list[LayoutRegion]) -> AgentInput:
    page_text_by_number = {
        int(page["page_number"]): str(page.get("text", ""))
        for page in ordered_text.get("pages", [])
        if isinstance(page, dict)
    }

    def _region_payload(region: LayoutRegion) -> dict[str, Any]:
        return {
            "region_id": region.region_id,
            "region_type": region.region_type,
            "page_number": region.page_number,
            "bbox": region.bbox.as_list(),
            "crop_path": region.crop_path,
            "context_text": page_text_by_number.get(region.page_number, "")[:500],
        }

    return AgentInput(
        ordered_ocr_text=str(ordered_text.get("full_text", "")),
        layout_regions=[_region_payload(region) for region in regions],
        available_tools=TOOL_DESCRIPTORS,
        crop_references=[
            _region_payload(region) for region in regions if region.crop_path and region.region_type in {"chart", "table"}
        ],
    )


class LangChainAgentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, agent_input: AgentInput) -> AgentExecutionResult:
        logger.info("Running OpenAI-backed agent with %s cropped visual region(s)", len(agent_input.crop_references))
        tool_results: list[dict[str, Any]] = []
        repair_client = build_agent_repair_client(self.settings)
        summary = repair_client.generate_structured(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_build_agent_summary_prompt(agent_input, tool_results),
            response_model=AgentAnalysisResult,
        ).model_copy(
            update={
                "tool_calls": [_normalize_tool_call(call) for call in tool_results],
                "tables_analyzed": _collect_table_outputs(tool_results),
                "charts_analyzed": _collect_chart_outputs(tool_results),
                "ordered_ocr_text_excerpt": agent_input.ordered_ocr_text[:500],
            }
        )
        validated_summary, validation_issue = _validate_agent_output(
            raw_output=summary.model_dump(mode="json"),
            agent_input=agent_input,
            tool_results=tool_results,
        )
        issues: list[ProcessingIssue] = [validation_issue] if validation_issue is not None else []

        return AgentExecutionResult(
            tool_results=tool_results,
            summary=validated_summary,
            model_name=self.settings.agent_llm_model,
            issues=issues,
        )


def build_document_artifacts(
    *,
    loaded: LoadedDocument,
    ocr_pages: list[OCRPageResult],
    ordered_text: dict[str, Any],
    regions: list[LayoutRegion],
    cropped_assets: list[CroppedRegionAsset],
    chunks: list[ProcessedChunk],
    agent_input: AgentInput,
    agent_result: AgentExecutionResult,
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
                "normalized_from": region.metadata.get("normalized_from", region.metadata.get("raw_type")),
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
        agent_input=agent_input.model_dump(mode="json"),
        agent_output=agent_result.summary.model_dump(mode="json"),
    )
    metadata = ProcessingMetadata(
        processing_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        schema_version="3.0.0",
        ocr_engine="paddleocr",
        reading_order_model=reading_order_model,
        layout_detection_model=layout_detection_model,
        agent_model=agent_result.model_name,
        confidence_summary=_confidence_summary(ocr_pages=ocr_pages, regions=regions, chunks=chunks),
        warnings=warnings,
        errors=errors,
    )
    return document, metadata


def export_artifacts(
    *,
    working_dir: Path,
    raw_ocr: list[OCRPageResult],
    reading_order: dict[str, Any],
    ordered_text: dict[str, Any],
    regions: list[LayoutRegion],
    region_associations: list[RegionAssociation],
    cropped_assets: list[CroppedRegionAsset],
    chunks: list[ProcessedChunk],
    document: ProcessedDocument,
    metadata: ProcessingMetadata,
) -> Path:
    ocr_dir = working_dir / "ocr"
    order_dir = working_dir / "order"
    layout_dir = working_dir / "layout"
    crops_dir = working_dir / "crops"
    for directory in (ocr_dir, order_dir, layout_dir, crops_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _write_json(ocr_dir / "raw_ocr.json", [page.model_dump(mode="json") for page in raw_ocr])
    _write_json(order_dir / "reading_order.json", reading_order)
    _write_json(order_dir / "ordered_text.json", ordered_text)
    _write_json(layout_dir / "regions.json", [region.model_dump(mode="json") for region in regions])
    _write_json(layout_dir / "region_associations.json", [assoc.model_dump(mode="json") for assoc in region_associations])
    _write_json(crops_dir / "cropped_assets.json", [asset.model_dump(mode="json") for asset in cropped_assets])
    _write_json(working_dir / "document.json", document.model_dump(mode="json"))
    _write_json(working_dir / "chunks.json", [chunk.model_dump(mode="json") for chunk in chunks])
    _write_json(working_dir / "metadata.json", metadata.model_dump(mode="json"))
    return working_dir / "document.json"


def _normalize_region_type(raw_type: str) -> str | None:
    if raw_type == "table":
        return "table"
    if raw_type in {"figure", "chart", "image"}:
        return "chart"
    if raw_type in {"text", "title", "list"}:
        return "text_block"
    return None


def _dedupe_regions(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    seen: set[tuple[int, str, tuple[float, float, float, float]]] = set()
    deduped: list[LayoutRegion] = []
    for region in regions:
        key = (region.page_number, region.region_type, tuple(round(value, 2) for value in region.bbox.as_list()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(region)
    return deduped


def _coerce_layoutreader_result(result: Any, items: list[OCRTextItem]) -> list[OCRTextItem] | None:
    if isinstance(result, dict) and "order" in result:
        result = result["order"]
    if not isinstance(result, list):
        return None
    if result and all(isinstance(value, int) for value in result) and all(0 <= value < len(items) for value in result):
        return [items[index] for index in result]
    if result and all(isinstance(value, str) for value in result):
        lookup = {item.item_id: item for item in items}
        if all(value in lookup for value in result):
            return [lookup[value] for value in result]
    return None


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


def _validate_agent_output(
    *,
    raw_output: Any,
    agent_input: AgentInput,
    tool_results: list[dict[str, Any]],
) -> tuple[AgentAnalysisResult, ProcessingIssue | None]:
    parsed = None
    if isinstance(raw_output, dict):
        parsed = raw_output
    elif isinstance(raw_output, str):
        cleaned = raw_output.strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    parsed = None

    if isinstance(parsed, dict):
        try:
            return AgentAnalysisResult.model_validate(
                {
                    **parsed,
                    "tool_calls": [_normalize_tool_call(call) for call in tool_results],
                    "tables_analyzed": _collect_table_outputs(tool_results),
                    "charts_analyzed": _collect_chart_outputs(tool_results),
                    "ordered_ocr_text_excerpt": parsed.get("ordered_ocr_text_excerpt") or agent_input.ordered_ocr_text[:500],
                }
            ), None
        except Exception as exc:
            raise RuntimeError(f"Agent output validation failed: {exc}") from exc

    raise RuntimeError("Agent output was not valid structured JSON.")


def _normalize_tool_call(raw_call: dict[str, Any]) -> AgentToolCall:
    tool_name = str(raw_call.get("tool_name") or "")
    if tool_name not in {"AnalyzeChartTool", "AnalyzeTableTool"}:
        tool_name = "AnalyzeChartTool" if "chart" in tool_name.lower() else "AnalyzeTableTool"
    region_id = ""
    tool_input = raw_call.get("tool_input")
    if isinstance(tool_input, dict):
        region_id = str(tool_input.get("region_id") or "")
    observation = raw_call.get("observation")
    return AgentToolCall(
        tool_name=tool_name,
        region_id=region_id,
        tool_input=tool_input if isinstance(tool_input, dict) else {"raw": tool_input},
        observation=observation if isinstance(observation, dict) else {"raw": observation},
    )


def _collect_chart_outputs(tool_results: list[dict[str, Any]]) -> list[ChartAnalysisOutput]:
    outputs: list[ChartAnalysisOutput] = []
    for call in tool_results:
        if call.get("tool_name") != "AnalyzeChartTool":
            continue
        observation = call.get("observation")
        if isinstance(observation, dict):
            try:
                outputs.append(ChartAnalysisOutput.model_validate(observation))
            except Exception:
                continue
    return outputs


def _collect_table_outputs(tool_results: list[dict[str, Any]]) -> list[TableAnalysisOutput]:
    outputs: list[TableAnalysisOutput] = []
    for call in tool_results:
        if call.get("tool_name") != "AnalyzeTableTool":
            continue
        observation = call.get("observation")
        if isinstance(observation, dict):
            try:
                outputs.append(TableAnalysisOutput.model_validate(observation))
            except Exception:
                continue
    return outputs


def _build_agent_summary_prompt(agent_input: AgentInput, tool_results: list[dict[str, Any]]) -> str:
    return (
        "Ordered OCR text:\n"
        f"{agent_input.ordered_ocr_text}\n\n"
        "Layout regions:\n"
        f"{json.dumps(agent_input.layout_regions, ensure_ascii=False, indent=2)}\n\n"
        "Available tools:\n"
        f"{json.dumps(agent_input.available_tools, ensure_ascii=False, indent=2)}\n\n"
        "Cropped visual regions:\n"
        f"{json.dumps(agent_input.crop_references, ensure_ascii=False, indent=2)}\n\n"
        "Tool results:\n"
        f"{json.dumps(tool_results, ensure_ascii=False, indent=2)}\n\n"
        "Use the tool results when present, answer only from the provided document content, and return valid JSON."
    )


def _build_native_pdf_tool(settings: Settings):
    try:
        from tools.native_pdf import NativePDFTool  # type: ignore
    except Exception:
        return None
    return NativePDFTool(min_text_chars=settings.min_pdf_text_chars)


def _build_paddleocr_tool(*, language: str):
    try:
        from tools.paddleocr import PaddleOCRTool  # type: ignore
    except Exception:
        return None
    return PaddleOCRTool(language=language)


def _load_pdf_with_system_pdfkit(path: Path, pages_dir: Path) -> list[PageContext]:
    try:
        import objc
        from Foundation import NSURL
        from AppKit import NSBitmapImageRep, NSPNGFileType
    except Exception:
        return []

    module_globals: dict[str, Any] = {}
    try:
        objc.loadBundle("PDFKit", module_globals, bundle_path="/System/Library/Frameworks/PDFKit.framework")
        PDFDocument = module_globals["PDFDocument"]
    except Exception:
        return []

    url = NSURL.fileURLWithPath_(str(path))
    document = PDFDocument.alloc().initWithURL_(url)
    if document is None:
        return []

    pages: list[PageContext] = []
    for page_index in range(int(document.pageCount())):
        page = document.pageAtIndex_(page_index)
        if page is None:
            continue
        page_number = page_index + 1
        page_text = str(page.string() or "").strip()
        bounds = page.boundsForBox_(0)
        image_path = pages_dir / f"page_{page_number}.png"
        thumbnail = page.thumbnailOfSize_forBox_((900, 1200), 0)
        if thumbnail is not None:
            rep = NSBitmapImageRep.alloc().initWithData_(thumbnail.TIFFRepresentation())
            png_data = rep.representationUsingType_properties_(NSPNGFileType, None)
            image_path.write_bytes(bytes(png_data))
        else:
            image_path = path
        pages.append(
            PageContext(
                page_number=page_number,
                width=float(bounds.size.width) if bounds is not None else None,
                height=float(bounds.size.height) if bounds is not None else None,
                page_image_path=image_path,
                native_text_blocks=[],
                native_text=page_text,
            )
        )
    return pages


def _extract_pdf_text_fallback(path: Path) -> str:
    xmp_text = _extract_pdf_xmp_packet(path)
    if xmp_text:
        xmp_candidates = _extract_xmp_text_candidates(xmp_text)
        if xmp_candidates and (len(xmp_candidates) >= 3 or sum(len(item) for item in xmp_candidates) >= 200):
            return "\n".join(xmp_candidates)

    try:
        completed = subprocess.run(
            ["strings", "-n", "8", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""

    lines = []
    for raw_line in completed.stdout.splitlines():
        line = re.sub(r"<[^>]+>", " ", raw_line)
        line = " ".join(line.split()).strip()
        if len(line) < 20:
            continue
        if _looks_like_pdf_syntax(line):
            continue
        if max((len(token) for token in line.split()), default=0) > 40:
            continue
        if not any(character.isalpha() for character in line):
            continue
        alpha_ratio = sum(character.isalpha() or character.isspace() for character in line) / max(len(line), 1)
        if alpha_ratio < 0.6:
            continue
        if len(re.findall(r"[A-Za-z]{3,}", line)) < 4:
            continue
        lines.append(line)
    return "\n".join(_dedupe_text_lines(lines))


def _build_native_text_items(page: PageContext) -> list[OCRTextItem]:
    segments = _split_native_text(page.native_text)
    items: list[OCRTextItem] = []
    for index, text in enumerate(segments, start=1):
        y0 = float((index - 1) * 20)
        items.append(
            OCRTextItem(
                item_id=f"p{page.page_number}_native_{index}",
                page_number=page.page_number,
                text=text,
                bbox=BoundingBox(x0=0.0, y0=y0, x1=1000.0, y1=y0 + 18.0),
                confidence=1.0,
                source="native_text_fallback",
            )
        )
    return items


def _split_native_text(text: str, *, max_chars: int = 500) -> list[str]:
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    if not paragraphs:
        paragraphs = [cleaned]
    segments: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            segments.append(paragraph)
            continue
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + max_chars)
            segments.append(paragraph[start:end].strip())
            start = end
    return [segment for segment in segments if segment]


def _dedupe_text_lines(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return deduped


def _looks_like_pdf_syntax(text: str) -> bool:
    lowered = text.lower()
    pdf_markers = (
        "%pdf",
        " obj",
        "endobj",
        "stream",
        "endstream",
        "/type/",
        "/filter/",
        "/length",
        "/xobject",
        "/mediabox",
        "/cropbox",
        "/resources",
        "/flatedecode",
        "xref",
        "startxref",
        "xpacket",
        "rdf:",
        "xmlns:",
        "adobe xmp",
        "jpeg",
        "base64",
    )
    if any(marker in lowered for marker in pdf_markers):
        return True
    symbol_ratio = sum(character in "<>/[]{}=_:" for character in text) / max(len(text), 1)
    return symbol_ratio > 0.15


def _extract_xmp_text_candidates(raw_text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r">([^<>]{20,})<", raw_text):
        text = html.unescape(" ".join(match.group(1).split())).strip()
        if len(text) < 20 or _looks_like_pdf_syntax(text):
            continue
        alpha_ratio = sum(character.isalpha() or character.isspace() for character in text) / max(len(text), 1)
        if alpha_ratio < 0.7:
            continue
        candidates.append(text)
    return _dedupe_text_lines(candidates)


def _extract_pdf_xmp_packet(path: Path) -> str:
    raw_text = path.read_bytes().decode("latin1", errors="ignore")
    match = re.search(r"<\?xpacket begin=.*?</x:xmpmeta>", raw_text, flags=re.DOTALL)
    if not match:
        return ""
    packet = match.group(0)
    packet = re.sub(r"<xmpGImg:image>.*?</xmpGImg:image>", " ", packet, flags=re.DOTALL)
    return packet
