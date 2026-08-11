from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Settings
from document_process.models import (
    LayoutRegion,
    OCRPageResult,
    OrderedTextBlock,
    ProcessingIssue,
    RegionAssociation,
)
from document_process.services import (
    AssociationService,
    CroppingService,
    DocumentLoaderService,
    LayoutDetectionService,
    OCRService,
    ReadingOrderService,
    build_chunks,
    build_document_artifacts,
    build_visual_summaries,
    export_artifacts,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingResult:
    document_id: str
    working_dir: Path
    document_json_path: Path
    page_count: int
    chunk_count: int
    warnings: list[str]


class DocumentPreprocessingPipeline:
    """Input -> PaddleOCR -> reading order -> Paddle layout -> crop -> frozen artifacts."""

    def __init__(
        self,
        settings: Settings,
        *,
        loader: DocumentLoaderService | None = None,
        ocr: OCRService | None = None,
        reading_order: ReadingOrderService | None = None,
        layout: LayoutDetectionService | None = None,
        association: AssociationService | None = None,
        cropping: CroppingService | None = None,
    ) -> None:
        self.settings = settings
        self.loader = loader or DocumentLoaderService(settings)
        self.ocr = ocr or OCRService()
        self.reading_order = reading_order or ReadingOrderService(settings)
        self.layout = layout or LayoutDetectionService()
        self.association = association or AssociationService()
        self.cropping = cropping or CroppingService()

    def run(
        self,
        source_path: Path,
        *,
        document_id: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> PreprocessingResult:
        logger.info("Starting document preprocessing for %s", source_path)
        loaded = self.loader.load(source_path, document_id=document_id)
        issues: list[ProcessingIssue] = []

        batch_size = self.settings.preprocess_page_batch_size
        all_pages = loaded.pages
        total_pages = len(all_pages)
        batches = [all_pages[i : i + batch_size] for i in range(0, total_pages, batch_size)]
        total_batches = len(batches)

        all_ocr_pages: list[OCRPageResult] = []
        all_regions: list[LayoutRegion] = []
        all_associations: list[RegionAssociation] = []
        all_ordered_blocks: list[OrderedTextBlock] = []
        merged_ro_pages: list[dict[str, Any]] = []
        merged_ro_ids: list[str] = []
        merged_ordered_text_pages: list[dict[str, Any]] = []
        layout_model = "unknown"
        batch_resolver = "unknown"
        next_block_index = 1
        pages_done = 0

        def _on_page_done() -> None:
            nonlocal pages_done
            pages_done += 1
            if on_progress is not None and total_pages > 0:
                on_progress(pages_done, total_pages)

        for batch_num, batch in enumerate(batches, start=1):
            first_page = batch[0].page_number
            last_page = batch[-1].page_number
            logger.info(
                "Batch %d/%d: pages %d-%d",
                batch_num,
                total_batches,
                first_page,
                last_page,
            )

            # Pass the PDF so OCRService can read its embedded text layer where
            # one exists; None for images, which have nothing to read.
            ocr_batch, ocr_issues = self.ocr.extract(
                batch,
                pdf_path=(loaded.original_copy_path if loaded.original_copy_path.suffix.lower() == ".pdf" else None),
                on_page_done=_on_page_done,
            )
            issues.extend(ocr_issues)

            ro_batch, ro_issues = self.reading_order.resolve(ocr_batch)
            issues.extend(ro_issues)

            regions_batch, layout_issues, layout_model = self.layout.detect(batch, ocr_batch)
            issues.extend(layout_issues)

            assocs_batch, blocks_batch, ordered_text_batch = self.association.associate(
                ocr_batch, ro_batch, regions_batch, start_index=next_block_index
            )

            batch_resolver = ro_batch.get("resolver", "unknown")
            all_ocr_pages.extend(ocr_batch)
            all_regions.extend(regions_batch)
            all_associations.extend(assocs_batch)
            all_ordered_blocks.extend(blocks_batch)
            merged_ro_ids.extend(ro_batch.get("document_order_item_ids", []))
            merged_ro_pages.extend(ro_batch.get("pages", []))
            merged_ordered_text_pages.extend(ordered_text_batch.get("pages", []))

            next_block_index += len(blocks_batch)

        reading_order = {
            "resolver": batch_resolver,
            "document_order_item_ids": merged_ro_ids,
            "pages": merged_ro_pages,
        }
        ordered_text = {
            "pages": merged_ordered_text_pages,
            "full_text": "\n\n".join(p["text"] for p in merged_ordered_text_pages if p.get("text")).strip(),
        }

        cropped_assets, crop_issues = self.cropping.crop_visual_regions(
            pages=all_pages,
            regions=all_regions,
            output_dir=loaded.working_dir / "crops",
        )
        issues.extend(crop_issues)

        intel_result = None
        if (
            self.settings.use_document_intelligence
            or self.settings.use_vlm_summaries
            or self.settings.use_adaptive_chunking
        ):
            from document_process.intelligence_service import DocumentIntelligenceService

            intel_service = DocumentIntelligenceService(self.settings)
            intel_result = intel_service.process(
                regions=all_regions,
                document_id=loaded.document_id,
                file_name=loaded.original_copy_path.name,
                page_count=total_pages,
                ordered_blocks=all_ordered_blocks,
            )

        target_chars = self.settings.preprocess_chunk_size
        overlap_chars = self.settings.preprocess_chunk_overlap
        if intel_result is not None:
            target_chars = int(intel_result.strategy.get("chunk_size", target_chars))
            overlap_chars = int(intel_result.strategy.get("overlap", overlap_chars))

        chunks = build_chunks(
            document_id=loaded.document_id,
            source_file=loaded.original_copy_path.name,
            ordered_blocks=all_ordered_blocks,
            regions=all_regions,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )

        if intel_result is not None:
            region_parent: dict[str, tuple[str, str | None]] = {
                region.region_id: (
                    str(region.metadata.get("parent_title") or ""),
                    region.metadata.get("parent_subtitle"),
                )
                for region in all_regions
                if region.metadata.get("parent_title") is not None
            }
            for chunk in chunks:
                for region_id in chunk.source_region_ids:
                    if region_id in region_parent:
                        parent_title, parent_subtitle = region_parent[region_id]
                        chunk.metadata["parent_title"] = parent_title
                        if parent_subtitle is not None:
                            chunk.metadata["parent_subtitle"] = parent_subtitle
                        break

        visual_summaries = build_visual_summaries(
            regions=all_regions,
            ordered_blocks=all_ordered_blocks,
            chunks=chunks,
            cropped_assets=cropped_assets,
        )

        if intel_result is not None and intel_result.visual_summaries:
            intel_vs = intel_result.visual_summaries
            visual_summaries = [
                summary.model_copy(
                    update={"metadata": {**summary.metadata, "intelligence": intel_vs[summary.crop_path]}}
                )
                if summary.crop_path and summary.crop_path in intel_vs
                else summary
                for summary in visual_summaries
            ]

        if self.settings.use_vlm_summaries:
            from document_process.vlm import enrich_summaries_with_vlm

            visual_summaries = enrich_summaries_with_vlm(visual_summaries, settings=self.settings)

        document, metadata = build_document_artifacts(
            loaded=loaded,
            ocr_pages=all_ocr_pages,
            ordered_text=ordered_text,
            regions=all_regions,
            cropped_assets=cropped_assets,
            chunks=chunks,
            reading_order_model=reading_order.get("resolver", "unknown"),
            layout_detection_model=layout_model,
            issues=issues,
        )

        document_json_path = export_artifacts(
            working_dir=loaded.working_dir,
            loaded=loaded,
            raw_ocr=all_ocr_pages,
            reading_order=reading_order,
            ordered_text=ordered_text,
            regions=all_regions,
            region_associations=all_associations,
            cropped_assets=cropped_assets,
            visual_summaries=visual_summaries,
            chunks=chunks,
            document=document,
            metadata=metadata,
            descriptor=intel_result.descriptor if intel_result is not None else None,
            summary_embedding=intel_result.summary_embedding if intel_result is not None else None,
        )

        logger.info(
            "Finished preprocessing document %s with %s page(s) and %s chunk(s)",
            loaded.document_id,
            total_pages,
            len(chunks),
        )
        return PreprocessingResult(
            document_id=loaded.document_id,
            working_dir=loaded.working_dir,
            document_json_path=document_json_path,
            page_count=total_pages,
            chunk_count=len(chunks),
            warnings=[issue.message for issue in issues if issue.level == "warning"],
        )


def preprocess_document(
    source_name_or_path: str | Path,
    *,
    settings: Settings | None = None,
    document_id: str | None = None,
    force: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> PreprocessingResult:
    resolved_settings = settings or Settings()
    source_path = Path(source_name_or_path)
    if not source_path.is_absolute() and source_path.parent == Path("."):
        source_path = resolved_settings.raw_documents_dir / source_path
    loader = DocumentLoaderService(resolved_settings)
    resolved_id = document_id or loader._build_document_id(source_path)
    working_dir = resolved_settings.processed_documents_dir / resolved_id
    manifest_path = working_dir / "manifest.json"
    chunks_path = working_dir / "chunks.json"
    document_path = working_dir / "document.json"
    if not force and manifest_path.exists() and chunks_path.exists() and document_path.exists():
        logger.info("Reusing frozen preprocessing for %s", source_path)
        chunk_count = len(json.loads(chunks_path.read_text(encoding="utf-8")))
        metadata_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return PreprocessingResult(
            document_id=resolved_id,
            working_dir=working_dir,
            document_json_path=document_path,
            page_count=int(metadata_payload.get("page_count") or 0),
            chunk_count=chunk_count,
            warnings=[],
        )
    pipeline = DocumentPreprocessingPipeline(resolved_settings, loader=loader)
    return pipeline.run(source_path, document_id=resolved_id, on_progress=on_progress)
