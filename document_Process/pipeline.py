from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from config import Settings
from document_Process.models import ProcessingIssue
from document_Process.services import (
    AssociationService,
    CroppingService,
    DocumentLoaderService,
    LangChainAgentService,
    LayoutDetectionService,
    OCRService,
    ReadingOrderService,
    build_agent_input,
    build_chunks,
    build_document_artifacts,
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
    """Deterministic preprocessing pipeline plus a two-tool LangChain agent stage."""

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
        agent: LangChainAgentService | None = None,
    ) -> None:
        self.settings = settings
        self.loader = loader or DocumentLoaderService(settings)
        self.ocr = ocr or OCRService()
        self.reading_order = reading_order or ReadingOrderService()
        self.layout = layout or LayoutDetectionService()
        self.association = association or AssociationService()
        self.cropping = cropping or CroppingService()
        self.agent = agent or LangChainAgentService(settings)

    def run(self, source_path: Path, *, document_id: str | None = None) -> PreprocessingResult:
        logger.info("Starting document preprocessing for %s", source_path)
        loaded = self.loader.load(source_path, document_id=document_id)
        issues: list[ProcessingIssue] = []

        ocr_pages, ocr_issues = self.ocr.extract(loaded.pages)
        issues.extend(ocr_issues)

        reading_order, order_issues = self.reading_order.resolve(ocr_pages)
        issues.extend(order_issues)

        regions, layout_issues, layout_model = self.layout.detect(loaded.pages, ocr_pages)
        issues.extend(layout_issues)

        associations, ordered_blocks, ordered_text = self.association.associate(ocr_pages, reading_order, regions)

        cropped_assets, crop_issues = self.cropping.crop_visual_regions(
            pages=loaded.pages,
            regions=regions,
            output_dir=loaded.working_dir / "crops",
        )
        issues.extend(crop_issues)

        chunks = build_chunks(
            document_id=loaded.document_id,
            source_file=loaded.original_copy_path.name,
            ordered_blocks=ordered_blocks,
            regions=regions,
            target_chars=self.settings.preprocess_chunk_size,
            overlap_chars=self.settings.preprocess_chunk_overlap,
        )

        agent_input = build_agent_input(ordered_text=ordered_text, regions=regions)
        agent_result = self.agent.run(agent_input)
        issues.extend(agent_result.issues)

        document, metadata = build_document_artifacts(
            loaded=loaded,
            ocr_pages=ocr_pages,
            ordered_text=ordered_text,
            regions=regions,
            cropped_assets=cropped_assets,
            chunks=chunks,
            agent_input=agent_input,
            agent_result=agent_result,
            reading_order_model=reading_order.get("resolver", "unknown"),
            layout_detection_model=layout_model,
            issues=issues,
        )

        document_json_path = export_artifacts(
            working_dir=loaded.working_dir,
            raw_ocr=ocr_pages,
            reading_order=reading_order,
            ordered_text=ordered_text,
            regions=regions,
            region_associations=associations,
            cropped_assets=cropped_assets,
            chunks=chunks,
            document=document,
            metadata=metadata,
        )
        logger.info(
            "Finished preprocessing document %s with %s page(s) and %s chunk(s)",
            loaded.document_id,
            len(loaded.pages),
            len(chunks),
        )
        return PreprocessingResult(
            document_id=loaded.document_id,
            working_dir=loaded.working_dir,
            document_json_path=document_json_path,
            page_count=len(loaded.pages),
            chunk_count=len(chunks),
            warnings=[issue.message for issue in issues if issue.level == "warning"],
        )


def preprocess_document(
    source_name_or_path: str | Path,
    *,
    settings: Settings | None = None,
    document_id: str | None = None,
) -> PreprocessingResult:
    resolved_settings = settings or Settings()
    source_path = Path(source_name_or_path)
    if not source_path.is_absolute() and source_path.parent == Path("."):
        source_path = resolved_settings.raw_documents_dir / source_path
    pipeline = DocumentPreprocessingPipeline(resolved_settings)
    return pipeline.run(source_path, document_id=document_id)
