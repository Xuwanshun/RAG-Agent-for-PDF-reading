from __future__ import annotations

from document_Process.clients import JSONModelClient
from document_Process.models import ChartAnalysisOutput, TableAnalysisOutput


TOOL_DESCRIPTORS = [
    {
        "name": "AnalyzeChartTool",
        "description": "Analyze a cropped chart image and return chart type, axes, data points, and trends.",
    },
    {
        "name": "AnalyzeTableTool",
        "description": "Analyze a cropped table image and return headers, rows, values, and notes.",
    },
]


CHART_PROMPT = (
    "Analyze this cropped chart image and return JSON only with the keys "
    "region_id, page_number, region_type, chart_type, axes, data_points, trend_summary, and notes."
)

TABLE_PROMPT = (
    "Analyze this cropped table image and return JSON only with the keys "
    "region_id, page_number, region_type, headers, rows, values, and notes."
)


def analyze_chart_region(
    *,
    vlm_client: JSONModelClient,
    region_id: str,
    page_number: int,
    bbox: list[float],
    crop_path: str,
    context_text: str = "",
) -> dict:
    payload = vlm_client.analyze_image(
        image_path=crop_path,
        system_prompt="You are a chart analysis model. Return valid JSON only.",
        user_prompt=f"{CHART_PROMPT}\nRegion metadata: region_id={region_id}, page_number={page_number}, bbox={bbox}. Nearby OCR text: {context_text}",
        response_model=ChartAnalysisOutput,
    )
    return payload.model_copy(update={"region_id": region_id, "page_number": page_number}).model_dump(mode="json")


def analyze_table_region(
    *,
    vlm_client: JSONModelClient,
    region_id: str,
    page_number: int,
    bbox: list[float],
    crop_path: str,
    context_text: str = "",
) -> dict:
    payload = vlm_client.analyze_image(
        image_path=crop_path,
        system_prompt="You are a table extraction model. Return valid JSON only.",
        user_prompt=f"{TABLE_PROMPT}\nRegion metadata: region_id={region_id}, page_number={page_number}, bbox={bbox}. Nearby OCR text: {context_text}",
        response_model=TableAnalysisOutput,
    )
    return payload.model_copy(update={"region_id": region_id, "page_number": page_number}).model_dump(mode="json")
