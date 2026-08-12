"""
Tests for the OCR model selection and its provenance label.

Two defects motivated these:

  * The code pinned PP-OCRv4_mobile — the oldest and smallest models available —
    while the installed paddleocr 3.7 defaults to PP-OCRv6_medium. Reported gains
    are roughly +13pp end-to-end for v5 over v4, and a further +4.6% detection /
    +5.1% recognition for v6_medium over v5_server.
  * OCRPageResult was stamped text_source="paddleocr_ppocrv5_mobile" while v4
    models actually ran, so every OCR-produced artifact recorded a model version
    it never used. The label is now derived from the configured model names so it
    cannot drift from reality again.
"""

from __future__ import annotations

from document_process.services import ocr_text_source_label


def test_label_is_derived_from_the_configured_models(tmp_settings):
    """e.g. PP-OCRv6_medium_det + PP-OCRv6_medium_rec -> 'paddleocr_v6_medium'."""
    label = ocr_text_source_label(tmp_settings)

    assert label.startswith("paddleocr_")
    assert "v6" in label and "medium" in label


def test_label_changes_when_the_model_changes(tmp_settings):
    """The previous hardcoded string claimed v5 while v4 ran."""
    v6 = ocr_text_source_label(tmp_settings(ocr_detection_model="PP-OCRv6_medium_det"))
    v4 = ocr_text_source_label(tmp_settings(ocr_detection_model="PP-OCRv4_mobile_det"))

    assert v6 != v4


def test_label_records_the_recognition_model_too():
    """Detection and recognition can be mixed; both belong in provenance."""
    from config import Settings

    label = ocr_text_source_label(
        Settings(
            openai_api_key="x",
            ocr_detection_model="PP-OCRv6_medium_det",
            ocr_recognition_model="PP-OCRv5_server_rec",
        )
    )

    assert "v6" in label.lower()
    assert "v5" in label.lower()


def test_defaults_are_the_current_generation():
    """paddleocr 3.7 ships PP-OCRv6_medium as its default; do not pin older."""
    from config import Settings

    settings = Settings(openai_api_key="x")

    assert settings.ocr_detection_model == "PP-OCRv6_medium_det"
    assert settings.ocr_recognition_model == "PP-OCRv6_medium_rec"
