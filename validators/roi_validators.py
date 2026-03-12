from __future__ import annotations
from typing import Any


class RoiValidationError(ValueError):
    """Erreur de validation métier sur les métriques ROI."""


def validate_roi_metrics_or_raise(roi_metrics: dict[str, Any]) -> None:
    required_fields = [
        "heures_economisees_par_mois",
        "economies_mensuelles",
        "projection_annuelle",
        "detail_calcul",
        "confidence",
    ]

    for field in required_fields:
        if field not in roi_metrics:
            raise RoiValidationError(f"Champ ROI manquant: {field}")

    numeric_fields = [
        "heures_economisees_par_mois",
        "economies_mensuelles",
        "projection_annuelle",
    ]

    for field in numeric_fields:
        value = roi_metrics[field]
        try:
            value = float(value)
        except (TypeError, ValueError) as e:
            raise RoiValidationError(f"Champ ROI invalide: {field}={value!r}") from e

        if value < 0:
            raise RoiValidationError(f"Champ ROI négatif interdit: {field}={value}")

    expected_projection = round(float(roi_metrics["economies_mensuelles"]) * 12.0, 2)
    actual_projection = round(float(roi_metrics["projection_annuelle"]), 2)

    if actual_projection != expected_projection:
        raise RoiValidationError(
            "projection_annuelle incohérente avec economies_mensuelles * 12."
        )

    if not isinstance(roi_metrics["detail_calcul"], str) or not roi_metrics["detail_calcul"].strip():
        raise RoiValidationError("detail_calcul vide ou invalide.")

    confidence = roi_metrics["confidence"]
    if not isinstance(confidence, dict):
        raise RoiValidationError("confidence doit être un dict.")

    if "roi_mode" not in confidence:
        raise RoiValidationError("confidence.roi_mode manquant.")

    if confidence["roi_mode"] not in {"full", "partial", "blocked"}:
        raise RoiValidationError(
            f"confidence.roi_mode invalide: {confidence['roi_mode']!r}"
        )
