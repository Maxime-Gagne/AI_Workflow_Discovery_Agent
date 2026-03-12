from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from schemas import DiagnosticAnalyste, GainsEstimes, RoiConfidence


class DiagnosticValidationError(ValueError):
    """Erreur de validation métier sur la sortie Analyst."""


def parse_percent(text: str) -> float:
    """
    Convertit une chaîne du type '72%' ou '72.5 %' en float.
    """
    if not isinstance(text, str):
        raise DiagnosticValidationError(f"Pourcentage invalide (type): {type(text)!r}")

    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$", text)
    if not match:
        raise DiagnosticValidationError(f"Pourcentage invalide: {text!r}")

    value = float(match.group(1))
    if not (0.0 <= value <= 100.0):
        raise DiagnosticValidationError(f"Pourcentage hors bornes: {text!r}")

    return value


def _require_non_empty_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DiagnosticValidationError(f"{field_name} vide ou invalide.")


def _validate_steps(diagnostic: DiagnosticAnalyste) -> None:
    steps = diagnostic.etapes_actuelles

    if not steps:
        raise DiagnosticValidationError("No current steps detected.")

    ids = [step.id for step in steps]
    if len(ids) != len(set(ids)):
        raise DiagnosticValidationError("Duplicate step IDs detected.")

    seen_signatures: set[tuple[str, str]] = set()

    for step in steps:
        _require_non_empty_str(step.id, "current_steps[].id")
        _require_non_empty_str(step.nom, f"Step {step.id}.name")
        _require_non_empty_str(step.acteur, f"Step {step.id}.actor")

        if step.friction is not None:
            _require_non_empty_str(step.friction, f"Step {step.id}.friction")

        signature = (step.nom.strip().lower(), step.acteur.strip().lower())
        if signature in seen_signatures:
            raise DiagnosticValidationError(
                f"Duplicate steps detected (name/actor): {step.nom!r} / {step.acteur!r}"
            )
        seen_signatures.add(signature)


def _validate_metriques_cles(diagnostic: DiagnosticAnalyste) -> None:
    metrics = diagnostic.metriques_cles

    _require_non_empty_str(metrics.volume_total, "metriques_cles.volume_total")
    _require_non_empty_str(metrics.duree_totale_estimee, "metriques_cles.duree_totale_estimee")
    _require_non_empty_str(metrics.taux_manuel, "metriques_cles.taux_manuel")
    _require_non_empty_str(metrics.principale_friction, "metriques_cles.principale_friction")

    parse_percent(metrics.taux_manuel)

def _validate_roi_confidence(confidence: RoiConfidence) -> None:
    if confidence.roi_mode not in {"full", "partial", "blocked"}:
        raise DiagnosticValidationError(
            f"gains_estimes.confidence.roi_mode invalide: {confidence.roi_mode!r}"
        )

    levels = {
        "monthly_volume": confidence.monthly_volume,
        "manual_time_minutes": confidence.manual_time_minutes,
        "activity_duration_minutes": confidence.activity_duration_minutes,
        "case_cycle_time_minutes": confidence.case_cycle_time_minutes,
        "waiting_time_minutes": confidence.waiting_time_minutes,
    }

    allowed = {"provided", "observed", "estimated", "missing"}
    for field_name, level in levels.items():
        if level not in allowed:
            raise DiagnosticValidationError(
                f"gains_estimes.confidence.{field_name} invalide: {level!r}"
            )

def _validate_gains_estimes(
    gains: GainsEstimes,
    roi_metrics: Optional[Mapping[str, Any]] = None,
    tol: float = 1e-6,
) -> None:
    numeric_values = {
        "heures_economisees_par_mois": gains.heures_economisees_par_mois,
        "economies_mensuelles_devise": gains.economies_mensuelles_devise,
        "projection_annuelle": gains.projection_annuelle,
    }

    for field_name, value in numeric_values.items():
        if value < 0:
            raise DiagnosticValidationError(f"{field_name} ne peut pas être négatif: {value}")

    if gains.mois_retour_investissement is not None and gains.mois_retour_investissement <= 0:
        raise DiagnosticValidationError(
            "mois_retour_investissement doit être > 0 quand il est fourni."
        )

    expected_projection = gains.economies_mensuelles_devise * 12
    if abs(gains.projection_annuelle - expected_projection) > tol:
        raise DiagnosticValidationError(
            "projection_annuelle incohérente avec economies_mensuelles_devise * 12."
        )

    _require_non_empty_str(gains.detail_du_calcul, "gains_estimes.detail_du_calcul")

    if not gains.assumptions:
        raise DiagnosticValidationError("gains_estimes.assumptions est vide.")

    if any(not isinstance(item, str) or not item.strip() for item in gains.assumptions):
        raise DiagnosticValidationError(
            "gains_estimes.assumptions contient une valeur vide."
        )

    _validate_roi_confidence(gains.confidence)

    if not gains.principales_opportunites:
        raise DiagnosticValidationError("gains_estimes.principales_opportunites est vide.")

    if any(not isinstance(item, str) or not item.strip() for item in gains.principales_opportunites):
        raise DiagnosticValidationError(
            "gains_estimes.principales_opportunites contient une valeur vide."
        )

    if roi_metrics is None:
        return

    mapping = {
        "heures_economisees_par_mois": "heures_economisees_par_mois",
        "economies_mensuelles": "economies_mensuelles_devise",
        "projection_annuelle": "projection_annuelle",
    }

    for roi_key, gains_attr in mapping.items():
        if roi_key not in roi_metrics:
            raise DiagnosticValidationError(
                f"roi_metrics incomplet: clé manquante {roi_key!r}"
            )

        expected_value = float(roi_metrics[roi_key])
        actual_value = float(getattr(gains, gains_attr))

        if abs(actual_value - expected_value) > tol:
            raise DiagnosticValidationError(
                f"Le LLM a altéré {gains_attr}: attendu={expected_value}, obtenu={actual_value}"
            )

    if "assumptions" not in roi_metrics:
        raise DiagnosticValidationError("roi_metrics incomplet: clé manquante 'assumptions'")

    if "confidence" not in roi_metrics:
        raise DiagnosticValidationError("roi_metrics incomplet: clé manquante 'confidence'")

    expected_assumptions = list(roi_metrics["assumptions"])
    if gains.assumptions != expected_assumptions:
        raise DiagnosticValidationError(
            "Le LLM a altéré gains_estimes.assumptions par rapport au ROI déterministe."
        )

    expected_confidence = dict(roi_metrics["confidence"])
    actual_confidence = gains.confidence.model_dump()

    if actual_confidence != expected_confidence:
        raise DiagnosticValidationError(
            "Le LLM a altéré gains_estimes.confidence par rapport au ROI déterministe."
        )


def validate_diagnostic_or_raise(
    diagnostic: DiagnosticAnalyste,
    roi_metrics: Optional[Mapping[str, Any]] = None,
    tol: float = 1e-6,
) -> None:
    """
    Validation métier complète de la sortie Analyst.

    - valide la structure utile des étapes
    - borne les métriques clés
    - borne le score d'automatisation
    - vérifie la cohérence de GainsEstimes
    - force l'alignement avec le ROI déterministe si roi_metrics est fourni
    """
    if not isinstance(diagnostic, DiagnosticAnalyste):
        raise DiagnosticValidationError(
            f"DiagnosticAnalyste attendu, reçu: {type(diagnostic)!r}"
        )

    _require_non_empty_str(diagnostic.titre_processus, "titre_processus")
    _require_non_empty_str(diagnostic.description, "description")

    _validate_steps(diagnostic)
    _validate_metriques_cles(diagnostic)
    _validate_gains_estimes(diagnostic.gains_estimes, roi_metrics=roi_metrics, tol=tol)


def audit_diagnostic(diagnostic: DiagnosticAnalyste) -> list[str]:
    warnings: list[str] = []

    if len(diagnostic.etapes_actuelles) < 3:
        warnings.append("The diagnostic contains very few steps; granularity may be insufficient.")

    if len(diagnostic.gains_estimes.principales_opportunites) > 5:
        warnings.append("Too many opportunities were identified; consider stricter prioritization.")

    try:
        taux = parse_percent(diagnostic.metriques_cles.taux_manuel)
        if taux < 10:
            warnings.append("The manual workload is very low; verify that the automation potential is realistic.")
        if taux > 90:
            warnings.append("The manual workload is very high; verify that the source did not oversimplify the process.")
    except DiagnosticValidationError:
        warnings.append("The manual workload percentage could not be interpreted.")

    if diagnostic.gains_estimes.mois_retour_investissement is not None:
        if diagnostic.gains_estimes.mois_retour_investissement > 24:
            warnings.append("Long payback period (>24 months).")

    return warnings
