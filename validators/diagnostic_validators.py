from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from schemas import DiagnosticAnalyste, GainsEstimes


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
        raise DiagnosticValidationError("Aucune étape actuelle détectée.")

    ids = [step.id for step in steps]
    if len(ids) != len(set(ids)):
        raise DiagnosticValidationError("IDs d'étapes dupliqués.")

    seen_signatures: set[tuple[str, str]] = set()

    for step in steps:
        _require_non_empty_str(step.id, "etapes_actuelles[].id")
        _require_non_empty_str(step.nom, f"Étape {step.id}.nom")
        _require_non_empty_str(step.acteur, f"Étape {step.id}.acteur")
        _require_non_empty_str(step.friction, f"Étape {step.id}.friction")

        signature = (step.nom.strip().lower(), step.acteur.strip().lower())
        if signature in seen_signatures:
            raise DiagnosticValidationError(
                f"Étapes dupliquées détectées (nom/acteur): {step.nom!r} / {step.acteur!r}"
            )
        seen_signatures.add(signature)


def _validate_metriques_cles(diagnostic: DiagnosticAnalyste) -> None:
    metrics = diagnostic.metriques_cles

    _require_non_empty_str(metrics.volume_total, "metriques_cles.volume_total")
    _require_non_empty_str(metrics.duree_totale_estimee, "metriques_cles.duree_totale_estimee")
    _require_non_empty_str(metrics.taux_manuel, "metriques_cles.taux_manuel")
    _require_non_empty_str(metrics.principale_friction, "metriques_cles.principale_friction")

    parse_percent(metrics.taux_manuel)


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
    """
    Audit non bloquant pour signaler des zones fragiles sans rejeter la sortie.
    """
    warnings: list[str] = []

    if len(diagnostic.etapes_actuelles) < 3:
        warnings.append("Le diagnostic contient très peu d'étapes; la granularité peut être insuffisante.")

    if len(diagnostic.gains_estimes.principales_opportunites) > 5:
        warnings.append("Le nombre d'opportunités est élevé; envisager une priorisation plus stricte.")

    try:
        taux = parse_percent(diagnostic.metriques_cles.taux_manuel)
        if taux < 10:
            warnings.append("Le taux manuel est très faible; vérifier si le potentiel d'automatisation est réaliste.")
        if taux > 90:
            warnings.append("Le taux manuel est très élevé; vérifier que la source n'a pas sur-simplifié le processus.")
    except DiagnosticValidationError:
        warnings.append("Le taux manuel n'a pas pu être interprété.")

    if diagnostic.gains_estimes.mois_retour_investissement is not None:
        if diagnostic.gains_estimes.mois_retour_investissement > 24:
            warnings.append("Retour sur investissement long (>24 mois).")

    return warnings
