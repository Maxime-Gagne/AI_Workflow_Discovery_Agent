from __future__ import annotations

from schemas import TimeMetric, TimeMetricsReport


class TimeValidationError(ValueError):
    """Erreur de validation métier sur les métriques temporelles."""


def _assert_allowed_level(metric: TimeMetric) -> None:
    allowed = {"provided", "observed", "estimated", "missing"}
    if metric.confidence_level not in allowed:
        raise TimeValidationError(
            f"{metric.name}: confidence_level invalide ({metric.confidence_level})."
        )


def _assert_non_negative_numeric(metric: TimeMetric) -> None:
    if metric.confidence_level == "missing":
        return

    if not isinstance(metric.value, (int, float)):
        raise TimeValidationError(
            f"{metric.name}: value doit être numérique pour une métrique temporelle exploitable."
        )

    if metric.value < 0:
        raise TimeValidationError(
            f"{metric.name}: value négative interdite ({metric.value})."
        )


def _assert_missing_consistency(metric: TimeMetric) -> None:
    if metric.confidence_level == "missing":
        if metric.value is not None:
            raise TimeValidationError(
                f"{metric.name}: une métrique 'missing' ne doit pas avoir de valeur."
            )
    else:
        if metric.value is None:
            raise TimeValidationError(
                f"{metric.name}: une métrique non 'missing' doit avoir une valeur."
            )


def _assert_estimated_has_hypothesis(metric: TimeMetric) -> None:
    if metric.confidence_level == "estimated":
        if metric.hypothesis is None or not metric.hypothesis.strip():
            raise TimeValidationError(
                f"{metric.name}: une métrique 'estimated' doit documenter une hypothèse."
            )


def validate_time_metric_or_raise(metric: TimeMetric) -> None:
    if not isinstance(metric, TimeMetric):
        raise TimeValidationError(f"TimeMetric attendu, reçu: {type(metric)!r}")

    if not metric.name.strip():
        raise TimeValidationError("TimeMetric.name vide.")
    if not metric.unit.strip():
        raise TimeValidationError(f"{metric.name}: unit vide.")
    if not metric.source.strip():
        raise TimeValidationError(f"{metric.name}: source vide.")

    _assert_allowed_level(metric)
    _assert_missing_consistency(metric)
    _assert_estimated_has_hypothesis(metric)
    _assert_non_negative_numeric(metric)


def _assert_metric_units(report: TimeMetricsReport) -> None:
    expected_units = {
        "monthly_volume": {"cases/month", "items/month", "events/month"},
        "activity_duration_minutes": {"minutes", "minute"},
        "case_cycle_time_minutes": {"minutes", "minute"},
        "manual_time_minutes": {"minutes", "minute"},
        "waiting_time_minutes": {"minutes", "minute"},
    }

    metrics = {
        "monthly_volume": report.monthly_volume,
        "activity_duration_minutes": report.activity_duration_minutes,
        "case_cycle_time_minutes": report.case_cycle_time_minutes,
        "manual_time_minutes": report.manual_time_minutes,
        "waiting_time_minutes": report.waiting_time_minutes,
    }

    for key, metric in metrics.items():
        if metric.unit not in expected_units[key]:
            raise TimeValidationError(
                f"{metric.name}: unité inattendue pour {key} ({metric.unit})."
            )


def _assert_time_logic(report: TimeMetricsReport) -> None:
    activity = report.activity_duration_minutes
    manual = report.manual_time_minutes
    waiting = report.waiting_time_minutes
    cycle = report.case_cycle_time_minutes

    comparable = [activity, manual, waiting, cycle]
    if any(m.confidence_level == "missing" for m in comparable):
        return

    activity_value = float(activity.value)
    manual_value = float(manual.value)
    waiting_value = float(waiting.value)
    cycle_value = float(cycle.value)

    if manual_value > cycle_value:
        raise TimeValidationError(
            "manual_time_minutes ne peut pas dépasser case_cycle_time_minutes."
        )

    if activity_value > cycle_value:
        raise TimeValidationError(
            "activity_duration_minutes ne peut pas dépasser case_cycle_time_minutes."
        )

    if waiting_value > cycle_value:
        raise TimeValidationError(
            "waiting_time_minutes ne peut pas dépasser case_cycle_time_minutes."
        )


def _assert_roi_flags(report: TimeMetricsReport) -> None:
    critical = [
        report.monthly_volume,
        report.activity_duration_minutes,
        report.manual_time_minutes,
    ]
    levels = [m.confidence_level for m in critical]

    full_ok = all(level in {"provided", "observed"} for level in levels)
    partial_ok = all(level in {"provided", "observed", "estimated"} for level in levels) and not any(
        level == "missing" for level in levels
    )

    if report.can_compute_full_roi and not full_ok:
        raise TimeValidationError(
            "can_compute_full_roi=True incohérent avec les niveaux de confiance critiques."
        )

    if report.can_compute_partial_roi and not partial_ok:
        raise TimeValidationError(
            "can_compute_partial_roi=True incohérent avec les niveaux de confiance critiques."
        )

    if report.can_compute_full_roi and not report.can_compute_partial_roi:
        raise TimeValidationError(
            "Un ROI complet implique nécessairement qu'un ROI partiel soit possible."
        )


def validate_time_metrics_report_or_raise(report: TimeMetricsReport) -> None:
    if not isinstance(report, TimeMetricsReport):
        raise TimeValidationError(
            f"TimeMetricsReport attendu, reçu: {type(report)!r}"
        )

    metrics = [
        report.monthly_volume,
        report.activity_duration_minutes,
        report.case_cycle_time_minutes,
        report.manual_time_minutes,
        report.waiting_time_minutes,
    ]

    for metric in metrics:
        validate_time_metric_or_raise(metric)

    _assert_metric_units(report)
    _assert_time_logic(report)
    _assert_roi_flags(report)


def audit_time_metrics_report(report: TimeMetricsReport) -> list[str]:
    warnings: list[str] = []

    metrics = [
        report.monthly_volume,
        report.activity_duration_minutes,
        report.case_cycle_time_minutes,
        report.manual_time_minutes,
        report.waiting_time_minutes,
    ]

    estimated_metrics = [m.name for m in metrics if m.confidence_level == "estimated"]
    missing_metrics = [m.name for m in metrics if m.confidence_level == "missing"]

    if estimated_metrics:
        warnings.append(
            f"Métriques estimées présentes: {estimated_metrics}. Vérifier les hypothèses."
        )

    if missing_metrics:
        warnings.append(
            f"Métriques manquantes: {missing_metrics}. Le ROI peut être partiel ou bloqué."
        )

    if report.can_compute_partial_roi and not report.can_compute_full_roi:
        warnings.append("ROI partiel seulement: certaines métriques critiques sont estimées.")

    if not report.can_compute_partial_roi:
        warnings.append("ROI désactivé: données temporelles insuffisantes.")

    if (
        report.activity_duration_minutes.confidence_level != "missing"
        and report.waiting_time_minutes.confidence_level != "missing"
        and float(report.waiting_time_minutes.value) > float(report.activity_duration_minutes.value) * 3
    ):
        warnings.append(
            "Le temps d'attente dépasse largement le temps d'activité; possible goulot de validation/transfert."
        )

    return warnings
