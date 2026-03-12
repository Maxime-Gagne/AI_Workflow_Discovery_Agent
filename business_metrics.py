from __future__ import annotations
from typing import Any, Optional
from schemas import TimeMetricsReport



class BusinessMetricsError(ValueError):
    """Erreur métier sur le calcul de ROI à partir des métriques temporelles."""


def _safe_float(value: Any, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as e:
        raise BusinessMetricsError(
            f"Valeur invalide pour {field_name}: {value!r}"
        ) from e

    return out


def _require_metric_value(time_report: TimeMetricsReport, metric_name: str) -> float:
    metric = getattr(time_report, metric_name)

    if metric.value is None:
        raise BusinessMetricsError(
            f"Métrique requise absente: {metric_name}"
        )

    return _safe_float(metric.value, metric_name)


def _build_roi_mode(time_report: TimeMetricsReport) -> str:
    if time_report.can_compute_full_roi:
        return "full"
    if time_report.can_compute_partial_roi:
        return "partial"
    return "blocked"


def compute_roi_from_time_report(
    time_report: TimeMetricsReport,
    taux_horaire: float,
    automation_rate: float = 1.0,
    residual_manual_time_minutes: float = 0.0,
    implementation_cost: Optional[float] = None,
) -> dict[str, Any]:
    """
    Calcule un ROI à partir des métriques temporelles validées.

    Hypothèses:
    - monthly_volume = nombre de cas mensuels
    - manual_time_minutes = temps manuel actuel par cas
    - automation_rate = part du temps manuel réellement automatisable
    - residual_manual_time_minutes = temps manuel restant par cas après automatisation
    - taux_horaire = coût horaire chargé

    Retourne un dict simple, prêt à être injecté dans l'Analyst.
    """
    if not isinstance(time_report, TimeMetricsReport):
        raise BusinessMetricsError(
            f"TimeMetricsReport attendu, reçu: {type(time_report)!r}"
        )

    taux_horaire = _safe_float(taux_horaire, "taux_horaire")
    automation_rate = _safe_float(automation_rate, "automation_rate")
    residual_manual_time_minutes = _safe_float(
        residual_manual_time_minutes,
        "residual_manual_time_minutes",
    )

    if implementation_cost is not None:
        implementation_cost = _safe_float(implementation_cost, "implementation_cost")

    if taux_horaire <= 0:
        raise BusinessMetricsError("taux_horaire doit être > 0.")

    if not (0.0 <= automation_rate <= 1.0):
        raise BusinessMetricsError("automation_rate doit être compris entre 0 et 1.")

    if residual_manual_time_minutes < 0:
        raise BusinessMetricsError(
            "residual_manual_time_minutes ne peut pas être négatif."
        )

    if not time_report.can_compute_partial_roi:
        raise BusinessMetricsError(
            "ROI impossible: métriques temporelles insuffisantes."
        )

    monthly_volume = _require_metric_value(time_report, "monthly_volume")
    manual_time_minutes = _require_metric_value(time_report, "manual_time_minutes")

    if monthly_volume < 0:
        raise BusinessMetricsError("monthly_volume ne peut pas être négatif.")
    if manual_time_minutes < 0:
        raise BusinessMetricsError("manual_time_minutes ne peut pas être négatif.")

    automatable_minutes_per_case = manual_time_minutes * automation_rate
    saved_minutes_per_case = max(
        automatable_minutes_per_case - residual_manual_time_minutes,
        0.0,
    )

    saved_minutes_per_month = monthly_volume * saved_minutes_per_case
    saved_hours_per_month = saved_minutes_per_month / 60.0
    monthly_savings = saved_hours_per_month * taux_horaire
    annual_projection = monthly_savings * 12.0

    mois_retour_investissement: Optional[float] = None
    if implementation_cost is not None:
        if monthly_savings > 0:
            mois_retour_investissement = round(
                implementation_cost / monthly_savings,
                2,
            )
        else:
            mois_retour_investissement = None

    roi_mode = _build_roi_mode(time_report)

    confidence = {
        "roi_mode": roi_mode,
        "monthly_volume": time_report.monthly_volume.confidence_level,
        "manual_time_minutes": time_report.manual_time_minutes.confidence_level,
        "activity_duration_minutes": time_report.activity_duration_minutes.confidence_level,
        "case_cycle_time_minutes": time_report.case_cycle_time_minutes.confidence_level,
        "waiting_time_minutes": time_report.waiting_time_minutes.confidence_level,
    }

    assumptions: list[str] = [
        f"automation_rate={automation_rate:.2f}",
        f"residual_manual_time_minutes={residual_manual_time_minutes:.2f}",
        f"taux_horaire={taux_horaire:.2f}",
    ]

    if implementation_cost is not None:
        assumptions.append(f"implementation_cost={implementation_cost:.2f}")

    return {
        "heures_economisees_par_mois": round(saved_hours_per_month, 2),
        "economies_mensuelles": round(monthly_savings, 2),
        "projection_annuelle": round(annual_projection, 2),
        "mois_retour_investissement": mois_retour_investissement,
        "detail_calcul": (
            f"{monthly_volume:.2f} cas/mois × "
            f"{saved_minutes_per_case:.2f} min économisées/cas "
            f"= {saved_minutes_per_month:.2f} min/mois "
            f"= {saved_hours_per_month:.2f} h/mois"
        ),
        "assumptions": assumptions,
        "confidence": confidence,
    }

