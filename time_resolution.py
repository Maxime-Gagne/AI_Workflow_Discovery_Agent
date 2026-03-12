from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import pandas as pd


ResolutionMode = Literal["provided", "observed", "estimated", "insufficient"]

DURATION_MINUTES_CANDIDATES = [
    "duration_minutes", "duree_minutes", "temps_minutes", "manual_time_minutes"
]

DURATION_HOURS_CANDIDATES = [
    "duration_hours", "duree_heures", "temps_heures", "manual_time_hours"
]

DURATION_DAYS_CANDIDATES = [
    "duration_days", "duree_jours", "duree_moyenne_jours", "temps_jours", "jours"
]

@dataclass
class ResolvedTimeContext:
    normalized_data: list[dict[str, Any]]
    case_id_col: Optional[str]
    activity_col: Optional[str]
    timestamp_col: Optional[str]
    start_timestamp_col: Optional[str]
    end_timestamp_col: Optional[str]
    duration_minutes_col: Optional[str] = None
    mode: ResolutionMode = "insufficient"
    requires_user_input: bool = False
    can_run_workflow_analysis: bool = True
    can_run_time_quantification: bool = False
    warnings: list[str] = field(default_factory=list)
    user_message: Optional[str] = None


CASE_ID_CANDIDATES = [
    "case_id", "ticket_id", "id", "process_id", "instance_id", "dossier_id",
    "lead_id", "candidate_id", "client_id", "opportunity_id", "acteur"
]
ACTIVITY_CANDIDATES = [
    "activity", "action", "event", "step", "status",
    "content", "description", "label",
    "nom", "titre", "etape", "category"
]
TIMESTAMP_CANDIDATES = [
    "timestamp", "created_at", "created", "date", "datetime", "time", "ts"
]
START_TIMESTAMP_CANDIDATES = [
    "start_timestamp", "started_at", "timestamp_start", "date_debut", "opened_at"
]
END_TIMESTAMP_CANDIDATES = [
    "end_timestamp", "ended_at", "timestamp_end", "resolved", "date_fin", "closed_at"
]


def _to_dataframe(raw_data: Any) -> pd.DataFrame:
    if isinstance(raw_data, pd.DataFrame):
        return raw_data.copy()

    if isinstance(raw_data, list):
        return pd.DataFrame(raw_data)

    if isinstance(raw_data, dict):
        for value in raw_data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return pd.DataFrame(value)

    raise ValueError(f"Format raw_data non supporté pour la résolution temporelle: {type(raw_data)!r}")


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _has_minimum_user_metrics(user_metrics: dict[str, Any]) -> bool:
    required = ("monthly_volume", "manual_time_minutes")
    return all(user_metrics.get(k) is not None for k in required)


def resolve_time_context(
    raw_data: Any,
    user_metrics: Optional[dict[str, Any]] = None,
    source_key: str = "",
) -> ResolvedTimeContext:
    user_metrics = user_metrics or {}
    df = _to_dataframe(raw_data)
    warnings: list[str] = []

    activity_col = _pick_column(df, ACTIVITY_CANDIDATES)
    timestamp_col = _pick_column(df, TIMESTAMP_CANDIDATES)
    start_timestamp_col = _pick_column(df, START_TIMESTAMP_CANDIDATES)
    end_timestamp_col = _pick_column(df, END_TIMESTAMP_CANDIDATES)
    case_id_col = _pick_column(df, CASE_ID_CANDIDATES)

    duration_minutes_col = _pick_column(df, DURATION_MINUTES_CANDIDATES)
    duration_hours_col = _pick_column(df, DURATION_HOURS_CANDIDATES)
    duration_days_col = _pick_column(df, DURATION_DAYS_CANDIDATES)

    if activity_col is None:
        return ResolvedTimeContext(
            normalized_data=df.to_dict(orient="records"),
            case_id_col=None,
            activity_col=None,
            timestamp_col=None,
            start_timestamp_col=None,
            end_timestamp_col=None,
            duration_minutes_col=None,
            mode="insufficient",
            requires_user_input=False,
            can_run_workflow_analysis=True,
            can_run_time_quantification=False,
            warnings=[
                "Aucune colonne d’activité détectée pour la quantification temporelle."
            ],
            user_message=(
                "L’analyse structurelle du workflow continue, "
                "mais la quantification du temps est ignorée faute de champ activité explicite."
            ),
        )
    work = df.copy()

    if activity_col != "activity":
        work["activity"] = work[activity_col]
        activity_col = "activity"

    # Cas A — timestamps observables
    if timestamp_col is not None:
        if timestamp_col != "timestamp":
            work["timestamp"] = work[timestamp_col]
            timestamp_col = "timestamp"

        if start_timestamp_col and start_timestamp_col != "start_timestamp":
            work["start_timestamp"] = work[start_timestamp_col]
            start_timestamp_col = "start_timestamp"

        if end_timestamp_col and end_timestamp_col != "end_timestamp":
            work["end_timestamp"] = work[end_timestamp_col]
            end_timestamp_col = "end_timestamp"

        if case_id_col is None:
            work["case_id"] = "global_case_1"
            case_id_col = "case_id"
            warnings.append(
                "Aucun identifiant de cas détecté : un case_id global a été synthétisé. "
                "Les métriques de cycle resteront approximatives."
            )
            mode: ResolutionMode = "estimated"
        else:
            if case_id_col != "case_id":
                work["case_id"] = work[case_id_col]
                case_id_col = "case_id"
            mode = "observed"

        if start_timestamp_col is None:
            warnings.append(
                "Aucun start_timestamp détecté : durée d’activité individuelle non directement observable."
            )
            if mode == "observed":
                mode = "estimated"

        return ResolvedTimeContext(
            normalized_data=work.to_dict(orient="records"),
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
            start_timestamp_col=start_timestamp_col,
            end_timestamp_col=end_timestamp_col,
            duration_minutes_col=None,
            mode=mode,
            requires_user_input=False,
            can_run_workflow_analysis=True,
            can_run_time_quantification=True,
            warnings=warnings,
            user_message=None,
        )

    # Cas B — durée explicite en minutes
    if duration_minutes_col is not None:
        if duration_minutes_col != "duration_minutes":
            work["duration_minutes"] = pd.to_numeric(work[duration_minutes_col], errors="coerce")
            duration_minutes_col = "duration_minutes"

        if case_id_col is None:
            work["case_id"] = "global_case_1"
            case_id_col = "case_id"

        if case_id_col != "case_id":
            work["case_id"] = work[case_id_col]
            case_id_col = "case_id"

        warnings.append(
            "Aucun horodatage détecté, mais une durée explicite en minutes a été trouvée. "
            "La quantification temporelle restera partielle."
        )

        return ResolvedTimeContext(
            normalized_data=work.to_dict(orient="records"),
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=None,
            start_timestamp_col=None,
            end_timestamp_col=None,
            duration_minutes_col="duration_minutes",
            mode="estimated",
            requires_user_input=False,
            can_run_workflow_analysis=True,
            can_run_time_quantification=True,
            warnings=warnings,
            user_message=None,
        )

    # Cas C — durée explicite en heures
    if duration_hours_col is not None:
        work["duration_minutes"] = pd.to_numeric(work[duration_hours_col], errors="coerce") * 60.0
        if case_id_col is None:
            work["case_id"] = "global_case_1"
            case_id_col = "case_id"
        if case_id_col != "case_id":
            work["case_id"] = work[case_id_col]
            case_id_col = "case_id"

        warnings.append(
            "Aucun horodatage détecté, mais une durée explicite en heures a été trouvée. "
            "La quantification temporelle restera partielle."
        )

        return ResolvedTimeContext(
            normalized_data=work.to_dict(orient="records"),
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=None,
            start_timestamp_col=None,
            end_timestamp_col=None,
            duration_minutes_col="duration_minutes",
            mode="estimated",
            requires_user_input=False,
            can_run_workflow_analysis=True,
            can_run_time_quantification=True,
            warnings=warnings,
            user_message=None,
        )

    # Cas D — durée explicite en jours
    if duration_days_col is not None:
        work["duration_minutes"] = pd.to_numeric(work[duration_days_col], errors="coerce") * 24.0 * 60.0
        if case_id_col is None:
            work["case_id"] = "global_case_1"
            case_id_col = "case_id"
        if case_id_col != "case_id":
            work["case_id"] = work[case_id_col]
            case_id_col = "case_id"

        warnings.append(
            "Aucun horodatage détecté, mais une durée explicite en jours a été trouvée. "
            "La quantification temporelle restera partielle."
        )

        return ResolvedTimeContext(
            normalized_data=work.to_dict(orient="records"),
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=None,
            start_timestamp_col=None,
            end_timestamp_col=None,
            duration_minutes_col="duration_minutes",
            mode="estimated",
            requires_user_input=False,
            can_run_workflow_analysis=True,
            can_run_time_quantification=True,
            warnings=warnings,
            user_message=None,
        )

    # Cas E — aucune structure temps, mais métriques user minimales
    if _has_minimum_user_metrics(user_metrics):
        warnings.append("Aucune structure temporelle détectée : bascule en mode métriques fournies par l’utilisateur.")
        return ResolvedTimeContext(
            normalized_data=work.to_dict(orient="records"),
            case_id_col=None,
            activity_col=activity_col,
            timestamp_col=None,
            start_timestamp_col=None,
            end_timestamp_col=None,
            duration_minutes_col=None,
            mode="provided",
            requires_user_input=False,
            can_run_workflow_analysis=True,
            can_run_time_quantification=True,
            warnings=warnings,
            user_message=None,
        )

    # Cas F — workflow possible, quantification insuffisante
    return ResolvedTimeContext(
        normalized_data=work.to_dict(orient="records"),
        case_id_col=None,
        activity_col=activity_col,
        timestamp_col=None,
        start_timestamp_col=None,
        end_timestamp_col=None,
        duration_minutes_col=None,
        mode="insufficient",
        requires_user_input=False,
        can_run_workflow_analysis=True,
        can_run_time_quantification=False,
        warnings=["Aucune donnée temporelle exploitable détectée."],
        user_message=(
            "L’analyse structurelle du workflow reste possible, "
            "mais le chiffrage du gain de temps sera limité ou indisponible."
        ),
    )
