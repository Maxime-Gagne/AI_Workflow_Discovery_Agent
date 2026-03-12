from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import pandas as pd


ResolutionMode = Literal["provided", "observed", "estimated", "insufficient"]


@dataclass
class ResolvedTimeContext:
    normalized_data: list[dict[str, Any]]
    case_id_col: Optional[str]
    activity_col: Optional[str]
    timestamp_col: Optional[str]
    start_timestamp_col: Optional[str]
    end_timestamp_col: Optional[str]
    mode: ResolutionMode
    requires_user_input: bool
    warnings: list[str] = field(default_factory=list)
    user_message: Optional[str] = None


CASE_ID_CANDIDATES = [
    "case_id", "ticket_id", "id", "process_id", "instance_id", "dossier_id",
    "lead_id", "candidate_id", "client_id", "opportunity_id", "acteur"
]
ACTIVITY_CANDIDATES = [
    "activity", "action", "event", "step", "status", "content", "description", "label"
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

    if activity_col is None:
        return ResolvedTimeContext(
            normalized_data=[],
            case_id_col=None,
            activity_col=None,
            timestamp_col=None,
            start_timestamp_col=None,
            end_timestamp_col=None,
            mode="insufficient",
            requires_user_input=True,
            warnings=["Aucune colonne d’activité détectée."],
            user_message=(
                "Le pipeline n’a pas trouvé de champ décrivant les actions/étapes. "
                "Ajoute une colonne de type action/activity/status ou fournis des métriques temporelles manuelles."
            ),
        )

    # Cas 100% fourni par l'utilisateur
    if timestamp_col is None and _has_minimum_user_metrics(user_metrics):
        warnings.append("Aucun horodatage détecté : bascule en mode métriques fournies par l’utilisateur.")
        return ResolvedTimeContext(
            normalized_data=df.to_dict(orient="records"),
            case_id_col=None,
            activity_col=activity_col,
            timestamp_col=None,
            start_timestamp_col=None,
            end_timestamp_col=None,
            mode="provided",
            requires_user_input=False,
            warnings=warnings,
            user_message=None,
        )

    if timestamp_col is None:
        return ResolvedTimeContext(
            normalized_data=df.to_dict(orient="records"),
            case_id_col=None,
            activity_col=activity_col,
            timestamp_col=None,
            start_timestamp_col=None,
            end_timestamp_col=None,
            mode="insufficient",
            requires_user_input=True,
            warnings=["Aucun horodatage détecté."],
            user_message=(
                "Le pipeline ne dispose pas d’horodatages exploitables. "
                "Ajoute des timestamps ou fournis au minimum : volume mensuel et temps manuel par tâche."
            ),
        )

    work = df.copy()

    # Canonicalisation activity
    if activity_col != "activity":
        work["activity"] = work[activity_col]
        activity_col = "activity"

    # Canonicalisation timestamp
    if timestamp_col != "timestamp":
        work["timestamp"] = work[timestamp_col]
        timestamp_col = "timestamp"

    # Canonicalisation start/end si présents
    if start_timestamp_col:
        if start_timestamp_col != "start_timestamp":
            work["start_timestamp"] = work[start_timestamp_col]
            start_timestamp_col = "start_timestamp"

    if end_timestamp_col:
        if end_timestamp_col != "end_timestamp":
            work["end_timestamp"] = work[end_timestamp_col]
            end_timestamp_col = "end_timestamp"

    # Case ID : si absent, on synthétise un seul cas global
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

    # SOP : pas de magie temporelle purement textuelle
    if source_key == "sop_text" and not _has_minimum_user_metrics(user_metrics):
        return ResolvedTimeContext(
            normalized_data=work.to_dict(orient="records"),
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
            start_timestamp_col=start_timestamp_col,
            end_timestamp_col=end_timestamp_col,
            mode="insufficient",
            requires_user_input=True,
            warnings=warnings + ["Mode SOP sans métriques temporelles utilisateur."],
            user_message=(
                "Le mode SOP ne peut pas produire un chiffrage crédible sans métriques temporelles manuelles. "
                "Ajoute au minimum : volume mensuel et temps manuel par tâche."
            ),
        )

    return ResolvedTimeContext(
        normalized_data=work.to_dict(orient="records"),
        case_id_col=case_id_col,
        activity_col=activity_col,
        timestamp_col=timestamp_col,
        start_timestamp_col=start_timestamp_col,
        end_timestamp_col=end_timestamp_col,
        mode=mode,
        requires_user_input=False,
        warnings=warnings,
        user_message=None,
    )
