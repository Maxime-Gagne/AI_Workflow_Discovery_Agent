from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import pm4py
from pm4py.statistics.sojourn_time.pandas import get as pd_sojourn_time
import pandas as pd

from schemas import TimeMetric, TimeMetricsReport
from validators.time_validators import validate_time_metrics_report_or_raise


@dataclass
class TimeColumnMapping:
    case_id_col: str = "case_id"
    activity_col: str = "activity"
    timestamp_col: str = "timestamp"
    start_timestamp_col: Optional[str] = None
    end_timestamp_col: Optional[str] = None
    resource_col: Optional[str] = None


class AgentTempsError(ValueError):
    """Erreur métier liée à l'analyse temporelle."""


class AgentTemps:
    """
    Agent déterministe-first pour la gestion des métriques temporelles.

    Rôle:
    - normaliser les colonnes temps
    - intégrer les métriques fournies par l'utilisateur
    - calculer les métriques observées via pandas / PM4Py
    - retourner un TimeMetricsReport validé
    """

    def __init__(self, use_business_hours: bool = False) -> None:
        self.use_business_hours = use_business_hours

    def build_time_metrics_report(
        self,
        raw_data: Any,
        user_metrics: Optional[dict[str, Any]] = None,
        case_id_col: str = "case_id",
        activity_col: str = "activity",
        timestamp_col: str = "timestamp",
        start_timestamp_col: Optional[str] = None,
        end_timestamp_col: Optional[str] = None,
        resource_col: Optional[str] = None,
    ) -> TimeMetricsReport:
        mapping = TimeColumnMapping(
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
            start_timestamp_col=start_timestamp_col,
            end_timestamp_col=end_timestamp_col,
            resource_col=resource_col,
        )

        user_metrics = user_metrics or {}
        df = self._to_dataframe(raw_data)
        df = self._normalize_dataframe(df, mapping)

        observed_metrics: dict[str, TimeMetric] = {}
        warnings: list[str] = []

        # 1) Volume mensuel
        observed_metrics["monthly_volume"] = self._build_monthly_volume_metric(df, mapping)

        # 2) Métriques PM4Py
        pm_metrics = self._compute_pm4py_metrics(df, mapping)
        warnings.extend(pm_metrics.get("warnings", []))

        observed_metrics["activity_duration_minutes"] = self._metric_from_pm4py_value(
            name="activity_duration_minutes",
            unit="minutes",
            value=pm_metrics.get("activity_duration_median_minutes"),
            notes=[
                (
                    "Durée médiane issue de PM4Py sojourn_time."
                    if pm_metrics.get("activity_duration_median_minutes") is not None
                    else "Durée d'activité non calculable via PM4Py."
                )
            ],
        )

        observed_metrics["case_cycle_time_minutes"] = self._metric_from_pm4py_value(
            name="case_cycle_time_minutes",
            unit="minutes",
            value=pm_metrics.get("case_cycle_time_median_minutes"),
            notes=[
                (
                    "Temps de cycle médian par cas."
                    if pm_metrics.get("case_cycle_time_median_minutes") is not None
                    else "Temps de cycle non calculable."
                )
            ],
        )

        observed_metrics["waiting_time_minutes"] = self._metric_from_pm4py_value(
            name="waiting_time_minutes",
            unit="minutes",
            value=pm_metrics.get("waiting_time_median_minutes"),
            notes=[
                (
                    "Temps d'attente médian entre événements successifs."
                    if pm_metrics.get("waiting_time_median_minutes") is not None
                    else "Temps d'attente non calculable."
                )
            ],
        )

        # 3) Temps manuel
        observed_metrics["manual_time_minutes"] = self._build_manual_time_metric(
            observed_metrics["activity_duration_minutes"],
            user_metrics=user_metrics,
        )

        # 4) Priorité aux métriques fournies par l'utilisateur
        merged_metrics = self._merge_with_user_metrics(
            observed_metrics=observed_metrics,
            user_metrics=user_metrics,
        )

        # 5) Assemblage + validation finale
        report = self._assemble_report(merged_metrics, warnings)
        validate_time_metrics_report_or_raise(report)

        return report

    def _to_dataframe(self, raw_data: Any) -> pd.DataFrame:
        if isinstance(raw_data, pd.DataFrame):
            return raw_data.copy()

        if isinstance(raw_data, list):
            return pd.DataFrame(raw_data)

        raise AgentTempsError(
            f"Format raw_data non supporté: {type(raw_data)!r}. "
            "Attendu: pandas.DataFrame ou list[dict]."
        )

    def _normalize_dataframe(
        self,
        df: pd.DataFrame,
        mapping: TimeColumnMapping,
    ) -> pd.DataFrame:
        required = [mapping.case_id_col, mapping.activity_col, mapping.timestamp_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise AgentTempsError(f"Colonnes requises manquantes: {missing}")

        df = df.copy()
        df[mapping.timestamp_col] = pd.to_datetime(df[mapping.timestamp_col], errors="coerce")

        if mapping.start_timestamp_col and mapping.start_timestamp_col in df.columns:
            df[mapping.start_timestamp_col] = pd.to_datetime(
                df[mapping.start_timestamp_col], errors="coerce"
            )

        if mapping.end_timestamp_col and mapping.end_timestamp_col in df.columns:
            df[mapping.end_timestamp_col] = pd.to_datetime(
                df[mapping.end_timestamp_col], errors="coerce"
            )

        df = df.dropna(subset=[mapping.case_id_col, mapping.activity_col, mapping.timestamp_col])
        df = df.sort_values([mapping.case_id_col, mapping.timestamp_col]).reset_index(drop=True)
        return df

    def _build_monthly_volume_metric(
        self,
        df: pd.DataFrame,
        mapping: TimeColumnMapping,
    ) -> TimeMetric:
        ts = df[mapping.timestamp_col]
        if ts.empty:
            return self._missing_metric(
                name="monthly_volume",
                unit="cases/month",
                source="agent_temps",
                notes=["Aucun timestamp exploitable."]
            )

        start_ts = ts.min()
        end_ts = ts.max()
        unique_cases = df[mapping.case_id_col].nunique()

        if pd.isna(start_ts) or pd.isna(end_ts) or unique_cases == 0:
            return self._missing_metric(
                name="monthly_volume",
                unit="cases/month",
                source="agent_temps",
                notes=["Fenêtre temporelle ou volume inexploitable."]
            )

        days = max((end_ts - start_ts).days + 1, 1)
        monthly_value = float(unique_cases / days * 30.0)

        if days >= 28:
            return TimeMetric(
                name="monthly_volume",
                unit="cases/month",
                value=round(monthly_value, 2),
                confidence_level="estimated",
                source="rule_engine",
                hypothesis=f"Projection mensuelle basée sur {days} jours observés.",
                notes=[f"{unique_cases} cas uniques observés sur {days} jours."],
            )

        return TimeMetric(
            name="monthly_volume",
            unit="cases/month",
            value=round(monthly_value, 2),
            confidence_level="estimated",
            source="rule_engine",
            hypothesis=f"Projection mensuelle faible fenêtre basée sur {days} jours observés.",
            notes=["Fenêtre temporelle courte: prudence sur la projection."],
        )

    def _build_activity_duration_metric(
        self,
        df: pd.DataFrame,
        mapping: TimeColumnMapping,
    ) -> tuple[TimeMetric, list[str]]:
        warnings: list[str] = []

        # Cas fort: start/end explicites
        if mapping.start_timestamp_col and mapping.start_timestamp_col in df.columns:
            valid = df.dropna(subset=[mapping.start_timestamp_col, mapping.timestamp_col]).copy()
            if not valid.empty:
                durations = (
                    valid[mapping.timestamp_col] - valid[mapping.start_timestamp_col]
                ).dt.total_seconds() / 60.0
                durations = durations[durations >= 0]
                if not durations.empty:
                    return (
                        TimeMetric(
                            name="activity_duration_minutes",
                            unit="minutes",
                            value=round(float(durations.median()), 2),
                            confidence_level="observed",
                            source="event_log",
                            notes=["Durée médiane calculée à partir de start_timestamp et timestamp."],
                        ),
                        warnings,
                    )

        # Fallback prudent: end_timestamp - start_timestamp dédié
        if mapping.end_timestamp_col and mapping.end_timestamp_col in df.columns and mapping.start_timestamp_col:
            valid = df.dropna(subset=[mapping.start_timestamp_col, mapping.end_timestamp_col]).copy()
            if not valid.empty:
                durations = (
                    valid[mapping.end_timestamp_col] - valid[mapping.start_timestamp_col]
                ).dt.total_seconds() / 60.0
                durations = durations[durations >= 0]
                if not durations.empty:
                    return (
                        TimeMetric(
                            name="activity_duration_minutes",
                            unit="minutes",
                            value=round(float(durations.median()), 2),
                            confidence_level="observed",
                            source="event_log",
                            notes=["Durée médiane calculée à partir de start_timestamp et end_timestamp."],
                        ),
                        warnings,
                    )

        warnings.append(
            "Impossible de mesurer proprement activity_duration_minutes sans start/end explicites."
        )
        return (
            self._missing_metric(
                name="activity_duration_minutes",
                unit="minutes",
                source="agent_temps",
                notes=["Aucune paire start/end exploitable."],
            ),
            warnings,
        )

    def _build_case_cycle_time_metric(
        self,
        df: pd.DataFrame,
        mapping: TimeColumnMapping,
    ) -> tuple[TimeMetric, list[str]]:
        warnings: list[str] = []

        grouped = df.groupby(mapping.case_id_col)[mapping.timestamp_col]
        starts = grouped.min()
        ends = grouped.max()

        if starts.empty or ends.empty:
            return (
                self._missing_metric(
                    name="case_cycle_time_minutes",
                    unit="minutes",
                    source="agent_temps",
                    notes=["Impossible de calculer les bornes des cas."],
                ),
                warnings,
            )

        durations = (ends - starts).dt.total_seconds() / 60.0
        durations = durations[durations >= 0]

        if durations.empty:
            return (
                self._missing_metric(
                    name="case_cycle_time_minutes",
                    unit="minutes",
                    source="agent_temps",
                    notes=["Aucune durée de cycle positive disponible."],
                ),
                warnings,
            )

        return (
            TimeMetric(
                name="case_cycle_time_minutes",
                unit="minutes",
                value=round(float(durations.median()), 2),
                confidence_level="observed",
                source="event_log",
                notes=["Temps de cycle médian calculé par cas."],
            ),
            warnings,
        )

    def _build_waiting_time_metric(
        self,
        df: pd.DataFrame,
        mapping: TimeColumnMapping,
    ) -> tuple[TimeMetric, list[str]]:
        warnings: list[str] = []

        work = df[[mapping.case_id_col, mapping.timestamp_col]].copy()
        work["prev_timestamp"] = work.groupby(mapping.case_id_col)[mapping.timestamp_col].shift(1)
        waits = (work[mapping.timestamp_col] - work["prev_timestamp"]).dt.total_seconds() / 60.0
        waits = waits.dropna()
        waits = waits[waits >= 0]

        if waits.empty:
            return (
                self._missing_metric(
                    name="waiting_time_minutes",
                    unit="minutes",
                    source="agent_temps",
                    notes=["Aucun délai inter-événements calculable."],
                ),
                warnings,
            )

        return (
            TimeMetric(
                name="waiting_time_minutes",
                unit="minutes",
                value=round(float(waits.median()), 2),
                confidence_level="observed",
                source="event_log",
                notes=["Temps d'attente médian entre événements successifs d'un même cas."],
            ),
            warnings,
        )

    def _merge_with_user_metrics(
        self,
        observed_metrics: dict[str, TimeMetric],
        user_metrics: dict[str, Any],
    ) -> dict[str, TimeMetric]:
        merged = dict(observed_metrics)

        # Priorité: provided > observed > estimated > missing
        for metric_name, current in list(merged.items()):
            if metric_name in user_metrics:
                merged[metric_name] = TimeMetric(
                    name=metric_name,
                    unit=current.unit,
                    value=float(user_metrics[metric_name]),
                    confidence_level="provided",
                    source="user_input",
                    notes=["Valeur prioritaire fournie par l'utilisateur."],
                )

        return merged

    def _missing_metric(
        self,
        name: str,
        unit: str,
        source: str,
        notes: list[str],
    ) -> TimeMetric:
        return TimeMetric(
            name=name,
            unit=unit,
            value=None,
            confidence_level="missing",
            source=source,
            hypothesis=None,
            notes=notes,
        )

    def _assemble_report(
        self,
        metrics: dict[str, TimeMetric],
        warnings: list[str],
    ) -> TimeMetricsReport:
        critical = [
            metrics["monthly_volume"],
            metrics["activity_duration_minutes"],
            metrics["manual_time_minutes"],
        ]
        levels = [m.confidence_level for m in critical]

        can_compute_full_roi = all(level in {"provided", "observed"} for level in levels)
        can_compute_partial_roi = (
            all(level in {"provided", "observed", "estimated"} for level in levels)
            and not any(level == "missing" for level in levels)
        )

        return TimeMetricsReport(
            monthly_volume=metrics["monthly_volume"],
            activity_duration_minutes=metrics["activity_duration_minutes"],
            case_cycle_time_minutes=metrics["case_cycle_time_minutes"],
            manual_time_minutes=metrics["manual_time_minutes"],
            waiting_time_minutes=metrics["waiting_time_minutes"],
            warnings=warnings,
            can_compute_full_roi=can_compute_full_roi,
            can_compute_partial_roi=can_compute_partial_roi,
        )
# ==============================================
# Pm4PY
# ==============================================

    def _compute_pm4py_metrics(
        self,
        df: pd.DataFrame,
        mapping: TimeColumnMapping,
    ) -> dict[str, Any]:
        """
        Calcule des métriques process mining déterministes via PM4Py.

        Retourne un dict brut intermédiaire, pas encore des TimeMetric.
        """
        result: dict[str, Any] = {
            "activity_duration_median_minutes": None,
            "case_cycle_time_median_minutes": None,
            "waiting_time_median_minutes": None,
            "variants_count": None,
            "top_variant_share_pct": None,
            "dfg_edges_count": None,
            "warnings": [],
        }

        work = df.copy()

        try:
            # PM4Py préfère les DataFrames pandas formatés pour le process mining
            # avec case_id / activity / timestamp / start_timestamp si présent.
            start_key = mapping.start_timestamp_col or mapping.timestamp_col

            work = pm4py.format_dataframe(
                work,
                case_id=mapping.case_id_col,
                activity_key=mapping.activity_col,
                timestamp_key=mapping.timestamp_col,
                start_timestamp_key=start_key,
            )
        except Exception as e:
            result["warnings"].append(f"PM4Py format_dataframe failed: {e}")
            return result

        # 1) Sojourn / durée d'activité si start_timestamp exploitable
        if mapping.start_timestamp_col and mapping.start_timestamp_col in work.columns:
            try:
                params = {
                    pd_sojourn_time.Parameters.ACTIVITY_KEY: mapping.activity_col,
                    pd_sojourn_time.Parameters.START_TIMESTAMP_KEY: mapping.start_timestamp_col,
                    pd_sojourn_time.Parameters.TIMESTAMP_KEY: mapping.timestamp_col,
                    pd_sojourn_time.Parameters.BUSINESS_HOURS: self.use_business_hours,
                }

                sojourn = pd_sojourn_time.apply(work, parameters=params)

                # sojourn est généralement un dict {activity: seconds}
                if sojourn:
                    values_minutes = [
                        float(v) / 60.0
                        for v in sojourn.values()
                        if v is not None and float(v) >= 0
                    ]
                    if values_minutes:
                        result["activity_duration_median_minutes"] = round(
                            float(pd.Series(values_minutes).median()), 2
                        )
            except Exception as e:
                result["warnings"].append(f"PM4Py sojourn_time failed: {e}")
        else:
            result["warnings"].append(
                "PM4Py sojourn_time skipped: start_timestamp absent."
            )

        # 2) Temps de cycle par cas (déterministe)
        try:
            grouped = work.groupby(mapping.case_id_col)[mapping.timestamp_col]
            starts = grouped.min()
            ends = grouped.max()
            cycle_minutes = ((ends - starts).dt.total_seconds() / 60.0)
            cycle_minutes = cycle_minutes[cycle_minutes >= 0]
            if not cycle_minutes.empty:
                result["case_cycle_time_median_minutes"] = round(
                    float(cycle_minutes.median()), 2
                )
        except Exception as e:
            result["warnings"].append(f"Cycle time computation failed: {e}")

        # 3) Temps d’attente inter-événements par cas
        try:
            temp = work[[mapping.case_id_col, mapping.timestamp_col]].copy()
            temp["prev_timestamp"] = temp.groupby(mapping.case_id_col)[mapping.timestamp_col].shift(1)
            waits = (
                temp[mapping.timestamp_col] - temp["prev_timestamp"]
            ).dt.total_seconds() / 60.0
            waits = waits.dropna()
            waits = waits[waits >= 0]
            if not waits.empty:
                result["waiting_time_median_minutes"] = round(float(waits.median()), 2)
        except Exception as e:
            result["warnings"].append(f"Waiting time computation failed: {e}")

        # 4) Variants
        try:
            variants_dict = pm4py.get_variants_as_tuples(work)
            if variants_dict:
                counts = sorted(
                    [len(case_list) for case_list in variants_dict.values()],
                    reverse=True,
                )
                total_cases = sum(counts)
                result["variants_count"] = len(variants_dict)
                if total_cases > 0:
                    result["top_variant_share_pct"] = round((counts[0] / total_cases) * 100.0, 2)
        except Exception as e:
            result["warnings"].append(f"Variant discovery failed: {e}")

        # 5) DFG simple
        try:
            dfg, start_activities, end_activities = pm4py.discover_dfg(work)
            result["dfg_edges_count"] = len(dfg) if dfg is not None else 0
        except Exception as e:
            result["warnings"].append(f"DFG discovery failed: {e}")

        return result

    def _metric_from_pm4py_value(
        self,
        name: str,
        unit: str,
        value: Optional[float],
        notes: list[str],
    ) -> TimeMetric:
        if value is None:
            return self._missing_metric(
                name=name,
                unit=unit,
                source="pm4py",
                notes=notes,
            )

        return TimeMetric(
            name=name,
            unit=unit,
            value=round(float(value), 2),
            confidence_level="observed",
            source="pm4py",
            notes=notes,
        )

    def _build_manual_time_metric(
        self,
        activity_duration_metric: TimeMetric,
        user_metrics: dict[str, Any],
    ) -> TimeMetric:
        if "manual_time_minutes" in user_metrics and user_metrics["manual_time_minutes"] is not None:
            try:
                value = float(user_metrics["manual_time_minutes"])
            except (TypeError, ValueError) as e:
                raise AgentTempsError(
                    f"Valeur utilisateur invalide pour manual_time_minutes: "
                    f"{user_metrics['manual_time_minutes']!r}"
                ) from e

            if value < 0:
                raise AgentTempsError("manual_time_minutes fourni par l'utilisateur ne peut pas être négatif.")

            return TimeMetric(
                name="manual_time_minutes",
                unit="minutes",
                value=value,
                confidence_level="provided",
                source="user_input",
                hypothesis=None,
                notes=["Valeur fournie explicitement par l'utilisateur."],
            )

        if activity_duration_metric.confidence_level == "observed":
            if activity_duration_metric.value is None:
                raise AgentTempsError(
                    "activity_duration_minutes observée mais sans valeur exploitable."
                )

            value = float(activity_duration_metric.value)
            if value < 0:
                raise AgentTempsError("activity_duration_minutes observée négative.")

            return TimeMetric(
                name="manual_time_minutes",
                unit="minutes",
                value=value,
                confidence_level="estimated",
                source="rule_engine",
                hypothesis=(
                    "Proxy initial basé sur la durée observée d'activité, "
                    "en attendant une vraie métrique manuelle."
                ),
                notes=["À remplacer dès qu'une vraie métrique manuelle est disponible."],
            )

        return self._missing_metric(
            name="manual_time_minutes",
            unit="minutes",
            source="agent_temps",
            notes=["Aucune métrique manuelle fiable fournie ou calculable."],
        )

    def build_time_metrics_from_user_input(
        self,
        user_metrics: dict[str, Any],
    ) -> TimeMetricsReport:
        if not user_metrics:
            raise AgentTempsError("user_metrics vide pour build_time_metrics_from_user_input().")

        def _provided(name: str, unit: str) -> TimeMetric:
            raw = user_metrics.get(name)
            if raw is None:
                return self._missing_metric(
                    name=name,
                    unit=unit,
                    source="user_input",
                    notes=["Métrique non fournie par l'utilisateur."],
                )

            try:
                value = float(raw)
            except (TypeError, ValueError) as e:
                raise AgentTempsError(f"Valeur utilisateur invalide pour {name}: {raw!r}") from e

            if value < 0:
                raise AgentTempsError(f"Valeur négative interdite pour {name}: {value}")

            return TimeMetric(
                name=name,
                unit=unit,
                value=value,
                confidence_level="provided",
                source="user_input",
                hypothesis=None,
                notes=["Valeur fournie explicitement par l'utilisateur."],
            )

        monthly_volume = _provided("monthly_volume", "cases/month")
        manual_time = _provided("manual_time_minutes", "minutes")
        waiting_time = _provided("waiting_time_minutes", "minutes")

        # Proxy accepté pour rester utilisable en portfolio
        if manual_time.confidence_level == "provided":
            activity_duration = TimeMetric(
                name="activity_duration_minutes",
                unit="minutes",
                value=float(manual_time.value),
                confidence_level="estimated",
                source="rule_engine",
                hypothesis="Proxy initial basé sur le temps manuel fourni par l'utilisateur.",
                notes=["À remplacer par un start/end timestamp si disponible."],
            )
        else:
            activity_duration = self._missing_metric(
                name="activity_duration_minutes",
                unit="minutes",
                source="agent_temps",
                notes=["Impossible de reconstruire activity_duration_minutes sans donnée source."],
            )

        if manual_time.confidence_level == "provided" and waiting_time.confidence_level == "provided":
            cycle_value = float(manual_time.value) + float(waiting_time.value)
            case_cycle_time = TimeMetric(
                name="case_cycle_time_minutes",
                unit="minutes",
                value=cycle_value,
                confidence_level="estimated",
                source="rule_engine",
                hypothesis="Somme du temps manuel fourni et du temps d'attente fourni.",
                notes=["Approximation contrôlée à partir des métriques utilisateur."],
            )
        else:
            case_cycle_time = self._missing_metric(
                name="case_cycle_time_minutes",
                unit="minutes",
                source="agent_temps",
                notes=["Cycle time non reconstructible à partir des seules métriques fournies."],
            )

        critical = [monthly_volume, activity_duration, manual_time]
        levels = [m.confidence_level for m in critical]

        can_compute_full_roi = all(level in {"provided", "observed"} for level in levels)
        can_compute_partial_roi = (
            all(level in {"provided", "observed", "estimated"} for level in levels)
            and not any(level == "missing" for level in levels)
        )

        report = TimeMetricsReport(
            monthly_volume=monthly_volume,
            activity_duration_minutes=activity_duration,
            case_cycle_time_minutes=case_cycle_time,
            manual_time_minutes=manual_time,
            waiting_time_minutes=waiting_time,
            warnings=["Rapport temporel construit à partir de métriques utilisateur."],
            can_compute_full_roi=can_compute_full_roi,
            can_compute_partial_roi=can_compute_partial_roi,
        )
        validate_time_metrics_report_or_raise(report)
        return report
