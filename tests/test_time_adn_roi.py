import pytest
from schemas import TimeMetric, TimeMetricsReport
from validators.time_validators import validate_time_metrics_report_or_raise, TimeValidationError
from business_metrics import compute_roi_from_time_report, BusinessMetricsError


def make_metric(name, unit, value, level, source="test", hypothesis=None):
    return TimeMetric(
        name=name,
        unit=unit,
        value=value,
        confidence_level=level,
        source=source,
        hypothesis=hypothesis,
        notes=["test"],
    )


def make_report(level_manual="provided", level_volume="provided", level_activity="provided"):
    return TimeMetricsReport(
        monthly_volume=make_metric("monthly_volume", "cases/month", 100, level_volume, hypothesis="x" if level_volume == "estimated" else None),
        activity_duration_minutes=make_metric("activity_duration_minutes", "minutes", 10, level_activity, hypothesis="x" if level_activity == "estimated" else None),
        case_cycle_time_minutes=make_metric("case_cycle_time_minutes", "minutes", 20, "observed"),
        manual_time_minutes=make_metric("manual_time_minutes", "minutes", 10, level_manual, hypothesis="x" if level_manual == "estimated" else None),
        waiting_time_minutes=make_metric("waiting_time_minutes", "minutes", 5, "observed"),
        warnings=[],
        can_compute_full_roi=(level_volume in {"provided", "observed"} and level_activity in {"provided", "observed"} and level_manual in {"provided", "observed"}),
        can_compute_partial_roi=(level_volume != "missing" and level_activity != "missing" and level_manual != "missing"),
    )


def test_estimated_requires_hypothesis():
    with pytest.raises(ValueError):
        TimeMetric(
            name="manual_time_minutes",
            unit="minutes",
            value=12,
            confidence_level="estimated",
            source="rule_engine",
            hypothesis=None,
            notes=["x"],
        )


def test_manual_cannot_exceed_cycle():
    report = TimeMetricsReport(
        monthly_volume=make_metric("monthly_volume", "cases/month", 100, "provided"),
        activity_duration_minutes=make_metric("activity_duration_minutes", "minutes", 10, "observed"),
        case_cycle_time_minutes=make_metric("case_cycle_time_minutes", "minutes", 8, "observed"),
        manual_time_minutes=make_metric("manual_time_minutes", "minutes", 12, "provided"),
        waiting_time_minutes=make_metric("waiting_time_minutes", "minutes", 2, "observed"),
        warnings=[],
        can_compute_full_roi=True,
        can_compute_partial_roi=True,
    )
    with pytest.raises(TimeValidationError):
        validate_time_metrics_report_or_raise(report)


def test_roi_full_with_observed_or_provided():
    report = make_report()
    roi = compute_roi_from_time_report(report, taux_horaire=50.0, automation_rate=1.0, residual_manual_time_minutes=0.0)
    assert roi["confidence"]["roi_mode"] == "full"
    assert roi["economies_mensuelles"] > 0


def test_roi_partial_with_estimated():
    report = make_report(level_manual="estimated")
    roi = compute_roi_from_time_report(report, taux_horaire=50.0, automation_rate=1.0, residual_manual_time_minutes=0.0)
    assert roi["confidence"]["roi_mode"] == "partial"


def test_roi_blocked_with_missing():
    report = TimeMetricsReport(
        monthly_volume=make_metric("monthly_volume", "cases/month", None, "missing"),
        activity_duration_minutes=make_metric("activity_duration_minutes", "minutes", 10, "observed"),
        case_cycle_time_minutes=make_metric("case_cycle_time_minutes", "minutes", 20, "observed"),
        manual_time_minutes=make_metric("manual_time_minutes", "minutes", 10, "provided"),
        waiting_time_minutes=make_metric("waiting_time_minutes", "minutes", 5, "observed"),
        warnings=[],
        can_compute_full_roi=False,
        can_compute_partial_roi=False,
    )
    with pytest.raises(BusinessMetricsError):
        compute_roi_from_time_report(report, taux_horaire=50.0)
