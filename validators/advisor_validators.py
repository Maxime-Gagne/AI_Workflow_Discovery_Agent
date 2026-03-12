from __future__ import annotations

from typing import Any, Iterable, Optional

from schemas import RapportAdvisor, WorkflowOptimise


class AdvisorValidationError(ValueError):
    """Erreur de validation métier sur la sortie Advisor."""


def _require_non_empty_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise AdvisorValidationError(f"{field_name} vide ou invalide.")


def _normalize_catalog_names(tool_catalog: Iterable[Any]) -> set[str]:
    """
    Accepte soit :
    - un set/list de noms d'outils
    - une liste de dicts issus de tools_catalog.json avec clé 'name'
    """
    names: set[str] = set()

    for item in tool_catalog:
        if isinstance(item, str):
            name = item.strip()
            if name:
                names.add(name)
        elif isinstance(item, dict):
            raw_name = item.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                names.add(raw_name.strip())
        else:
            raise AdvisorValidationError(
                f"Entrée de catalogue non supportée: {type(item)!r}"
            )

    if not names:
        raise AdvisorValidationError("Le catalogue d'outils est vide ou illisible.")

    return names


def _validate_workflow_refs(report: RapportAdvisor, workflow: WorkflowOptimise) -> None:
    node_ids = {n.id for n in workflow.noeuds}
    node_labels = {n.label for n in workflow.noeuds}

    if not report.recommandations:
        raise AdvisorValidationError("Aucune recommandation retournée par l'Advisor.")

    seen_node_ids: set[str] = set()

    for rec in report.recommandations:
        _require_non_empty_str(rec.noeud_id, "recommandations[].noeud_id")
        _require_non_empty_str(rec.noeud_label, f"recommandation {rec.noeud_id}.noeud_label")

        if rec.noeud_id not in node_ids:
            raise AdvisorValidationError(
                f"Recommandation liée à un noeud inconnu: {rec.noeud_id}"
            )

        if rec.noeud_label not in node_labels:
            raise AdvisorValidationError(
                f"noeud_label inconnu ou incohérent avec le workflow: {rec.noeud_label}"
            )

        if rec.noeud_id in seen_node_ids:
            raise AdvisorValidationError(
                f"Recommandations dupliquées pour le noeud {rec.noeud_id}"
            )
        seen_node_ids.add(rec.noeud_id)


def _validate_tools_against_catalog(
    report: RapportAdvisor,
    catalog_names: set[str],
) -> None:
    for rec in report.recommandations:
        if not rec.outils:
            raise AdvisorValidationError(
                f"Aucun outil recommandé pour le noeud {rec.noeud_id}"
            )

        seen_tools_for_node: set[str] = set()

        for outil in rec.outils:
            _require_non_empty_str(outil.nom, f"{rec.noeud_id}.outils[].nom")
            _require_non_empty_str(outil.justification, f"{rec.noeud_id}.outils[{outil.nom}].justification")

            if outil.nom not in catalog_names:
                raise AdvisorValidationError(
                    f"Outil hors catalogue détecté: {outil.nom}"
                )

            if outil.nom in seen_tools_for_node:
                raise AdvisorValidationError(
                    f"Outil dupliqué pour le noeud {rec.noeud_id}: {outil.nom}"
                )
            seen_tools_for_node.add(outil.nom)


def _validate_stack_fields(report: RapportAdvisor, catalog_names: set[str]) -> None:
    if not report.stack_minimale:
        raise AdvisorValidationError("stack_minimale vide.")
    if not report.stack_complete:
        raise AdvisorValidationError("stack_complete vide.")

    for name in report.stack_minimale:
        _require_non_empty_str(name, "stack_minimale[]")
        if name not in catalog_names:
            raise AdvisorValidationError(f"Outil hors catalogue dans stack_minimale: {name}")

    for name in report.stack_complete:
        _require_non_empty_str(name, "stack_complete[]")
        if name not in catalog_names:
            raise AdvisorValidationError(f"Outil hors catalogue dans stack_complete: {name}")

    if len(report.stack_minimale) != len(set(report.stack_minimale)):
        raise AdvisorValidationError("Doublons détectés dans stack_minimale.")

    if len(report.stack_complete) != len(set(report.stack_complete)):
        raise AdvisorValidationError("Doublons détectés dans stack_complete.")

    if not set(report.stack_minimale).issubset(set(report.stack_complete)):
        raise AdvisorValidationError(
            "stack_minimale doit être incluse dans stack_complete."
        )


def _validate_implementation_guidance(report: RapportAdvisor) -> None:
    _require_non_empty_str(report.conseil_implementation, "conseil_implementation")


def _validate_recommendation_coverage(
    report: RapportAdvisor,
    workflow: WorkflowOptimise,
) -> None:
    """
    Version volontairement stricte :
    - on exige une recommandation pour chaque noeud non trivial du workflow
    - on ignore trigger et fin
    """
    expected_ids = {
        n.id for n in workflow.noeuds
        if n.type_noeud not in {"trigger", "fin"}
    }
    got_ids = {rec.noeud_id for rec in report.recommandations}

    missing = expected_ids - got_ids
    if missing:
        raise AdvisorValidationError(
            f"Certains noeuds du workflow n'ont aucune recommandation: {sorted(missing)}"
        )


def validate_advisor_report_or_raise(
    report: RapportAdvisor,
    workflow: WorkflowOptimise,
    tool_catalog: Iterable[Any],
) -> None:
    """
    Validation métier complète de la sortie Advisor.

    Vérifie :
    - cohérence des références vers le workflow
    - conformité des outils au catalogue autorisé
    - cohérence de stack_minimale / stack_complete
    - présence de recommandations pour les noeuds utiles
    """
    if not isinstance(report, RapportAdvisor):
        raise AdvisorValidationError(
            f"RapportAdvisor attendu, reçu: {type(report)!r}"
        )

    if not isinstance(workflow, WorkflowOptimise):
        raise AdvisorValidationError(
            f"WorkflowOptimise attendu, reçu: {type(workflow)!r}"
        )

    catalog_names = _normalize_catalog_names(tool_catalog)

    _validate_workflow_refs(report, workflow)
    _validate_tools_against_catalog(report, catalog_names)
    _validate_stack_fields(report, catalog_names)
    _validate_implementation_guidance(report)
    _validate_recommendation_coverage(report, workflow)


def audit_advisor_report(
    report: RapportAdvisor,
    workflow: WorkflowOptimise,
    tool_catalog: Optional[Iterable[Any]] = None,
) -> list[str]:
    """
    Audit non bloquant : signale les zones fragiles sans rejeter la sortie.
    """
    warnings: list[str] = []

    if len(report.stack_complete) > 8:
        warnings.append("stack_complete volumineuse (>8 outils) : risque de sur-architecture.")

    if len(report.stack_minimale) == len(report.stack_complete):
        warnings.append("stack_minimale et stack_complete sont identiques : peu de différenciation de rollout.")

    # Détection simple de redondance conceptuelle
    orchestration_keywords = ("hubspot", "notion", "jira", "linear", "airtable", "monday", "clickup")
    orchestration_hits = [
        name for name in report.stack_complete
        if any(k in name.lower() for k in orchestration_keywords)
    ]
    if len(orchestration_hits) > 2:
        warnings.append(
            f"Plusieurs outils centraux potentiels détectés dans la stack_complete : {orchestration_hits}"
        )

    if tool_catalog is not None:
        catalog_names = _normalize_catalog_names(tool_catalog)
        unknown_in_complete = [x for x in report.stack_complete if x not in catalog_names]
        if unknown_in_complete:
            warnings.append(
                f"Outils non reconnus dans stack_complete : {unknown_in_complete}"
            )

    non_terminal_nodes = [
        n for n in workflow.noeuds if n.type_noeud not in {"trigger", "fin"}
    ]
    if len(report.recommandations) < max(1, len(non_terminal_nodes) // 2):
        warnings.append(
            "Couverture de recommandations faible par rapport au nombre de noeuds du workflow."
        )

    if len(report.conseil_implementation.split()) < 12:
        warnings.append("conseil_implementation très court : justification potentiellement insuffisante.")

    return warnings
