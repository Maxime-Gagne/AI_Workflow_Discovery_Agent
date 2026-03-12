from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from schemas import DiagnosticAnalyste, WorkflowOptimise
from agents.agent_mapper import compute_transformation


class WorkflowValidationError(ValueError):
    """Erreur de validation métier sur le workflow généré par le Mapper."""


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowValidationError(f"{field_name} vide ou invalide.")


def _validate_workflow_header(workflow: WorkflowOptimise) -> None:
    _require_non_empty_str(workflow.titre_workflow, "titre_workflow")
    _require_non_empty_str(workflow.description_transformation, "description_transformation")


def _validate_nodes(workflow: WorkflowOptimise) -> None:
    if not workflow.noeuds:
        raise WorkflowValidationError("Le workflow ne contient aucun noeud.")

    ids = [n.id for n in workflow.noeuds]
    if len(ids) != len(set(ids)):
        raise WorkflowValidationError("IDs de noeuds dupliqués.")

    trigger_count = 0
    fin_count = 0

    for node in workflow.noeuds:
        _require_non_empty_str(node.id, "noeuds[].id")
        _require_non_empty_str(node.label, f"noeud {node.id}.label")
        _require_non_empty_str(node.description, f"noeud {node.id}.description")

        if len(node.label.split()) > 5:
            raise WorkflowValidationError(
                f"Noeud {node.id}: label > 5 mots ({node.label!r})."
            )

        if node.type_noeud == "trigger":
            trigger_count += 1
        elif node.type_noeud == "fin":
            fin_count += 1

    if trigger_count == 0:
        raise WorkflowValidationError("Aucun noeud trigger détecté.")
    if fin_count == 0:
        raise WorkflowValidationError("Aucun noeud fin détecté.")


def _validate_original_step_refs(
    workflow: WorkflowOptimise,
    diagnostic: Optional[DiagnosticAnalyste],
) -> None:
    if diagnostic is None:
        return

    valid_step_ids = {e.id for e in diagnostic.etapes_actuelles}

    for node in workflow.noeuds:
        if node.etape_originale is None:
            continue
        if node.etape_originale not in valid_step_ids:
            raise WorkflowValidationError(
                f"Noeud {node.id}: etape_originale inconnue ({node.etape_originale})."
            )


def _build_graph(workflow: WorkflowOptimise):
    node_ids = {n.id for n in workflow.noeuds}
    graph = defaultdict(list)
    indegree = {nid: 0 for nid in node_ids}
    outgoing = {nid: 0 for nid in node_ids}
    incoming = {nid: 0 for nid in node_ids}

    for link in workflow.liens:
        if link.de not in node_ids:
            raise WorkflowValidationError(f"Lien invalide: source inconnue {link.de}.")
        if link.vers not in node_ids:
            raise WorkflowValidationError(f"Lien invalide: cible inconnue {link.vers}.")
        if link.de == link.vers:
            raise WorkflowValidationError(f"Self-loop interdit sur {link.de}.")

        graph[link.de].append(link.vers)
        indegree[link.vers] += 1
        outgoing[link.de] += 1
        incoming[link.vers] += 1

    return graph, indegree, outgoing, incoming


def _assert_acyclic(node_ids: set[str], graph: dict[str, list[str]], indegree: dict[str, int]) -> None:
    indegree_copy = dict(indegree)
    queue = deque([nid for nid in node_ids if indegree_copy[nid] == 0])
    visited = 0

    while queue:
        current = queue.popleft()
        visited += 1
        for nxt in graph[current]:
            indegree_copy[nxt] -= 1
            if indegree_copy[nxt] == 0:
                queue.append(nxt)

    if visited != len(node_ids):
        raise WorkflowValidationError("Le workflow contient un cycle.")


def _assert_node_connectivity(workflow: WorkflowOptimise, incoming: dict[str, int], outgoing: dict[str, int]) -> None:
    for node in workflow.noeuds:
        if node.type_noeud == "trigger":
            if incoming[node.id] != 0:
                raise WorkflowValidationError(
                    f"Noeud trigger {node.id} ne doit pas avoir d'entrée."
                )
            if outgoing[node.id] == 0:
                raise WorkflowValidationError(
                    f"Noeud trigger {node.id} doit avoir au moins une sortie."
                )

        elif node.type_noeud == "fin":
            if outgoing[node.id] != 0:
                raise WorkflowValidationError(
                    f"Noeud fin {node.id} ne doit pas avoir de sortie."
                )
            if incoming[node.id] == 0:
                raise WorkflowValidationError(
                    f"Noeud fin {node.id} doit être atteignable."
                )

        else:
            if incoming[node.id] == 0:
                raise WorkflowValidationError(
                    f"Noeud {node.id} ({node.type_noeud}) sans entrée."
                )
            if outgoing[node.id] == 0:
                raise WorkflowValidationError(
                    f"Noeud {node.id} ({node.type_noeud}) sans sortie."
                )


def _assert_path_from_trigger_to_fin(workflow: WorkflowOptimise, graph: dict[str, list[str]]) -> None:
    triggers = [n.id for n in workflow.noeuds if n.type_noeud == "trigger"]
    fins = {n.id for n in workflow.noeuds if n.type_noeud == "fin"}

    reachable = set()
    queue = deque(triggers)

    while queue:
        current = queue.popleft()
        if current in reachable:
            continue
        reachable.add(current)
        for nxt in graph[current]:
            queue.append(nxt)

    if not (reachable & fins):
        raise WorkflowValidationError("Aucun chemin complet trigger -> fin détecté.")

    unreachable_fins = fins - reachable
    if unreachable_fins:
        raise WorkflowValidationError(
            f"Certains noeuds fin sont inatteignables: {sorted(unreachable_fins)}"
        )


def _validate_decision_nodes(workflow: WorkflowOptimise) -> None:
    outgoing_by_node = defaultdict(int)
    conditioned_edges = defaultdict(int)

    for link in workflow.liens:
        outgoing_by_node[link.de] += 1
        if link.condition and link.condition.strip():
            conditioned_edges[link.de] += 1

    for node in workflow.noeuds:
        if node.type_noeud != "decision":
            continue

        if outgoing_by_node[node.id] < 2:
            raise WorkflowValidationError(
                f"Noeud decision {node.id} doit avoir au moins 2 sorties."
            )

        if conditioned_edges[node.id] == 0:
            raise WorkflowValidationError(
                f"Noeud decision {node.id} doit avoir au moins une condition explicite."
            )


def _validate_transformation_consistency(
    workflow: WorkflowOptimise,
    diagnostic: Optional[DiagnosticAnalyste],
) -> None:
    if diagnostic is None:
        return

    resume = compute_transformation(diagnostic, workflow)

    all_classified = (
        set(resume.etapes_automatisees)
        | set(resume.etapes_conservees_humaines)
        | set(resume.etapes_eliminees)
    )
    all_original = {e.id for e in diagnostic.etapes_actuelles}

    if all_classified != all_original:
        raise WorkflowValidationError(
            "La transformation ne couvre pas exactement toutes les étapes du diagnostic."
        )

    if set(resume.etapes_automatisees) & set(resume.etapes_conservees_humaines):
        raise WorkflowValidationError("Conflit: étapes classées à la fois automatiques et humaines.")
    if set(resume.etapes_automatisees) & set(resume.etapes_eliminees):
        raise WorkflowValidationError("Conflit: étapes classées à la fois automatiques et éliminées.")
    if set(resume.etapes_conservees_humaines) & set(resume.etapes_eliminees):
        raise WorkflowValidationError("Conflit: étapes classées à la fois humaines et éliminées.")


def validate_workflow_or_raise(
    workflow: WorkflowOptimise,
    diagnostic: Optional[DiagnosticAnalyste] = None,
) -> None:
    if not isinstance(workflow, WorkflowOptimise):
        raise WorkflowValidationError(
            f"WorkflowOptimise attendu, reçu: {type(workflow)!r}"
        )

    _validate_workflow_header(workflow)
    _validate_nodes(workflow)
    _validate_original_step_refs(workflow, diagnostic)

    graph, indegree, outgoing, incoming = _build_graph(workflow)
    node_ids = {n.id for n in workflow.noeuds}

    _assert_acyclic(node_ids, graph, indegree)
    _assert_node_connectivity(workflow, incoming, outgoing)
    _assert_path_from_trigger_to_fin(workflow, graph)
    _validate_decision_nodes(workflow)
    _validate_transformation_consistency(workflow, diagnostic)


def audit_workflow(workflow: WorkflowOptimise, diagnostic: Optional[DiagnosticAnalyste] = None) -> list[str]:
    warnings: list[str] = []

    ia_count = sum(1 for n in workflow.noeuds if n.type_noeud == "ia")
    human_count = sum(1 for n in workflow.noeuds if n.type_noeud == "humain")
    auto_count = sum(1 for n in workflow.noeuds if n.type_noeud == "automatique")

    if ia_count == 0:
        warnings.append("Aucun noeud IA dans le workflow cible.")
    if auto_count == 0:
        warnings.append("Aucun noeud automatique dans le workflow cible.")
    if human_count > auto_count:
        warnings.append("Le workflow conserve plus de noeuds humains que de noeuds automatiques.")

    if diagnostic is not None:
        mapped_steps = {
            n.etape_originale for n in workflow.noeuds if n.etape_originale is not None
        }
        all_steps = {e.id for e in diagnostic.etapes_actuelles}
        missing = all_steps - mapped_steps
        if missing:
            warnings.append(
                f"Certaines étapes originales ne sont référencées par aucun noeud: {sorted(missing)}"
            )

    if len(workflow.noeuds) > 20:
        warnings.append("Workflow volumineux (>20 noeuds); la lisibilité du diagramme peut chuter.")

    return warnings
