import os
import time
import instructor
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import DiagnosticAnalyste, WorkflowOptimise, ResumeTransformation, NoeudWorkflow, LienWorkflow

load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")

SYSTEM_PROMPT = SYSTEM_PROMPT = """Tu es un architecte logiciel expert en modélisation de processus.
Transforme le diagnostic fourni en un workflow technique optimisé (graphe strict).

CONTRAINTES TOPOLOGIQUES OBLIGATOIRES :
1. Le workflow doit contenir AU MOINS un noeud de type 'trigger'.
2. Le workflow doit contenir AU MOINS un noeud de type 'fin'.
3. Chaque chemin logique doit terminer sur un noeud 'fin'.
4. Aucun noeud intermédiaire ne doit rester sans sortie.
5. Les noeuds de décision doivent avoir au moins deux sorties conditionnelles si nécessaire.

RÈGLE CRITIQUE (transformation_resume) :
Chaque étape originale doit être classée dans EXACTEMENT UNE des listes :
- etapes_automatisees
- etapes_conservees_humaines
- etapes_eliminees

Priorité de classification si split :
etapes_conservees_humaines > etapes_automatisees.
"""

def compute_transformation(diagnostic, workflow):
    all_etapes = {e.id for e in diagnostic.etapes_actuelles}

    automatisees, humaines = set(), set()
    for noeud in workflow.noeuds:
        if not noeud.etape_originale:
            continue
        if noeud.type_noeud in ("automatique", "trigger"):
            automatisees.add(noeud.etape_originale)
        elif noeud.type_noeud == "humain":
            humaines.add(noeud.etape_originale)

    automatisees -= humaines
    eliminees = all_etapes - automatisees - humaines

    return ResumeTransformation(
        etapes_eliminees=sorted(eliminees),
        etapes_automatisees=sorted(automatisees),
        etapes_conservees_humaines=sorted(humaines),
    )

def _unique_node_id(existing_ids: set[str], prefix: str) -> str:
    i = 1
    while f"{prefix}_{i}" in existing_ids:
        i += 1
    return f"{prefix}_{i}"


def ensure_workflow_boundaries(workflow: WorkflowOptimise) -> WorkflowOptimise:
    """
    Répare de manière déterministe les bornes minimales du graphe :
    - garantit au moins un trigger
    - garantit au moins un fin
    - relie les noeuds feuilles non-fin à un fin
    - relie les noeuds racines non-trigger à un trigger
    """
    wf = workflow.model_copy(deep=True)

    node_ids = {n.id for n in wf.noeuds}
    incoming = {n.id: 0 for n in wf.noeuds}
    outgoing = {n.id: 0 for n in wf.noeuds}

    for link in wf.liens:
        if link.de in outgoing:
            outgoing[link.de] += 1
        if link.vers in incoming:
            incoming[link.vers] += 1

    triggers = [n for n in wf.noeuds if n.type_noeud == "trigger"]
    fins = [n for n in wf.noeuds if n.type_noeud == "fin"]

    # 1) Garantir un trigger
    if not triggers:
        trigger_id = _unique_node_id(node_ids, "trigger")
        node_ids.add(trigger_id)

        trigger_node = NoeudWorkflow(
            id=trigger_id,
            label="Start",
            type_noeud="trigger",
            description="Déclencheur automatique ajouté pour garantir une entrée valide.",
            etape_originale=None,
        )
        wf.noeuds.append(trigger_node)

        roots = [n for n in wf.noeuds if n.id != trigger_id and incoming.get(n.id, 0) == 0]
        if not roots and wf.noeuds:
            roots = [n for n in wf.noeuds if n.id != trigger_id][:1]

        for root in roots:
            wf.liens.append(LienWorkflow(de=trigger_id, vers=root.id, condition=None))

    # Recalcul après éventuel ajout du trigger
    incoming = {n.id: 0 for n in wf.noeuds}
    outgoing = {n.id: 0 for n in wf.noeuds}
    for link in wf.liens:
        if link.de in outgoing:
            outgoing[link.de] += 1
        if link.vers in incoming:
            incoming[link.vers] += 1

    fins = [n for n in wf.noeuds if n.type_noeud == "fin"]

    # 2) Garantir un fin
    if not fins:
        fin_id = _unique_node_id(node_ids, "fin")
        node_ids.add(fin_id)

        fin_node = NoeudWorkflow(
            id=fin_id,
            label="End",
            type_noeud="fin",
            description="Terminaison ajoutée automatiquement pour garantir une sortie valide.",
            etape_originale=None,
        )
        wf.noeuds.append(fin_node)
        fins = [fin_node]

    # 3) Relier les feuilles non-fin vers un fin
    fin_id = fins[0].id
    incoming = {n.id: 0 for n in wf.noeuds}
    outgoing = {n.id: 0 for n in wf.noeuds}
    for link in wf.liens:
        if link.de in outgoing:
            outgoing[link.de] += 1
        if link.vers in incoming:
            incoming[link.vers] += 1

    leaf_nodes = [
        n for n in wf.noeuds
        if n.type_noeud != "fin" and outgoing.get(n.id, 0) == 0
    ]

    existing_edges = {(l.de, l.vers) for l in wf.liens}
    for leaf in leaf_nodes:
        if (leaf.id, fin_id) not in existing_edges:
            wf.liens.append(LienWorkflow(de=leaf.id, vers=fin_id, condition=None))

    return wf

def map_workflow(diagnostic: DiagnosticAnalyste):
    native_client = genai.Client(api_key=api_key)
    client = instructor.from_genai(native_client)

    user_message = f"DIAGNOSTIC ACTUEL :\n{diagnostic.model_dump_json(indent=2)}"

    start_time = time.perf_counter()
    workflow, raw_response = client.chat.completions.create_with_completion(
        model="gemini-3.1-flash-lite-preview",
        messages=[{"role": "user", "content": user_message}],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
        response_model=WorkflowOptimise,
    )
    latency = time.perf_counter() - start_time

    workflow = ensure_workflow_boundaries(workflow)
    transformation_resume = compute_transformation(diagnostic, workflow)

    return workflow, transformation_resume, latency, raw_response.usage_metadata
