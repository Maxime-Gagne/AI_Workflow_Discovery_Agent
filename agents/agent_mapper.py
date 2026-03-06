import os
import instructor
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import DiagnosticAnalyste, WorkflowOptimise, ResumeTransformation
load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")

SYSTEM_PROMPT = """Tu es un architecte logiciel expert en modélisation de processus.
Ton rôle est de transformer un diagnostic de processus métier en un workflow technique optimisé.
Conçois l'architecture sous forme de graphe strict (noeuds et liens) en remplaçant les étapes manuelles redondantes par des solutions automatisées.

RÈGLE CRITIQUE — transformation_resume :
Chaque étape du diagnostic (E1, E2, etc.) doit apparaître dans EXACTEMENT UNE des trois listes suivantes :
- `etapes_automatisees` : l'étape est remplacée par un noeud automatique
- `etapes_conservees_humaines` : l'étape reste manuelle, exécutée par un humain
- `etapes_eliminees` : l'étape disparaît complètement sans équivalent dans le workflow

Une étape ne peut PAS apparaître dans deux listes à la fois.
Si une étape est splittée en un noeud automatique ET un noeud humain (ex: scheduling auto + entretien humain),
classe-la dans `etapes_conservees_humaines` car la partie humaine est conservée.
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

    # Tiebreak : si une étape a un noeud humain, elle prime
    automatisees -= humaines
    eliminees = all_etapes - automatisees - humaines

    return ResumeTransformation(
        etapes_eliminees=sorted(eliminees),
        etapes_automatisees=sorted(automatisees),
        etapes_conservees_humaines=sorted(humaines),
    )

def map_workflow(diagnostic: DiagnosticAnalyste) -> WorkflowOptimise:
    native_client = genai.Client(api_key=api_key)
    client = instructor.from_genai(native_client)

    user_message = f"""Voici le diagnostic du processus actuel :
{diagnostic.model_dump_json(indent=2)}

Génère le nouveau workflow optimisé selon le schéma requis."""

    workflow = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": user_message}],
        config=types.GenerateContentConfig(                # FIX: system_instruction via config, pas role "system"
            system_instruction=SYSTEM_PROMPT,
        ),
        response_model=WorkflowOptimise,
    )

    return workflow
