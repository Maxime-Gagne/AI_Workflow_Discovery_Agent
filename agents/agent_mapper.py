import os
import time
import instructor
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import DiagnosticAnalyste, WorkflowOptimise, ResumeTransformation

load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")

SYSTEM_PROMPT = """Tu es un architecte logiciel expert en modélisation de processus.
Transforme le diagnostic fourni en un workflow technique optimisé (graphe strict).
RÈGLE CRITIQUE (transformation_resume) :
Chaque étape originale doit être classée dans EXACTEMENT UNE des listes : etapes_automatisees, etapes_conservees_humaines, ou etapes_eliminees.
Priorité de classification si split : etapes_conservees_humaines > etapes_automatisees."""

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

    # Calcul déterministe du résumé de la transformation
    resume = compute_transformation(diagnostic, workflow)

    return workflow, resume, latency, raw_response.usage_metadata
