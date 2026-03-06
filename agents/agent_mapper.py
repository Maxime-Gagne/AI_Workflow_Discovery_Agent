import os
import instructor
from google import genai
from dotenv import load_dotenv
from schemas import DiagnosticAnalyste, WorkflowOptimise

load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")

SYSTEM_PROMPT = """Tu es un architecte logiciel expert en modélisation de processus.
Ton rôle est de transformer un diagnostic de processus métier en un workflow technique optimisé.
Conçois l'architecture sous forme de graphe strict (noeuds et liens) en remplaçant les étapes manuelles redondantes par des solutions automatisées."""

def map_workflow(diagnostic: DiagnosticAnalyste) -> WorkflowOptimise:
    native_client = genai.Client(api_key=api_key)
    client = instructor.from_genai(native_client)

    user_message = f"""Voici le diagnostic du processus actuel :
{diagnostic.model_dump_json(indent=2)}

Génère le nouveau workflow optimisé selon le schéma requis."""

    workflow = client.chat.completions.create(
        model="gemini-3.0-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        response_model=WorkflowOptimise,
    )

    return workflow
