import os
import instructor
import google.generativeai as genai
from dotenv import load_dotenv
from schemas import DiagnosticAnalyste, WorkflowOptimise

load_dotenv()
# Priorité à l'environnement (Hugging Face) puis au fichier .env local
api_key = os.environ.get("GOOGLE_AI_STUDIO_KEY")

SYSTEM_PROMPT = """Tu es un architecte logiciel expert en modélisation de processus.
Ton rôle est de transformer un diagnostic de processus métier en un workflow technique optimisé.
Conçois l'architecture sous forme de graphe strict (noeuds et liens) en remplaçant les étapes manuelles redondantes par des solutions automatisées."""

def map_workflow(diagnostic: DiagnosticAnalyste) -> WorkflowOptimise:
    """
    Reçoit le diagnostic Pydantic strict de l'agent_analyst.
    Génère une architecture de workflow optimisée via Gemini 1.5 Flash.
    """
    client = instructor.from_gemini(
        client=genai.GenerativeModel("models/gemini-3.0-flash"),
        mode=instructor.Mode.GEMINI_JSON,
        api_key=api_key
    )

    # La méthode .model_dump_json() sérialise l'objet Pydantic pour l'injection dans le prompt
    user_message = f"""Voici le diagnostic du processus actuel :
{diagnostic.model_dump_json(indent=2)}

Génère le nouveau workflow optimisé selon le schéma requis."""

    workflow = client.messages.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        response_model=WorkflowOptimise,
    )

    return workflow
