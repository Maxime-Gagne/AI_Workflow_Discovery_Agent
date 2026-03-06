import os
import instructor
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import WorkflowOptimise, RapportAdvisor

load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")

SYSTEM_PROMPT = """Tu es un CTO expert en intégration de systèmes et en sélection d'outils d'automatisation d'entreprise.
Ton rôle est de recommander la stack technologique exacte pour exécuter un workflow défini. Privilégie des outils modernes, fiables et interconnectables (API)."""

def advise(workflow: WorkflowOptimise) -> RapportAdvisor:
    with open("data/tools_catalog.json", "r", encoding="utf-8") as f:
        catalog = f.read()

    native_client = genai.Client(api_key=api_key)
    client = instructor.from_genai(native_client)

    user_message = f"""Voici le catalogue d'outils autorisés :
{catalog}

Voici l'architecture du workflow optimisé :
{workflow.model_dump_json(indent=2)}

En te basant PRIORITAIREMENT sur le catalogue fourni, recommande la stack technique."""

    rapport = client.chat.completions.create(
        model="gemini-2.5-flash",                          # FIX: gemini-3.0-flash n'existe pas
        messages=[{"role": "user", "content": user_message}],
        config=types.GenerateContentConfig(                # FIX: system_instruction via config, pas role "system"
            system_instruction=SYSTEM_PROMPT,
        ),
        response_model=RapportAdvisor,
    )

    return rapport
