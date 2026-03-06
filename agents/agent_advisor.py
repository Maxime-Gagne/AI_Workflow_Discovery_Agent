import os
import instructor
import google.generativeai as genai
from dotenv import load_dotenv
from schemas import WorkflowOptimise, RapportAdvisor

load_dotenv()
api_key = os.environ.get("GOOGLE_AI_STUDIO_KEY")

SYSTEM_PROMPT = """Tu es un CTO expert en intégration de systèmes et en sélection d'outils d'automatisation d'entreprise.
Ton rôle est de recommander la stack technologique exacte pour exécuter un workflow défini. Privilégie des outils modernes, fiables et interconnectables (API)."""

def advise(workflow: WorkflowOptimise) -> RapportAdvisor:
    """
    Reçoit le workflow optimisé Pydantic de l'agent_mapper.
    Sélectionne les outils appropriés pour chaque noeud technique.
    """
    with open("data/tools_catalog.json", "r", encoding="utf-8") as f:
        catalog = f.read()

    client = instructor.from_gemini(
        client=genai.GenerativeModel("models/gemini-3.0-flash"),
        mode=instructor.Mode.GEMINI_JSON,
        api_key=api_key
    )

    # On injecte le catalogue dans le message
    user_message = f"""Voici le catalogue d'outils autorisés :
{catalog}

Voici l'architecture du workflow optimisé :
{workflow.model_dump_json(indent=2)}

En te basant PRIORITAIREMENT sur le catalogue fourni, recommande la stack technique."""
    rapport = client.messages.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        response_model=RapportAdvisor,
    )

    return rapport
