import os
import time
import instructor
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import WorkflowOptimise, RapportAdvisor

load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")

SYSTEM_PROMPT = """Tu es un CTO expert en intégration de systèmes.
Recommande la stack technologique exacte pour exécuter ce workflow.
CONTRAINTE ARCHITECTURALE STRICTE : Impose une "Single Source of Truth". Élimine la redondance des BDD. Choisis un seul outil central pour l'orchestration humaine.
Privilégie les outils modernes du catalogue via API."""

def advise(workflow: WorkflowOptimise):
    # Calcule le chemin absolu de la racine du projet (remonte d'un dossier depuis 'agents')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalog_path = os.path.join(base_dir, "data", "tools_catalog.json")

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = f.read()

    native_client = genai.Client(api_key=api_key)
    client = instructor.from_genai(native_client)

    user_message = f"""CATALOGUE AUTORISÉ :
{catalog}

ARCHITECTURE DU WORKFLOW :
{workflow.model_dump_json(indent=2)}"""

    start_time = time.perf_counter()
    rapport, raw_response = client.chat.completions.create_with_completion(
        model="gemini-3.1-flash-lite-preview",
        messages=[{"role": "user", "content": user_message}],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
        response_model=RapportAdvisor,
    )
    latency = time.perf_counter() - start_time

    return rapport, latency, raw_response.usage_metadata
