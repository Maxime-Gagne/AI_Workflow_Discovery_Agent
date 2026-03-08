import os
import time
import instructor
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import DiagnosticAnalyste

load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")

SYSTEM_PROMPT = """Tu es un analyste expert en optimisation de processus métier.
Produis un diagnostic structuré du processus actuel à partir des données clusterisées.
CONTRAINTE ARCHITECTURALE STRICTE : Injecte IMPÉRATIVEMENT les valeurs exactes des MÉTRIQUES ROI fournies dans le modèle de réponse (GainsEstimes). Aucune altération, aucun recalcul."""

def analyze(condensed_payload: dict, source_label: str, roi_metrics: dict):
    native_client = genai.Client(api_key=api_key)
    client = instructor.from_genai(native_client)

    user_message = f"""Source: {source_label}
DONNÉES CLUSTERISÉES :
{condensed_payload}

MÉTRIQUES ROI DÉTERMINISTES :
{roi_metrics}"""

    start_time = time.perf_counter()
    diagnostic, raw_response = client.chat.completions.create_with_completion(
        model="gemini-3.1-flash-lite-preview",
        messages=[{"role": "user", "content": user_message}],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
        ),
        response_model=DiagnosticAnalyste,
    )
    latency = time.perf_counter() - start_time

    return diagnostic, latency, raw_response.usage_metadata

SOP_SYSTEM_PROMPT = """Tu es un ingénieur process expert en automatisation.
Modélise un diagnostic structuré à partir de cette Procédure Opérationnelle Standard (SOP) textuelle.
CONTRAINTES ARCHITECTURALES STRICTES :
1. Extrais séquentiellement chaque étape (acteur, action, type).
2. Déduis métriques de durée, volume et frictions du contexte sémantique.
3. Estime mathématiquement GainsEstimes via le taux horaire fourni. Justifie dans detail_du_calcul."""

def analyze_sop(texte_brut: str, source_label: str, taux_horaire: float):
    native_client = genai.Client(api_key=api_key)
    client = instructor.from_genai(native_client)

    user_message = f"""Source: {source_label}
TAUX HORAIRE: {taux_horaire} $/h
DOCUMENT SOP BRUT :
{texte_brut}"""

    start_time = time.perf_counter()
    diagnostic, raw_response = client.chat.completions.create_with_completion(
        model="gemini-3.1-flash-lite-preview",
        messages=[{"role": "user", "content": user_message}],
        config=types.GenerateContentConfig(
            system_instruction=SOP_SYSTEM_PROMPT,
            temperature=0.1,
        ),
        response_model=DiagnosticAnalyste,
    )
    latency = time.perf_counter() - start_time

    return diagnostic, latency, raw_response.usage_metadata
