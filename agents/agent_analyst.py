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

CONTRAINTES ARCHITECTURALES STRICTES :
1. Toute la gestion du temps et du ROI a déjà été calculée en amont par AgentTemps et le moteur déterministe.
2. Tu n'as PAS le droit de recalculer, corriger, arrondir, reformuler ou remplacer les métriques ROI fournies.
3. Si un bloc ROI déterministe est fourni, tu dois injecter IMPÉRATIVEMENT les valeurs exactes dans GainsEstimes.
4. Cette contrainte d'injection exacte concerne TOUS les champs de GainsEstimes :
   - heures_economisees_par_mois
   - economies_mensuelles_devise
   - projection_annuelle
   - mois_retour_investissement
   - detail_du_calcul
   - assumptions
   - confidence
5. Le champ confidence doit être recopié tel quel, y compris roi_mode et tous les niveaux de confiance détaillés.
6. Le champ assumptions doit être recopié tel quel, sans reformulation.
7. Les seules parties que tu peux interpréter librement sont :
   - titre_processus
   - description
   - etapes_actuelles
   - metriques_cles
   - potentiel_automatisation
   - principales_opportunites
"""

def analyze(condensed_payload: dict, source_label: str, roi_metrics: dict | None = None):
    native_client = genai.Client(api_key=api_key)
    client = instructor.from_genai(native_client)

    roi_block = roi_metrics if roi_metrics is not None else "AUCUNE"

    user_message = f"""Source: {source_label}
DONNÉES CLUSTERISÉES :
{condensed_payload}

MÉTRIQUES ROI DÉTERMINISTES :
{roi_block}
"""

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
2. N'invente jamais de durée, de volume, ni de métrique ROI.
3. Si des métriques ROI déterministes sont fournies, injecte IMPÉRATIVEMENT leurs valeurs exactes dans GainsEstimes.
4. Cette injection exacte concerne TOUS les champs déterministes de GainsEstimes :
   - heures_economisees_par_mois
   - economies_mensuelles_devise
   - projection_annuelle
   - mois_retour_investissement
   - detail_du_calcul
   - assumptions
   - confidence
5. Si aucune métrique ROI n'est fournie, limite-toi à un diagnostic structurel cohérent sans hallucination quantitative.
"""

def analyze_sop(
    texte_brut: str,
    source_label: str,
    taux_horaire: float,
    roi_metrics: dict | None = None,
):
    native_client = genai.Client(api_key=api_key)
    client = instructor.from_genai(native_client)

    user_message = f"""Source: {source_label}
TAUX HORAIRE: {taux_horaire} $/h
DOCUMENT SOP BRUT :
{texte_brut}

MÉTRIQUES ROI DÉTERMINISTES :
{roi_metrics if roi_metrics is not None else "AUCUNE"}
"""

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
