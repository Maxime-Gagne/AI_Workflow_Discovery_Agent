import instructor
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
import os

# Configuration du contrat de sortie
class AtomicAction(BaseModel):
    action: str = Field(description="Description concise de l'action effectuée (verbe à l'infinitif).")
    acteur: str = Field(description="Entité ou rôle ayant réalisé l'action.")
    timestamp: Optional[str] = Field(description="Horodatage ISO 8601 si disponible, sinon null.")
    contexte_supp: Optional[str] = Field(description="Informations critiques (outils, blocages, tags).")


class SourceQualityReport(BaseModel):
    is_processable: bool = Field(description="Vrai si le contenu décrit un processus métier identifiable.")
    detected_format: str = Field(description="Type de source identifiée (ex: Chat Log, CRM Export, Unstructured).")
    explanation: str = Field(description="Note technique sur la qualité et les transformations effectuées.")
    structured_data: List[AtomicAction] = Field(description="Liste intégrale des actions normalisées.")


def refine_source_data(raw_content: str, api_key: str) -> SourceQualityReport:
    client = genai.Client(api_key=api_key)
    instr_client = instructor.from_gemini(client=client, mode=instructor.Mode.GEMINI_JSON)

    prompt = f"""
    Tu es un expert en Process Mining et Architecture Cognitive.
    Ton rôle est d'extraire la structure métier d'un fichier brut, potentiellement bruité (logs, conversations, exports vrac).

    INSTRUCTIONS :
    1. Analyse l'intégralité des données fournies ci-dessous.
    2. Identifie chaque unité d'action (Atomic Action).
    3. Si les données proviennent d'un chat (Slack/Discord) : Synthétise les échanges en étapes de décision ou d'exécution.
    4. Supprime le bruit (salutations, hors-sujet) mais conserve les preuves de friction.
    5. Normalise le format de sortie pour que chaque ligne soit indépendante.

    DONNÉES BRUTES :
    {raw_content}
    """

    response = instr_client.chat.completions.create(
        model="gemini-3.1-flash-lite-preview",
        response_model=SourceQualityReport,
        messages=[{"role": "user", "content": prompt}],
    )

    return response
