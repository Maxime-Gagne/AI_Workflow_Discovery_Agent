import os
import instructor
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import DiagnosticAnalyste

load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")

SYSTEM_PROMPT = """Tu es un analyste expert en optimisation de processus métier et automatisation IA.
Tu reçois des données brutes exportées d'un outil métier et tu produis un diagnostic structuré du processus actuel."""

def analyze(condensed_payload: dict, source_label: str) -> DiagnosticAnalyste:
    native_client = genai.Client(api_key=api_key)
    client = instructor.from_genai(native_client)

    user_message = f"Analyse ce processus (Source: {source_label}) : {condensed_payload}"

    diagnostic = client.chat.completions.create(
        model="gemini-2.5-flash",                          # FIX: gemini-3.0-flash n'existe pas
        messages=[{"role": "user", "content": user_message}],
        config=types.GenerateContentConfig(                # FIX: system_instruction via config, pas role "system"
            system_instruction=SYSTEM_PROMPT,
        ),
        response_model=DiagnosticAnalyste,
    )

    return diagnostic
