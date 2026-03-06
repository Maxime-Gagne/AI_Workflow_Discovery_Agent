import os
import instructor
import google.generativeai as genai
from dotenv import load_dotenv
from schemas import DiagnosticAnalyste

load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")

def analyze(condensed_payload: dict, source_label: str) -> DiagnosticAnalyste:
    """
    Analyse les clusters via Gemini 3 Flash et valide la sortie avec Pydantic.
    """
    client = instructor.from_gemini(
        client=genai.GenerativeModel(
            model_name="models/gemini-3.0-flash",
        ),
        mode=instructor.Mode.GEMINI_JSON,
        api_key=api_key
    )

    user_message = f"Analyse ce processus (Source: {source_label}) : {condensed_payload}"

    diagnostic = client.messages.create(
        messages=[{"role": "user", "content": user_message}],
        response_model=DiagnosticAnalyste,
    )

    return diagnostic
