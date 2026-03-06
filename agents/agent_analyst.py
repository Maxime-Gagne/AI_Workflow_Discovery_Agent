import os
import instructor
from google import genai
from dotenv import load_dotenv
from schemas import DiagnosticAnalyste

load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")

def analyze(condensed_payload: dict, source_label: str) -> DiagnosticAnalyste:
    native_client = genai.Client(api_key=api_key)
    client = instructor.from_genai(native_client)

    user_message = f"Analyse ce processus (Source: {source_label}) : {condensed_payload}"

    diagnostic = client.chat.completions.create(
        model="gemini-3.0-flash",
        messages=[{"role": "user", "content": user_message}],
        response_model=DiagnosticAnalyste,
    )

    return diagnostic
