from enum import StrEnum
import json
import os
import re
from typing import Any, Any, List

from google import genai

GEMINI_MODEL = "gemini-2.5-flash"

class Sentiments(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class GeminiFlashModel:
    """
    Drop-in replacement for FlanT5LargeModel using Gemini.

    Contract:
        extract_top_features(reviews, sentiment, top_n) -> list[str]

    Responsibilities:
        - Prompt construction
        - API call
        - Output parsing

    Non-responsibilities:
        - Sampling
        - Sentiment filtering
        - DataFrame handling
    """

    MODEL_NAME = "gemini-2.5-flash"

    MAX_CHARS_PER_REVIEW = 512

    def __init__(self, api_key: str):
        print("Initializing GeminiFlashModel...")
        print("Starting Gemini client initialization...")
        self.client = genai.Client(api_key=api_key)

    def extract_top_features(
        self,
        reviews: List[str],
        sentiment: str,
        top_n: int
    ) -> List[str]:

        if not reviews:
            return [f"(No reviews found for sentiment '{sentiment}')"]

        prompt = self._build_prompt(reviews, sentiment, top_n)

        try:
            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
            )

            raw = response.text.strip()

            return self._parse_output(raw, top_n)

        except Exception as e:
            return [f"(Gemini error: {str(e)})"]

    # ─────────────────────────────────────────────
    # Prompt
    # ─────────────────────────────────────────────

    def _build_prompt(
        self,
        reviews: List[str],
        sentiment: str,
        top_n: int
    ) -> str:

        sentiment_label = {
            Sentiments.POSITIVE: "POSITIVAS (elogiosas)",
            Sentiments.NEGATIVE: "NEGATIVAS (críticas)",
        }.get(sentiment, sentiment)

        # Keep truncation responsibility local (same as FLAN parity)
        joined = "\n".join(
            f"- {r[:self.MAX_CHARS_PER_REVIEW].replace('\n', ' ')}"
            for r in reviews
        )

        count = len(reviews)

        return f"""
            Sos un analista de producto experto.

            Analizá las siguientes {count} reseñas {sentiment_label} de una aplicación móvil (escritas en español).
            Las reseñas están ordenadas por cantidad de likes (las más votadas primero — esas representan opiniones más compartidas). 
            Cada reseña indica sus likes y la versión de la app.

            Tu tarea:
            1. Identificá los {top_n} aspectos o características MÁS MENCIONADOS.
            2. Agrupá menciones similares bajo un mismo aspecto.
            3. Priorizá aspectos que aparecen en reseñas con más likes.

            Respondé ÚNICAMENTE con un JSON válido:

            {{
              "aspectos": [
                {{"aspecto": "nombre del aspecto", "menciones_aprox": 15, "ejemplo_reseña": "texto breve de ejemplo"}},
                ...
              ]
            }}

            REGLAS:
            - Todo en ESPAÑOL
            - Aspectos concisos y específicos
            - Ordená de más mencionado a menos mencionado
            - No incluyas texto fuera del JSON

            RESEÑAS:
            {joined}"""

    # ─────────────────────────────────────────────
    # Parsing
    # ─────────────────────────────────────────────

    def _parse_output(self, raw: str, top_n: int) -> List[str]:

        cleaned = self._strip_code_fences(raw)

        try:
            data = json.loads(cleaned)
            aspectos = data.get("aspectos", [])

            if isinstance(aspectos, list) and aspectos:
                return [
                    self._format_aspect(a)
                    for a in aspectos[:top_n]
                ]

        except json.JSONDecodeError:
            pass

        return self._fallback_parse(raw, top_n)

    def _strip_code_fences(self, raw_text: str) -> str:
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
        return raw_text.strip()

    
    def _format_aspect(self, a: Any) -> str:
        if isinstance(a, dict):
            nombre = a.get("aspecto", "?")
            menciones = a.get("menciones_aprox", "?")
            ejemplo = a.get("ejemplo_reseña", "")
    
            line = f"{nombre} (~{menciones} menciones)"
    
            if ejemplo:
                ejemplo = ejemplo.strip()
                line += (
                    f' → "{ejemplo[:80]}..."'
                    if len(ejemplo) > 80
                    else f' → "{ejemplo}"'
                )
    
            return line
    
        return str(a).strip()
    
    
    def _fallback_parse(self, raw: str, top_n: int) -> List[str]:

        lines = raw.splitlines()
        
        features = [
            re.sub(r"^[\d\.\)\-\•\–\s]+", "", l).strip()
            for l in lines
            if len(l.strip()) > 3
        ]

        return features[:top_n] if features else [raw]