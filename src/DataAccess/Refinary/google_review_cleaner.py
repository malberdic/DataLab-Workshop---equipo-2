import pandas as pd
import re


# Columnas esperadas del CSV (google-play-scraper)
CONTENT_COL = "content"
SCORE_COL   = "score"

class GoogleReviewCleaner:
    """Clase para limpiar y validar reseñas de Google Play."""

    @staticmethod
    def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia el DataFrame de reseñas de Google Play.
        - Elimina filas sin texto o con texto muy corto.
        - Elimina caracteres no deseados (ej. emojis, saltos de línea).
        - Normaliza espacios.
        """
        # Eliminar filas sin texto o con texto muy corto
        reviews = reviews.dropna(subset=[CONTENT_COL])
        reviews[CONTENT_COL] = reviews[CONTENT_COL].astype(str).str.strip()
        reviews = reviews[reviews[CONTENT_COL].str.len() > 5]

        # Limpiar texto: eliminar emojis, saltos de línea, etc.
        def clean_text(text):
            text = re.sub(r'\s+', ' ', text)  # Normalizar espacios
            text = re.sub(r'[^\w\s.,!?]', '', text)  # Eliminar caracteres no deseados
            return text.strip()

        reviews[CONTENT_COL] = reviews[CONTENT_COL].apply(clean_text)
        
        return reviews

    @staticmethod
    def validate_reviews(reviews: pd.DataFrame) -> bool:
        """Valida que el DataFrame tenga las columnas necesarias."""
        if CONTENT_COL not in reviews.columns:
            print(f"Error: No se encontró la columna '{CONTENT_COL}' en el DataFrame.") #TODO reemplazar por logger
            return False
        if SCORE_COL not in reviews.columns:
            print(f"Error: No se encontró la columna '{SCORE_COL}' en el DataFrame.") #TODO reemplazar por logger
            return False
        return True
