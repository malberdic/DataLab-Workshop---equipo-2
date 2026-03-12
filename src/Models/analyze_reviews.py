"""
analyze_reviews.py

"""

import argparse
import re
import sys
from collections import Counter

import pandas as pd
from transformers import pipeline


# Configuración

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SUMMARY_MODEL   = "google/flan-t5-large"

# Columnas esperadas del CSV (google-play-scraper)
CONTENT_COL = "content"
SCORE_COL   = "score"

# Cuántas reseñas mandarle al modelo de resumen (no saturar?)
MAX_REVIEWS_FOR_SUMMARY = 80
MAX_CHARS_PER_REVIEW    = 300


# Utilidades 

def load_csv(path: str) -> pd.DataFrame:
    """Carga el CSV y valida que tenga la columna 'content'."""
    df = pd.read_csv(path)
    if CONTENT_COL not in df.columns:
        sys.exit(
            f"Error. No tiene columna '{CONTENT_COL}'.\n"
            f"Columnas encontradas: {list(df.columns)}"
        )
    # Limpieza básica
    df = df.dropna(subset=[CONTENT_COL])
    df[CONTENT_COL] = df[CONTENT_COL].astype(str).str.strip()
    df = df[df[CONTENT_COL].str.len() > 5]
    print(f"Reseñas cargadas: {len(df)}")
    return df


def classify_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columna 'sentiment' con valores: positive / neutral / negative.
    Usa el score de estrellas como fallback si el texto es muy corto.
    """
    print(f"Modelo de sentimiento: {SENTIMENT_MODEL}")
    sentiment_pipe = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        truncation=True, #max chars
        max_length=512,
    )

    labels = []
    batch_size = 32
    texts = df[CONTENT_COL].tolist()

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = sentiment_pipe(batch)
        for r in results:
            labels.append(r["label"].lower())

        pct = min(i + batch_size, len(texts))

    df = df.copy()
    df["sentiment"] = labels
    return df


def build_summary_prompt(reviews: list[str], sentiment: str, top_n: int) -> str:
    """Construye el prompt para flan-t5."""
    joined = " | ".join(
        r[:MAX_CHARS_PER_REVIEW].replace("\n", " ") for r in reviews
    )
    return (
        f"You are analyzing user reviews of a mobile app. "
        f"Below are {sentiment} reviews separated by '|'. "
        f"List the top {top_n} most mentioned {sentiment} aspects or features "
        f"as a numbered list. Be concise.\n\nReviews: {joined}"
    ) # Primera prueba del prompt


def extract_top_features(df: pd.DataFrame, sentiment: str, top_n: int, summary_pipe) -> list[str]:
    """Filtra reseñas por sentimiento y extrae los temas más frecuentes con flan-t5."""
    subset = df[df["sentiment"] == sentiment][CONTENT_COL].tolist()

    if not subset:
        return [f"(No se encontraron reseñas '{sentiment}')"]

    # Muestra para no exceder el contexto del modelo
    sample = subset[:MAX_REVIEWS_FOR_SUMMARY]
    prompt = build_summary_prompt(sample, sentiment, top_n)

    output = summary_pipe(
        prompt,
        max_new_tokens=200,
        do_sample=False,
    )
    raw = output[0]["generated_text"].strip()

    # Parseo simple
    lines = re.split(r"\n|\d+[\.\)]", raw)
    features = [l.strip(" -–•") for l in lines if len(l.strip()) > 3]
    return features[:top_n] if features else [raw]


# Main

def main():
    parser = argparse.ArgumentParser(description="Analiza reseñas de Google Play Store")
    parser.add_argument("--csv",  required=True, help="Ruta al archivo CSV de reseñas")
    parser.add_argument("--top",  type=int, default=3, help="Cantidad de features a mostrar")
    args = parser.parse_args()

    # Cargar datos
    df = load_csv(args.csv)

    # Clasificar sentimiento
    df = classify_sentiment(df)

    # Estadísticas
    counts = df["sentiment"].value_counts()
    total  = len(df)
    print("\n Distribución de sentimiento")
    for label, count in counts.items():
        pct = count / total * 100
        print(f"  {label:<10} {count:>5} reseñas  ({pct:.1f}%)")

    # Cargar modelo de resumen
    print(f"\nModelo de resumen: {SUMMARY_MODEL}")
    summary_pipe = pipeline(
        "text2text-generation",
        model=SUMMARY_MODEL,
        truncation=True,
        max_length=1024,
    )

    # Extraer features positivas y negativas
    print("\nCaracterísticas más elogiadas")
    praised = extract_top_features(df, "positive", args.top, summary_pipe)

    print("Características más criticadas")
    criticized = extract_top_features(df, "negative", args.top, summary_pipe)

    # 5. Resultado final
    print(f"  RESUMEN DE RESEÑAS  ({total} reseñas analizadas)")

    print(f"\n{args.top} características ELOGIADAS:")
    for i, f in enumerate(praised, 1):
        print(f"  {i}. {f}")

    print(f"\n{args.top} características CRITICADAS:")
    for i, f in enumerate(criticized, 1):
        print(f"  {i}. {f}")


    # 6. Exportar resultado a CSV
    output_path = args.csv.replace(".csv", "_analyzed.csv")
    df.to_csv(output_path, index=False)
    print(f"\nCSV con sentimientos guardado en: {output_path}")


if __name__ == "__main__":
    main()