"""
analyze_reviews_v2.py

Pipeline de análisis de reseñas de Google Play Store.
- Clasificación de sentimiento (multilingüe, local con nlptown)
- Ponderación por thumbsUpCount (reseñas más votadas pesan más)
- Segmentación por versión de la app
- Extracción de aspectos con Google Gemini API
- Análisis FODA completo generado por Gemini

Uso:
    python analyze_reviews_v2.py --csv reviews.csv
    python analyze_reviews_v2.py --csv reviews.csv --top 5
"""

import argparse
import json
import os
import re
import sys

import pandas as pd
from google import genai
from transformers import pipeline


# ---
# Configuración
# ---

SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
GEMINI_MODEL = "gemini-2.5-flash"

CONTENT_COL  = "content"
SCORE_COL    = "score"
VERSION_COL  = "reviewCreatedVersion"
LIKES_COL    = "thumbsUpCount"
DATE_COL     = "at"

STAR_TO_SENTIMENT = {
    "1 star":  "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    sys.exit(
        "Error: Variable de entorno GEMINI_API_KEY no configurada.\n"
        "Configurala con:  set GEMINI_API_KEY=tu_clave_aqui  (CMD)\n"
        "              o:  $env:GEMINI_API_KEY='tu_clave_aqui'  (PowerShell)"
    )


# ---
# Carga y limpieza
# ---

def load_csv(path: str) -> pd.DataFrame:
    """Carga el CSV, valida columnas y limpia datos."""
    df = pd.read_csv(path)
    if CONTENT_COL not in df.columns:
        sys.exit(
            f"Error. No tiene columna '{CONTENT_COL}'.\n"
            f"Columnas encontradas: {list(df.columns)}"
        )

    df = df.dropna(subset=[CONTENT_COL])
    df[CONTENT_COL] = df[CONTENT_COL].astype(str).str.strip()
    df = df[df[CONTENT_COL].str.len() > 5]

    # Asegurar tipos numéricos
    if LIKES_COL in df.columns:
        df[LIKES_COL] = pd.to_numeric(df[LIKES_COL], errors="coerce").fillna(0).astype(int)

    if SCORE_COL in df.columns:
        df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce").fillna(3).astype(int)

    # Parsear fecha
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    print(f"Reseñas cargadas: {len(df)}")
    return df


# ---
# Sentimiento (corre local)
# ---

def classify_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Clasifica sentimiento con nlptown (multilingüe)."""
    print(f"Cargando modelo de sentimiento: {SENTIMENT_MODEL}")
    sentiment_pipe = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        truncation=True,
        max_length=512,
    )

    labels = []
    batch_size = 32
    texts = df[CONTENT_COL].tolist()
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        results = sentiment_pipe(batch)
        for r in results:
            star_label = r["label"].lower()
            sentiment = STAR_TO_SENTIMENT.get(star_label, "neutral")
            labels.append(sentiment)
        processed = min(i + batch_size, total)
        print(f"  Sentimiento: {processed}/{total} reseñas procesadas", end="\r")

    print()
    df = df.copy()
    df["sentiment"] = labels
    return df


# ---
# Análisis por versión de la app
# ---

def analyze_by_version(df: pd.DataFrame) -> str:
    """
    Analiza sentimiento agrupado por versión de la app.
    Devuelve un string formateado con la tabla de resultados.
    """
    if VERSION_COL not in df.columns:
        return "(Columna de versión no disponible)"

    df_v = df.dropna(subset=[VERSION_COL])
    df_v = df_v[df_v[VERSION_COL].str.strip() != ""]

    if df_v.empty:
        return "(No hay datos de versión)"

    # Agrupar por versión
    version_stats = (
        df_v.groupby(VERSION_COL)
        .agg(
            total=("sentiment", "size"),
            positivas=("sentiment", lambda x: (x == "positive").sum()),
            negativas=("sentiment", lambda x: (x == "negative").sum()),
            neutrales=("sentiment", lambda x: (x == "neutral").sum()),
            score_promedio=(SCORE_COL, "mean") if SCORE_COL in df_v.columns else ("sentiment", "size"),
        )
        .sort_values("total", ascending=False)
    )

    # Mostrar solo versiones con suficientes reseñas (≥5)
    version_stats = version_stats[version_stats["total"] >= 5]

    if version_stats.empty:
        return "(No hay versiones con suficientes reseñas para analizar)"

    lines = []
    lines.append(f"  {'Versión':<16} {'Total':>6} {'👍 Pos':>7} {'👎 Neg':>7} {'😐 Neu':>7} {'Score ⭐':>8}")
    lines.append("  " + "─" * 55)

    for version, row in version_stats.head(10).iterrows():
        pct_neg = row["negativas"] / row["total"] * 100
        # Marcar versiones con alto % negativo
        flag = " ⚠️" if pct_neg > 40 else ""
        score_str = f"{row['score_promedio']:.1f}" if "score_promedio" in row else "N/A"
        lines.append(
            f"  {str(version):<16} {int(row['total']):>6} "
            f"{int(row['positivas']):>7} {int(row['negativas']):>7} "
            f"{int(row['neutrales']):>7} {score_str:>8}{flag}"
        )

    return "\n".join(lines)


# ---
# Preparar reseñas ponderadas por likes
# ---

def prepare_weighted_reviews(df: pd.DataFrame, sentiment: str) -> str:
    """
    Prepara las reseñas filtradas por sentimiento, ordenadas por
    thumbsUpCount descendente. Incluye el peso de likes para que
    Gemini priorice las más votadas.
    """
    subset = df[df["sentiment"] == sentiment].copy()

    if subset.empty:
        return ""

    # Ordenar por likes (más votadas primero)
    if LIKES_COL in subset.columns:
        subset = subset.sort_values(LIKES_COL, ascending=False)

    lines = []
    for idx, (_, row) in enumerate(subset.iterrows()):
        text = str(row[CONTENT_COL])[:500]
        likes = int(row.get(LIKES_COL, 0))
        version = row.get(VERSION_COL, "?")

        if likes > 0:
            lines.append(f"{idx+1}. [👍{likes} likes | v{version}] {text}")
        else:
            lines.append(f"{idx+1}. [v{version}] {text}")

    return "\n".join(lines)


# ---
# Extracción de aspectos con Gemini
# ---

def extract_aspects_gemini(
    df: pd.DataFrame,
    sentiment: str,
    top_n: int,
    client: genai.Client,
) -> list[str]:
    """Extrae aspectos más mencionados usando Gemini."""
    reviews_text = prepare_weighted_reviews(df, sentiment)
    count = len(df[df["sentiment"] == sentiment])

    if not reviews_text:
        return [f"(No se encontraron reseñas '{sentiment}')"]

    sentiment_label = {
        "positive": "POSITIVAS (elogiosas)",
        "negative": "NEGATIVAS (críticas)",
    }.get(sentiment, sentiment)

    prompt = f"""Sos un analista de producto experto. Analizá las siguientes {count} reseñas {sentiment_label} de una aplicación móvil (escritas en español).

Las reseñas están ordenadas por cantidad de likes (las más votadas primero — esas representan opiniones más compartidas). Cada reseña indica sus likes y la versión de la app.

Tu tarea:
1. Identificá los {top_n} aspectos o características MÁS MENCIONADOS.
2. Priorizá aspectos que aparecen en reseñas con más likes.
3. Respondé ÚNICAMENTE con un JSON válido:

{{
  "aspectos": [
    {{"aspecto": "nombre del aspecto", "menciones_aprox": 15, "ejemplo_reseña": "texto breve de ejemplo"}},
    ...
  ]
}}

REGLAS:
- Nombres de aspectos en ESPAÑOL, concisos y específicos.
- Ordená de más mencionado a menos mencionado.
- No incluyas texto fuera del JSON.

RESEÑAS:
{reviews_text}"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        data = json.loads(raw_text)
        aspectos = data.get("aspectos", [])

        results = []
        for a in aspectos[:top_n]:
            nombre = a.get("aspecto", "?")
            menciones = a.get("menciones_aprox", "?")
            ejemplo = a.get("ejemplo_reseña", "")
            line = f"{nombre} (~{menciones} menciones)"
            if ejemplo:
                line += f'  →  "{ejemplo[:80]}..."' if len(ejemplo) > 80 else f'  →  "{ejemplo}"'
            results.append(line)

        return results if results else [raw_text]

    except json.JSONDecodeError:
        print("  ⚠ Gemini no devolvió JSON válido, mostrando respuesta cruda.")
        lines = [l.strip() for l in response.text.strip().split("\n") if l.strip()]
        return lines[:top_n]

    except Exception as e:
        return [f"Error al consultar Gemini: {e}"]


# ---
# Análisis FODA con Gemini
# ---

def generate_foda(
    df: pd.DataFrame,
    praised: list[str],
    criticized: list[str],
    version_analysis: str,
    client: genai.Client,
) -> dict:
    """
    Genera un análisis FODA completo usando Gemini, alimentado por
    los aspectos ya extraídos, la data de versiones, y las reseñas.
    """

    # Estadísticas generales para contexto
    total = len(df)
    counts = df["sentiment"].value_counts()
    pct_pos = counts.get("positive", 0) / total * 100
    pct_neg = counts.get("negative", 0) / total * 100
    pct_neu = counts.get("neutral", 0) / total * 100

    avg_score = df[SCORE_COL].mean() if SCORE_COL in df.columns else "N/A"
    avg_likes = df[LIKES_COL].mean() if LIKES_COL in df.columns else "N/A"

    # Muestra de reseñas neutrales (a veces contienen sugerencias valiosas)
    neutral_reviews = df[df["sentiment"] == "neutral"][CONTENT_COL].tolist()
    neutral_sample = "\n".join(f"- {r[:200]}" for r in neutral_reviews[:30])

    prompt = f"""Sos un consultor estratégico de producto digital. Con base en el análisis de {total} reseñas de una aplicación móvil, generá un análisis FODA (SWOT) completo.

DATOS DE ENTRADA:

Distribución de sentimiento:
- Positivas: {pct_pos:.1f}% ({counts.get('positive', 0)} reseñas)
- Negativas: {pct_neg:.1f}% ({counts.get('negative', 0)} reseñas)
- Neutrales: {pct_neu:.1f}% ({counts.get('neutral', 0)} reseñas)
- Score promedio: {avg_score:.2f}/5 ⭐
- Likes promedio por reseña: {avg_likes:.1f}

🟢 Aspectos más elogiados:
{chr(10).join(f'  {i+1}. {a}' for i, a in enumerate(praised))}

🔴 Aspectos más criticados:
{chr(10).join(f'  {i+1}. {a}' for i, a in enumerate(criticized))}

📱 Análisis por versión de la app:
{version_analysis}

😐 Muestra de reseñas neutrales (posibles sugerencias):
{neutral_sample}

INSTRUCCIONES:
Generá un JSON con la siguiente estructura exacta:

{{
  "fortalezas": [
    {{"punto": "descripción concisa", "evidencia": "dato o reseña que lo respalda"}},
    ...
  ],
  "oportunidades": [
    {{"punto": "descripción concisa", "evidencia": "dato o reseña que lo respalda"}},
    ...
  ],
  "debilidades": [
    {{"punto": "descripción concisa", "evidencia": "dato o reseña que lo respalda"}},
    ...
  ],
  "amenazas": [
    {{"punto": "descripción concisa", "evidencia": "dato o reseña que lo respalda"}},
    ...
  ],
  "resumen_ejecutivo": "Párrafo breve (3-4 oraciones) con la conclusión general del análisis."
}}

REGLAS:
- Entre 3 y 5 puntos por categoría.
- Todo en ESPAÑOL.
- Las Oportunidades deben derivarse de las sugerencias y necesidades no cubiertas de los usuarios.
- Las Amenazas deben inferirse de tendencias negativas (ej: si una actualización reciente genera muchas quejas, eso es una amenaza de churn).
- Incluí evidencia concreta (citas de reseñas, porcentajes, datos de versiones).
- No incluyas texto fuera del JSON.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        return json.loads(raw_text)

    except json.JSONDecodeError:
        print("  ⚠ Gemini no devolvió JSON válido para el FODA.")
        return {"error": response.text[:500]}

    except Exception as e:
        return {"error": str(e)}


def print_foda(foda: dict):
    """Imprime el análisis FODA de forma legible."""
    if "error" in foda:
        print(f"\n  ⚠ Error en FODA: {foda['error']}")
        return

    sections = [
        ("FORTALEZAS (Strengths)", "fortalezas"),
        ("OPORTUNIDADES (Opportunities)", "oportunidades"),
        ("DEBILIDADES (Weaknesses)", "debilidades"),
        ("AMENAZAS (Threats)", "amenazas"),
    ]

    for title, key in sections:
        items = foda.get(key, [])
        print(f"\n  {title}")
        print("  " + "─" * 50)
        for i, item in enumerate(items, 1):
            punto = item.get("punto", "?")
            evidencia = item.get("evidencia", "")
            print(f"  {i}. {punto}")
            if evidencia:
                print(f"{evidencia[:120]}")

    resumen = foda.get("resumen_ejecutivo", "")
    if resumen:
        print(f"\nRESUMEN")
        print("  " + "─" * 50)
        # Word wrap a ~80 chars
        words = resumen.split()
        line = "  "
        for w in words:
            if len(line) + len(w) + 1 > 85:
                print(line)
                line = "  " + w
            else:
                line += " " + w if line.strip() else "  " + w
        if line.strip():
            print(line)


# ---
# Main
# ---

def main():
    parser = argparse.ArgumentParser(
        description="Analiza reseñas de Google Play Store (v2 - FODA con Gemini)"
    )
    parser.add_argument("--csv", required=True, help="Ruta al archivo CSV de reseñas")
    parser.add_argument("--top", type=int, default=5, help="Cantidad de aspectos a extraer")
    args = parser.parse_args()

    # ── 1. Cargar datos ──
    df = load_csv(args.csv)

    # ── 2. Clasificar sentimiento (local) ──
    df = classify_sentiment(df)

    # ── 3. Estadísticas generales ──
    counts = df["sentiment"].value_counts()
    total  = len(df)
    print("\n══════════════════════════════════════════════════")
    print(f"  DISTRIBUCIÓN DE SENTIMIENTO ({total} reseñas)")
    print("══════════════════════════════════════════════════")
    for label, count in counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {label:<10} {count:>5}  ({pct:5.1f}%)  {bar}")

    # ── 4. Análisis por versión ──
    print("\n══════════════════════════════════════════════════")
    print("  SENTIMIENTO POR VERSIÓN DE LA APP")
    print("══════════════════════════════════════════════════")
    version_analysis = analyze_by_version(df)
    print(version_analysis)

    # ── 5. Inicializar Gemini ──
    print(f"\nConectando con Gemini ({GEMINI_MODEL})...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    # ── 6. Extraer aspectos (ponderados por likes) ──
    print(f"\n🟢 Analizando {len(df[df['sentiment'] == 'positive'])} reseñas positivas...")
    praised = extract_aspects_gemini(df, "positive", args.top, client)

    print(f"🔴 Analizando {len(df[df['sentiment'] == 'negative'])} reseñas negativas...")
    criticized = extract_aspects_gemini(df, "negative", args.top, client)

    # ── 7. Resumen de aspectos ──
    print("\n══════════════════════════════════════════════════")
    print(f"  ASPECTOS CLAVE ({total} reseñas analizadas)")
    print("══════════════════════════════════════════════════")

    print(f"\n  🟢 Top {args.top} aspectos MÁS ELOGIADOS:")
    for i, f in enumerate(praised, 1):
        print(f"  {i}. {f}")

    print(f"\n  🔴 Top {args.top} aspectos MÁS CRITICADOS:")
    for i, f in enumerate(criticized, 1):
        print(f"  {i}. {f}")

    # ---────────────────
    # TODO [PARKING LOT]: Análisis FODA (SWOT)
    # Descomentar este bloque para habilitar la generación del
    # reporte FODA completo. Pendiente: separar en modo --report
    # para que el usuario elija entre resumen rápido o informe
    # completo con FODA.
    # ---────────────────
    # print("\n══════════════════════════════════════════════════")
    # print("  ANÁLISIS FODA (SWOT)")
    # print("══════════════════════════════════════════════════")
    # print("\n  Generando análisis FODA con Gemini...")
    # foda = generate_foda(df, praised, criticized, version_analysis, client)
    # print_foda(foda)

    # ── 8. Exportar resultados ──
    output_csv = args.csv.replace(".csv", "_analyzed.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nCSV con sentimientos: {output_csv}")

    # TODO [PARKING LOT]: Exportar FODA como JSON
    # Descomentar junto con el bloque FODA de arriba.
    # if "error" not in foda:
    #     foda_path = args.csv.replace(".csv", "_foda.json")
    #     with open(foda_path, "w", encoding="utf-8") as f:
    #         json.dump(foda, f, ensure_ascii=False, indent=2)
    #     print(f"📁 FODA exportado: {foda_path}")

    print("\nAnálisis completo.")


if __name__ == "__main__":
    main()
