"""
app.py — PlayInsights · Dashboard de Análisis de Reseñas (Streamlit)
Réplica del prototipo Emergent, adaptado a Streamlit + Python.

Ejecutar:
    streamlit run src/app.py
"""

import sys, os, json, re, io, urllib.parse, collections, time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google_play_scraper import reviews, Sort, search
from google_play_scraper import app as gplay_app
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from fpdf import FPDF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Models"))
from analyze_reviews_v2 import (
    classify_sentiment, extract_aspects_gemini, generate_foda,
    analyze_by_version,
    CONTENT_COL, SCORE_COL, VERSION_COL, LIKES_COL, DATE_COL, GEMINI_MODEL,
)

# ───────────────────────────────────
# CONFIG
# ───────────────────────────────────
st.set_page_config(page_title="PlayInsights", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

COLORS = {"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"}
COUNTRIES = {
    "Argentina": "ar", "México": "mx", "España": "es", "Colombia": "co",
    "Chile": "cl", "Perú": "pe", "Uruguay": "uy", "Paraguay": "py",
    "Estados Unidos": "us", "Brasil": "br",
}
LANGUAGES = {"Español": "es", "English": "en", "Português": "pt"}

if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "history" not in st.session_state:
    st.session_state["history"] = []

# ───────────────────────────────────
# CSS
# ───────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    h1,h2,h3,h4 { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; color: #0f172a !important; }
    .block-container { padding-top: 4rem; }
    header[data-testid="stHeader"] { background: white; border-bottom: 1px solid #e2e8f0; }
    div[data-testid="stSidebar"] { display: none; }

    /* Cards */
    .stat-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 4px;
        padding: 1.2rem 1.4rem;
    }
    .stat-card .label { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.3rem; }
    .stat-card .value { font-size: 1.6rem; font-weight: 700; color: #0f172a; }
    .stat-card .sub { font-size: 0.75rem; color: #64748b; margin-top: 0.15rem; }

    .chart-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 4px;
        padding: 1.2rem; margin-bottom: 0.8rem;
    }
    .chart-card-title { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 1rem; }

    /* App result card */
    .app-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 4px;
        padding: 1rem 1.2rem; margin-bottom: 0.4rem;
        display: flex; align-items: center; gap: 1rem;
        transition: border-color 0.15s;
    }
    .app-card:hover { border-color: #94a3b8; }

    /* Review card */
    .review-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 4px;
        padding: 1rem 1.2rem; margin-bottom: 0.3rem;
    }

    /* Sentiment badges */
    .badge-positive { background: #dcfce7; color: #166534; padding: 1px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 500; border: 1px solid #bbf7d0; }
    .badge-negative { background: #fee2e2; color: #991b1b; padding: 1px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 500; border: 1px solid #fecaca; }
    .badge-neutral  { background: #fef3c7; color: #92400e; padding: 1px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 500; border: 1px solid #fde68a; }

    /* FODA */
    .foda-cell { border-radius: 4px; padding: 1rem; margin-bottom: 0.5rem; }
    .foda-s { background: #f0fdf4; border-left: 3px solid #22c55e; }
    .foda-o { background: #eff6ff; border-left: 3px solid #3b82f6; }
    .foda-w { background: #fefce8; border-left: 3px solid #eab308; }
    .foda-t { background: #fef2f2; border-left: 3px solid #ef4444; }

    /* Word cloud HTML */
    .html-wordcloud { display: flex; flex-wrap: wrap; gap: 6px 12px; justify-content: center; align-items: center; min-height: 180px; padding: 1rem; }
    .html-wordcloud span { color: #0f172a; line-height: 1.2; cursor: default; transition: opacity 0.15s; }
    .html-wordcloud span:hover { opacity: 1 !important; }

    /* Hero */
    .hero-label { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; }
    .hero-title { font-size: 2.5rem; font-weight: 800; color: #0f172a; line-height: 1.1; margin: 0.5rem 0; }
    .hero-sub { font-size: 0.95rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────
# HELPERS
# ───────────────────────────────────

STOPWORDS = {
    "de","la","el","en","y","a","que","es","un","una","los","las","por","con","para","del",
    "al","se","no","lo","le","su","me","más","pero","muy","ya","si","mi","como","sin","todo",
    "esta","the","is","it","to","and","of","in","for","on","my","i","this","that","you","not",
    "are","was","but","have","has","app","its","can","so","from","with","all","an","be","been",
    "hay","les","nos","te","tan","he","ha","ser","son","era","tiene","bien","hace","solo",
    "este","eso","esto","una","uno","cada","donde","cuando","entre","después",
}

def render_stat(label, value, sub="", color="#64748b"):
    sub_html = f'<div class="sub" style="color:{color}">{sub}</div>' if sub else ""
    st.markdown(f'<div class="stat-card"><div class="label">{label}</div><div class="value">{value}</div>{sub_html}</div>', unsafe_allow_html=True)

def clean_dataframe(df):
    df = df.dropna(subset=[CONTENT_COL])
    df[CONTENT_COL] = df[CONTENT_COL].astype(str).str.strip()
    df = df[df[CONTENT_COL].str.len() > 5]
    if LIKES_COL in df.columns: df[LIKES_COL] = pd.to_numeric(df[LIKES_COL], errors="coerce").fillna(0).astype(int)
    if SCORE_COL in df.columns: df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce").fillna(3).astype(int)
    if DATE_COL in df.columns: df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df

@st.cache_data(show_spinner=False, ttl=300)
def search_apps_cached(query):
    try: return search(query, lang="es", country="ar", n_hits=8)
    except: return []

@st.cache_data(show_spinner=False, ttl=600)
def scrape_reviews_cached(app_id, lang, country, country_name, count, sort_label):
    sort_order = Sort.MOST_RELEVANT if sort_label == "Más relevantes" else Sort.NEWEST
    all_revs, token = [], None
    while len(all_revs) < count:
        batch = min(200, count - len(all_revs))
        try:
            result, token = reviews(app_id, lang=lang, country=country, sort=sort_order, count=batch, continuation_token=token)
        except: break
        if not result: break
        all_revs.extend(result)
        if not token: break
    df = pd.DataFrame(all_revs[:count])
    if not df.empty: df["country"] = country_name
    return df

def get_word_freq(df, n=60):
    text = " ".join(df[CONTENT_COL].dropna().astype(str).str.lower())
    words = [w for w in re.findall(r"[a-záéíóúñü]+", text) if len(w) > 3 and w not in STOPWORDS]
    return collections.Counter(words).most_common(n)

def render_html_wordcloud(word_freq):
    """Renders an HTML word cloud with sized spans (like Emergent prototype)."""
    if not word_freq: return
    max_c = word_freq[0][1]
    min_c = word_freq[-1][1] if len(word_freq) > 1 else max_c
    spans = []
    for word, count in word_freq:
        ratio = (count - min_c) / (max_c - min_c) if max_c != min_c else 0.5
        size = round(12 + ratio * 30)
        opacity = round(0.45 + ratio * 0.55, 2)
        weight = "700" if ratio > 0.7 else "600" if ratio > 0.4 else "500"
        spans.append(f'<span style="font-size:{size}px;opacity:{opacity};font-weight:{weight}" title="{word}: {count}">{word}</span>')
    st.markdown(f'<div class="html-wordcloud">{"".join(spans)}</div>', unsafe_allow_html=True)

def extract_topics_gemini(df, client, app_name, n=8):
    """Extract named topics with sentiment using Gemini (like prototype analyzer.py)."""
    sample = df[df[CONTENT_COL].notna()].head(150)
    reviews_text = "\n---\n".join([
        f"Rating: {r.get(SCORE_COL, 3)}/5 | {str(r.get(CONTENT_COL, ''))[:150]}"
        for _, r in sample.iterrows()
    ])
    prompt = f"""Analiza las siguientes reseñas de la app "{app_name}" y extrae los {n} temas principales que mencionan los usuarios.

Para cada tema proporciona:
- "topic": nombre del tema (máximo 3 palabras, en español)
- "count": número estimado de reseñas que lo mencionan
- "sentiment": sentimiento predominante ("positive", "negative" o "neutral")

Responde ÚNICAMENTE con un array JSON válido. Sin markdown, sin texto extra.
Ejemplo: [{{"topic": "Diseño de interfaz", "count": 45, "sentiment": "positive"}}]

Reseñas:
{reviews_text}"""
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = resp.text.strip()
        if text.startswith("```"): text = re.sub(r"```[a-z]*\n?", "", text).strip("`").strip()
        topics = json.loads(text)
        return topics[:10] if isinstance(topics, list) else []
    except: return []


# ═══════════════════════════════════
# NAVBAR (top bar)
# ═══════════════════════════════════

current_page = st.session_state.get("page", "home")

def nav_pill(label, icon, page_key, active):
    bg = "#0f172a" if active else "transparent"
    color = "white" if active else "#475569"
    hover_bg = "#334155" if not active else "#0f172a"
    return f'<a href="#" onclick="return false;" style="display:inline-flex;align-items:center;gap:4px;padding:5px 12px;border-radius:4px;font-size:0.82rem;font-weight:500;background:{bg};color:{color};text-decoration:none;transition:all 0.15s">{icon} {label}</a>'

col_logo, col_spacer, col_nav = st.columns([2, 4, 2])
with col_logo:
    st.markdown('<div style="display:flex;align-items:center;gap:0.5rem;padding-top:0.2rem"><div style="width:28px;height:28px;background:#0f172a;border-radius:4px;display:flex;align-items:center;justify-content:center"><span style="color:white;font-size:0.8rem">📊</span></div><span style="font-weight:700;font-size:1rem;color:#0f172a">PlayInsights</span></div>', unsafe_allow_html=True)
with col_nav:
    nc1, nc2 = st.columns(2)
    with nc1:
        if st.button("🔍 Buscar", key="nav_search", use_container_width=True, type="primary" if current_page=="home" else "secondary"):
            st.session_state["page"] = "home"; st.session_state.pop("search_results", None); st.rerun()
    with nc2:
        if st.button("📋 Historial", key="nav_hist", use_container_width=True, type="primary" if current_page=="history" else "secondary"):
            st.session_state["page"] = "history"; st.rerun()

st.markdown("<hr style='margin:0 0 1rem;border-color:#e2e8f0'>", unsafe_allow_html=True)


# ═══════════════════════════════════
# PAGE: HOME (Search + Config)
# ═══════════════════════════════════

def get_api_key():
    """Load Gemini API key from secrets, env var, or return empty."""
    try: return st.secrets["general"]["GEMINI_API_KEY"]
    except: return os.environ.get("GEMINI_API_KEY", "")


if st.session_state["page"] == "home":
    col_pad_l, col_main, col_pad_r = st.columns([1, 3, 1])
    with col_main:
        # Hero
        st.markdown('<div class="hero-label">Google Play Store</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Análisis de Reseñas<br>con IA</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Busca cualquier app, extrae sus reseñas y obtén análisis profundos con Gemini AI.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Search form ──
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            query = st.text_input("Buscar", placeholder="Nombre de la app o URL de Google Play...", label_visibility="collapsed")
        with col_btn:
            search_clicked = st.button("Buscar", type="primary", use_container_width=True, key="search_btn")

        # Config row with labels (matches prototype)
        col_lang, col_count, col_countries, col_sort = st.columns(4)
        with col_lang:
            st.markdown('<span style="font-size:0.72rem;color:#64748b;font-weight:500">Idioma:</span>', unsafe_allow_html=True)
            lang_name = st.selectbox("Idioma", list(LANGUAGES.keys()), index=0, label_visibility="collapsed")
        with col_count:
            st.markdown('<span style="font-size:0.72rem;color:#64748b;font-weight:500">Reseñas:</span>', unsafe_allow_html=True)
            review_count = st.selectbox("Reseñas", [100, 200, 500, 1000], index=0, label_visibility="collapsed", format_func=lambda x: f"{x:,} reseñas")
        with col_countries:
            st.markdown('<span style="font-size:0.72rem;color:#64748b;font-weight:500">Países:</span>', unsafe_allow_html=True)
            countries = st.multiselect("Países", list(COUNTRIES.keys()), default=["Argentina"], label_visibility="collapsed")
        with col_sort:
            st.markdown('<span style="font-size:0.72rem;color:#64748b;font-weight:500">Ordenar por:</span>', unsafe_allow_html=True)
            sort_order = st.selectbox("Orden", ["Más relevantes", "Más recientes"], label_visibility="collapsed")

        # Clear results when query is emptied
        if not query.strip():
            st.session_state.pop("search_results", None)

        # Search results
        if search_clicked and query.strip():
            results = search_apps_cached(query.strip())
            if not results:
                st.markdown("""<div style="display:flex;align-items:center;gap:0.5rem;color:#dc2626;background:#fef2f2;border:1px solid #fecaca;border-radius:4px;padding:0.8rem 1rem;margin:1rem 0;font-size:0.85rem">
                    <span>⚠️</span><span>No se encontraron apps. Probá con otro término.</span>
                </div>""", unsafe_allow_html=True)
            st.session_state["search_results"] = results

        if st.session_state.get("search_results"):
            results = st.session_state["search_results"]
            st.markdown(f'<p style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin:1rem 0 0.8rem">{len(results)} resultado{"s" if len(results)!=1 else ""} encontrado{"s" if len(results)!=1 else ""}</p>', unsafe_allow_html=True)

            for idx, app_data in enumerate(results):
                app_id = app_data.get("appId", "") or app_data.get("id", "")
                title = app_data.get("title", app_id)
                icon = app_data.get("icon", "")
                dev = app_data.get("developer", "")
                score = app_data.get("score", 0) or 0
                installs = app_data.get("installs", "")
                genre = app_data.get("genre", "")

                if not app_id:
                    continue

                col_card, col_action = st.columns([5, 1])
                with col_card:
                    badge = f'<span style="font-size:0.65rem;color:#64748b;border:1px solid #e2e8f0;padding:1px 6px;border-radius:3px;margin-left:0.5rem">{genre}</span>' if genre else ""
                    score_html = f'<span style="font-size:0.7rem;color:#64748b">⭐ {score:.1f}</span>' if score else ""
                    inst_html = f'<span style="font-size:0.7rem;color:#94a3b8">⬇ {installs}</span>' if installs and installs != 'N/A' else ""
                    st.markdown(f"""<div class="app-card">
                        <img src="{icon}" style="width:48px;height:48px;border-radius:4px" onerror="this.style.display='none'">
                        <div style="flex:1;min-width:0">
                            <div style="display:flex;align-items:center;gap:0.3rem"><span style="font-weight:600;font-size:0.9rem;color:#0f172a">{title}</span>{badge}</div>
                            <div style="font-size:0.75rem;color:#64748b">{dev}</div>
                            <div style="display:flex;gap:0.8rem;margin-top:0.2rem">{score_html}{inst_html}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col_action:
                    st.markdown("<div style='padding-top:1rem'></div>", unsafe_allow_html=True)
                    if st.button("Analizar ›", key=f"analyze_{app_id}_{idx}", use_container_width=True):
                        st.session_state.update({
                            "sel_app_id": app_id, "sel_app_name": title, "sel_app_icon": icon,
                            "sel_app_dev": dev, "sel_app_genre": genre, "sel_app_score": score,
                            "sel_app_installs": installs, "sel_lang": LANGUAGES[lang_name],
                            "sel_count": review_count, "sel_countries": countries,
                            "sel_sort": sort_order,
                        })
                        st.session_state["page"] = "loading"
                        st.rerun()
        else:
            # Empty state (like prototype)
            st.markdown("""<div style="text-align:center;padding:3rem 0">
                <div style="width:48px;height:48px;background:#f1f5f9;border-radius:4px;display:flex;align-items:center;justify-content:center;margin:0 auto 1rem"><span style="font-size:1.2rem;color:#94a3b8">🔍</span></div>
                <p style="color:#94a3b8;font-size:0.9rem">Busca una app para comenzar el análisis</p>
                <p style="color:#cbd5e1;font-size:0.78rem;margin-top:0.3rem">Ej: "Spotify", "WhatsApp" o pegá la URL de Google Play</p>
            </div>""", unsafe_allow_html=True)

    st.stop()


# ═══════════════════════════════════
# PAGE: LOADING (scrape + analyze)
# ═══════════════════════════════════

if st.session_state["page"] == "loading":
    app_id = st.session_state.get("sel_app_id", "")
    app_name = st.session_state.get("sel_app_name", app_id)
    lang = st.session_state.get("sel_lang", "es")
    count = st.session_state.get("sel_count", 100)
    sel_countries = st.session_state.get("sel_countries", ["Argentina"])
    sort_label = st.session_state.get("sel_sort", "Más relevantes")
    api_key = get_api_key()

    # Handle URL
    clean_id = str(app_id).strip()
    if "play.google.com" in clean_id:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(clean_id).query)
        clean_id = params.get("id", [clean_id])[0]

    # Try to get app info
    if not st.session_state.get("sel_app_icon"):
        try:
            info = gplay_app(clean_id, lang=lang, country="ar")
            st.session_state["sel_app_name"] = info.get("title", clean_id)
            st.session_state["sel_app_icon"] = info.get("icon", "")
            st.session_state["sel_app_dev"] = info.get("developer", "")
            st.session_state["sel_app_genre"] = info.get("genre", "")
            st.session_state["sel_app_score"] = info.get("score", 0)
            st.session_state["sel_app_installs"] = info.get("installs", "")
            app_name = st.session_state["sel_app_name"]
        except: pass

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(f"""<div style="text-align:center;padding:3rem 0">
            <div style="width:48px;height:48px;border:2px solid #0f172a;border-top-color:transparent;border-radius:50%;margin:0 auto 1rem;animation:spin 1s linear infinite"></div>
            <h2 style="font-size:1.3rem;margin-bottom:0.3rem">Analizando {app_name}</h2>
            <p style="color:#64748b;font-size:0.9rem">Esto puede tardar unos minutos...</p>
        </div>
        <style>@keyframes spin {{ 0%{{transform:rotate(0deg)}} 100%{{transform:rotate(360deg)}} }}</style>""", unsafe_allow_html=True)

        progress = st.progress(0, text="Iniciando...")
        n_countries = len(sel_countries) if sel_countries else 1
        per_country = max(50, count // n_countries)

        # Step 1: Scrape
        dfs = []
        for i, country_name in enumerate(sel_countries):
            pct = int(10 + (i / n_countries) * 40)
            progress.progress(pct, text=f"📱 Descargando reseñas de {country_name} ({per_country})...")
            try:
                df_part = scrape_reviews_cached(clean_id, lang, COUNTRIES[country_name], country_name, per_country, sort_label)
                if not df_part.empty: dfs.append(df_part)
            except Exception as e:
                st.warning(f"Error en {country_name}: {e}")

        if not dfs:
            st.error("No se pudieron obtener reseñas.")
            if st.button("← Volver"): st.session_state["page"] = "home"; st.rerun()
            st.stop()

        df = pd.concat(dfs, ignore_index=True)
        df = clean_dataframe(df)

        # Step 2: Classify sentiment
        progress.progress(60, text="🧠 Clasificando sentimientos...")
        df = classify_sentiment(df)

        # Step 3: Extract AI topics
        topics = []
        if api_key:
            progress.progress(75, text="🤖 Extrayendo temas con IA (Gemini)...")
            try:
                client = genai.Client(api_key=api_key)
                topics = extract_topics_gemini(df, client, app_name)
            except: pass

        # Step 4: Generate deep analysis (with retry for 503)
        ai_analysis = ""
        if api_key:
            progress.progress(85, text="📝 Generando análisis profundo...")
            client = genai.Client(api_key=api_key)
            counts = df["sentiment"].value_counts()
            total_r = len(df)
            n_pos_r, n_neg_r, n_neu_r = counts.get("positive", 0), counts.get("negative", 0), counts.get("neutral", 0)
            avg = df[SCORE_COL].mean() if SCORE_COL in df.columns else 0
            topics_text = ", ".join([t.get("topic", "") for t in topics[:6]])

            pos_sample = df[df["sentiment"]=="positive"].head(15)
            neg_sample = df[df["sentiment"]=="negative"].head(15)
            pos_text = "\n".join([f"- [{r.get(SCORE_COL,0)}★] {str(r.get(CONTENT_COL,''))[:200]}" for _,r in pos_sample.iterrows()])
            neg_text = "\n".join([f"- [{r.get(SCORE_COL,0)}★] {str(r.get(CONTENT_COL,''))[:200]}" for _,r in neg_sample.iterrows()])

            prompt = f"""Realiza un análisis profundo de las reseñas de la app "{app_name}" en Google Play Store.

## Datos clave:
- Reseñas analizadas: {total_r}
- Rating promedio: {avg:.1f}/5
- Sentimiento: {n_pos_r/total_r*100:.0f}% positivo, {n_neg_r/total_r*100:.0f}% negativo, {n_neu_r/total_r*100:.0f}% neutro
- Temas principales: {topics_text}

## Muestra de reseñas POSITIVAS:
{pos_text}

## Muestra de reseñas NEGATIVAS:
{neg_text}

Proporciona un análisis estructurado con estas secciones usando formato markdown:

## Resumen Ejecutivo
(2-3 párrafos con el estado general)

## Fortalezas Principales
(lista de puntos)

## Problemas Críticos
(lista con impacto)

## Perfil del Usuario
(qué tipo de usuarios)

## Recomendaciones de Mejora
(acciones concretas priorizadas)

Responde en español. Sé específico y accionable."""

            for attempt in range(3):
                try:
                    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                    ai_analysis = resp.text
                    break
                except Exception as e:
                    if "503" in str(e) and attempt < 2:
                        progress.progress(85, text=f"📝 Reintentando análisis ({attempt+2}/3)...")
                        time.sleep(3 * (attempt + 1))
                    else:
                        ai_analysis = f"Error al generar análisis: {str(e)[:200]}"

        progress.progress(95, text="💾 Guardando resultados...")

        # Store results
        st.session_state["df_analyzed"] = df
        st.session_state["topics"] = topics
        st.session_state["ai_analysis"] = ai_analysis

        # Add to history
        entry = {"name": app_name, "icon": st.session_state.get("sel_app_icon",""), "count": len(df),
                 "dev": st.session_state.get("sel_app_dev",""), "score": st.session_state.get("sel_app_score",0)}
        if not any(h["name"] == app_name for h in st.session_state["history"]):
            st.session_state["history"].append(entry)

        progress.progress(100, text="✅ ¡Análisis completado!")
        st.session_state["page"] = "dashboard"
        st.rerun()

    st.stop()


# ═══════════════════════════════════
# PAGE: HISTORY
# ═══════════════════════════════════

if st.session_state["page"] == "history":
    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        st.markdown('<div class="hero-label">Análisis guardados</div>', unsafe_allow_html=True)
        st.markdown('<h1 style="font-size:2rem;margin:0.3rem 0 1.5rem">Historial</h1>', unsafe_allow_html=True)

        hist = st.session_state.get("history", [])
        if not hist:
            st.markdown("""<div style="text-align:center;padding:4rem 0">
                <div style="font-size:2rem;margin-bottom:0.5rem">📋</div>
                <p style="color:#94a3b8;font-size:0.9rem">No hay análisis guardados</p>
            </div>""", unsafe_allow_html=True)
            if st.button("🔍 Buscar primera app", use_container_width=True):
                st.session_state["page"] = "home"; st.rerun()
        else:
            for entry in reversed(hist):
                icon_html = f'<img src="{entry["icon"]}" style="width:40px;height:40px;border-radius:4px">' if entry.get("icon") else ""
                score_html = f'⭐ {entry["score"]:.1f}' if entry.get("score") else ""
                st.markdown(f"""<div class="app-card">
                    {icon_html}
                    <div style="flex:1">
                        <div style="font-weight:600;font-size:0.9rem;color:#0f172a">{entry["name"]}</div>
                        <div style="font-size:0.75rem;color:#64748b">{entry.get("dev","")} · {entry["count"]} reseñas · {score_html}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════

if st.session_state["page"] != "dashboard":
    st.stop()

df = st.session_state["df_analyzed"]
topics = st.session_state.get("topics", [])
ai_analysis = st.session_state.get("ai_analysis", "")
api_key = st.session_state.get("api_key", "")

total = len(df)
counts = df["sentiment"].value_counts()
n_pos, n_neg, n_neu = counts.get("positive",0), counts.get("negative",0), counts.get("neutral",0)
avg_score = df[SCORE_COL].mean() if SCORE_COL in df.columns else 0

CHART_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#475569", family="Inter"), margin=dict(t=10,b=20,l=40,r=20))

# ── App header ──
app_icon = st.session_state.get("sel_app_icon","")
app_name = st.session_state.get("sel_app_name","")
app_dev = st.session_state.get("sel_app_dev","")
app_genre = st.session_state.get("sel_app_genre","")
app_play_score = st.session_state.get("sel_app_score","")
app_installs = st.session_state.get("sel_app_installs","")

col_back, col_header, col_rating = st.columns([0.5, 5, 2])
with col_back:
    if st.button("←", key="dash_back"):
        for k in ["df_analyzed","topics","ai_analysis","praised","criticized","foda"]:
            st.session_state.pop(k, None)
        st.session_state["page"] = "home"; st.rerun()
with col_header:
    icon_html = f'<img src="{app_icon}" style="width:48px;height:48px;border-radius:4px">' if app_icon else ""
    genre_html = f' · {app_genre}' if app_genre else ""
    st.markdown(f'<div style="display:flex;align-items:center;gap:1rem">{icon_html}<div><div style="font-weight:700;font-size:1.2rem;color:#0f172a">{app_name}</div><div style="font-size:0.78rem;color:#64748b">{app_dev}{genre_html}</div></div></div>', unsafe_allow_html=True)
with col_rating:
    score_html = f'<span style="background:#f8fafc;border:1px solid #e2e8f0;padding:3px 10px;border-radius:4px;font-weight:600;font-size:0.85rem">⭐ {app_play_score:.1f}</span>' if app_play_score else ""
    inst_html = f'<span style="color:#94a3b8;font-size:0.78rem;margin-left:0.5rem">{app_installs}</span>' if app_installs else ""
    st.markdown(f'<div style="text-align:right;padding-top:0.5rem">{score_html}{inst_html}</div>', unsafe_allow_html=True)

# ── Tabs (Análisis IA first, then Métricas, then Reseñas) ──
tab_ia, tab_met, tab_rev = st.tabs(["🤖 Análisis IA", "📊 Métricas", f"💬 Reseñas ({total:,})"])

# ── TAB: ANÁLISIS IA (first) ──
with tab_ia:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid #f1f5f9">
        <div style="width:24px;height:24px;background:#0f172a;border-radius:4px;display:flex;align-items:center;justify-content:center"><span style="color:white;font-size:0.7rem">🤖</span></div>
        <span style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1.5px;font-weight:600">Análisis Profundo — Gemini AI</span>
    </div>""", unsafe_allow_html=True)

    api_key = get_api_key()
    if ai_analysis and not ai_analysis.startswith("Error"):
        # Parse markdown sections into styled cards
        sections = re.split(r'(?=^## )', ai_analysis, flags=re.MULTILINE)
        section_icons = {"resumen": "📋", "fortaleza": "💪", "problema": "⚠️", "perfil": "👥", "recomendacion": "🎯"}

        for section in sections:
            section = section.strip()
            if not section:
                continue
            # Extract title if it starts with ##
            match = re.match(r'^## (.+)', section)
            if match:
                sec_title = match.group(1).strip()
                sec_body = section[match.end():].strip()
                # Pick icon
                icon = "📊"
                for keyword, ic in section_icons.items():
                    if keyword in sec_title.lower():
                        icon = ic; break
                st.markdown(f"""<div class="chart-card" style="margin-bottom:0.8rem">
                    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.8rem">
                        <span style="font-size:1.1rem">{icon}</span>
                        <span style="font-weight:700;font-size:1rem;color:#0f172a">{sec_title}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.markdown(sec_body)
                st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.markdown(section)

        # Download buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📝 Descargar Markdown", data=ai_analysis.encode("utf-8"),
                             file_name=f"analisis_{app_name.replace(' ','_')}.md", mime="text/markdown", use_container_width=True)
        with col2:
            try:
                pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Helvetica", size=16); pdf.cell(0, 10, f"Analisis - {app_name}", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", size=10)
                for line in ai_analysis.replace("##","").replace("**","").replace("*","").replace("#","").split("\n"):
                    if line.strip():
                        pdf.multi_cell(0, 5, line.strip().encode('latin-1', errors='replace').decode('latin-1')); pdf.ln(1)
                st.download_button("📄 Descargar PDF", data=bytes(pdf.output()),
                                 file_name=f"analisis_{app_name.replace(' ','_')}.pdf", mime="application/pdf", use_container_width=True)
            except: pass
    elif not api_key:
        st.markdown("""<div style="text-align:center;padding:3rem 0">
            <div style="font-size:2.5rem;margin-bottom:0.5rem">🔑</div>
            <p style="color:#64748b;font-size:0.95rem;margin-bottom:0.3rem">API Key de Gemini no configurada</p>
            <p style="color:#94a3b8;font-size:0.8rem">Configurá tu key en <code>.streamlit/secrets.toml</code> para habilitar el análisis con IA</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning(ai_analysis if ai_analysis else "No se pudo generar el análisis.")


# ── TAB: MÉTRICAS ──
with tab_met:
    c1,c2,c3,c4 = st.columns(4)
    with c1: render_stat("Total Reseñas", f"{total:,}")
    with c2: render_stat("Rating Promedio", f"{avg_score:.2f}/5", color="#f59e0b")
    with c3: render_stat("Sentimiento Positivo", f"{n_pos/total*100:.1f}%", f"{n_pos:,} reseñas", "#22c55e")
    with c4: render_stat("Sentimiento Negativo", f"{n_neg/total*100:.1f}%", f"{n_neg:,} reseñas", "#ef4444")
    st.markdown("<br>", unsafe_allow_html=True)

    # Donut + Ratings (compact)
    col_d, col_r = st.columns(2)
    with col_d:
        st.markdown('<div class="chart-card"><div class="chart-card-title">Distribución de Sentimiento</div>', unsafe_allow_html=True)
        sent_df = pd.DataFrame({"Sentimiento":["Positivo","Neutro","Negativo"],"Cantidad":[n_pos,n_neu,n_neg]})
        fig = px.pie(sent_df, values="Cantidad", names="Sentimiento", color="Sentimiento",
                     color_discrete_map={"Positivo":COLORS["positive"],"Neutro":COLORS["neutral"],"Negativo":COLORS["negative"]}, hole=0.55)
        fig.update_layout(**CHART_LAYOUT, showlegend=True, height=220)
        fig.update_traces(textinfo="percent+value", textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="chart-card"><div class="chart-card-title">Distribución de Ratings</div>', unsafe_allow_html=True)
        if SCORE_COL in df.columns:
            sc = df[SCORE_COL].value_counts().sort_index()
            fig2 = px.bar(x=[f"{i}★" for i in sc.index], y=sc.values, labels={"x":"","y":"Cantidad"})
            fig2.update_traces(marker_color="#0f172a")
            fig2.update_layout(**CHART_LAYOUT, showlegend=False, height=220)
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Rating over time (monthly)
    if DATE_COL in df.columns and df[DATE_COL].notna().sum() > 5:
        st.markdown('<div class="chart-card"><div class="chart-card-title">Evolución del Rating en el Tiempo</div>', unsafe_allow_html=True)
        df_t = df.dropna(subset=[DATE_COL]).copy()
        df_t["month"] = df_t[DATE_COL].dt.to_period("M")
        monthly = df_t.groupby("month")[SCORE_COL].mean().reset_index() if SCORE_COL in df_t.columns else pd.DataFrame()
        if not monthly.empty:
            monthly["dt"] = monthly["month"].dt.to_timestamp()
            monthly = monthly.sort_values("dt").tail(18)
            fig3 = px.line(monthly, x="dt", y=SCORE_COL, labels={"dt":"","score":"Score"})
            fig3.update_traces(line_color="#0f172a", line_width=2, mode="lines+markers", marker_size=5)
            fig3.update_layout(**CHART_LAYOUT, height=200, yaxis_range=[1,5.2], xaxis=dict(dtick="M1",tickformat="%Y-%m"))
            st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Topics + Word Cloud (compact)
    col_topics, col_wc = st.columns(2)
    with col_topics:
        st.markdown('<div class="chart-card"><div class="chart-card-title">Temas Más Mencionados</div>', unsafe_allow_html=True)
        if topics:
            topics_df = pd.DataFrame(topics[:8])
            if "topic" in topics_df.columns and "count" in topics_df.columns:
                fig4 = px.bar(topics_df, x="count", y="topic", orientation="h", labels={"count":"Menciones","topic":""})
                colors_list = [COLORS.get(t.get("sentiment",""), "#334155") for t in topics[:8]]
                fig4.update_traces(marker_color=colors_list)
                fig4.update_layout(**CHART_LAYOUT, height=220, showlegend=False, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig4, use_container_width=True)
        else:
            wf = get_word_freq(df, 10)
            if wf:
                wdf = pd.DataFrame(wf, columns=["Palabra","Frecuencia"])
                fig4 = px.bar(wdf, x="Frecuencia", y="Palabra", orientation="h", color="Frecuencia", color_continuous_scale=["#ef4444","#22c55e"])
                fig4.update_layout(**CHART_LAYOUT, height=220, showlegend=False, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_wc:
        st.markdown('<div class="chart-card"><div class="chart-card-title">Nube de Palabras</div>', unsafe_allow_html=True)
        wf = get_word_freq(df, 60)
        render_html_wordcloud(wf)
        st.markdown('</div>', unsafe_allow_html=True)


# ── TAB 3: RESEÑAS ──
BADGE_MAP = {"positive":"badge-positive","negative":"badge-negative","neutral":"badge-neutral"}
LABEL_MAP = {"positive":"Positivo","negative":"Negativo","neutral":"Neutro"}

with tab_rev:
    col_s, col_f1, col_f2, col_cnt = st.columns([3, 1.5, 1.5, 1])
    with col_s: rev_search = st.text_input("Buscar", placeholder="Buscar en reseñas...", label_visibility="collapsed", key="rev_search")
    with col_f1: sent_f = st.selectbox("Sent", ["Todos","Positivo","Neutro","Negativo"], label_visibility="collapsed", key="rev_sent")
    with col_f2: score_f = st.selectbox("Score", ["Todos"]+[f"{i}★" for i in range(5,0,-1)], label_visibility="collapsed", key="rev_score")

    filtered = df.copy()
    if rev_search: filtered = filtered[filtered[CONTENT_COL].str.lower().str.contains(rev_search.lower(), na=False)]
    if sent_f != "Todos":
        m = {"Positivo":"positive","Neutro":"neutral","Negativo":"negative"}
        filtered = filtered[filtered["sentiment"]==m.get(sent_f)]
    if score_f != "Todos" and SCORE_COL in filtered.columns:
        filtered = filtered[filtered[SCORE_COL]==int(score_f[0])]
    with col_cnt: st.markdown(f'<div style="padding-top:0.5rem;color:#94a3b8;font-size:0.78rem;text-align:right">{len(filtered):,} reseña{"s" if len(filtered)!=1 else ""}</div>', unsafe_allow_html=True)

    # Pagination
    PER_PAGE = 20
    total_p = max(1, (len(filtered)+PER_PAGE-1)//PER_PAGE)
    if "rev_page" not in st.session_state: st.session_state["rev_page"] = 1
    pg = min(st.session_state["rev_page"], total_p)
    page_df = filtered.iloc[(pg-1)*PER_PAGE : pg*PER_PAGE]

    for _, row in page_df.iterrows():
        content = str(row.get(CONTENT_COL,""))
        score = row.get(SCORE_COL, 0)
        sent = row.get("sentiment","neutral")
        uname = row.get("userName","Anónimo")
        date = row.get(DATE_COL, None)
        ver = row.get(VERSION_COL, "")
        likes = row.get(LIKES_COL, 0)

        stars = "".join([f'<span style="color:{"#f59e0b" if i<score else "#cbd5e1"};font-size:0.65rem">★</span>' for i in range(5)])
        badge = f'<span class="{BADGE_MAP.get(sent,"")}">{LABEL_MAP.get(sent,sent)}</span>'
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d") if pd.notna(date) else ""
        ver_html = f'<div style="font-size:0.68rem;color:#cbd5e1;margin-top:0.2rem">v{ver}</div>' if ver else ""
        likes_html = f'<span style="color:#94a3b8;font-size:0.7rem">👍 {int(likes)}</span>' if likes else ""

        st.markdown(f"""<div class="review-card">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem">
                <span style="font-size:0.78rem;font-weight:600;color:#334155">{uname}</span>
                <span>{stars}</span> {badge}
                <span style="margin-left:auto;font-size:0.7rem;color:#94a3b8">{likes_html} {date_str}</span>
            </div>
            <div style="font-size:0.8rem;color:#475569;line-height:1.6">{content[:500]}{'...' if len(content)>500 else ''}</div>
            {ver_html}
        </div>""", unsafe_allow_html=True)

    if total_p > 1:
        col_pv, col_pi, col_nx = st.columns([1,3,1])
        with col_pv:
            if st.button("← Anterior", disabled=pg<=1, key="pg_prev", use_container_width=True): st.session_state["rev_page"]=pg-1; st.rerun()
        with col_pi: st.markdown(f'<div style="text-align:center;padding-top:0.5rem;color:#64748b;font-size:0.8rem">Página {pg} de {total_p}</div>', unsafe_allow_html=True)
        with col_nx:
            if st.button("Siguiente →", disabled=pg>=total_p, key="pg_next", use_container_width=True): st.session_state["rev_page"]=pg+1; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    csv = filtered[[c for c in [CONTENT_COL,"sentiment",SCORE_COL,LIKES_COL,DATE_COL,VERSION_COL,"userName"] if c in filtered.columns]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", data=csv, file_name="reviews.csv", mime="text/csv")
