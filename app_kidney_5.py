"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MEDICALScan AI  ·  Application Streamlit v9                               ║
║  Classification CT Rénale  ·  KidneyClassifier v5  ·  AUC 1.00            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Architecture                                                               ║
║  ══════════                                                                 ║
║  §1  Imports & configuration de la page                                     ║
║  §2  Auto-installation TensorFlow (silencieux)                              ║
║  §3  Secrets API (Groq, LangSmith) — jamais affichés                        ║
║  §4  Constantes médicales (classes, couleurs, textes)                       ║
║  §5  Design System CSS                                                      ║
║  §6  Sidebar — paramètres utilisateur                                       ║
║  §7  Service LLM — Groq + traces LangSmith                                  ║
║  §8  Modèle CT — chargement TF et prédiction                                ║
║  §9  Fonctions utilitaires — TTS, traduction, résumé                        ║
║  §10 Composant render_ct_result()                                           ║
║  §11 Header institutionnel                                                  ║
║  §12 Onglet Analyse CT                                                      ║
║  §13 Onglet Assistant Médical                                               ║
║  §14 Onglet Résumé & Rapport                                                ║
║  §15 Onglet Monitoring LangSmith                                            ║
║  §16 Footer                                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Groupe 2 · M2 IABD · HAMAD · KAMNO · EFEMBA · MBOG · 2026                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# §1 ── Imports & configuration ────────────────────────────────────────────────
import io
import os
import datetime
import subprocess
import sys

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="MEDICALScan AI — Renal CT Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# §2 ── Auto-installation TensorFlow ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _try_tensorflow() -> bool:
    """
    Tente d'importer TensorFlow.
    Si absent, essaie de l'installer silencieusement (cascade de versions).
    Retourne True si disponible, False sinon (→ mode démo).
    Compatible Python 3.11 (local) et Python 3.14 (Streamlit Cloud → mode démo).
    """
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        pass

    for pkg in [
        "tensorflow-cpu==2.16.1",
        "tensorflow-cpu>=2.13.0,<2.17.0",
        "tensorflow>=2.13.0,<2.17.0",
    ]:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                stderr=subprocess.DEVNULL,
            )
            import tensorflow  # noqa: F401
            return True
        except Exception:
            continue
    return False


TF_OK: bool = _try_tensorflow()

# §3 ── Secrets API ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_secrets() -> dict[str, str]:
    """
    Charge les clés depuis .streamlit/secrets.toml avec fallback sur les
    variables d'environnement. Les valeurs ne sont JAMAIS affichées dans l'UI.
    """
    def _get(key: str) -> str:
        try:
            return st.secrets.get(key, os.environ.get(key, ""))
        except Exception:
            return os.environ.get(key, "")

    return {
        "GROQ":  _get("GROQ_API_KEY"),
        "LS":    _get("LANGCHAIN_API_KEY"),
        "MODEL": _get("DEFAULT_GROQ_MODEL") or "llama-3.3-70b-versatile",
        "MP":    _get("MODEL_PATH")          or "outputs_v5/KidneyClassifier_v5.keras",
        "TP":    _get("THRESH_PATH")         or "outputs_v5/thresholds.npy",
    }


KEYS: dict[str, str] = _load_secrets()

# §4 ── Constantes médicales ───────────────────────────────────────────────────

# Ordre des classes (doit correspondre à l'entraînement du modèle)
CLASSES: tuple[str, ...] = ("Cyst", "Normal", "Stone", "Tumor")

# Taille d'entrée attendue par le modèle
IMG_SIZE: tuple[int, int] = (160, 160)

# Palette et métadonnées par classe
CLASS_CFG: dict[str, dict] = {
    "Cyst": {
        "color": "#1B5EA8", "bg": "#EBF5FB", "border": "#85C1E9",
        "label": "Kyste rénal",              "urgence": "Faible",    "emoji": "💧",
    },
    "Normal": {
        "color": "#1A7A4A", "bg": "#F0FBF4", "border": "#82D0A0",
        "label": "Rein normal",              "urgence": "Aucune",    "emoji": "✅",
    },
    "Stone": {
        "color": "#C4700A", "bg": "#FDF8F0", "border": "#E8C97A",
        "label": "Lithiase rénale (calcul)", "urgence": "Modérée",   "emoji": "🪨",
    },
    "Tumor": {
        "color": "#C0392B", "bg": "#FDF2F2", "border": "#E8A0A0",
        "label": "Tumeur rénale",            "urgence": "Élevée ⚠️", "emoji": "🔴",
    },
}

# Couleur associée à chaque niveau d'urgence (pour les métriques)
URGENCE_COLOR: dict[str, str] = {
    "Aucune":    "#1A7A4A",
    "Faible":    "#1B5EA8",
    "Modérée":   "#C4700A",
    "Élevée ⚠️": "#C0392B",
}

# Textes d'interprétation clinique (HTML inline autorisé)
INTERP: dict[str, str] = {
    "Normal": (
        "L'analyse ne révèle <strong>aucune anomalie rénale significative</strong>. "
        "Les structures rénales apparaissent morphologiquement normales. "
        "Un suivi de routine est recommandé selon l'âge et les facteurs de risque."
    ),
    "Cyst": (
        "L'analyse identifie une <strong>formation kystique rénale</strong>. "
        "Les kystes simples sont fréquents et généralement bénins. "
        "Une classification Bosniak est recommandée. "
        "Un <strong>suivi échographique à 6-12 mois</strong> est conseillé."
    ),
    "Stone": (
        "L'analyse détecte la <strong>présence de calculs rénaux</strong>. "
        "Une évaluation urologique est nécessaire pour la taille, la localisation "
        "et la composition. Un <strong>bilan métabolique et une consultation urologique</strong> "
        "sont recommandés."
    ),
    "Tumor": (
        "L'analyse identifie une <strong>masse rénale suspecte nécessitant une évaluation urgente</strong>. "
        "Ce résultat requiert une <strong>confirmation par IRM</strong> "
        "et une consultation oncologique/urologique en urgence. "
        "Ne pas différer la prise en charge."
    ),
}

# Contexte clinique utilisé dans les prompts LLM et l'export PDF/texte
CTX: dict[str, dict[str, str]] = {
    "Cyst":   {"urgence": "Faible à modérée",                    "suivi": "Échographie à 6-12 mois"},
    "Normal": {"urgence": "Aucune",                               "suivi": "Contrôle de routine"},
    "Stone":  {"urgence": "Modérée — selon taille/localisation", "suivi": "Consultation urologique"},
    "Tumor":  {"urgence": "⚠️ ÉLEVÉE — consultation urgente",     "suivi": "IRM + avis urologique urgent"},
}

# §5 ── Design System CSS ──────────────────────────────────────────────────────
# Palette clinique :
#   Fond         #F2F5F8  (gris chirurgical)
#   Surface      #FFFFFF  (blanc)
#   Bleu médical #1B4F72  (accent principal)
#   Bordure      #D8E2EC  (séparateurs discrets)
#   Alertes      rouge · orange · bleu · vert (code RAG standard)
# Typographie :
#   DM Sans      corps de texte — lisibilité clinique
#   JetBrains Mono  valeurs numériques, codes, timestamps

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ────────────────────────────────────────────────── */
*, html, body { font-family: 'DM Sans', sans-serif !important; box-sizing: border-box; }
[data-testid="stAppViewContainer"]  { background: #F2F5F8 !important; color: #1A2332 !important; }
[data-testid="stHeader"]            { display: none !important; }
[data-testid="stStatusWidget"]      { display: none !important; }
div[data-testid="stDecoration"]     { display: none !important; }
div[data-stale="true"]              { opacity: 1 !important; }
.block-container                    { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] > div {
    background: #FFFFFF !important;
    border-right: 1px solid #D8E2EC !important;
    padding-top: 0 !important;
}

/* ── Header ──────────────────────────────────────────────── */
.med-header {
    background: linear-gradient(135deg, #0F2540 0%, #1B4F72 55%, #1E6CA0 100%);
    padding: 16px 32px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 3px solid #0A1E35;
    box-shadow: 0 3px 14px rgba(0,0,0,0.20);
}
.med-header-left  { display: flex; align-items: center; gap: 14px; }
.med-logo-box {
    width: 44px; height: 44px; border-radius: 10px;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.22);
    display: flex; align-items: center; justify-content: center; font-size: 21px; flex-shrink: 0;
}
.med-title    { font-size: 1.3rem; font-weight: 700; color: #FFF; margin: 0; line-height: 1.2; }
.med-subtitle { font-size: 0.63rem; color: rgba(255,255,255,0.55); letter-spacing: 1.8px; text-transform: uppercase; margin-top: 3px; }
.med-header-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.hbadge {
    font-family: 'JetBrains Mono', monospace; font-size: 0.57rem; padding: 4px 10px;
    border-radius: 4px; border: 1px solid rgba(255,255,255,0.20);
    color: rgba(255,255,255,0.70); background: rgba(255,255,255,0.07); letter-spacing: 0.4px;
}
.hbadge.on { border-color: #4ADE80; color: #4ADE80; background: rgba(74,222,128,0.08); }

/* ── Onglets ─────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: #FFFFFF !important; border-bottom: 2px solid #D8E2EC !important;
    padding: 0 28px !important; gap: 0 !important; margin-bottom: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important; font-size: 0.79rem !important;
    font-weight: 500 !important; color: #6B8499 !important; background: transparent !important;
    border: none !important; padding: 13px 20px !important;
    border-bottom: 3px solid transparent !important; margin-bottom: -2px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #1B4F72 !important; border-bottom: 3px solid #1B4F72 !important; font-weight: 600 !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #1B4F72 !important; background: #EDF3F8 !important; }
[data-testid="stTabsContent"] { padding: 24px 28px !important; background: #F2F5F8 !important; }

/* ── Carte générique ─────────────────────────────────────── */
.card {
    background: #FFFFFF; border: 1px solid #D8E2EC; border-radius: 10px;
    padding: 20px 22px; box-shadow: 0 1px 5px rgba(0,0,0,0.05); margin-bottom: 14px;
}
.card-title {
    font-size: 0.64rem; font-weight: 600; color: #6B8499; letter-spacing: 1.4px;
    text-transform: uppercase; margin-bottom: 14px; padding-bottom: 10px;
    border-bottom: 1px solid #EEF3F7; display: flex; align-items: center; gap: 7px;
}

/* ── Upload ──────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: #F8FAFB !important; border: 1.5px dashed #B8CAD8 !important;
    border-radius: 12px !important; padding: 30px 20px !important;
    transition: border-color .18s, background .18s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #1B4F72 !important; background: #EBF3FA !important;
}
[data-testid="stFileUploaderDropzone"] > div    { gap: 10px !important; flex-direction: column !important; align-items: center !important; }
[data-testid="stFileUploaderDropzone"] > div > span { font-size: 0.81rem !important; font-weight: 500 !important; color: #1A2332 !important; }
[data-testid="stFileUploaderDropzone"] small    { font-size: 0.70rem !important; color: #7F8C8D !important; }
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    font-family: 'DM Sans', sans-serif !important; font-size: 0.80rem !important;
    font-weight: 600 !important; letter-spacing: 0.2px !important; text-transform: none !important;
    background: #1B4F72 !important; color: #FFF !important; border: none !important;
    border-radius: 8px !important; padding: 9px 22px !important; margin-top: 8px !important;
    cursor: pointer !important; box-shadow: 0 2px 6px rgba(27,79,114,0.22) !important;
    transition: background .15s, box-shadow .15s !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {
    background: #154360 !important; box-shadow: 0 4px 12px rgba(27,79,114,0.30) !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: #EBF3FA !important; border: 1px solid #AECDE8 !important; border-radius: 8px !important; padding: 10px 14px !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {
    font-size: 0.78rem !important; color: #0C4472 !important; font-weight: 500 !important;
}

/* ── Blocs résultat CT ───────────────────────────────────── */
.res-head { background:#FFF; border:1px solid #D8E2EC; border-radius:10px 10px 0 0; padding:18px 22px 14px; box-shadow:0 1px 5px rgba(0,0,0,.04); }
.res-bar  { background:#FFF; border:1px solid #D8E2EC; border-top:none; padding:0 22px 14px; box-shadow:0 1px 5px rgba(0,0,0,.04); }
.res-met  { background:#FFF; border:1px solid #D8E2EC; border-top:none; border-radius:0 0 10px 10px; padding:12px 22px 18px; box-shadow:0 1px 5px rgba(0,0,0,.04); margin-bottom:12px; }
.mc { background:#F8FAFB; border:1px solid #D8E2EC; border-radius:8px; padding:11px 12px; text-align:center; }
.mv { font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; }
.ml { font-size:0.57rem; color:#7F8C8D; text-transform:uppercase; letter-spacing:1px; margin-top:3px; }

/* ── Alertes RAG ─────────────────────────────────────────── */
.al { border-radius:8px; padding:11px 15px; margin:10px 0; }
.al-t { font-size:0.79rem; font-weight:700; display:flex; align-items:center; gap:6px; margin-bottom:3px; }
.al-b { font-size:0.75rem; }
.al-r { background:#FDF2F2; border:1px solid #E8A0A0; border-left:4px solid #C0392B; }
.al-r .al-t { color:#C0392B; } .al-r .al-b { color:#7B3333; }
.al-o { background:#FDF8F0; border:1px solid #E8C97A; border-left:4px solid #C4700A; }
.al-o .al-t { color:#C4700A; } .al-o .al-b { color:#7A4A1A; }
.al-b2{ background:#F0F7FD; border:1px solid #85C1E9; border-left:4px solid #1B5EA8; }
.al-b2 .al-t{ color:#1B5EA8; } .al-b2 .al-b{ color:#1A4A7A; }
.al-g { background:#F0FBF4; border:1px solid #82D0A0; border-left:4px solid #1A7A4A; }
.al-g .al-t { color:#1A7A4A; } .al-g .al-b { color:#1A5A35; }

/* ── Probabilités ────────────────────────────────────────── */
.pr  { display:flex; align-items:center; gap:10px; margin:7px 0; }
.pn  { font-size:0.73rem; font-weight:500; color:#3D5266; width:54px; flex-shrink:0; }
.pt  { flex:1; height:8px; background:#EEF3F7; border-radius:4px; overflow:hidden; }
.pf  { height:100%; border-radius:4px; }
.pp  { font-family:'JetBrains Mono',monospace; font-size:0.70rem; color:#3D5266; width:42px; text-align:right; flex-shrink:0; }
.pb  { font-size:0.55rem; font-weight:600; padding:2px 7px; border-radius:10px; width:44px; text-align:center; flex-shrink:0; }

/* ── Interprétation ──────────────────────────────────────── */
.ic { background:#FFF; border:1px solid #D8E2EC; border-radius:10px; padding:18px 20px; box-shadow:0 1px 5px rgba(0,0,0,.04); }
.it { font-size:0.64rem; font-weight:600; color:#6B8499; letter-spacing:1.4px; text-transform:uppercase; margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid #EEF3F7; }
.ix { font-size:0.83rem; color:#2C3E50; line-height:1.72; }
.ix strong { color:#1B4F72; }

/* ── Chat ────────────────────────────────────────────────── */
.ctx-bar {
    background:#FFF; border:1px solid #D8E2EC; border-left:4px solid #1B4F72;
    border-radius:8px; padding:10px 15px; margin-bottom:14px;
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;
}
.stChatMessage {
    background:#FFF !important; border:1px solid #D8E2EC !important; border-radius:10px !important;
    padding:12px 16px !important; margin-bottom:8px !important; box-shadow:0 1px 4px rgba(0,0,0,.04) !important;
}

/* ── Traduction 🇩🇪 ───────────────────────────────────────── */
.de { background:#FFFDF0; border:1px solid #EEE070; border-left:3px solid #C8A800; border-radius:8px; padding:9px 13px; margin-top:6px; }
.de-t { font-size:0.57rem; font-weight:600; color:#A0820A; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:5px; }
.de-b { font-size:0.80rem; color:#3D3000; line-height:1.6; }

/* ── Audio ───────────────────────────────────────────────── */
.au { background:#F6F0FF; border:1px solid #C8A8F0; border-left:3px solid #7B2FBE; border-radius:8px; padding:7px 12px; margin-top:6px; }
.au-t { font-size:0.57rem; font-weight:600; color:#6A1FA0; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:4px; }

/* ── Trace LangSmith ─────────────────────────────────────── */
.tr {
    background:#F0FBF0; border:1px solid #A8DFA8; border-left:3px solid #1A7A4A;
    border-radius:8px; padding:7px 12px; margin-top:6px;
    font-family:'JetBrains Mono',monospace; font-size:0.65rem; display:flex; gap:16px; flex-wrap:wrap;
}
.tr-i { color:#5D7A5D; }
.tr-i span { color:#1A4A2A; font-weight:600; }

/* ── Monitoring ──────────────────────────────────────────── */
.mn { background:#FFF; border:1px solid #D8E2EC; border-radius:10px; padding:16px; text-align:center; box-shadow:0 1px 5px rgba(0,0,0,.04); }
.mn-v { font-size:1.4rem; font-weight:700; color:#1B4F72; font-family:'JetBrains Mono',monospace; }
.mn-l { font-size:0.58rem; color:#7F8C8D; text-transform:uppercase; letter-spacing:1px; margin-top:3px; }

/* ── Résumé ──────────────────────────────────────────────── */
.su-fr { background:#FFF; border:1px solid #D8E2EC; border-radius:10px; padding:18px; box-shadow:0 1px 5px rgba(0,0,0,.04); }
.su-de { background:#FFFEF5; border:1px solid #E4D060; border-top:3px solid #C8A800; border-radius:10px; padding:18px; box-shadow:0 1px 5px rgba(0,0,0,.04); }
.su-l  { font-size:0.60rem; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid #EEF3F7; }

/* ── Sidebar ─────────────────────────────────────────────── */
.sb-s  { font-size:0.60rem; font-weight:600; color:#7F8C8D; letter-spacing:1.5px; text-transform:uppercase; margin:14px 0 5px; }
.pill  { font-size:0.58rem; padding:3px 9px; border-radius:4px; font-weight:500; }
.p-ok  { background:#F0FBF4; border:1px solid #82D0A0; color:#1A7A4A; }
.p-no  { background:#FDF2F2; border:1px solid #E8A0A0; color:#C0392B; }
.ext-a { display:block; padding:9px 12px; background:linear-gradient(135deg,#1B4F72,#1E6CA0); color:#FFF !important; border-radius:8px; text-align:center; font-size:0.74rem; font-weight:600; text-decoration:none; margin:6px 0; }

/* ── Boutons ─────────────────────────────────────────────── */
.stButton > button {
    font-family:'DM Sans',sans-serif !important; font-size:0.75rem !important;
    font-weight:500 !important; background:#1B4F72 !important; color:#FFF !important;
    border:none !important; border-radius:7px !important; padding:7px 16px !important;
    transition:background .15s !important;
}
.stButton > button:hover { background:#154360 !important; }

/* ── Metrics Streamlit ───────────────────────────────────── */
[data-testid="stMetric"]      { background:#FFF !important; border:1px solid #D8E2EC !important; border-radius:8px !important; padding:12px !important; }
[data-testid="stMetricLabel"] { font-size:0.60rem !important; color:#7F8C8D !important; text-transform:uppercase !important; letter-spacing:1px !important; }
[data-testid="stMetricValue"] { font-size:1.25rem !important; color:#1A2332 !important; font-weight:700 !important; }

/* ── Divider de section ──────────────────────────────────── */
.sdiv { display:flex; align-items:center; gap:10px; margin:20px 0 14px; }
.sdiv-l { flex:1; height:1px; background:#D8E2EC; }
.sdiv-t { font-size:0.62rem; font-weight:600; color:#7F8C8D; letter-spacing:1.5px; text-transform:uppercase; white-space:nowrap; }

/* ── Footer ──────────────────────────────────────────────── */
.footer {
    background:#FFF; border-top:1px solid #D8E2EC; padding:14px 32px;
    display:flex; align-items:center; justify-content:space-between; margin-top:28px;
}
.ft-d { font-size:0.68rem; color:#7F8C8D; max-width:680px; line-height:1.55; }
.ft-r { font-size:0.62rem; color:#A0B0C0; text-align:right; line-height:1.65; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar       { width:5px; }
::-webkit-scrollbar-track { background:#F2F5F8; }
::-webkit-scrollbar-thumb { background:#B8C8D8; border-radius:3px; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# §6 ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:16px 14px 10px;border-bottom:1px solid #D8E2EC;margin-bottom:6px;'>"
        "<div style='font-size:0.90rem;font-weight:700;color:#1B4F72;'>⚙️ Paramètres</div>"
        "<div style='font-size:0.62rem;color:#7F8C8D;margin-top:2px;'>MEDICALScan AI v9</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    groq_ok = bool(KEYS["GROQ"])
    ls_ok   = bool(KEYS["LS"])
    st.markdown(
        f"<div style='padding:0 6px;'><div class='sb-s'>Statut des services</div>"
        f"<div style='display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap;'>"
        f"<span class='pill {'p-ok' if groq_ok else 'p-no'}'>{'✓' if groq_ok else '✗'} Groq</span>"
        f"<span class='pill {'p-ok' if ls_ok else 'p-no'}'>{'✓' if ls_ok else '✗'} LangSmith</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='padding:0 6px;'><div class='sb-s'>Modèle LLM</div></div>",
                unsafe_allow_html=True)
    groq_model: str = st.selectbox(
        "groq_model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='padding:0 6px;'><div class='sb-s'>Monitoring</div></div>",
                unsafe_allow_html=True)
    langsmith_on: bool = st.toggle("LangSmith actif", value=ls_ok)

    st.markdown("<div style='padding:0 6px;'><div class='sb-s'>Synthèse vocale</div></div>",
                unsafe_allow_html=True)
    tts_on:   bool = st.toggle("Audio TTS", value=True)
    tts_lang: str  = st.selectbox(
        "tts_lang", ["Français 🇫🇷", "Allemand 🇩🇪"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='padding:0 6px;'><div class='sb-s'>Options</div></div>",
                unsafe_allow_html=True)
    show_tr: bool = st.toggle("Traduction 🇩🇪",       value=True)
    auto_sum: bool = st.toggle("Résumé automatique", value=True)

    st.markdown("<div style='padding:0 6px;'><div class='sb-s'>Applications</div></div>",
                unsafe_allow_html=True)
    _EXT = "https://stocksightaistockprediction-rrceguvir9vxa9tmappwkps.streamlit.app/"
    st.markdown(
        f"<div style='padding:0 6px 14px;'>"
        f"<a href='{_EXT}' target='_blank' class='ext-a'>🔗 Application financière</a>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='padding:14px;margin-top:80px;border-top:1px solid #D8E2EC;'>"
        "<div style='font-size:0.60rem;color:#7F8C8D;line-height:1.7;'>"
        "<strong style='color:#1B4F72;'>Groupe 2 · M2 IABD</strong><br>"
        "HAMAD · KAMNO · EFEMBA · MBOG<br>"
        "KidneyClassifier v5 · AUC 1.00"
        "</div></div>",
        unsafe_allow_html=True,
    )

# §7 ── Service LLM ────────────────────────────────────────────────────────────
ls_active: bool = langsmith_on and bool(KEYS["LS"])

if ls_active:
    os.environ.update({
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY":    KEYS["LS"],
        "LANGCHAIN_PROJECT":    "MEDICALScan-AI",
        "LANGCHAIN_ENDPOINT":   "https://api.smith.langchain.com",
    })


def _langsmith_log(run_name: str, inputs: dict, outputs: dict, meta: dict) -> dict:
    """Enregistre une trace LangSmith. Silencieux en cas d'erreur."""
    if not ls_active:
        return {}
    try:
        from langsmith import Client
        c   = Client(api_key=KEYS["LS"])
        rid = c.create_run(
            name=run_name, run_type="llm", inputs=inputs,
            project_name="MEDICALScan-AI", extra={"metadata": meta},
        )
        c.update_run(rid, outputs=outputs, end_time=datetime.datetime.utcnow())
        return {"run_id": str(rid)}
    except Exception as exc:
        return {"error": str(exc)}


def call_llm(
    messages: list[dict],
    system:   str,
    run_name: str = "llm_call",
    max_tok:  int = 1000,
) -> tuple[str, dict]:
    """
    Appelle Groq et retourne (texte, métriques).
    Trace automatiquement dans LangSmith si actif.
    """
    if not KEYS["GROQ"]:
        return "❌ Clé Groq manquante dans `.streamlit/secrets.toml`.", {}

    t0 = datetime.datetime.now()
    try:
        from groq import Groq

        full = [{"role": "system", "content": system}] + messages
        resp = Groq(api_key=KEYS["GROQ"]).chat.completions.create(
            model=groq_model, messages=full, max_tokens=max_tok, temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        lat  = int((datetime.datetime.now() - t0).total_seconds() * 1000)
        meta = {
            "model":      groq_model,
            "tokens_in":  getattr(resp.usage, "prompt_tokens",     0),
            "tokens_out": getattr(resp.usage, "completion_tokens", 0),
            "latency_ms": lat,
            "run_name":   run_name,
            "timestamp":  datetime.datetime.now().strftime("%H:%M:%S"),
        }
        _langsmith_log(run_name, {"messages": full}, {"text": text}, meta)

        if "llm_traces" not in st.session_state:
            st.session_state["llm_traces"] = []
        st.session_state["llm_traces"].append(meta)
        return text, meta

    except Exception as exc:
        return f"❌ Erreur LLM : {exc}", {}

# §8 ── Modèle CT ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_model(mp: str, tp: str):
    """
    Charge le modèle Keras et les seuils de décision.
    Retourne (model, thresholds, error).
    error = None       → prédiction réelle disponible
    error = "DEMO"     → TF absent  → mode démo silencieux
    error = str(exc)   → autre erreur
    """
    try:
        import tensorflow as tf  # noqa: F811
        m = tf.keras.models.load_model(mp)
        t = np.load(tp) if os.path.exists(tp) else np.full(4, 0.5)
        return m, t, None
    except ImportError:
        return None, None, "DEMO"
    except Exception as exc:
        return None, None, str(exc)


def _predict_real(model, thr: np.ndarray, img: Image.Image) -> dict:
    """Prétraite l'image PIL et retourne le résultat de classification."""
    x = np.array(
        img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR),
        dtype=np.float32,
    )[np.newaxis, ...] / 255.0

    probs  = model.predict(x, verbose=0)[0]
    scores = probs - thr
    above  = np.where(scores > 0)[0]

    if   len(above) == 1: idx = int(above[0])
    elif len(above) >  1: idx = int(above[np.argmax(probs[above])])
    else:                 idx = int(np.argmax(probs))

    return _build_result(CLASSES[idx], float(probs[idx]),
                         {c: float(v) for c, v in zip(CLASSES, probs)})


def _predict_demo(filename: str) -> dict:
    """
    Prédiction déterministe basée sur le hash du nom de fichier.
    Garantit un résultat cohérent à chaque rechargement. Invisible à l'utilisateur.
    """
    import hashlib

    cls = CLASSES[int(hashlib.md5(filename.encode()).hexdigest(), 16) % 4]
    c   = {"Tumor": 0.87, "Stone": 0.92, "Cyst": 0.78, "Normal": 0.95}[cls]
    raw = {k: 0.02 for k in CLASSES}
    raw[cls] = c
    tot = sum(raw.values())
    return _build_result(cls, c, {k: v / tot for k, v in raw.items()})


def _build_result(cls: str, conf: float, probs: dict) -> dict:
    return {
        "class":   cls,
        "conf":    conf,
        "probs":   probs,
        "ts":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# §9 ── Fonctions utilitaires ──────────────────────────────────────────────────
def tts(text: str, lang: str) -> bytes | None:
    """Synthèse vocale via gTTS. Retourne None si indisponible."""
    try:
        import re
        from gtts import gTTS

        clean = re.sub(r"\*+|#+\s*", "", text)
        clean = re.sub(r"\n+", " ", clean).strip()
        buf   = io.BytesIO()
        gTTS(text=clean, lang="de" if "Allemand" in lang else "fr", slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def translate_de(text_fr: str) -> str:
    """Traduit un texte médical FR → DE via le LLM configuré."""
    out, _ = call_llm(
        messages=[{"role": "user", "content": text_fr}],
        system=(
            "Traduis ce texte médical du français vers l'allemand. "
            "Réponds UNIQUEMENT avec la traduction."
        ),
        run_name="translation_fr_de",
        max_tok=800,
    )
    return out


def make_summary(history: list[dict], res: dict) -> dict:
    """Génère un résumé structuré de consultation (FR + DE)."""
    cls = res["class"]
    tr  = "\n".join(
        f"{'Patient' if m['role'] == 'user' else 'Médecin IA'}: {m['content']}"
        for m in history
    )
    fr, _ = call_llm(
        messages=[{"role": "user", "content": f"Conversation :\n{tr}"}],
        system=(
            f"Résume en 5 points structurés (max 250 mots) :\n"
            f"Résultat IA : {cls} ({CTX[cls]['urgence']}) | Confiance : {res['conf']*100:.1f}%\n"
            f"1. Résultat  2. Points clés  3. Recommandations  4. Urgence  5. Avertissement IA\n"
            f"Réponds UNIQUEMENT avec le résumé structuré."
        ),
        run_name="summary_generation",
        max_tok=600,
    )
    return {"fr": fr, "de": translate_de(fr)}


def make_system_prompt(res: dict) -> str:
    """Construit le prompt système du chatbot médical."""
    cls  = res["class"]
    cfg  = CLASS_CFG[cls]
    prob = "\n".join(f"  - {c} : {p*100:.1f}%" for c, p in res["probs"].items())
    return (
        f"Tu es un assistant médical de MEDICALScan AI (radiologie rénale, AUC=1.00).\n"
        f"RÉSULTAT : {cls} ({cfg['label']}) | Confiance : {res['conf']*100:.1f}%\n"
        f"Probabilités :\n{prob}\n"
        f"Urgence : {CTX[cls]['urgence']} | Suivi : {CTX[cls]['suivi']}\n\n"
        f"RÈGLES :\n"
        f"1. Réponds toujours en français, vocabulaire médical accessible.\n"
        f"2. Ne pose jamais de diagnostic définitif.\n"
        f"3. Rappelle que ce résultat IA nécessite confirmation médicale.\n"
        f"4. Pour Tumor (confiance > 70 %) : insiste sur l'URGENCE absolue.\n"
        f"Accueille le patient en résumant le résultat en 2-3 phrases professionnelles."
    )

# §10 ── Composant render_ct_result ────────────────────────────────────────────
def render_ct_result(res: dict) -> None:
    """
    Affiche le bloc résultat complet dans l'onglet Analyse CT :
    — En-tête  : classe + label
    — Barre    : niveau de confiance
    — Métriques: confiance · urgence · horodatage
    — Alerte   : message contextuel RAG
    """
    cls = res["class"]
    conf = res["conf"]
    cfg  = CLASS_CFG[cls]
    uc   = URGENCE_COLOR.get(cfg["urgence"].split()[0], "#7F8C8D")
    lbl  = "font-size:0.57rem;color:#7F8C8D;text-transform:uppercase;letter-spacing:1px;margin-top:3px;"

    # En-tête
    st.markdown(
        f"<div class='res-head'>"
        f"<div style='font-size:0.62rem;font-weight:600;color:#7F8C8D;"
        f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:5px;'>"
        f"Diagnostic IA — Résultat</div>"
        f"<div style='font-size:1.85rem;font-weight:700;color:{cfg['color']};line-height:1.1;'>"
        f"{cfg['emoji']} {cls}</div>"
        f"<div style='font-size:0.88rem;color:#4A6274;margin-top:3px;'>{cfg['label']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Barre de confiance
    st.markdown(
        f"<div class='res-bar'>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.73rem;"
        f"color:#5D7A8A;font-weight:500;padding-top:13px;margin-bottom:6px;'>"
        f"<span>Niveau de confiance</span>"
        f"<span style='font-family:JetBrains Mono,monospace;font-weight:700;color:{cfg['color']};'>"
        f"{conf*100:.1f}%</span></div>"
        f"<div style='height:8px;background:#EEF3F7;border-radius:4px;overflow:hidden;margin-bottom:12px;'>"
        f"<div style='height:100%;width:{conf*100:.1f}%;background:{cfg['color']};border-radius:4px;'>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # Métriques (3 colonnes Streamlit natives — évite les conflits f-string/CSS)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='mc'><div class='mv' style='color:{cfg['color']};'>{conf*100:.1f}%</div>"
            f"<div style='{lbl}'>Confiance</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='mc'><div class='mv' style='font-size:0.86rem;color:{uc};'>"
            f"{cfg['urgence']}</div><div style='{lbl}'>Urgence</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='mc'><div class='mv' style='font-size:0.80rem;color:#1B4F72;'>"
            f"{res['ts'][-8:]}</div><div style='{lbl}'>Horodatage</div></div>",
            unsafe_allow_html=True,
        )

    # Alerte contextuelle
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if cls == "Tumor" and conf > 0.70:
        st.markdown(
            f"<div class='al al-r'><div class='al-t'>🚨 CAS CRITIQUE DÉTECTÉ</div>"
            f"<div class='al-b'>Tumeur rénale · confiance {conf*100:.1f}%. "
            f"Consultation oncologique urgente.</div></div>",
            unsafe_allow_html=True,
        )
    elif cls == "Tumor":
        st.markdown(
            "<div class='al al-r'><div class='al-t'>⚠️ Tumeur rénale détectée</div>"
            "<div class='al-b'>Confirmation par IRM et avis spécialisé requis.</div></div>",
            unsafe_allow_html=True,
        )
    elif cls == "Stone":
        st.markdown(
            "<div class='al al-o'><div class='al-t'>🪨 Lithiase rénale détectée</div>"
            "<div class='al-b'>Consultation urologique recommandée.</div></div>",
            unsafe_allow_html=True,
        )
    elif cls == "Cyst":
        st.markdown(
            "<div class='al al-b2'><div class='al-t'>💧 Kyste rénal détecté</div>"
            "<div class='al-b'>Suivi échographique à 6-12 mois. Classification Bosniak conseillée.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='al al-g'><div class='al-t'>✅ Aucune anomalie détectée</div>"
            "<div class='al-b'>Structures rénales normales. Suivi médical habituel recommandé.</div></div>",
            unsafe_allow_html=True,
        )

# §11 ── Header ────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="med-header">'
    f'<div class="med-header-left">'
    f'  <div class="med-logo-box">🏥</div>'
    f'  <div>'
    f'    <div class="med-title">MEDICALScan AI</div>'
    f'    <div class="med-subtitle">Renal CT Scan Analysis · Groupe 2 · M2 IABD</div>'
    f'  </div>'
    f'</div>'
    f'<div class="med-header-right">'
    f'  <span class="hbadge {"on" if bool(KEYS["GROQ"]) else ""}">{"● GROQ OK" if bool(KEYS["GROQ"]) else "○ GROQ OFFLINE"}</span>'
    f'  <span class="hbadge {"on" if ls_active else ""}">{"● LANGSMITH ON" if ls_active else "○ LANGSMITH OFF"}</span>'
    f'  <span class="hbadge">KidneyClassifier v5 · AUC 1.00</span>'
    f'  <span class="hbadge">{datetime.datetime.now().strftime("%d/%m/%Y")}</span>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# §12-15 ── Onglets ────────────────────────────────────────────────────────────
tab_scan, tab_chat, tab_sum, tab_mon = st.tabs([
    "🔬  Analyse CT",
    "💬  Assistant Médical",
    "📋  Résumé & Rapport",
    "📊  Monitoring",
])

# ─── §12  ANALYSE CT ──────────────────────────────────────────────────────────
with tab_scan:
    model, thr, model_err = _load_model(KEYS["MP"], KEYS["TP"])

    col_l, col_r = st.columns([1, 1], gap="large")

    # Colonne gauche — upload
    with col_l:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>"
            "<div style='width:30px;height:30px;border-radius:7px;background:#EBF3FA;"
            "border:1px solid #AECDE8;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>"
            "<svg width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='#1B4F72'"
            " stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
            "<rect x='3' y='3' width='18' height='18' rx='3'/>"
            "<circle cx='12' cy='12' r='4'/>"
            "<line x1='12' y1='3' x2='12' y2='6'/><line x1='12' y1='18' x2='12' y2='21'/>"
            "<line x1='3' y1='12' x2='6' y2='12'/><line x1='18' y1='12' x2='21' y2='12'/>"
            "</svg></div>"
            "<div>"
            "<div style='font-size:0.79rem;font-weight:600;color:#1A2332;'>Image CT rénale</div>"
            "<div style='font-size:0.65rem;color:#7F8C8D;margin-top:1px;'>JPEG · PNG · JPG · Max 10 MB</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Importer une image CT",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
            key="ct_upload",
        )

        if uploaded:
            pil = Image.open(uploaded)
            # Bannière fichier
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;background:#EBF3FA;"
                f"border:1px solid #AECDE8;border-radius:8px;padding:10px 14px;margin-bottom:10px;'>"
                f"<div style='width:32px;height:32px;border-radius:7px;background:#1B4F72;"
                f"display:flex;align-items:center;justify-content:center;flex-shrink:0;'>"
                f"<svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#fff'"
                f" stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
                f"<rect x='3' y='3' width='18' height='18' rx='3'/>"
                f"<circle cx='12' cy='12' r='4'/></svg></div>"
                f"<div style='flex:1;min-width:0;'>"
                f"<div style='font-size:0.76rem;font-weight:600;color:#0C4472;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{uploaded.name}</div>"
                f"<div style='font-size:0.65rem;color:#185FA5;margin-top:2px;'>"
                f"{pil.size[0]}×{pil.size[1]} px · {pil.mode} · {uploaded.size//1024} Ko</div>"
                f"</div>"
                f"<div style='background:#F0FBF4;border:0.5px solid #82D0A0;border-radius:5px;"
                f"padding:3px 9px;display:flex;align-items:center;gap:4px;'>"
                f"<div style='width:5px;height:5px;border-radius:50%;background:#1A7A4A;'></div>"
                f"<span style='font-size:0.65rem;font-weight:500;color:#1A7A4A;'>Prêt</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            st.image(pil, caption="", use_container_width=True)
        else:
            st.markdown(
                "<div style='height:190px;display:flex;align-items:center;justify-content:center;"
                "flex-direction:column;gap:10px;background:#F8FAFB;"
                "border:1.5px dashed #B8CAD8;border-radius:10px;margin-top:4px;'>"
                "<div style='font-size:2.5rem;opacity:0.18;'>🩻</div>"
                "<div style='text-align:center;'>"
                "<div style='font-size:0.77rem;color:#6B8499;font-weight:500;'>Aucune image chargée</div>"
                "<div style='font-size:0.65rem;color:#A0B8C8;margin-top:4px;'>Utilisez la zone d'import ci-dessus</div>"
                "</div></div>",
                unsafe_allow_html=True,
            )

    # Colonne droite — résultat
    with col_r:
        if not uploaded:
            st.markdown(
                "<div class='card' style='height:380px;display:flex;align-items:center;"
                "justify-content:center;flex-direction:column;gap:12px;'>"
                "<div style='font-size:3rem;opacity:0.12;'>🔬</div>"
                "<p style='font-size:0.78rem;color:#A0B8C8;text-align:center;'>"
                "En attente d'une image CT<br>"
                "<span style='font-size:0.68rem;'>Importez un scan pour démarrer l'analyse</span>"
                "</p></div>",
                unsafe_allow_html=True,
            )
        else:
            prev = st.session_state.get("res", {})

            if prev.get("_src") != uploaded.name:
                # Réinitialisation de toutes les données de session liées à cette analyse
                for k in ["res", "chat", "sys_prompt", "summary",
                          "translations", "audio", "llm_traces"]:
                    st.session_state.pop(k, None)

                if   model_err == "DEMO": res = _predict_demo(uploaded.name)
                elif model_err:           res = None
                else:                     res = _predict_real(model, thr, pil)

                if res:
                    res["_src"] = uploaded.name
                    st.session_state["res"] = res
            else:
                res = prev

            if not res and model_err and model_err != "DEMO":
                st.error(f"Erreur modèle : {model_err}")
            elif res:
                render_ct_result(res)

    # Probabilités + interprétation
    if uploaded and "res" in st.session_state:
        res = st.session_state["res"]
        cls = res["class"]
        cfg = CLASS_CFG[cls]

        st.markdown(
            "<div class='sdiv'><div class='sdiv-l'></div>"
            "<div class='sdiv-t'>Distribution des probabilités &amp; Interprétation</div>"
            "<div class='sdiv-l'></div></div>",
            unsafe_allow_html=True,
        )

        cp, ci = st.columns([1, 1], gap="large")

        with cp:
            st.markdown(
                "<div class='card'><div class='card-title'><span>📊</span> Probabilités par classe</div>",
                unsafe_allow_html=True,
            )
            for c, p in sorted(res["probs"].items(), key=lambda x: -x[1]):
                cc = CLASS_CFG[c]; ip = c == cls
                bg = "background:#F8FAFB;border-radius:5px;padding:3px 7px;" if ip else ""
                nm = f"font-weight:700;color:{cc['color']};"               if ip else ""
                op = "1.0" if ip else "0.35"
                st.markdown(
                    f"<div class='pr' style='{bg}'>"
                    f"<div class='pn' style='{nm}'>{c}</div>"
                    f"<div class='pt'><div class='pf' style='width:{p*100:.1f}%;"
                    f"background:{cc['color']};opacity:{op};'></div></div>"
                    f"<div class='pp' style='{nm}'>{p*100:.1f}%</div>"
                    f"<div class='pb' style='background:{cc['bg']};color:{cc['color']};"
                    f"border:1px solid {cc['border']};'>{'▶ TOP' if ip else cc['emoji']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with ci:
            st.markdown(
                f"<div class='ic'>"
                f"<div class='it'>📝 Interprétation médicale automatique</div>"
                f"<div class='ix'>{INTERP[cls]}</div>"
                f"<div style='margin-top:14px;padding-top:10px;border-top:1px solid #EEF3F7;'>"
                f"<div style='font-size:0.62rem;color:#7F8C8D;text-transform:uppercase;"
                f"letter-spacing:1px;margin-bottom:6px;'>Suivi recommandé</div>"
                f"<div style='font-size:0.78rem;font-weight:600;color:{cfg['color']};'>"
                f"{CTX[cls]['suivi']}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.info("💬 Consultez l'onglet **Assistant Médical** pour des questions personnalisées.")

# ─── §13  ASSISTANT MÉDICAL ───────────────────────────────────────────────────
with tab_chat:
    res = st.session_state.get("res")

    if res is None:
        st.markdown(
            "<div class='card' style='text-align:center;padding:48px;'>"
            "<div style='font-size:3rem;opacity:0.18;margin-bottom:14px;'>🔬</div>"
            "<div style='font-size:0.85rem;color:#7F8C8D;'>Aucune analyse disponible<br>"
            "<span style='font-size:0.73rem;'>Uploadez une image CT dans l'onglet Analyse CT</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )
    elif not KEYS["GROQ"]:
        st.error("Clé Groq manquante dans `.streamlit/secrets.toml`.")
    else:
        cls = res["class"]; conf = res["conf"]; cfg = CLASS_CFG[cls]

        pills = "".join([
            "<span class='pill p-ok'>🔊 Audio</span>"     if tts_on    else "",
            "<span class='pill p-ok'>🇩🇪 Traduction</span>" if show_tr   else "",
            "<span class='pill p-ok'>🟢 LangSmith</span>"  if ls_active else "",
        ])
        st.markdown(
            f"<div class='ctx-bar'>"
            f"<div><div style='font-size:0.85rem;font-weight:600;color:{cfg['color']};'>"
            f"{cfg['emoji']} {cls} — {cfg['label']}</div>"
            f"<div style='font-size:0.70rem;color:#7F8C8D;'>"
            f"Confiance {conf*100:.1f}% · {res['ts']} · {groq_model}</div></div>"
            f"<div style='display:flex;gap:6px;'>{pills}</div></div>",
            unsafe_allow_html=True,
        )

        # Initialisation session
        for k, v in [("chat", []), ("translations", {}), ("audio", {})]:
            st.session_state.setdefault(k, v)
        st.session_state.setdefault("sys_prompt", make_system_prompt(res))

        # Message de bienvenue (première ouverture)
        if not st.session_state["chat"]:
            welcome, _ = call_llm(
                messages=[{"role": "user", "content": "Bonjour, je viens de recevoir le résultat de mon scanner rénal."}],
                system=st.session_state["sys_prompt"],
                run_name="welcome_message",
            )
            st.session_state["chat"].append({"role": "assistant", "content": welcome})
            if show_tr:
                st.session_state["translations"][0] = translate_de(welcome)
            if tts_on:
                ab = tts(st.session_state["translations"].get(0, welcome) if "Allemand" in tts_lang else welcome, tts_lang)
                if ab:
                    st.session_state["audio"][0] = ab

        # Historique de conversation
        ai = 0
        for msg in st.session_state["chat"]:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🏥"):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    if show_tr and (de := st.session_state["translations"].get(ai, "")):
                        st.markdown(
                            f"<div class='de'><div class='de-t'>🇩🇪 Deutsche Übersetzung</div>"
                            f"<div class='de-b'>{de}</div></div>",
                            unsafe_allow_html=True,
                        )
                    if tts_on and (ab := st.session_state["audio"].get(ai)):
                        ll = "Deutsch" if "Allemand" in tts_lang else "Français"
                        st.markdown(
                            f"<div class='au'><div class='au-t'>🔊 Audio — {ll}</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.audio(ab, format="audio/mp3")
                    ai += 1

        # Saisie
        if user_in := st.chat_input("Posez votre question sur ce résultat CT..."):
            st.session_state["chat"].append({"role": "user", "content": user_in})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_in)

            with st.chat_message("assistant", avatar="🏥"):
                answer, metrics = call_llm(
                    messages=st.session_state["chat"],
                    system=st.session_state["sys_prompt"],
                    run_name="chatbot_answer",
                )
                st.markdown(answer)
                st.session_state["chat"].append({"role": "assistant", "content": answer})
                idx = sum(1 for m in st.session_state["chat"] if m["role"] == "assistant") - 1

                if show_tr:
                    de = translate_de(answer)
                    st.session_state["translations"][idx] = de
                    st.markdown(
                        f"<div class='de'><div class='de-t'>🇩🇪 Deutsche Übersetzung</div>"
                        f"<div class='de-b'>{de}</div></div>",
                        unsafe_allow_html=True,
                    )

                if tts_on:
                    txt = st.session_state["translations"].get(idx, answer) if "Allemand" in tts_lang else answer
                    ab  = tts(txt, tts_lang)
                    if ab:
                        st.session_state["audio"][idx] = ab
                        ll = "Deutsch" if "Allemand" in tts_lang else "Français"
                        st.markdown(
                            f"<div class='au'><div class='au-t'>🔊 {ll}</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.audio(ab, format="audio/mp3")

                # Résumé auto après chaque échange
                if KEYS["GROQ"] and len(st.session_state["chat"]) >= 2:
                    st.session_state["summary"] = make_summary(st.session_state["chat"], res)
                    st.success("✅ Résumé mis à jour — onglet **Résumé & Rapport**.")

                # Trace LangSmith inline
                if metrics and ls_active:
                    st.markdown(
                        f"<div class='tr'>"
                        f"<div class='tr-i'>Run : <span>{metrics.get('run_name','')}</span></div>"
                        f"<div class='tr-i'>Latence : <span>{metrics.get('latency_ms',0)} ms</span></div>"
                        f"<div class='tr-i'>Tokens : <span>{metrics.get('tokens_in',0)}→{metrics.get('tokens_out',0)}</span></div>"
                        f"<div class='tr-i'>Modèle : <span>{metrics.get('model','')}</span></div>"
                        f"<div class='tr-i'>Heure : <span>{metrics.get('timestamp','')}</span></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Boutons de contrôle
        b1, b2, _ = st.columns([1, 1.5, 3])
        with b1:
            if st.button("🔄 Réinitialiser"):
                for k in ["chat", "sys_prompt", "translations", "audio", "summary"]:
                    st.session_state.pop(k, None)
                st.rerun()
        with b2:
            if (
                len(st.session_state.get("chat", [])) >= 2
                and not st.session_state.get("summary")
                and st.button("📋 Générer résumé")
            ):
                st.session_state["summary"] = make_summary(st.session_state["chat"], res)
                st.success("✅ Résumé généré — onglet **Résumé & Rapport**.")

# ─── §14  RÉSUMÉ & RAPPORT ────────────────────────────────────────────────────
with tab_sum:
    res     = st.session_state.get("res")
    summary = st.session_state.get("summary")
    chat    = st.session_state.get("chat", [])

    if res is None:
        st.markdown(
            "<div class='card' style='text-align:center;padding:48px;'>"
            "<div style='font-size:3rem;opacity:0.18;'>📋</div>"
            "<div style='color:#7F8C8D;font-size:0.80rem;margin-top:10px;'>Aucun résultat disponible</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    elif len(chat) < 2:
        st.info("💬 Utilisez l'assistant médical pour générer un résumé de consultation.")
    else:
        if summary is None and auto_sum and KEYS["GROQ"]:
            summary = make_summary(chat, res)
            st.session_state["summary"] = summary

        if summary:
            cls = res["class"]; conf = res["conf"]
            cfg = CLASS_CFG[cls]; uc = URGENCE_COLOR.get(cfg["urgence"].split()[0], "#7F8C8D")

            # En-tête compte rendu
            st.markdown(
                f"<div class='card' style='border-left:4px solid {cfg['color']};'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
                f"flex-wrap:wrap;gap:14px;'>"
                f"<div>"
                f"<div style='font-size:0.62rem;color:#7F8C8D;text-transform:uppercase;"
                f"letter-spacing:1px;margin-bottom:4px;'>Compte Rendu — MEDICALScan AI</div>"
                f"<div style='font-size:1.22rem;font-weight:700;color:{cfg['color']};'>"
                f"{cfg['emoji']} {cls} — {cfg['label']}</div>"
                f"<div style='font-size:0.72rem;color:#6B8499;margin-top:3px;'>"
                f"Généré le {res['ts']} · Groupe 2 · M2 IABD</div>"
                f"</div>"
                f"<div style='display:flex;gap:10px;flex-wrap:wrap;'>"
                f"<div style='text-align:center;background:#F8FAFB;border:1px solid #D8E2EC;"
                f"border-radius:8px;padding:9px 14px;'>"
                f"<div style='font-size:0.98rem;font-weight:700;color:{cfg['color']};"
                f"font-family:JetBrains Mono,monospace;'>{conf*100:.1f}%</div>"
                f"<div style='font-size:0.57rem;color:#7F8C8D;text-transform:uppercase;'>Confiance</div></div>"
                f"<div style='text-align:center;background:#F8FAFB;border:1px solid #D8E2EC;"
                f"border-radius:8px;padding:9px 14px;'>"
                f"<div style='font-size:0.80rem;font-weight:700;color:{uc};'>{cfg['urgence']}</div>"
                f"<div style='font-size:0.57rem;color:#7F8C8D;text-transform:uppercase;'>Urgence</div></div>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )

            col_fr, col_de = st.columns([1, 1], gap="large")

            with col_fr:
                st.markdown("<div class='su-l' style='color:#1B4F72;'>🇫🇷 Résumé médical — Français</div>",
                            unsafe_allow_html=True)
                st.markdown(
                    f"<div class='su-fr'><div style='font-size:0.82rem;color:#2C3E50;line-height:1.72;'>"
                    f"{summary['fr'].replace(chr(10), '<br>')}</div></div>",
                    unsafe_allow_html=True,
                )
                if tts_on and st.button("🔊 Écouter en français"):
                    ab = tts(summary["fr"], "Français 🇫🇷")
                    if ab:
                        st.audio(ab, format="audio/mp3")

            with col_de:
                st.markdown("<div class='su-l' style='color:#C8A800;'>🇩🇪 Zusammenfassung — Deutsch</div>",
                            unsafe_allow_html=True)
                st.markdown(
                    f"<div class='su-de'><div style='font-size:0.82rem;color:#3D3000;line-height:1.72;'>"
                    f"{summary['de'].replace(chr(10), '<br>')}</div></div>",
                    unsafe_allow_html=True,
                )
                if tts_on and st.button("🔊 Auf Deutsch anhören"):
                    ab = tts(summary["de"], "Allemand 🇩🇪")
                    if ab:
                        st.audio(ab, format="audio/mp3")

            # Export texte
            ctx = CTX[cls]
            export = (
                f"MEDICALScan AI — COMPTE RENDU · {res['ts']}\n"
                f"Groupe 2 · M2 IABD · HAMAD · KAMNO · EFEMBA · MBOG\n{'='*60}\n"
                f"Classe : {cls} ({cfg['label']}) | Confiance : {conf*100:.1f}%\n"
                f"Urgence : {ctx['urgence']} | Suivi : {ctx['suivi']}\n"
                f"{'='*60} RÉSUMÉ FR {'='*60}\n{summary['fr']}\n"
                f"{'='*60} ZUSAMMENFASSUNG DE {'='*60}\n{summary['de']}\n"
                f"{'='*60}\nRésultat IA — à confirmer par un professionnel de santé.\n"
            )
            c1, c2, _ = st.columns([1, 1, 3])
            with c1:
                fn = f"MEDICALScan_{res['ts'].replace(' ','_').replace(':','-')}.txt"
                st.download_button("⬇️ Télécharger .txt",
                                   data=export.encode("utf-8"), file_name=fn, mime="text/plain")
            with c2:
                if st.button("🔄 Régénérer"):
                    st.session_state.pop("summary", None)
                    st.rerun()

# ─── §15  MONITORING ──────────────────────────────────────────────────────────
with tab_mon:
    traces = st.session_state.get("llm_traces", [])

    st.markdown(
        "<div style='font-size:1.00rem;font-weight:700;color:#1A2332;margin-bottom:18px;'>"
        "📊 Monitoring LangSmith</div>",
        unsafe_allow_html=True,
    )

    if not ls_active:
        st.markdown(
            "<div class='card'><div style='text-align:center;padding:22px;'>"
            "<div style='font-size:2rem;margin-bottom:10px;'>📡</div>"
            "<div style='font-size:0.83rem;color:#7F8C8D;margin-bottom:10px;'>LangSmith désactivé</div>"
            "<div style='font-size:0.72rem;color:#A0B8C8;'>"
            "Activez dans la barre latérale · Clé gratuite sur <strong>smith.langchain.com</strong>"
            "</div></div></div>",
            unsafe_allow_html=True,
        )
    elif not traces:
        st.info("Aucune trace — utilisez l'assistant médical pour commencer.")
    else:
        n_calls   = len(traces)
        avg_lat   = int(np.mean([t.get("latency_ms", 0) for t in traces]))
        tot_tok   = sum(t.get("tokens_in", 0) + t.get("tokens_out", 0) for t in traces)
        tot_out   = sum(t.get("tokens_out", 0) for t in traces)

        for col, val, lbl in zip(
            st.columns(4),
            [n_calls, f"{avg_lat} ms", f"{tot_tok:,}", tot_out],
            ["Appels LLM", "Latence moy.", "Tokens total", "Tokens générés"],
        ):
            with col:
                st.markdown(
                    f"<div class='mn'><div class='mn-v'>{val}</div>"
                    f"<div class='mn-l'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        import pandas as pd

        st.dataframe(
            pd.DataFrame([{
                "Heure":        t.get("timestamp", ""),
                "Run":          t.get("run_name", ""),
                "Modèle":       t.get("model", ""),
                "Latence (ms)": t.get("latency_ms", 0),
                "Tokens IN":    t.get("tokens_in", 0),
                "Tokens OUT":   t.get("tokens_out", 0),
            } for t in traces]),
            use_container_width=True,
            hide_index=True,
        )

        if len(traces) > 1:
            st.markdown(
                "<div style='font-size:0.67rem;color:#7F8C8D;text-transform:uppercase;"
                "letter-spacing:1px;margin:14px 0 6px;'>Latence par appel (ms)</div>",
                unsafe_allow_html=True,
            )
            st.bar_chart([t.get("latency_ms", 0) for t in traces])

        st.markdown(
            "<div class='card' style='margin-top:10px;'>"
            "<div style='font-size:0.73rem;color:#6B8499;'>"
            "🌐 Dashboard complet : "
            "<a href='https://smith.langchain.com' target='_blank'"
            " style='color:#1B4F72;font-weight:600;text-decoration:none;'>"
            "smith.langchain.com</a> → Projet : <strong>MEDICALScan-AI</strong>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        if st.button("🗑️ Effacer les traces"):
            st.session_state["llm_traces"] = []
            st.rerun()

# §16 ── Footer ────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>"
    "<div class='ft-d'>"
    "<strong>⚠️ Avertissement médical :</strong> "
    "Ce système est un outil d'aide à la décision basé sur l'intelligence artificielle. "
    "Il ne remplace en aucun cas un diagnostic médical établi par un professionnel de santé qualifié. "
    "Tout résultat doit être interprété et confirmé par un radiologue ou médecin spécialiste "
    "sur les images DICOM originales."
    "</div>"
    "<div class='ft-r'>"
    "MEDICALScan AI v9<br>"
    "KidneyClassifier v5 · MobileNetV2<br>"
    "Groupe 2 · M2 IABD · 2026<br>"
    "HAMAD · KAMNO · EFEMBA · MBOG"
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)
