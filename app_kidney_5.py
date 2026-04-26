"""
==============================================================================
 MEDICALScan AI — Application Streamlit v6
 Design médical professionnel — style PACS / dashboard hospitalier
 CORRECTION : card résultat principal scindée en blocs séparés
==============================================================================
 Groupe 2 · M2 IABD · HAMAD · KAMNO · EFEMBA · MBOG
==============================================================================
"""

import os, io, datetime, time
import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CLÉS API — chargées depuis .streamlit/secrets.toml, jamais affichées
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _secrets():
    def g(k):
        try: return st.secrets.get(k, os.environ.get(k, ""))
        except: return os.environ.get(k, "")
    return {
        'GROQ':  g("GROQ_API_KEY"),
        'LS':    g("LANGCHAIN_API_KEY"),
        'MODEL': g("DEFAULT_GROQ_MODEL") or "llama-3.3-70b-versatile",
        'MP':    g("MODEL_PATH")         or "outputs_v5/KidneyClassifier_v5.keras",
        'TP':    g("THRESH_PATH")        or "outputs_v5/thresholds.npy",
    }

K = _secrets()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MEDICALScan AI — Renal CT Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — DESIGN MÉDICAL PROFESSIONNEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}
[data-testid="stAppViewContainer"] {
    background: #F0F4F8 !important;
    color: #1A2332 !important;
}
[data-testid="stHeader"] { background: transparent !important; display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Sidebar */
[data-testid="stSidebar"] > div {
    background: #FFFFFF !important;
    border-right: 1px solid #DDE3EA !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label { display: none; }

/* Header */
.med-header {
    background: linear-gradient(135deg, #1B3A5C 0%, #1B4F72 60%, #2471A3 100%);
    padding: 20px 40px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 3px solid #1A5276;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.med-header-left { display: flex; align-items: center; gap: 16px; }
.med-logo-box {
    width: 52px; height: 52px;
    background: rgba(255,255,255,0.15);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    border: 1px solid rgba(255,255,255,0.25);
}
.med-title {
    font-size: 1.5rem; font-weight: 700; color: #FFFFFF;
    letter-spacing: -0.3px; margin: 0; line-height: 1.2;
}
.med-subtitle {
    font-size: 0.72rem; color: rgba(255,255,255,0.65);
    letter-spacing: 1.5px; text-transform: uppercase; margin-top: 2px;
}
.med-header-right { display: flex; align-items: center; gap: 10px; }
.header-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; padding: 4px 12px; border-radius: 4px;
    letter-spacing: 0.5px;
    border: 1px solid rgba(255,255,255,0.25);
    color: rgba(255,255,255,0.8);
    background: rgba(255,255,255,0.08);
}
.header-badge.active {
    border-color: #52BE80; color: #52BE80;
    background: rgba(82,190,128,0.1);
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: #FFFFFF !important;
    border-bottom: 2px solid #DDE3EA !important;
    padding: 0 32px !important;
    gap: 0 !important; margin-bottom: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    color: #5D7A8A !important; background: transparent !important;
    border: none !important; padding: 14px 22px !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    letter-spacing: 0 !important; text-transform: none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #1B4F72 !important;
    border-bottom: 3px solid #1B4F72 !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: #1B4F72 !important; background: #F0F4F8 !important;
}
[data-testid="stTabsContent"] {
    padding: 28px 32px !important; background: #F0F4F8 !important;
}

/* Cards */
.med-card {
    background: #FFFFFF; border: 1px solid #DDE3EA;
    border-radius: 10px; padding: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 16px;
}
.med-card-title {
    font-size: 0.72rem; font-weight: 600; color: #5D7A8A;
    letter-spacing: 1.2px; text-transform: uppercase;
    margin-bottom: 16px; padding-bottom: 10px;
    border-bottom: 1px solid #EEF2F5;
    display: flex; align-items: center; gap: 8px;
}

/* Upload */
[data-testid="stFileUploadDropzone"] {
    background: #F8FAFC !important;
    border: 2px dashed #B8C8D8 !important;
    border-radius: 10px !important; transition: border-color 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #1B4F72 !important; background: #EBF5FB !important;
}

/* Alertes */
.alert-critical {
    background: #FDF2F2; border: 1px solid #E8A0A0;
    border-left: 4px solid #C0392B;
    border-radius: 8px; padding: 14px 18px; margin: 12px 0;
}
.alert-critical-title {
    font-size: 0.85rem; font-weight: 700; color: #C0392B;
    display: flex; align-items: center; gap: 8px; margin-bottom: 4px;
}
.alert-critical-text { font-size: 0.82rem; color: #7B3333; }

.alert-warning {
    background: #FDF8F0; border: 1px solid #E8C97A;
    border-left: 4px solid #C4700A;
    border-radius: 8px; padding: 14px 18px; margin: 12px 0;
}
.alert-warning-title { font-size: 0.85rem; font-weight: 700; color: #C4700A; display:flex; align-items:center; gap:8px; margin-bottom:4px; }
.alert-warning-text  { font-size: 0.82rem; color: #7A4A1A; }

.alert-info {
    background: #F0F7FD; border: 1px solid #85C1E9;
    border-left: 4px solid #1B5EA8;
    border-radius: 8px; padding: 14px 18px; margin: 12px 0;
}
.alert-info-title { font-size: 0.85rem; font-weight: 700; color: #1B5EA8; display:flex; align-items:center; gap:8px; margin-bottom:4px; }
.alert-info-text  { font-size: 0.82rem; color: #1A4A7A; }

.alert-success {
    background: #F0FBF4; border: 1px solid #82D0A0;
    border-left: 4px solid #1A7A4A;
    border-radius: 8px; padding: 14px 18px; margin: 12px 0;
}
.alert-success-title { font-size: 0.85rem; font-weight: 700; color: #1A7A4A; display:flex; align-items:center; gap:8px; margin-bottom:4px; }
.alert-success-text  { font-size: 0.82rem; color: #1A5A35; }

/* Probabilités */
.prob-row { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
.prob-cls-name { font-size: 0.78rem; font-weight: 500; color: #3D5266; width: 60px; flex-shrink: 0; }
.prob-track { flex: 1; height: 10px; background: #EEF2F5; border-radius: 5px; overflow: hidden; }
.prob-fill  { height: 100%; border-radius: 5px; }
.prob-pct   { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #3D5266; width: 48px; text-align: right; flex-shrink: 0; }
.prob-badge { font-size: 0.6rem; font-weight: 600; padding: 2px 8px; border-radius: 10px; width: 50px; text-align: center; flex-shrink: 0; }

/* Interprétation */
.interp-card {
    background: #FFFFFF; border: 1px solid #DDE3EA;
    border-radius: 10px; padding: 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.interp-title {
    font-size: 0.72rem; font-weight: 600; color: #5D7A8A;
    letter-spacing: 1.2px; text-transform: uppercase;
    margin-bottom: 14px; padding-bottom: 10px;
    border-bottom: 1px solid #EEF2F5;
}
.interp-text { font-size: 0.88rem; color: #2C3E50; line-height: 1.7; }
.interp-text strong { color: #1B4F72; }

/* Mode simulation */
.sim-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #D4AC0D; animation: pulse-dot 1.5s infinite;
    flex-shrink: 0;
}
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.5; transform:scale(1.3); }
}

/* Chat */
.chat-ctx-bar {
    background: #FFFFFF; border: 1px solid #DDE3EA;
    border-left: 4px solid #1B4F72; border-radius: 8px;
    padding: 12px 18px; margin-bottom: 16px;
    display: flex; align-items: center; justify-content: space-between;
}
.chat-ctx-cls  { font-size: 0.9rem; font-weight: 600; }
.chat-ctx-meta { font-size: 0.75rem; color: #7F8C8D; }

.stChatMessage {
    background: #FFFFFF !important; border: 1px solid #DDE3EA !important;
    border-radius: 10px !important; padding: 14px 18px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

/* Traduction */
.de-card {
    background: #FFFDF0; border: 1px solid #F0E080;
    border-left: 3px solid #C8A800;
    border-radius: 8px; padding: 12px 16px; margin-top: 8px;
}
.de-label { font-size: 0.62rem; font-weight: 600; color: #A0820A; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px; }
.de-text  { font-size: 0.85rem; color: #3D3000; line-height: 1.6; }

/* Audio */
.audio-card {
    background: #F8F0FF; border: 1px solid #D0B0F0;
    border-left: 3px solid #7B2FBE;
    border-radius: 8px; padding: 10px 14px; margin-top: 8px;
}
.audio-label { font-size: 0.62rem; font-weight: 600; color: #6A1FA0; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px; }

/* LangSmith trace */
.trace-bar {
    background: #F0FBF0; border: 1px solid #B0DFB0;
    border-left: 3px solid #1A7A4A; border-radius: 8px;
    padding: 10px 14px; margin-top: 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    display: flex; gap: 20px; flex-wrap: wrap;
}
.trace-item       { color: #5D7A5D; }
.trace-item span  { color: #1A4A2A; font-weight: 600; }

/* Monitoring */
.mon-metric {
    background: #FFFFFF; border: 1px solid #DDE3EA;
    border-radius: 10px; padding: 18px;
    text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.mon-val { font-size: 1.6rem; font-weight: 700; color: #1B4F72; font-family: 'JetBrains Mono', monospace; }
.mon-lbl { font-size: 0.65rem; color: #7F8C8D; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* Résumé */
.sum-card-fr {
    background: #FFFFFF; border: 1px solid #DDE3EA;
    border-radius: 10px; padding: 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.sum-card-de {
    background: #FFFEF5; border: 1px solid #E8D870;
    border-top: 3px solid #C8A800;
    border-radius: 10px; padding: 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.sum-lang-label {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 12px;
    padding-bottom: 8px; border-bottom: 1px solid #EEF2F5;
}

/* Sidebar pills */
.sb-section { font-size: 0.65rem; font-weight: 600; color: #7F8C8D; letter-spacing: 1.5px; text-transform: uppercase; margin: 16px 0 6px 0; }
.sb-key-status { display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
.sb-pill { font-size: 0.62rem; padding: 3px 10px; border-radius: 4px; font-weight: 500; }
.sb-ok   { background: #F0FBF4; border: 1px solid #82D0A0; color: #1A7A4A; }
.sb-nok  { background: #FDF2F2; border: 1px solid #E8A0A0; color: #C0392B; }

/* Buttons */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important; font-weight: 500 !important;
    background: #1B4F72 !important; color: #FFFFFF !important;
    border: none !important; border-radius: 6px !important;
    padding: 8px 18px !important;
    letter-spacing: 0 !important; text-transform: none !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #154360 !important; }

/* Metrics Streamlit */
[data-testid="stMetric"] {
    background: #FFFFFF !important; border: 1px solid #DDE3EA !important;
    border-radius: 8px !important; padding: 14px !important;
}
[data-testid="stMetricLabel"] { font-size: 0.65rem !important; color: #7F8C8D !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #1A2332 !important; font-weight: 700 !important; }

/* Footer */
.med-footer {
    background: #FFFFFF; border-top: 1px solid #DDE3EA;
    padding: 16px 40px;
    display: flex; align-items: center; justify-content: space-between;
    margin-top: 32px;
}
.footer-disclaimer { font-size: 0.72rem; color: #7F8C8D; max-width: 700px; line-height: 1.5; }
.footer-right { font-size: 0.68rem; color: #A0B0C0; text-align: right; }

/* Divider */
.sec-divider { display: flex; align-items: center; gap: 12px; margin: 24px 0 16px 0; }
.sec-divider-line { flex: 1; height: 1px; background: #DDE3EA; }
.sec-divider-text { font-size: 0.68rem; font-weight: 600; color: #7F8C8D; letter-spacing: 1.5px; text-transform: uppercase; white-space: nowrap; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F0F4F8; }
::-webkit-scrollbar-thumb { background: #B8C8D8; border-radius: 3px; }

/* Spinner médical */
@keyframes spin { to { transform: rotate(360deg); } }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 16px 12px 16px;border-bottom:1px solid #DDE3EA;margin-bottom:8px;'>
        <div style='font-size:1rem;font-weight:700;color:#1B4F72;'>⚙️ Paramètres</div>
        <div style='font-size:0.68rem;color:#7F8C8D;margin-top:2px;'>MEDICALScan AI v6</div>
    </div>
    """, unsafe_allow_html=True)

    groq_ok = bool(K['GROQ'])
    ls_ok   = bool(K['LS'])
    st.markdown(f"""
    <div style='padding:0 8px;'>
        <div class="sb-section">Statut des services</div>
        <div class="sb-key-status">
            <span class="sb-pill {'sb-ok' if groq_ok else 'sb-nok'}">{'✓' if groq_ok else '✗'} Groq LLM</span>
            <span class="sb-pill {'sb-ok' if ls_ok else 'sb-nok'}">{'✓' if ls_ok else '✗'} LangSmith</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:0 8px;'><div class='sb-section'>Modèle LLM</div></div>", unsafe_allow_html=True)
    groq_model_choice = st.selectbox("", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ], label_visibility="collapsed")

    st.markdown("<div style='padding:0 8px;'><div class='sb-section'>Monitoring</div></div>", unsafe_allow_html=True)
    langsmith_on = st.toggle("LangSmith actif", value=ls_ok)

    st.markdown("<div style='padding:0 8px;'><div class='sb-section'>Synthèse vocale</div></div>", unsafe_allow_html=True)
    tts_on   = st.toggle("Audio TTS", value=True)
    tts_lang = st.selectbox("Langue", ["Français 🇫🇷", "Allemand 🇩🇪"])

    st.markdown("<div style='padding:0 8px;'><div class='sb-section'>Options</div></div>", unsafe_allow_html=True)
    show_translation = st.toggle("Traduction 🇩🇪", value=True)
    auto_summary     = st.toggle("Résumé automatique", value=True)
    sim_mode         = st.toggle("Mode simulation patient", value=False)

    st.markdown("""
    <div style='padding:16px;margin-top:auto;border-top:1px solid #DDE3EA;
                position:absolute;bottom:0;left:0;right:0;background:#FFF;'>
        <div style='font-size:0.65rem;color:#7F8C8D;line-height:1.6;'>
            <strong style='color:#1B4F72;'>Groupe 2 · M2 IABD</strong><br>
            HAMAD · KAMNO · EFEMBA · MBOG<br>
            KidneyClassifier v5 · AUC 1.00
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES MÉDICALES
# ─────────────────────────────────────────────────────────────────────────────
CLASSES  = ('Cyst', 'Normal', 'Stone', 'Tumor')
IMG_SIZE = (160, 160)

CLASS_CONFIG = {
    'Cyst':   {'color':'#1B5EA8','bg':'#EBF5FB','border':'#85C1E9','label':'Kyste rénal',             'urgence':'Faible',    'emoji':'💧'},
    'Normal': {'color':'#1A7A4A','bg':'#F0FBF4','border':'#82D0A0','label':'Rein normal',              'urgence':'Aucune',    'emoji':'✅'},
    'Stone':  {'color':'#C4700A','bg':'#FDF8F0','border':'#E8C97A','label':'Lithiase rénale (calcul)', 'urgence':'Modérée',   'emoji':'🪨'},
    'Tumor':  {'color':'#C0392B','bg':'#FDF2F2','border':'#E8A0A0','label':'Tumeur rénale',            'urgence':'Élevée ⚠️', 'emoji':'🔴'},
}
URGENCE_COLORS = {
    'Aucune':'#1A7A4A', 'Faible':'#1B5EA8',
    'Modérée':'#C4700A', 'Élevée ⚠️':'#C0392B'
}
INTERP_TEXTS = {
    'Normal': "L'analyse de l'image CT ne révèle <strong>aucune anomalie rénale significative</strong>. Les structures rénales apparaissent morphologiquement normales. Un suivi de routine est recommandé selon l'âge et les facteurs de risque du patient.",
    'Cyst':   "L'analyse identifie une <strong>formation kystique rénale</strong>. Les kystes rénaux simples sont fréquents et généralement bénins. Une classification selon Bosniak est recommandée pour stratifier le risque. Un <strong>suivi échographique à 6-12 mois</strong> est conseillé.",
    'Stone':  "L'analyse détecte la <strong>présence d'un ou plusieurs calculs rénaux</strong>. Une évaluation urologique est nécessaire pour déterminer la taille, la localisation et la composition du calcul. Un <strong>bilan métabolique et une consultation urologique</strong> sont recommandés.",
    'Tumor':  "L'analyse identifie une <strong>masse rénale suspecte nécessitant une évaluation urgente</strong>. Ce résultat requiert une <strong>confirmation par IRM de caractérisation</strong> et une consultation oncologique/urologique en urgence. Ne pas différer la prise en charge.",
}
MEDICAL_CONTEXT = {
    'Cyst':   {'urgence':'Faible à modérée',                    'suivi':'Échographie à 6-12 mois'},
    'Normal': {'urgence':'Aucune',                               'suivi':'Contrôle de routine'},
    'Stone':  {'urgence':'Modérée — selon taille/localisation', 'suivi':'Consultation urologique'},
    'Tumor':  {'urgence':'⚠️ ÉLEVÉE — consultation urgente',     'suivi':'IRM + avis urologique urgent'},
}

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
ls_active  = langsmith_on and bool(K['LS'])
groq_badge = 'active' if bool(K['GROQ']) else ''
ls_badge   = 'active' if ls_active else ''
groq_txt   = '● GROQ CONNECTED' if bool(K['GROQ']) else '○ GROQ OFFLINE'
ls_txt     = '● LANGSMITH ON'   if ls_active       else '○ LANGSMITH OFF'

st.markdown(f"""
<div class="med-header">
    <div class="med-header-left">
        <div class="med-logo-box">🏥</div>
        <div>
            <div class="med-title">MEDICALScan AI</div>
            <div class="med-subtitle">Renal CT Scan Analysis System · Groupe 2 · M2 IABD</div>
        </div>
    </div>
    <div class="med-header-right">
        <span class="header-badge {groq_badge}">{groq_txt}</span>
        <span class="header-badge {ls_badge}">{ls_txt}</span>
        <span class="header-badge">MobileNetV2 · AUC 1.00</span>
        <span class="header-badge">{datetime.datetime.now().strftime('%d/%m/%Y')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

if sim_mode:
    st.markdown("""
    <div style='background:#FFF8E7;border-bottom:1px solid #F0D080;padding:8px 32px;
                display:flex;align-items:center;gap:10px;'>
        <div class="sim-dot"></div>
        <div style='font-size:0.78rem;color:#7D6608;font-weight:500;'>
            MODE SIMULATION PATIENT ACTIVÉ — Les résultats sont à titre éducatif uniquement
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LANGSMITH & LLM
# ─────────────────────────────────────────────────────────────────────────────
if ls_active:
    os.environ.update({
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY":    K['LS'],
        "LANGCHAIN_PROJECT":    "MEDICALScan-AI",
        "LANGCHAIN_ENDPOINT":   "https://api.smith.langchain.com",
    })


def log_langsmith(run_name, inputs, outputs, meta) -> dict:
    if not ls_active: return {}
    try:
        from langsmith import Client
        c   = Client(api_key=K['LS'])
        rid = c.create_run(name=run_name, run_type="llm", inputs=inputs,
                           project_name="MEDICALScan-AI", extra={"metadata": meta})
        c.update_run(rid, outputs=outputs, end_time=datetime.datetime.utcnow())
        return {'run_id': str(rid)}
    except Exception as e:
        return {'error': str(e)}


def call_llm(messages, system, run_name="llm", max_tokens=1000):
    if not K['GROQ']:
        return "❌ Clé Groq manquante dans secrets.toml.", {}
    t0 = datetime.datetime.now()
    try:
        from groq import Groq
        full = [{"role": "system", "content": system}] + messages
        r    = Groq(api_key=K['GROQ']).chat.completions.create(
            model=groq_model_choice, messages=full,
            max_tokens=max_tokens, temperature=0.3)
        lat  = int((datetime.datetime.now() - t0).total_seconds() * 1000)
        txt  = r.choices[0].message.content.strip()
        ti   = getattr(r.usage, 'prompt_tokens',     0)
        to   = getattr(r.usage, 'completion_tokens', 0)
        meta = {
            'model': groq_model_choice, 'tokens_in': ti, 'tokens_out': to,
            'latency_ms': lat, 'run_name': run_name,
            'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
        }
        log_langsmith(run_name, {"messages": full}, {"text": txt}, meta)
        if 'llm_traces' not in st.session_state:
            st.session_state['llm_traces'] = []
        st.session_state['llm_traces'].append(meta)
        return txt, meta
    except Exception as e:
        return f"❌ Erreur : {e}", {}

# ─────────────────────────────────────────────────────────────────────────────
# MODÈLE CT
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ct_model(mp, tp):
    try:
        import tensorflow as tf
        m = tf.keras.models.load_model(mp)
        t = np.load(tp) if os.path.exists(tp) else np.full(4, 0.5)
        return m, t, None
    except Exception as e:
        return None, None, str(e)


def predict_ct(model, thr, pil):
    img = pil.convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
    x   = np.array(img, dtype=np.float32)[np.newaxis, ...] / 255.0
    p   = model.predict(x, verbose=0)[0]
    s   = p - thr
    ab  = np.where(s > 0)[0]
    idx = (int(ab[0]) if len(ab) == 1
           else int(ab[np.argmax(p[ab])]) if len(ab) > 1
           else int(np.argmax(p)))
    return {
        'class':         CLASSES[idx],
        'class_idx':     idx,
        'confidence':    float(p[idx]),
        'probabilities': {c: float(v) for c, v in zip(CLASSES, p)},
        'timestamp':     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# TTS & TRADUCTION
# ─────────────────────────────────────────────────────────────────────────────
def tts(text, lang_choice):
    try:
        from gtts import gTTS
        import re
        clean = re.sub(r'\*+|#+\s*', '', text)
        clean = re.sub(r'\n+', ' ', clean).strip()
        buf   = io.BytesIO()
        gTTS(text=clean, lang='de' if 'Allemand' in lang_choice else 'fr',
             slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except:
        return None


def translate_de(text_fr):
    r, _ = call_llm(
        [{"role": "user", "content": text_fr}],
        "Traduis ce texte médical du français vers l'allemand. Réponds UNIQUEMENT avec la traduction.",
        run_name="translation_fr_de", max_tokens=800)
    return r


def generate_summary(history, result):
    cls = result['class']
    ctx = MEDICAL_CONTEXT[cls]
    tr  = "\n".join([
        f"{'Patient' if m['role']=='user' else 'Médecin IA'}: {m['content']}"
        for m in history
    ])
    fr, _ = call_llm(
        [{"role": "user", "content": f"Conversation:\n{tr}"}],
        f"""Résume en 5 points (max 250 mots) :
Résultat IA : {cls} ({ctx['urgence']}) | Confiance : {result['confidence']*100:.1f}%
1. Résultat  2. Points clés  3. Recommandations  4. Urgence  5. Avertissement IA
Réponds UNIQUEMENT avec le résumé structuré.""",
        run_name="summary_generation", max_tokens=600)
    return {'fr': fr, 'de': translate_de(fr)}


def build_system_prompt(result):
    cls  = result['class']
    ctx  = MEDICAL_CONTEXT[cls]
    cfg  = CLASS_CONFIG[cls]
    prob = "\n".join([f"  - {c}: {p*100:.1f}%" for c, p in result['probabilities'].items()])
    return f"""Tu es un assistant médical de MEDICALScan AI (radiologie rénale, AUC=1.00).
RÉSULTAT : {cls} ({cfg['label']}) | Confiance : {result['confidence']*100:.1f}%
Probabilités :\n{prob}
Urgence : {ctx['urgence']} | Suivi : {ctx['suivi']}
RÈGLES : Français uniquement. Jamais de diagnostic définitif. Toujours rappeler confirmation médicale.
Pour Tumor avec confiance > 70% : insiste sur l'URGENCE absolue.
Accueille le patient en résumant le résultat clairement en 2-3 phrases professionnelles."""

# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────────────────────────────────────
tab_scan, tab_chat, tab_sum, tab_mon = st.tabs([
    "🔬  Analyse CT",
    "💬  Assistant Médical",
    "📋  Résumé & Rapport",
    "📊  Monitoring",
])

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — ANALYSE CT
# ══════════════════════════════════════════════════════════════════════════════
with tab_scan:

    st.markdown("""
    <div class="sec-divider">
        <div class="sec-divider-line"></div>
        <div class="sec-divider-text">Zone d'analyse</div>
        <div class="sec-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Colonne gauche : Upload ──────────────────────────────────────────────
    with col_left:
        st.markdown("""
        <div class="med-card">
            <div class="med-card-title"><span>📁</span> Image CT — Upload</div>
            <p style='font-size:0.78rem;color:#7F8C8D;margin-bottom:12px;'>
                Formats acceptés : JPEG · PNG · JPG
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"],
                                    label_visibility="collapsed")
        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption=f"Image CT — {uploaded.name}",
                     use_container_width=True)
            st.markdown(f"""
            <div style='display:flex;gap:12px;margin-top:10px;'>
                <div style='flex:1;background:#F8FAFC;border:1px solid #DDE3EA;
                            border-radius:6px;padding:10px;text-align:center;'>
                    <div style='font-size:0.7rem;color:#7F8C8D;'>DIMENSIONS</div>
                    <div style='font-size:0.85rem;font-weight:600;color:#1A2332;
                                font-family:monospace;'>{pil_img.size[0]}x{pil_img.size[1]} px</div>
                </div>
                <div style='flex:1;background:#F8FAFC;border:1px solid #DDE3EA;
                            border-radius:6px;padding:10px;text-align:center;'>
                    <div style='font-size:0.7rem;color:#7F8C8D;'>FORMAT</div>
                    <div style='font-size:0.85rem;font-weight:600;color:#1A2332;'>{pil_img.mode}</div>
                </div>
                <div style='flex:1;background:#F8FAFC;border:1px solid #DDE3EA;
                            border-radius:6px;padding:10px;text-align:center;'>
                    <div style='font-size:0.7rem;color:#7F8C8D;'>TAILLE</div>
                    <div style='font-size:0.85rem;font-weight:600;color:#1A2332;'>{uploaded.size//1024} Ko</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='height:200px;display:flex;align-items:center;
                        justify-content:center;flex-direction:column;gap:12px;'>
                <div style='font-size:2.5rem;opacity:0.3;'>🫁</div>
                <p style='font-size:0.78rem;color:#A0B0C0;text-align:center;'>
                    Glissez une image CT rénale<br>ou cliquez pour sélectionner
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Colonne droite : Résultats ───────────────────────────────────────────
    with col_right:
        if not uploaded:
            st.markdown("""
            <div class="med-card" style='height:420px;display:flex;align-items:center;
                        justify-content:center;flex-direction:column;gap:12px;'>
                <div style='font-size:3rem;opacity:0.2;'>🔬</div>
                <p style='font-size:0.82rem;color:#A0B0C0;text-align:center;'>
                    En attente d'une image CT<br>
                    <span style='font-size:0.72rem;'>Uploadez un scan pour démarrer l'analyse</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Animation chargement
            with st.spinner(""):
                st.markdown("""
                <div style='display:flex;align-items:center;gap:10px;background:#EBF5FB;
                            border:1px solid #85C1E9;border-radius:8px;
                            padding:12px 16px;margin-bottom:12px;'>
                    <div style='width:16px;height:16px;border:2px solid #1B4F72;
                                border-top-color:transparent;border-radius:50%;
                                animation:spin 0.8s linear infinite;'></div>
                    <div style='font-size:0.8rem;color:#1B4F72;font-weight:500;'>
                        Analyse en cours par KidneyClassifier v5...
                    </div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.3)
                model_ct, thr, err = load_ct_model(K['MP'], K['TP'])

            if err:
                st.error(f"Erreur modèle : {err}")
            else:
                result = predict_ct(model_ct, thr, pil_img)
                for k in ['last_result', 'chat_history', 'chat_system',
                          'summary', 'translations', 'audio_data', 'llm_traces']:
                    st.session_state.pop(k, None)
                st.session_state['last_result'] = result

                cls  = result['class']
                conf = result['confidence']
                cfg  = CLASS_CONFIG[cls]
                ctx  = MEDICAL_CONTEXT[cls]

                # ── BLOC 1 : Titre + Classe + Label ──────────────────────────
                # Séparé en st.markdown indépendant pour éviter le conflit
                # f-string Python vs accolades CSS dans un même bloc HTML
                st.markdown(
                    f"<div style='background:#FFFFFF;border:1px solid #DDE3EA;"
                    f"border-radius:10px;padding:24px 24px 0 24px;"
                    f"box-shadow:0 1px 4px rgba(0,0,0,0.06);'>"
                    f"<div style='font-size:0.68rem;font-weight:600;color:#7F8C8D;"
                    f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;'>"
                    f"Diagnostic IA — Résultat</div>"
                    f"<div style='font-size:2.2rem;font-weight:700;color:{cfg['color']};"
                    f"line-height:1.1;margin-bottom:4px;'>{cfg['emoji']} {cls}</div>"
                    f"<div style='font-size:1rem;color:#4A6274;margin-bottom:20px;'>"
                    f"{cfg['label']}</div></div>",
                    unsafe_allow_html=True,
                )

                # ── BLOC 2 : Barre de confiance ───────────────────────────────
                st.markdown(
                    f"<div style='background:#FFFFFF;border:1px solid #DDE3EA;"
                    f"border-radius:0 0 0 0;padding:0 24px 0 24px;"
                    f"box-shadow:0 1px 4px rgba(0,0,0,0.06);margin-top:2px;'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:0.78rem;color:#5D7A8A;margin-bottom:8px;"
                    f"font-weight:500;padding-top:16px;'>"
                    f"<span>Niveau de confiance</span>"
                    f"<span style='font-family:monospace;font-weight:700;"
                    f"color:{cfg['color']};font-size:1rem;'>{conf*100:.1f}%</span></div>"
                    f"<div style='height:10px;background:#EEF2F5;"
                    f"border-radius:5px;overflow:hidden;margin-bottom:20px;'>"
                    f"<div style='height:100%;width:{conf*100:.1f}%;"
                    f"background:{cfg['color']};border-radius:5px;'>"
                    f"</div></div></div>",
                    unsafe_allow_html=True,
                )

                # ── BLOC 3 : Métriques — colonnes natives Streamlit ───────────
                # Évite tout conflit f-string / CSS en utilisant st.columns
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(
                        f"<div style='background:#FFFFFF;border:1px solid #DDE3EA;"
                        f"border-radius:8px;padding:14px 16px;text-align:center;"
                        f"box-shadow:0 1px 3px rgba(0,0,0,0.04);'>"
                        f"<div style='font-size:1.6rem;font-weight:700;"
                        f"color:{cfg['color']};font-family:monospace;'>{conf*100:.1f}%</div>"
                        f"<div style='font-size:0.65rem;color:#7F8C8D;"
                        f"text-transform:uppercase;letter-spacing:1px;"
                        f"margin-top:4px;'>Confiance</div></div>",
                        unsafe_allow_html=True,
                    )
                with m2:
                    urg_color = URGENCE_COLORS.get(cfg['urgence'].split()[0], '#7F8C8D')
                    st.markdown(
                        f"<div style='background:#FFFFFF;border:1px solid #DDE3EA;"
                        f"border-radius:8px;padding:14px 16px;text-align:center;"
                        f"box-shadow:0 1px 3px rgba(0,0,0,0.04);'>"
                        f"<div style='font-size:1rem;font-weight:700;"
                        f"color:{urg_color};'>{cfg['urgence']}</div>"
                        f"<div style='font-size:0.65rem;color:#7F8C8D;"
                        f"text-transform:uppercase;letter-spacing:1px;"
                        f"margin-top:4px;'>Urgence</div></div>",
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f"<div style='background:#FFFFFF;border:1px solid #DDE3EA;"
                        f"border-radius:8px;padding:14px 16px;text-align:center;"
                        f"box-shadow:0 1px 3px rgba(0,0,0,0.04);'>"
                        f"<div style='font-size:0.9rem;font-weight:700;"
                        f"color:#1B4F72;font-family:monospace;'>"
                        f"{result['timestamp'][-8:]}</div>"
                        f"<div style='font-size:0.65rem;color:#7F8C8D;"
                        f"text-transform:uppercase;letter-spacing:1px;"
                        f"margin-top:4px;'>Horodatage</div></div>",
                        unsafe_allow_html=True,
                    )

                # ── BLOC 4 : Alertes ──────────────────────────────────────────
                if cls == "Tumor" and conf > 0.70:
                    st.markdown(
                        f"<div class='alert-critical'>"
                        f"<div class='alert-critical-title'>🚨 CAS CRITIQUE DÉTECTÉ</div>"
                        f"<div class='alert-critical-text'>Tumeur rénale détectée avec une "
                        f"confiance de <strong>{conf*100:.1f}%</strong>. "
                        f"Consultation oncologique/urologique en urgence requise. "
                        f"Ne pas différer.</div></div>",
                        unsafe_allow_html=True,
                    )
                elif cls == "Tumor":
                    st.markdown(
                        "<div class='alert-critical'>"
                        "<div class='alert-critical-title'>⚠️ Tumeur rénale détectée</div>"
                        "<div class='alert-critical-text'>Confirmation par IRM et "
                        "avis spécialisé requis.</div></div>",
                        unsafe_allow_html=True,
                    )
                elif cls == "Stone":
                    st.markdown(
                        "<div class='alert-warning'>"
                        "<div class='alert-warning-title'>🪨 Lithiase rénale détectée</div>"
                        "<div class='alert-warning-text'>Consultation urologique "
                        "recommandée pour évaluation et prise en charge.</div></div>",
                        unsafe_allow_html=True,
                    )
                elif cls == "Cyst":
                    st.markdown(
                        "<div class='alert-info'>"
                        "<div class='alert-info-title'>💧 Kyste rénal détecté</div>"
                        "<div class='alert-info-text'>Suivi échographique à 6-12 mois "
                        "recommandé. Classification Bosniak conseillée.</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='alert-success'>"
                        "<div class='alert-success-title'>✅ Aucune anomalie détectée</div>"
                        "<div class='alert-success-text'>Les structures rénales apparaissent "
                        "normales. Continuer le suivi médical habituel.</div></div>",
                        unsafe_allow_html=True,
                    )

    # ── Section probabilités & interprétation ────────────────────────────────
    if uploaded and 'last_result' in st.session_state:
        result = st.session_state['last_result']
        cls    = result['class']
        cfg    = CLASS_CONFIG[cls]
        ctx    = MEDICAL_CONTEXT[cls]

        st.markdown("""
        <div class="sec-divider">
            <div class="sec-divider-line"></div>
            <div class="sec-divider-text">Distribution des probabilités</div>
            <div class="sec-divider-line"></div>
        </div>
        """, unsafe_allow_html=True)

        col_prob, col_interp = st.columns([1, 1], gap="large")

        with col_prob:
            st.markdown("""
            <div class="med-card">
                <div class="med-card-title"><span>📊</span> Probabilités par classe</div>
            </div>
            """, unsafe_allow_html=True)
            for c, p in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
                c_cfg    = CLASS_CONFIG[c]
                is_pred  = (c == cls)
                bg_style = "background:#F8FAFC;border-radius:6px;padding:4px 8px;" if is_pred else ""
                nm_style = f"font-weight:700;color:{c_cfg['color']};" if is_pred else ""
                pc_style = f"font-weight:700;color:{c_cfg['color']};" if is_pred else ""
                opacity  = "1.0" if is_pred else "0.4"
                badge_lbl = "▶ TOP" if is_pred else c_cfg['emoji']
                st.markdown(
                    f"<div class='prob-row' style='{bg_style}'>"
                    f"<div class='prob-cls-name' style='{nm_style}'>{c}</div>"
                    f"<div class='prob-track'>"
                    f"<div class='prob-fill' style='width:{p*100:.1f}%;"
                    f"background:{c_cfg['color']};opacity:{opacity};'></div></div>"
                    f"<div class='prob-pct' style='{pc_style}'>{p*100:.1f}%</div>"
                    f"<div class='prob-badge' style='background:{c_cfg['bg']};"
                    f"color:{c_cfg['color']};border:1px solid {c_cfg['border']};'>"
                    f"{badge_lbl}</div></div>",
                    unsafe_allow_html=True,
                )

        with col_interp:
            st.markdown(
                f"<div class='interp-card'>"
                f"<div class='interp-title'>📝 Interprétation médicale automatique</div>"
                f"<div class='interp-text'>{INTERP_TEXTS[cls]}</div>"
                f"<div style='margin-top:16px;padding-top:12px;"
                f"border-top:1px solid #EEF2F5;'>"
                f"<div style='font-size:0.68rem;color:#7F8C8D;"
                f"text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>"
                f"Suivi recommandé</div>"
                f"<div style='font-size:0.82rem;font-weight:600;"
                f"color:{cfg['color']};'>{ctx['suivi']}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.info("💬 Consultez l'onglet **Assistant Médical** pour des questions personnalisées sur ce résultat.")

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    result = st.session_state.get('last_result', None)

    if result is None:
        st.markdown("""
        <div class="med-card" style='text-align:center;padding:48px;'>
            <div style='font-size:3rem;opacity:0.3;margin-bottom:16px;'>🔬</div>
            <div style='font-size:0.9rem;color:#7F8C8D;'>
                Aucune analyse disponible<br>
                <span style='font-size:0.78rem;'>
                    Uploadez et analysez une image CT dans l'onglet "Analyse CT"
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if not K['GROQ']:
            st.error("Clé Groq manquante dans `.streamlit/secrets.toml`.")
            st.stop()

        cls  = result['class']
        conf = result['confidence']
        cfg  = CLASS_CONFIG[cls]
        ctx  = MEDICAL_CONTEXT[cls]

        tts_pill  = '<span class="sb-pill sb-ok">🔊 Audio</span>'       if tts_on        else ''
        tr_pill   = '<span class="sb-pill sb-ok">🇩🇪 Traduction</span>'  if show_translation else ''
        ls_pill   = '<span class="sb-pill sb-ok">🟢 LangSmith</span>'   if ls_active     else ''

        st.markdown(
            f"<div class='chat-ctx-bar'>"
            f"<div>"
            f"<div class='chat-ctx-cls' style='color:{cfg['color']};'>"
            f"{cfg['emoji']} {cls} — {cfg['label']}</div>"
            f"<div class='chat-ctx-meta'>"
            f"Confiance : {conf*100:.1f}% · {result['timestamp']} · {groq_model_choice}"
            f"</div></div>"
            f"<div style='display:flex;gap:8px;'>{tts_pill}{tr_pill}{ls_pill}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if sim_mode:
            st.markdown("""
            <div style='background:#FFF8E7;border:1px solid #F0D080;border-left:4px solid #D4AC0D;
                        border-radius:8px;padding:10px 16px;display:flex;align-items:center;
                        gap:10px;margin-bottom:16px;'>
                <div class="sim-dot"></div>
                <div style='font-size:0.78rem;color:#7D6608;font-weight:500;'>
                    Mode simulation — réponses à visée pédagogique
                </div>
            </div>
            """, unsafe_allow_html=True)

        for k, v in [('chat_history', []), ('translations', {}), ('audio_data', {})]:
            if k not in st.session_state:
                st.session_state[k] = v
        if 'chat_system' not in st.session_state:
            st.session_state['chat_system'] = build_system_prompt(result)

        # Accueil automatique
        if len(st.session_state['chat_history']) == 0:
            with st.spinner("Initialisation de l'assistant médical..."):
                welcome, _ = call_llm(
                    [{"role": "user", "content": "Bonjour, je viens de recevoir le résultat de mon scanner rénal."}],
                    st.session_state['chat_system'],
                    run_name="welcome_message",
                )
                st.session_state['chat_history'].append({"role": "assistant", "content": welcome})
                if show_translation:
                    st.session_state['translations'][0] = translate_de(welcome)
                if tts_on:
                    txt = st.session_state['translations'].get(0, welcome) if 'Allemand' in tts_lang else welcome
                    ab  = tts(txt, tts_lang)
                    if ab:
                        st.session_state['audio_data'][0] = ab

        # Affichage historique
        ai = 0
        for msg in st.session_state['chat_history']:
            with st.chat_message(msg['role'], avatar="👤" if msg['role'] == "user" else "🏥"):
                st.markdown(msg['content'])
                if msg['role'] == "assistant":
                    if show_translation:
                        de = st.session_state['translations'].get(ai, "")
                        if de:
                            st.markdown(
                                f"<div class='de-card'>"
                                f"<div class='de-label'>🇩🇪 Deutsche Übersetzung</div>"
                                f"<div class='de-text'>{de}</div></div>",
                                unsafe_allow_html=True,
                            )
                    if tts_on:
                        ab = st.session_state['audio_data'].get(ai)
                        if ab:
                            lang_lbl = 'Deutsch' if 'Allemand' in tts_lang else 'Français'
                            st.markdown(
                                f"<div class='audio-card'>"
                                f"<div class='audio-label'>🔊 Audio — {lang_lbl}</div></div>",
                                unsafe_allow_html=True,
                            )
                            st.audio(ab, format="audio/mp3")
                    ai += 1

        # Input utilisateur
        user_input = st.chat_input("Posez votre question sur ce résultat CT...")
        if user_input:
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

            with st.chat_message("assistant", avatar="🏥"):
                with st.spinner("Analyse de votre question..."):
                    answer, metrics = call_llm(
                        st.session_state['chat_history'],
                        st.session_state['chat_system'],
                        run_name="chatbot_answer",
                    )
                st.markdown(answer)
                st.session_state['chat_history'].append({"role": "assistant", "content": answer})
                idx = sum(1 for m in st.session_state['chat_history'] if m['role'] == "assistant") - 1

                if show_translation:
                    with st.spinner("Traduction 🇩🇪..."):
                        de = translate_de(answer)
                    st.session_state['translations'][idx] = de
                    st.markdown(
                        f"<div class='de-card'>"
                        f"<div class='de-label'>🇩🇪 Deutsche Übersetzung</div>"
                        f"<div class='de-text'>{de}</div></div>",
                        unsafe_allow_html=True,
                    )

                if tts_on:
                    txt = st.session_state['translations'].get(idx, answer) if 'Allemand' in tts_lang else answer
                    with st.spinner("Synthèse audio..."):
                        ab = tts(txt, tts_lang)
                    if ab:
                        st.session_state['audio_data'][idx] = ab
                        lang_lbl = 'Deutsch' if 'Allemand' in tts_lang else 'Français'
                        st.markdown(
                            f"<div class='audio-card'>"
                            f"<div class='audio-label'>🔊 {lang_lbl}</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.audio(ab, format="audio/mp3")

                if metrics and ls_active:
                    st.markdown(
                        f"<div class='trace-bar'>"
                        f"<div class='trace-item'>Run : <span>{metrics.get('run_name','')}</span></div>"
                        f"<div class='trace-item'>Latence : <span>{metrics.get('latency_ms',0)} ms</span></div>"
                        f"<div class='trace-item'>Tokens : <span>{metrics.get('tokens_in',0)} → {metrics.get('tokens_out',0)}</span></div>"
                        f"<div class='trace-item'>Modèle : <span>{metrics.get('model','')}</span></div>"
                        f"<div class='trace-item'>Heure : <span>{metrics.get('timestamp','')}</span></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.session_state.pop('summary', None)

        col_r, col_s, _ = st.columns([1, 1.5, 3])
        with col_r:
            if st.button("🔄 Réinitialiser"):
                for k in ['chat_history', 'chat_system', 'translations', 'audio_data', 'summary']:
                    st.session_state.pop(k, None)
                st.rerun()
        with col_s:
            if len(st.session_state.get('chat_history', [])) >= 2:
                if st.button("📋 Générer résumé"):
                    with st.spinner("Génération du résumé médical..."):
                        st.session_state['summary'] = generate_summary(
                            st.session_state['chat_history'], result)
                    st.success("✅ Résumé généré — consultez l'onglet Résumé & Rapport")

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — RÉSUMÉ & RAPPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_sum:
    result  = st.session_state.get('last_result', None)
    summary = st.session_state.get('summary', None)
    history = st.session_state.get('chat_history', [])

    if result is None:
        st.markdown("""
        <div class="med-card" style='text-align:center;padding:48px;'>
            <div style='font-size:3rem;opacity:0.3;'>📋</div>
            <div style='color:#7F8C8D;font-size:0.85rem;margin-top:12px;'>Aucun résultat disponible</div>
        </div>
        """, unsafe_allow_html=True)
    elif len(history) < 2:
        st.info("💬 Utilisez l'assistant médical puis cliquez sur 'Générer résumé'.")
    else:
        if summary is None and auto_summary and K['GROQ']:
            with st.spinner("Génération automatique du résumé..."):
                summary = generate_summary(history, result)
                st.session_state['summary'] = summary

        if summary:
            cls  = result['class']
            conf = result['confidence']
            cfg  = CLASS_CONFIG[cls]
            ctx  = MEDICAL_CONTEXT[cls]
            urg_c = URGENCE_COLORS.get(cfg['urgence'].split()[0], '#7F8C8D')

            st.markdown(
                f"<div class='med-card' style='border-left:4px solid {cfg['color']};'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:flex-start;flex-wrap:wrap;gap:16px;'>"
                f"<div>"
                f"<div style='font-size:0.68rem;color:#7F8C8D;text-transform:uppercase;"
                f"letter-spacing:1px;margin-bottom:4px;'>Compte Rendu — MEDICALScan AI</div>"
                f"<div style='font-size:1.4rem;font-weight:700;color:{cfg['color']};'>"
                f"{cfg['emoji']} {cls} — {cfg['label']}</div>"
                f"<div style='font-size:0.8rem;color:#5D7A8A;margin-top:4px;'>"
                f"Généré le {result['timestamp']} · Groupe 2 · M2 IABD</div>"
                f"</div>"
                f"<div style='display:flex;gap:12px;flex-wrap:wrap;'>"
                f"<div style='text-align:center;background:#F8FAFC;border:1px solid #DDE3EA;"
                f"border-radius:8px;padding:10px 16px;'>"
                f"<div style='font-size:1.1rem;font-weight:700;color:{cfg['color']};"
                f"font-family:monospace;'>{conf*100:.1f}%</div>"
                f"<div style='font-size:0.62rem;color:#7F8C8D;text-transform:uppercase;'>Confiance</div>"
                f"</div>"
                f"<div style='text-align:center;background:#F8FAFC;border:1px solid #DDE3EA;"
                f"border-radius:8px;padding:10px 16px;'>"
                f"<div style='font-size:0.85rem;font-weight:700;color:{urg_c};'>{cfg['urgence']}</div>"
                f"<div style='font-size:0.62rem;color:#7F8C8D;text-transform:uppercase;'>Urgence</div>"
                f"</div></div></div></div>",
                unsafe_allow_html=True,
            )

            col_fr, col_de = st.columns([1, 1], gap="large")
            with col_fr:
                st.markdown("<div class='sum-lang-label' style='color:#1B4F72;'>🇫🇷 Résumé médical — Français</div>",
                            unsafe_allow_html=True)
                st.markdown(
                    f"<div class='sum-card-fr'><div style='font-size:0.88rem;color:#2C3E50;line-height:1.7;'>"
                    f"{summary['fr'].replace(chr(10), '<br>')}</div></div>",
                    unsafe_allow_html=True,
                )
                if tts_on and st.button("🔊 Écouter en français"):
                    with st.spinner("Synthèse..."):
                        ab = tts(summary['fr'], 'Français 🇫🇷')
                    if ab:
                        st.audio(ab, format="audio/mp3")

            with col_de:
                st.markdown("<div class='sum-lang-label' style='color:#C8A800;'>🇩🇪 Zusammenfassung — Deutsch</div>",
                            unsafe_allow_html=True)
                st.markdown(
                    f"<div class='sum-card-de'><div style='font-size:0.88rem;color:#3D3000;line-height:1.7;'>"
                    f"{summary['de'].replace(chr(10), '<br>')}</div></div>",
                    unsafe_allow_html=True,
                )
                if tts_on and st.button("🔊 Auf Deutsch anhören"):
                    with st.spinner("Synthese..."):
                        ab = tts(summary['de'], 'Allemand 🇩🇪')
                    if ab:
                        st.audio(ab, format="audio/mp3")

            export = (
                f"MEDICALScan AI — COMPTE RENDU · {result['timestamp']}\n"
                f"Groupe 2 · M2 IABD · HAMAD · KAMNO · EFEMBA · MBOG\n"
                f"{'='*60}\n"
                f"Classe : {cls} ({cfg['label']}) | Confiance : {conf*100:.1f}%\n"
                f"Urgence : {ctx['urgence']} | Suivi : {ctx['suivi']}\n"
                f"{'='*60} RESUME FR {'='*60}\n"
                f"{summary['fr']}\n"
                f"{'='*60} ZUSAMMENFASSUNG DE {'='*60}\n"
                f"{summary['de']}\n"
                f"{'='*60}\n"
                f"Resultat IA — a confirmer par un professionnel de sante.\n"
            )
            c1, c2, _ = st.columns([1, 1, 3])
            with c1:
                st.download_button(
                    "⬇️ Télécharger rapport .txt",
                    data=export.encode('utf-8'),
                    file_name=f"MEDICALScan_{result['timestamp'].replace(' ','_').replace(':','-')}.txt",
                    mime="text/plain",
                )
            with c2:
                if st.button("🔄 Régénérer"):
                    st.session_state.pop('summary', None)
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 4 — MONITORING
# ══════════════════════════════════════════════════════════════════════════════
with tab_mon:
    traces = st.session_state.get('llm_traces', [])
    st.markdown("""
    <div style='font-size:1.1rem;font-weight:700;color:#1A2332;margin-bottom:20px;'>
        📊 Monitoring LangSmith
    </div>
    """, unsafe_allow_html=True)

    if not ls_active:
        st.markdown("""
        <div class="med-card">
            <div style='text-align:center;padding:24px;'>
                <div style='font-size:2rem;margin-bottom:12px;'>📡</div>
                <div style='font-size:0.9rem;color:#7F8C8D;margin-bottom:16px;'>LangSmith désactivé</div>
                <div style='font-size:0.78rem;color:#A0B0C0;'>
                    Activez-le dans la barre latérale et ajoutez votre clé
                    <code>ls__...</code> dans <code>.streamlit/secrets.toml</code><br>
                    Clé gratuite sur <strong>smith.langchain.com</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif not traces:
        st.info("Aucune trace disponible — utilisez l'assistant médical pour commencer.")
    else:
        total = len(traces)
        avg_l = int(np.mean([t.get('latency_ms', 0) for t in traces]))
        tok_t = sum(t.get('tokens_in', 0) + t.get('tokens_out', 0) for t in traces)
        tok_o = sum(t.get('tokens_out', 0) for t in traces)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, total,          "Appels LLM"),
            (c2, f"{avg_l} ms",  "Latence moy."),
            (c3, f"{tok_t:,}",   "Tokens total"),
            (c4, tok_o,          "Tokens générés"),
        ]:
            with col:
                st.markdown(
                    f"<div class='mon-metric'>"
                    f"<div class='mon-val'>{val}</div>"
                    f"<div class='mon-lbl'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        import pandas as pd
        df = pd.DataFrame([{
            'Heure':        t.get('timestamp', ''),
            'Run':          t.get('run_name',  ''),
            'Modèle':       t.get('model',     ''),
            'Latence (ms)': t.get('latency_ms', 0),
            'Tokens IN':    t.get('tokens_in',  0),
            'Tokens OUT':   t.get('tokens_out', 0),
        } for t in traces])
        st.dataframe(df, use_container_width=True, hide_index=True)

        if len(traces) > 1:
            st.markdown("""
            <div style='font-size:0.72rem;color:#7F8C8D;text-transform:uppercase;
                        letter-spacing:1px;margin:16px 0 8px;'>Latence par appel (ms)</div>
            """, unsafe_allow_html=True)
            st.bar_chart([t.get('latency_ms', 0) for t in traces])

        st.markdown("""
        <div class="med-card" style='margin-top:12px;'>
            <div style='font-size:0.78rem;color:#5D7A8A;'>
                🌐 Dashboard complet :
                <a href="https://smith.langchain.com" target="_blank"
                   style='color:#1B4F72;font-weight:600;text-decoration:none;'>
                   smith.langchain.com
                </a>
                → Projet : <strong>MEDICALScan-AI</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Effacer les traces"):
            st.session_state['llm_traces'] = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER MÉDICAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="med-footer">
    <div class="footer-disclaimer">
        <strong>⚠️ Avertissement médical :</strong>
        Ce système est un outil d'aide à la décision basé sur l'intelligence artificielle.
        Il ne remplace en aucun cas un diagnostic médical établi par un professionnel de santé qualifié.
        Tout résultat doit être interprété et confirmé par un radiologue ou médecin spécialiste
        sur les images DICOM originales.
    </div>
    <div class="footer-right">
        MEDICALScan AI v6<br>
        KidneyClassifier v5 · MobileNetV2<br>
        Groupe 2 · M2 IABD · 2026<br>
        HAMAD · KAMNO · EFEMBA · MBOG
    </div>
</div>
""", unsafe_allow_html=True)
