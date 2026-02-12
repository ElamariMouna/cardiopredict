"""
================================================================================
🏥 DASHBOARD PROFESSIONNEL - PRÉDICTION DE Niveau de risque de MORTALITÉ CARDIOVASCULAIRE EN USI
================================================================================
Version 3.0 - Design médical professionnel
Modèle: LightGBM (AUC = 0.92) - MIMIC-IV v3.1

Lancer: streamlit run dashboard_app.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="CardioPredict USI — Aide à la Décision Clinique et a la prédiction de niveau de risque de mortalité cardiovasculaire en USI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS PROFESSIONNEL - THÈME CARDIOLOGIE
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* === GLOBAL === */
    .stApp {
        background: linear-gradient(170deg, #0a0e27 0%, #0f1638 30%, #131d42 60%, #0d1230 100%);
    }
    
    .stApp > header { background: transparent !important; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0d2137 30%, #0a1f35 60%, #071520 100%) !important;
        border-right: 1px solid rgba(0,180,180,0.15);
    }
    
    [data-testid="stSidebar"] * {
        color: #c8dce8 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #7eb8d0 !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Sidebar inputs styling */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select {
        background: #0d2a40 !important;
        border: 1px solid rgba(0,180,180,0.25) !important;
        border-radius: 8px !important;
        color: #00ffcc !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] input:focus,
    [data-testid="stSidebar"] select:focus {
        border-color: #00b4b4 !important;
        box-shadow: 0 0 10px rgba(0,180,180,0.25) !important;
        background: #0f3050 !important;
    }
    
    /* Number input value text */
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        color: #00ffcc !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Selectbox text */
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #00ffcc !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 500 !important;
    }
    
    /* Slider value */
    [data-testid="stSidebar"] [data-testid="stThumbValue"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #00ffcc !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
    }
    
    /* Checkbox text */
    [data-testid="stSidebar"] .stCheckbox span {
        color: #c8dce8 !important;
        font-weight: 400 !important;
    }
    
    /* Sidebar checkboxes */
    [data-testid="stSidebar"] .stCheckbox {
        padding: 0.15rem 0 !important;
    }
    
    /* Sidebar section headers */
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00b4b4 !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        padding-bottom: 0.3rem !important;
        border-bottom: 1px solid rgba(0,180,180,0.15) !important;
        margin-top: 1rem !important;
    }
    
    /* Sidebar slider */
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: linear-gradient(90deg, #00b4b4, #0077b6) !important;
    }
    
    /* === HERO BANNER === */
    .hero-banner {
        background: linear-gradient(135deg, #1a1f4e 0%, #2d1b69 40%, #1a3a6e 70%, #0f2847 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(220,50,70,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero-banner::after {
        content: '🫀';
        position: absolute;
        top: 15px;
        right: 25px;
        font-size: 3rem;
        opacity: 0.15;
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #ffffff;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-family: 'Source Sans 3', sans-serif;
        color: rgba(255,255,255,0.55);
        font-size: 0.95rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* === METRIC CARDS === */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        flex: 1;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: 14px 14px 0 0;
    }
    
    .metric-card.prob::before { background: linear-gradient(90deg, #e74c3c, #c0392b); }
    .metric-card.risk::before { background: linear-gradient(90deg, #f39c12, #e67e22); }
    .metric-card.pred::before { background: linear-gradient(90deg, #3498db, #2980b9); }
    
    .metric-label {
        font-family: 'Source Sans 3', sans-serif;
        color: rgba(255,255,255,0.45);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 500;
        line-height: 1.2;
    }
    
    .metric-delta {
        font-family: 'Source Sans 3', sans-serif;
        color: rgba(255,255,255,0.35);
        font-size: 0.75rem;
        margin-top: 0.3rem;
    }
    
    /* === RISK BADGES === */
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    .risk-tres-faible { background: rgba(39,174,96,0.2); color: #2ecc71; border: 1px solid rgba(39,174,96,0.3); }
    .risk-faible { background: rgba(241,196,15,0.2); color: #f1c40f; border: 1px solid rgba(241,196,15,0.3); }
    .risk-modere { background: rgba(230,126,34,0.2); color: #e67e22; border: 1px solid rgba(230,126,34,0.3); }
    .risk-eleve { background: rgba(231,76,60,0.2); color: #e74c3c; border: 1px solid rgba(231,76,60,0.3); }
    .risk-tres-eleve { background: rgba(142,68,173,0.2); color: #9b59b6; border: 1px solid rgba(142,68,173,0.3); }
    
    /* === RESULT PANEL === */
    .result-panel {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .result-panel h2 {
        font-family: 'Playfair Display', serif;
        margin: 0 0 0.5rem 0;
        font-size: 1.4rem;
    }
    
    .result-panel h3 {
        font-family: 'JetBrains Mono', monospace;
        margin: 0 0 0.3rem 0;
        font-size: 1.6rem;
    }
    
    .result-panel p {
        font-family: 'Source Sans 3', sans-serif;
        margin: 0;
        font-size: 0.95rem;
    }
    
    /* === SECTIONS === */
    .section-header {
        font-family: 'Playfair Display', serif;
        color: #ffffff;
        font-size: 1.3rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* === ALERT CARDS === */
    .alert-card {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border-left: 3px solid;
        font-family: 'Source Sans 3', sans-serif;
    }
    
    .alert-card.critical { border-color: #e74c3c; background: rgba(231,76,60,0.08); }
    .alert-card.warning { border-color: #e67e22; background: rgba(230,126,34,0.08); }
    .alert-card.info { border-color: #f1c40f; background: rgba(241,196,15,0.08); }
    
    .alert-title {
        color: #ffffff;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .alert-detail {
        color: rgba(255,255,255,0.5);
        font-size: 0.8rem;
    }
    
    /* === RECOMMENDATION === */
    .reco-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        font-family: 'Source Sans 3', sans-serif;
    }
    
    .reco-box h4 {
        color: rgba(255,255,255,0.6);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 0 0 0.5rem 0;
    }
    
    .reco-box p {
        color: #ffffff;
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
        line-height: 1.5;
    }
    
    /* === EXPORT BUTTON === */
    .stDownloadButton button {
        background: linear-gradient(135deg, #1a3a6e, #2d1b69) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #2d4a8e, #3d2b89) !important;
        box-shadow: 0 4px 15px rgba(45,27,105,0.4) !important;
    }
    
    /* === FOOTER === */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    
    .footer p {
        font-family: 'Source Sans 3', sans-serif;
        color: rgba(255,255,255,0.25);
        font-size: 0.8rem;
        margin: 0.2rem 0;
    }
    
    /* === SIDEBAR STYLING === */
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-family: 'Playfair Display', serif !important;
        color: #ffffff !important;
        font-size: 1.3rem !important;
        border-bottom: 2px solid rgba(0,180,180,0.3) !important;
        padding-bottom: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h3:not(.sidebar-styled) {
        font-family: 'Source Sans 3', sans-serif !important;
        color: #00b4b4 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ECG Line Animation */
    @keyframes ecg-pulse {
        0% { opacity: 0.3; }
        50% { opacity: 0.6; }
        100% { opacity: 0.3; }
    }
    
    .ecg-decoration {
        position: absolute;
        bottom: 10px;
        left: 0;
        right: 0;
        height: 30px;
        opacity: 0.1;
        animation: ecg-pulse 3s infinite;
    }
    
    /* Plotly chart background */
    .js-plotly-plot .plotly .main-svg { background: transparent !important; }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
    }
    
    div[data-testid="stMetric"] label {
        color: rgba(255,255,255,0.5) !important;
        font-family: 'Source Sans 3', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.7rem !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: rgba(255,255,255,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CHARGER LE MODÈLE
# ============================================================================
@st.cache_resource
def load_model():
    model = joblib.load('lightgbm_final_sans_scores.pkl')
    features = joblib.load('feature_names.pkl')
    threshold = joblib.load('optimal_threshold.pkl')
    return model, features, threshold

model, features, threshold = load_model()

# ============================================================================
# PROFILS
# ============================================================================
PROFILES = {
    "🟢 Patient Stable": {
        'desc': "Signes vitaux normaux, pas de support thérapeutique",
        'gender': 0, 'age': 55, 'weight': 70,
        'myocardial_infarction': 0, 'heart_failure': 0, 'arrhythmia': 0,
        'valvular_disease': 0, 'coronary_artery_disease': 0,
        'troponin_t': 0.01, 'ck_mb': 3.0,
        'lactate_max': 1.2, 'lactate_min': 0.8,
        'creatinine_max': 0.9, 'creatinine_min': 0.7,
        'bun_max': 15.0, 'bun_min': 10.0,
        'sbp_min': 115, 'sbp_max': 140, 'dbp_min': 65, 'dbp_max': 85,
        'map_min': 80, 'map_max': 100, 'hr_min': 60, 'hr_max': 85,
        'resp_rate_min': 12, 'resp_rate_max': 18,
        'temperature_min': 36.5, 'temperature_max': 37.2,
        'spo2_min': 96, 'spo2_max': 99, 'glucose_min': 90, 'glucose_max': 140,
        'norepinephrine': 0, 'epinephrine': 0, 'dopamine': 0, 'dobutamine': 0,
        'vasopressor_count': 0, 'mechanical_ventilation': 0,
        'troponin_t_measured': 1, 'ck_mb_measured': 1,
        'lactate_max_measured': 1, 'lactate_min_measured': 1,
        'diag_Arrhythmia': 0, 'diag_Heart_Failure': 0, 'diag_Myocardial_Infarction': 0
    },
    "🟡 Infarctus Modéré": {
        'desc': "IDM, troponine élevée, hémodynamique préservée",
        'gender': 1, 'age': 62, 'weight': 82,
        'myocardial_infarction': 1, 'heart_failure': 0, 'arrhythmia': 0,
        'valvular_disease': 0, 'coronary_artery_disease': 1,
        'troponin_t': 2.5, 'ck_mb': 45.0,
        'lactate_max': 2.5, 'lactate_min': 1.5,
        'creatinine_max': 1.3, 'creatinine_min': 1.0,
        'bun_max': 25.0, 'bun_min': 18.0,
        'sbp_min': 95, 'sbp_max': 145, 'dbp_min': 55, 'dbp_max': 80,
        'map_min': 68, 'map_max': 95, 'hr_min': 70, 'hr_max': 110,
        'resp_rate_min': 14, 'resp_rate_max': 22,
        'temperature_min': 36.3, 'temperature_max': 37.5,
        'spo2_min': 93, 'spo2_max': 98, 'glucose_min': 110, 'glucose_max': 200,
        'norepinephrine': 0, 'epinephrine': 0, 'dopamine': 0, 'dobutamine': 0,
        'vasopressor_count': 0, 'mechanical_ventilation': 0,
        'troponin_t_measured': 1, 'ck_mb_measured': 1,
        'lactate_max_measured': 1, 'lactate_min_measured': 1,
        'diag_Arrhythmia': 0, 'diag_Heart_Failure': 0, 'diag_Myocardial_Infarction': 1
    },
    "🔴 Choc Cardiogénique": {
        'desc': "IC sévère, hypotension, vasopresseurs, ventilation",
        'gender': 1, 'age': 72, 'weight': 78,
        'myocardial_infarction': 1, 'heart_failure': 1, 'arrhythmia': 1,
        'valvular_disease': 0, 'coronary_artery_disease': 1,
        'troponin_t': 5.0, 'ck_mb': 80.0,
        'lactate_max': 7.5, 'lactate_min': 4.0,
        'creatinine_max': 3.5, 'creatinine_min': 2.0,
        'bun_max': 55.0, 'bun_min': 30.0,
        'sbp_min': 65, 'sbp_max': 110, 'dbp_min': 35, 'dbp_max': 65,
        'map_min': 45, 'map_max': 75, 'hr_min': 55, 'hr_max': 130,
        'resp_rate_min': 16, 'resp_rate_max': 32,
        'temperature_min': 35.5, 'temperature_max': 38.5,
        'spo2_min': 82, 'spo2_max': 96, 'glucose_min': 60, 'glucose_max': 280,
        'norepinephrine': 1, 'epinephrine': 1, 'dopamine': 0, 'dobutamine': 1,
        'vasopressor_count': 3, 'mechanical_ventilation': 1,
        'troponin_t_measured': 1, 'ck_mb_measured': 1,
        'lactate_max_measured': 1, 'lactate_min_measured': 1,
        'diag_Arrhythmia': 1, 'diag_Heart_Failure': 1, 'diag_Myocardial_Infarction': 1
    },
    "🟣 Sepsis Cardiaque": {
        'desc': "Infection sévère, atteinte cardiaque, défaillance multi-organes",
        'gender': 0, 'age': 68, 'weight': 65,
        'myocardial_infarction': 0, 'heart_failure': 1, 'arrhythmia': 1,
        'valvular_disease': 0, 'coronary_artery_disease': 0,
        'troponin_t': 1.2, 'ck_mb': 25.0,
        'lactate_max': 9.0, 'lactate_min': 5.0,
        'creatinine_max': 4.5, 'creatinine_min': 2.5,
        'bun_max': 70.0, 'bun_min': 35.0,
        'sbp_min': 60, 'sbp_max': 100, 'dbp_min': 30, 'dbp_max': 55,
        'map_min': 40, 'map_max': 65, 'hr_min': 90, 'hr_max': 145,
        'resp_rate_min': 20, 'resp_rate_max': 38,
        'temperature_min': 35.0, 'temperature_max': 39.5,
        'spo2_min': 78, 'spo2_max': 94, 'glucose_min': 50, 'glucose_max': 320,
        'norepinephrine': 1, 'epinephrine': 1, 'dopamine': 1, 'dobutamine': 0,
        'vasopressor_count': 3, 'mechanical_ventilation': 1,
        'troponin_t_measured': 1, 'ck_mb_measured': 1,
        'lactate_max_measured': 1, 'lactate_min_measured': 1,
        'diag_Arrhythmia': 1, 'diag_Heart_Failure': 1, 'diag_Myocardial_Infarction': 0
    },
    "🟠 IC Décompensée": {
        'desc': "IC chronique décompensée, surcharge, rein altéré",
        'gender': 1, 'age': 78, 'weight': 90,
        'myocardial_infarction': 0, 'heart_failure': 1, 'arrhythmia': 1,
        'valvular_disease': 1, 'coronary_artery_disease': 1,
        'troponin_t': 0.15, 'ck_mb': 8.0,
        'lactate_max': 3.5, 'lactate_min': 2.0,
        'creatinine_max': 2.8, 'creatinine_min': 1.8,
        'bun_max': 50.0, 'bun_min': 28.0,
        'sbp_min': 85, 'sbp_max': 130, 'dbp_min': 45, 'dbp_max': 70,
        'map_min': 58, 'map_max': 85, 'hr_min': 65, 'hr_max': 115,
        'resp_rate_min': 16, 'resp_rate_max': 28,
        'temperature_min': 36.0, 'temperature_max': 37.8,
        'spo2_min': 88, 'spo2_max': 96, 'glucose_min': 80, 'glucose_max': 220,
        'norepinephrine': 0, 'epinephrine': 0, 'dopamine': 0, 'dobutamine': 1,
        'vasopressor_count': 1, 'mechanical_ventilation': 0,
        'troponin_t_measured': 1, 'ck_mb_measured': 1,
        'lactate_max_measured': 1, 'lactate_min_measured': 1,
        'diag_Arrhythmia': 1, 'diag_Heart_Failure': 1, 'diag_Myocardial_Infarction': 0
    },
    "✏️ Personnalisé": {
        'desc': "Entrer manuellement les valeurs",
        'gender': 1, 'age': 65, 'weight': 75,
        'myocardial_infarction': 0, 'heart_failure': 0, 'arrhythmia': 0,
        'valvular_disease': 0, 'coronary_artery_disease': 0,
        'troponin_t': 0.01, 'ck_mb': 3.0,
        'lactate_max': 1.5, 'lactate_min': 1.0,
        'creatinine_max': 1.0, 'creatinine_min': 0.8,
        'bun_max': 20.0, 'bun_min': 15.0,
        'sbp_min': 110, 'sbp_max': 145, 'dbp_min': 60, 'dbp_max': 85,
        'map_min': 75, 'map_max': 100, 'hr_min': 65, 'hr_max': 90,
        'resp_rate_min': 14, 'resp_rate_max': 20,
        'temperature_min': 36.5, 'temperature_max': 37.2,
        'spo2_min': 96, 'spo2_max': 99, 'glucose_min': 90, 'glucose_max': 150,
        'norepinephrine': 0, 'epinephrine': 0, 'dopamine': 0, 'dobutamine': 0,
        'vasopressor_count': 0, 'mechanical_ventilation': 0,
        'troponin_t_measured': 1, 'ck_mb_measured': 1,
        'lactate_max_measured': 1, 'lactate_min_measured': 1,
        'diag_Arrhythmia': 0, 'diag_Heart_Failure': 0, 'diag_Myocardial_Infarction': 0
    }
}

# ============================================================================
# FONCTIONS
# ============================================================================
def get_risk(proba):
    if proba < 0.10: return "Très Faible", "#2ecc71", "risk-tres-faible", "Surveillance standard. Monitoring de routine."
    elif proba < 0.25: return "Faible", "#f1c40f", "risk-faible", "Surveillance rapprochée. Biomarqueurs / 12h."
    elif proba < 0.50: return "Modéré", "#e67e22", "risk-modere", "Monitoring intensif. Réévaluation / 4-6h. Imagerie."
    elif proba < 0.75: return "Élevé", "#e74c3c", "risk-eleve", "Alerte USI. Intervention rapide. Discussion multidisciplinaire."
    else: return "Très Élevé", "#9b59b6", "risk-tres-eleve", "Intervention urgente. Considérer soins palliatifs si approprié."

def make_gauge(proba, threshold):
    rl, color, _, _ = get_risk(proba)
    import math
    
    fig = go.Figure()
    
    # Colored arc zones (background)
    zone_configs = [
        (0, 10, '#2ecc71', 0.35, 'Très Faible'),
        (10, 25, '#f1c40f', 0.30, 'Faible'),
        (25, 50, '#e67e22', 0.30, 'Modéré'),
        (50, 75, '#e74c3c', 0.35, 'Élevé'),
        (75, 100, '#9b59b6', 0.35, 'Très Élevé'),
    ]
    
    for lo, hi, zcolor, opacity, label in zone_configs:
        theta0 = 180 - (lo / 100) * 180
        theta1 = 180 - (hi / 100) * 180
        n_pts = 30
        r_outer, r_inner = 0.95, 0.65
        angles = [theta1 + (theta0 - theta1) * i / n_pts for i in range(n_pts + 1)]
        xs = [r_outer * math.cos(math.radians(a)) for a in angles]
        ys = [r_outer * math.sin(math.radians(a)) for a in angles]
        xs += [r_inner * math.cos(math.radians(a)) for a in reversed(angles)]
        ys += [r_inner * math.sin(math.radians(a)) for a in reversed(angles)]
        xs.append(xs[0]); ys.append(ys[0])
        
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill='toself', fillcolor=zcolor,
            opacity=opacity, line=dict(width=0),
            hoverinfo='skip', showlegend=False
        ))
        
        # Zone labels
        mid_angle = 180 - ((lo + hi) / 2 / 100) * 180
        lx = 1.08 * math.cos(math.radians(mid_angle))
        ly = 1.08 * math.sin(math.radians(mid_angle))
        fig.add_annotation(
            x=lx, y=ly, text=f"<b>{label}</b>",
            font=dict(size=9, color=zcolor), showarrow=False, opacity=0.7
        )
    
    # Threshold marker line
    th_angle = 180 - (threshold * 100 / 100) * 180
    fig.add_trace(go.Scatter(
        x=[0.63 * math.cos(math.radians(th_angle)), 0.97 * math.cos(math.radians(th_angle))],
        y=[0.63 * math.sin(math.radians(th_angle)), 0.97 * math.sin(math.radians(th_angle))],
        mode='lines', line=dict(color='#ffffff', width=2, dash='dot'),
        hoverinfo='skip', showlegend=False
    ))
    fig.add_annotation(
        x=1.05 * math.cos(math.radians(th_angle)),
        y=1.05 * math.sin(math.radians(th_angle)),
        text=f"Seuil<br>{threshold:.0%}", font=dict(size=8, color='rgba(255,255,255,0.5)'),
        showarrow=False
    )
    
    # RED NEEDLE
    needle_angle = 180 - (proba * 100 / 100) * 180
    needle_len = 0.85
    nx = needle_len * math.cos(math.radians(needle_angle))
    ny = needle_len * math.sin(math.radians(needle_angle))
    
    # Needle body (thick red line)
    fig.add_trace(go.Scatter(
        x=[0, nx], y=[0, ny],
        mode='lines', line=dict(color='#ff1744', width=5),
        hoverinfo='skip', showlegend=False
    ))
    
    # Needle tip (triangle)
    tip_w = 0.03
    perp_angle = needle_angle + 90
    fig.add_trace(go.Scatter(
        x=[nx, tip_w * math.cos(math.radians(perp_angle)), -tip_w * math.cos(math.radians(perp_angle)), nx],
        y=[ny, tip_w * math.sin(math.radians(perp_angle)), -tip_w * math.sin(math.radians(perp_angle)), ny],
        fill='toself', fillcolor='#ff1744', line=dict(width=0),
        hoverinfo='skip', showlegend=False
    ))
    
    # Center circle
    circle_pts = 30
    cx = [0.06 * math.cos(math.radians(i * 360 / circle_pts)) for i in range(circle_pts + 1)]
    cy = [0.06 * math.sin(math.radians(i * 360 / circle_pts)) for i in range(circle_pts + 1)]
    fig.add_trace(go.Scatter(
        x=cx, y=cy, fill='toself', fillcolor='#ff1744',
        line=dict(color='#cc0033', width=2),
        hoverinfo='skip', showlegend=False
    ))
    
    # Tick marks
    for tick in range(0, 101, 10):
        a = 180 - (tick / 100) * 180
        fig.add_trace(go.Scatter(
            x=[0.60 * math.cos(math.radians(a)), 0.65 * math.cos(math.radians(a))],
            y=[0.60 * math.sin(math.radians(a)), 0.65 * math.sin(math.radians(a))],
            mode='lines', line=dict(color='rgba(255,255,255,0.3)', width=1),
            hoverinfo='skip', showlegend=False
        ))
        fig.add_annotation(
            x=0.54 * math.cos(math.radians(a)),
            y=0.54 * math.sin(math.radians(a)),
            text=str(tick), font=dict(size=9, color='rgba(255,255,255,0.35)'),
            showarrow=False
        )
    
    # Value display
    fig.add_annotation(
        x=0, y=-0.15,
        text=f"<b style='font-size:2.5rem;color:white;font-family:JetBrains Mono'>{proba:.1%}</b>",
        font=dict(size=36, color='white', family='JetBrains Mono'),
        showarrow=False
    )
    fig.add_annotation(
        x=0, y=-0.32,
        text=f"<b style='color:{color}'>● {rl}</b>",
        font=dict(size=14, color=color),
        showarrow=False
    )
    
    fig.update_layout(
        height=340, margin=dict(l=20, r=20, t=20, b=50),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[-1.3, 1.3], visible=False, fixedrange=True),
        yaxis=dict(range=[-0.45, 1.2], visible=False, fixedrange=True, scaleanchor='x'),
        showlegend=False, font=dict(color='white')
    )
    return fig

def make_organ_radar(d):
    """Graphique radar d'évaluation des organes"""
    
    # Score de sévérité par organe (0=normal, 100=critique)
    def cardiac_score():
        s = 0
        if d['troponin_t'] > 0.1: s += min(d['troponin_t'] / 5.0 * 40, 40)
        if d['hr_max'] > 100: s += min((d['hr_max'] - 100) / 50 * 20, 20)
        if d['hr_min'] < 50: s += 15
        if d['myocardial_infarction']: s += 15
        if d['heart_failure']: s += 15
        if d['arrhythmia']: s += 10
        return min(s, 100)
    
    def pulmonary_score():
        s = 0
        if d['spo2_min'] < 95: s += min((95 - d['spo2_min']) / 15 * 50, 50)
        if d['resp_rate_max'] > 20: s += min((d['resp_rate_max'] - 20) / 20 * 25, 25)
        if d['mechanical_ventilation']: s += 30
        return min(s, 100)
    
    def renal_score():
        s = 0
        if d['creatinine_max'] > 1.2: s += min((d['creatinine_max'] - 1.2) / 3.8 * 50, 50)
        if d['bun_max'] > 20: s += min((d['bun_max'] - 20) / 60 * 30, 30)
        return min(s, 100)
    
    def metabolic_score():
        s = 0
        if d['lactate_max'] > 2: s += min((d['lactate_max'] - 2) / 8 * 60, 60)
        if d['glucose_max'] > 180: s += min((d['glucose_max'] - 180) / 200 * 20, 20)
        if d['temperature_max'] > 38: s += min((d['temperature_max'] - 38) / 2 * 15, 15)
        if d['temperature_min'] < 36: s += 10
        return min(s, 100)
    
    def perfusion_score():
        s = 0
        if d['map_min'] < 65: s += min((65 - d['map_min']) / 30 * 40, 40)
        if d['sbp_min'] < 90: s += min((90 - d['sbp_min']) / 30 * 25, 25)
        s += min(d['vasopressor_count'] * 15, 40)
        return min(s, 100)
    
    categories = ['🫀 Cardiaque', '🫁 Pulmonaire', '🧫 Rénal', '⚗️ Métabolique', '🩸 Perfusion']
    scores = [cardiac_score(), pulmonary_score(), renal_score(), metabolic_score(), perfusion_score()]
    
    # Colors based on severity
    def score_color(s):
        if s < 20: return '#2ecc71'
        elif s < 40: return '#f1c40f'
        elif s < 60: return '#e67e22'
        elif s < 80: return '#e74c3c'
        else: return '#9b59b6'
    
    colors = [score_color(s) for s in scores]
    
    # Close the polygon
    cats_closed = categories + [categories[0]]
    scores_closed = scores + [scores[0]]
    
    fig = go.Figure()
    
    # Background zones
    for zone_val, zone_color, zone_name in [(100, 'rgba(155,89,182,0.08)', ''), 
                                             (80, 'rgba(231,76,60,0.08)', ''),
                                             (60, 'rgba(230,126,34,0.08)', ''),
                                             (40, 'rgba(241,196,15,0.06)', ''),
                                             (20, 'rgba(46,204,113,0.06)', '')]:
        fig.add_trace(go.Scatterpolar(
            r=[zone_val] * 6, theta=cats_closed,
            fill='toself', fillcolor=zone_color,
            line=dict(color='rgba(255,255,255,0.05)', width=1),
            hoverinfo='skip', showlegend=False
        ))
    
    # Main data
    fig.add_trace(go.Scatterpolar(
        r=scores_closed, theta=cats_closed,
        fill='toself',
        fillcolor='rgba(231,76,60,0.15)',
        line=dict(color='#e74c3c', width=2.5),
        hoverinfo='skip', showlegend=False
    ))
    
    # Data points with individual colors
    fig.add_trace(go.Scatterpolar(
        r=scores, theta=categories,
        mode='markers+text',
        marker=dict(size=12, color=colors, line=dict(color='white', width=1.5)),
        text=[f'{s:.0f}' for s in scores],
        textposition='top center',
        textfont=dict(color='white', size=11, family='JetBrains Mono'),
        hovertemplate='%{theta}: %{r:.0f}/100<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=['20', '40', '60', '80', '100'],
                tickfont=dict(size=8, color='rgba(255,255,255,0.25)'),
                gridcolor='rgba(255,255,255,0.06)',
                linecolor='rgba(255,255,255,0.05)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='rgba(255,255,255,0.8)', family='Source Sans 3'),
                gridcolor='rgba(255,255,255,0.06)',
                linecolor='rgba(255,255,255,0.05)'
            )
        ),
        height=380, margin=dict(l=60, r=60, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig, categories, scores, colors

def get_alerts(d):
    a = []
    if d['lactate_max'] > 4: a.append(("critical", "Hyperlactatémie sévère", f"Lactate = {d['lactate_max']:.1f} mmol/L"))
    elif d['lactate_max'] > 2: a.append(("warning", "Lactate élevé", f"Lactate = {d['lactate_max']:.1f} mmol/L"))
    if d['map_min'] < 65: a.append(("critical", "Hypotension", f"MAP = {d['map_min']} mmHg"))
    if d['spo2_min'] < 90: a.append(("critical", "Hypoxémie sévère", f"SpO2 = {d['spo2_min']}%"))
    elif d['spo2_min'] < 94: a.append(("warning", "Hypoxémie", f"SpO2 = {d['spo2_min']}%"))
    if d['creatinine_max'] > 2: a.append(("warning", "Insuffisance rénale", f"Créat = {d['creatinine_max']:.1f} mg/dL"))
    if d['troponin_t'] > 0.1: a.append(("warning", "Troponine élevée", f"TnT = {d['troponin_t']:.2f} ng/mL"))
    if d['vasopressor_count'] >= 2: a.append(("critical", "Vasopresseurs multiples", f"n = {d['vasopressor_count']}"))
    if d['mechanical_ventilation'] == 1: a.append(("info", "Ventilation mécanique", "Patient intubé"))
    if d['sbp_min'] < 90: a.append(("critical", "Hypotension systolique", f"SBP = {d['sbp_min']} mmHg"))
    if d['hr_max'] > 120: a.append(("warning", "Tachycardie", f"FC = {d['hr_max']} bpm"))
    return a

def gen_report(d, proba, rl, rc, reco, alerts, sv, feats):
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    pred = "DÉCÈS" if proba >= threshold else "SURVIE"
    sh = ""
    if sv is not None:
        idx = np.argsort(np.abs(sv))[::-1][:10]
        sh = "<table border='1' cellpadding='5' style='border-collapse:collapse;width:100%;font-size:13px'>"
        sh += "<tr style='background:#1a1f4e;color:white'><th>Variable</th><th>Valeur</th><th>Impact</th><th>Direction</th></tr>"
        for i in idx:
            v = d.get(feats[i], 'N/A'); imp = sv[i]
            c = "#e74c3c" if imp > 0 else "#27ae60"
            di = "↑ Risque" if imp > 0 else "↓ Protection"
            sh += f"<tr><td>{feats[i]}</td><td>{v}</td><td style='color:{c}'>{imp:+.4f}</td><td style='color:{c}'>{di}</td></tr>"
        sh += "</table>"
    ah = "".join([f"<p>{'🔴' if t=='critical' else '🟠' if t=='warning' else '🟡'} <b>{ti}</b> — {de}</p>" for t, ti, de in alerts]) or "<p>✅ Aucune alerte</p>"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>CardioPredict — Rapport</title>
    <style>body{{font-family:Arial;margin:40px;color:#333}}
    .hdr{{text-align:center;border-bottom:3px solid #1a1f4e;padding-bottom:20px;margin-bottom:30px}}
    .hdr h1{{color:#1a1f4e}}.res{{background:{rc}15;border:2px solid {rc};border-radius:10px;padding:20px;text-align:center;margin:20px 0}}
    .res .p{{font-size:3rem;font-weight:bold;color:{rc}}}.s{{margin:25px 0}}.s h3{{color:#1a1f4e;border-bottom:1px solid #ddd;padding-bottom:5px}}
    table{{width:100%}}th,td{{padding:6px 10px;text-align:left}}.ft{{text-align:center;margin-top:40px;color:#999;font-size:12px}}</style></head>
    <body><div class="hdr"><h1>🫀 CardioPredict — Rapport de Prédiction</h1><p>Date: {now} | LightGBM (AUC=0.92) | MIMIC-IV v3.1</p></div>
    <div class="res"><div class="p">{proba:.1%}</div><h2>Risque {rl} — {pred}</h2><p>Seuil: {threshold:.3f}</p></div>
    <div class="s"><h3>📋 Recommandation</h3><p style="background:#f8f9fa;padding:15px;border-left:4px solid {rc};border-radius:8px"><b>{reco}</b></p></div>
    <div class="s"><h3>⚠️ Alertes</h3>{ah}</div>
    <div class="s"><h3>📊 Données</h3><table border="1" cellpadding="5" style="border-collapse:collapse">
    <tr style="background:#ecf0f1"><th colspan="4">Patient</th></tr>
    <tr><td>Sexe</td><td>{"H" if d['gender']==1 else "F"}</td><td>Âge</td><td>{d['age']}</td></tr>
    <tr><td>Lactate</td><td>{d['lactate_max']:.1f}/{d['lactate_min']:.1f}</td><td>Créat</td><td>{d['creatinine_max']:.1f}/{d['creatinine_min']:.1f}</td></tr>
    <tr><td>MAP</td><td>{d['map_min']}-{d['map_max']}</td><td>SpO2</td><td>{d['spo2_min']}-{d['spo2_max']}%</td></tr>
    <tr><td>Vaso</td><td>{d['vasopressor_count']}</td><td>Vent</td><td>{"Oui" if d['mechanical_ventilation'] else "Non"}</td></tr>
    </table></div>
    <div class="s"><h3>🔍 SHAP</h3>{sh}</div>
    <div class="ft"><p>⚠️ Ne remplace pas le jugement clinique</p><p>CardioPredict — Thèse de doctorat</p></div></body></html>"""

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("## 🫀 CardioPredict")
st.sidebar.markdown("*Aide à la décision clinique*")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🎯 Profil Clinique")
sel = st.sidebar.selectbox("Profil:", list(PROFILES.keys()))
p = PROFILES[sel]
st.sidebar.caption(f"📝 {p['desc']}")
st.sidebar.markdown("---")

st.sidebar.markdown("### 👤 Démographie")
gender = st.sidebar.selectbox("Sexe", ["Homme", "Femme"], index=0 if p['gender']==1 else 1)
age = st.sidebar.slider("Âge", 18, 100, p['age'])
weight = st.sidebar.slider("Poids (kg)", 30, 200, p['weight'])

st.sidebar.markdown("### ❤️ Diagnostics")
mi = st.sidebar.checkbox("Infarctus", value=bool(p['myocardial_infarction']))
hf = st.sidebar.checkbox("Insuffisance cardiaque", value=bool(p['heart_failure']))
arr = st.sidebar.checkbox("Arythmie", value=bool(p['arrhythmia']))
vd = st.sidebar.checkbox("Valvulopathie", value=bool(p['valvular_disease']))
cad = st.sidebar.checkbox("Coronaropathie", value=bool(p['coronary_artery_disease']))
diag_arr = st.sidebar.checkbox("Diag: Arythmie", value=bool(p['diag_Arrhythmia']))
diag_hf = st.sidebar.checkbox("Diag: IC", value=bool(p['diag_Heart_Failure']))
diag_mi = st.sidebar.checkbox("Diag: IDM", value=bool(p['diag_Myocardial_Infarction']))

st.sidebar.markdown("### 🩸 Biomarqueurs")
trop = st.sidebar.number_input("Troponine T", 0.0, 50.0, float(p['troponin_t']), 0.01)
ckmb = st.sidebar.number_input("CK-MB", 0.0, 500.0, float(p['ck_mb']), 1.0)
lactate_max = st.sidebar.number_input("Lactate Max", 0.0, 30.0, float(p['lactate_max']), 0.1)
lactate_min = st.sidebar.number_input("Lactate Min", 0.0, 30.0, float(p['lactate_min']), 0.1)
creat_max = st.sidebar.number_input("Créatinine Max", 0.0, 20.0, float(p['creatinine_max']), 0.1)
creat_min = st.sidebar.number_input("Créatinine Min", 0.0, 20.0, float(p['creatinine_min']), 0.1)
bun_max = st.sidebar.number_input("BUN Max", 0.0, 200.0, float(p['bun_max']), 1.0)
bun_min = st.sidebar.number_input("BUN Min", 0.0, 200.0, float(p['bun_min']), 1.0)

st.sidebar.markdown("### 💉 Hémodynamique")
sbp_min = st.sidebar.number_input("SBP Min", 30, 250, int(p['sbp_min']))
sbp_max = st.sidebar.number_input("SBP Max", 30, 300, int(p['sbp_max']))
dbp_min = st.sidebar.number_input("DBP Min", 20, 150, int(p['dbp_min']))
dbp_max = st.sidebar.number_input("DBP Max", 20, 200, int(p['dbp_max']))
map_min = st.sidebar.number_input("MAP Min", 20, 150, int(p['map_min']))
map_max = st.sidebar.number_input("MAP Max", 20, 200, int(p['map_max']))
hr_min = st.sidebar.number_input("FC Min", 20, 200, int(p['hr_min']))
hr_max = st.sidebar.number_input("FC Max", 20, 250, int(p['hr_max']))

st.sidebar.markdown("### 🫁 Respiratoire")
resp_min = st.sidebar.number_input("FR Min", 5, 50, int(p['resp_rate_min']))
resp_max = st.sidebar.number_input("FR Max", 5, 60, int(p['resp_rate_max']))
spo2_min = st.sidebar.number_input("SpO2 Min", 50, 100, int(p['spo2_min']))
spo2_max = st.sidebar.number_input("SpO2 Max", 50, 100, int(p['spo2_max']))
temp_min = st.sidebar.number_input("Temp Min", 32.0, 42.0, float(p['temperature_min']), 0.1)
temp_max = st.sidebar.number_input("Temp Max", 32.0, 42.0, float(p['temperature_max']), 0.1)
gluc_min = st.sidebar.number_input("Glucose Min", 20, 500, int(p['glucose_min']))
gluc_max = st.sidebar.number_input("Glucose Max", 20, 800, int(p['glucose_max']))

st.sidebar.markdown("### 💊 Support")
norepi = st.sidebar.checkbox("Norépinéphrine", value=bool(p['norepinephrine']))
epi = st.sidebar.checkbox("Épinéphrine", value=bool(p['epinephrine']))
dopa = st.sidebar.checkbox("Dopamine", value=bool(p['dopamine']))
dobu = st.sidebar.checkbox("Dobutamine", value=bool(p['dobutamine']))
vaso_count = st.sidebar.number_input("Nb vasopresseurs", 0, 5, int(p['vasopressor_count']))
mech_vent = st.sidebar.checkbox("Ventilation mécanique", value=bool(p['mechanical_ventilation']))
trop_m = st.sidebar.checkbox("Troponine mesurée", value=True)
ckmb_m = st.sidebar.checkbox("CK-MB mesurée", value=True)
lm = st.sidebar.checkbox("Lactate max mesuré", value=True)
ln = st.sidebar.checkbox("Lactate min mesuré", value=True)

# ============================================================================
# PATIENT + PRÉDICTION
# ============================================================================
dat = {
    'gender': 1 if gender=="Homme" else 0, 'age': age, 'weight': weight,
    'myocardial_infarction': int(mi), 'heart_failure': int(hf),
    'arrhythmia': int(arr), 'valvular_disease': int(vd), 'coronary_artery_disease': int(cad),
    'troponin_t': trop, 'ck_mb': ckmb,
    'lactate_max': lactate_max, 'lactate_min': lactate_min,
    'creatinine_max': creat_max, 'creatinine_min': creat_min,
    'bun_max': bun_max, 'bun_min': bun_min,
    'sbp_min': sbp_min, 'sbp_max': sbp_max, 'dbp_min': dbp_min, 'dbp_max': dbp_max,
    'map_min': map_min, 'map_max': map_max, 'hr_min': hr_min, 'hr_max': hr_max,
    'resp_rate_min': resp_min, 'resp_rate_max': resp_max,
    'temperature_min': temp_min, 'temperature_max': temp_max,
    'spo2_min': spo2_min, 'spo2_max': spo2_max, 'glucose_min': gluc_min, 'glucose_max': gluc_max,
    'norepinephrine': int(norepi), 'epinephrine': int(epi),
    'dopamine': int(dopa), 'dobutamine': int(dobu),
    'vasopressor_count': vaso_count, 'mechanical_ventilation': int(mech_vent),
    'troponin_t_measured': int(trop_m), 'ck_mb_measured': int(ckmb_m),
    'lactate_max_measured': int(lm), 'lactate_min_measured': int(ln),
    'diag_Arrhythmia': int(diag_arr), 'diag_Heart_Failure': int(diag_hf),
    'diag_Myocardial_Infarction': int(diag_mi)
}

pdf = pd.DataFrame({k: [v] for k, v in dat.items()}, columns=features)
proba = model.predict_proba(pdf)[:, 1][0]
rl, rc, rcl, reco = get_risk(proba)
pred = "DÉCÈS" if proba >= threshold else "SURVIE"
alerts = get_alerts(dat)

# ============================================================================
# AFFICHAGE
# ============================================================================

# Hero Banner
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">CardioPredict</div>
    <div class="hero-subtitle">Système d'Aide à la Décision Clinique — Unité de Soins Intensifs Cardiovasculaire</div>
    <div style="margin-top:0.8rem;">
        <span class="risk-badge {rcl}">{sel}</span>
        <span style="color:rgba(255,255,255,0.3); margin:0 0.5rem;">|</span>
        <span style="color:rgba(255,255,255,0.4); font-family:'Source Sans 3',sans-serif; font-size:0.85rem;">
            LightGBM · AUC 0.92 · MIMIC-IV v3.1 · 13,569 patients
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Probabilité de décès", f"{proba:.1%}")
col2.metric("Niveau de risque", rl)
col3.metric("Prédiction", pred, f"Seuil: {threshold:.3f}")

# Gauge + Result
col1, col2 = st.columns([1, 1])
with col1:
    st.plotly_chart(make_gauge(proba, threshold), use_container_width=True)

with col2:
    st.markdown(f"""
    <div class="result-panel" style="background:linear-gradient(135deg,{rc}22,{rc}11); margin-top:0.5rem;">
        <h2 style="color:{rc}">Risque {rl}</h2>
        <h3 style="color:white">{proba:.1%}</h3>
        <p style="color:rgba(255,255,255,0.6)">Prédiction: <strong style="color:white">{pred}</strong></p>
    </div>
    <div class="reco-box">
        <h4>📋 RECOMMANDATION CLINIQUE</h4>
        <p>{reco}</p>
    </div>
    """, unsafe_allow_html=True)

# Alerts
st.markdown('<div class="section-header">⚠️ Alertes Cliniques</div>', unsafe_allow_html=True)
if alerts:
    cols = st.columns(min(len(alerts), 3))
    for i, (sev, title, detail) in enumerate(alerts):
        cols[i % 3].markdown(f'<div class="alert-card {sev}"><span class="alert-title">{title}</span><br><span class="alert-detail">{detail}</span></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert-card info"><span class="alert-title">✅ Aucune alerte clinique majeure</span></div>', unsafe_allow_html=True)

# Organ Radar
st.markdown('<div class="section-header">🏥 Évaluation par Système Organique</div>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    fig_radar, organ_names, organ_scores, organ_colors = make_organ_radar(dat)
    st.plotly_chart(fig_radar, use_container_width=True)
with col2:
    st.markdown("""
    <div style="padding:1rem 0;">
        <p style="color:rgba(255,255,255,0.5); font-family:'Source Sans 3',sans-serif; font-size:0.85rem; margin-bottom:1rem;">
            Score de sévérité par organe (0 = normal, 100 = critique)
        </p>
    </div>
    """, unsafe_allow_html=True)
    for name, score, color in zip(organ_names, organ_scores, organ_colors):
        if score < 20: status = "Normal"
        elif score < 40: status = "Léger"
        elif score < 60: status = "Modéré"
        elif score < 80: status = "Sévère"
        else: status = "Critique"
        pct = score / 100 * 100
        st.markdown(f"""
        <div style="margin:0.5rem 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
                <span style="color:rgba(255,255,255,0.8); font-family:'Source Sans 3',sans-serif; font-size:0.9rem;">{name}</span>
                <span style="color:{color}; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">{score:.0f}/100 · {status}</span>
            </div>
            <div style="background:rgba(255,255,255,0.06); border-radius:4px; height:8px; overflow:hidden;">
                <div style="background:{color}; width:{pct}%; height:100%; border-radius:4px; transition:width 0.5s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# SHAP
st.markdown('<div class="section-header">🔍 Explication de la Prédiction (SHAP)</div>', unsafe_allow_html=True)
try:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(pdf[features].astype(float))
    shap_vals = sv[1][0] if isinstance(sv, list) else sv[0]
    
    idx = np.argsort(np.abs(shap_vals))[::-1][:12]
    tf = [features[i] for i in idx]
    tv = [shap_vals[i] for i in idx]
    labs = [f"{f} = {dat.get(f,''):.2f}" if isinstance(dat.get(f,''), float) else f"{f} = {dat.get(f,'')}" for f in tf]
    cols = ['#e74c3c' if v > 0 else '#2ecc71' for v in tv]
    
    fig = go.Figure(go.Bar(x=tv, y=labs, orientation='h', marker_color=cols,
                           text=[f'{v:+.3f}' for v in tv], textposition='outside',
                           textfont=dict(color='rgba(255,255,255,0.7)', size=11)))
    fig.update_layout(
        height=420, margin=dict(l=200, r=60, t=20, b=30),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Impact SHAP', color='rgba(255,255,255,0.5)',
                   gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(autorange='reversed', color='rgba(255,255,255,0.7)',
                   tickfont=dict(family='Source Sans 3', size=12)),
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 Augmente le risque de décès | 🟢 Diminue le risque (effet protecteur)")
except Exception as e:
    shap_vals = None
    st.warning(f"SHAP: {e}")

# Export
st.markdown('<div class="section-header">📄 Export du Rapport</div>', unsafe_allow_html=True)
html = gen_report(dat, proba, rl, rc, reco, alerts, shap_vals, features)
st.download_button("📥 Télécharger le Rapport", html, f"cardiopredict_{datetime.now().strftime('%Y%m%d_%H%M')}.html", "text/html", use_container_width=True)
st.caption("Ouvrez le HTML → Ctrl+P → Enregistrer en PDF")

# Footer
st.markdown(f"""
<div class="footer">
    <p>⚠️ Outil d'aide à la décision et prediction le niveau de risque de mortalité — Ne remplace pas le jugement clinique du médecin</p>
    <p>CardioPredict · LightGBM · MIMIC-IV v3.1 · 13,569 patients · 45 variables · AUC 0.92</p>
    <p style="margin-top:0.5rem;">projet de maitrise — Prédiction de niveau de risque de mortalité cardiovasculaire en USI</p>
</div>
""", unsafe_allow_html=True)
