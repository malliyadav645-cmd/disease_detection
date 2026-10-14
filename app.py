"""VitaScan AI — Next-Generation Precision Multi-Disease Screening Dashboard.

Features:
- High-precision ML ensemble models (ExtraTrees, Random Forest, HistGradientBoosting, Soft Voting).
- Real-time biomarker range validation & anomaly flags.
- Explicit diagnostic calculation pipeline with step-by-step clinical verification.
- Comprehensive precision analytics (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, 5-Fold CV).
- Interactive risk stratification gauge and downloadable clinical screening dossiers.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="VitaScan AI | Clinical Screening Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional Clinical UI CSS ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    * { box-sizing: border-box; }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 14px;
    }

    /* ── Hide ALL Streamlit chrome ─────────────────────── */
    #MainMenu { visibility: hidden !important; }
    header[data-testid="stHeader"] { display: none !important; }
    footer { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    .stDeployButton { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    .viewerBadge_container__1QSob { display: none !important; }
    .css-1dp5vir { display: none !important; }

    /* ── Background ─────────────────────────────────────── */
    .stApp {
        background: #080e1a;
        color: #e2e8f0;
    }

    /* Dot grid overlay */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(56, 189, 248, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56, 189, 248, 0.025) 1px, transparent 1px);
        background-size: 52px 52px;
        pointer-events: none;
        z-index: 0;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #060b15 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.1) !important;
    }

    [data-testid="stSidebarContent"] { padding: 0 !important; }

    /* ── Typography ──────────────────────────────────────── */
    h1, h2, h3, h4, h5 {
        font-family: 'Syne', sans-serif !important;
        letter-spacing: -0.025em;
        color: #f1f5f9 !important;
    }

    /* ── Form inputs ─────────────────────────────────────── */
    .stNumberInput input,
    .stTextInput input {
        background: #0f1829 !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-size: 0.875rem !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: rgba(56, 189, 248, 0.45) !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.08) !important;
        outline: none !important;
    }

    .stNumberInput label,
    .stTextInput label,
    .stSelectbox label {
        color: #64748b !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }

    [data-baseweb="select"] > div {
        background: #0f1829 !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton > button {
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        transition: all 0.2s ease !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        background: #0f1829 !important;
        color: #94a3b8 !important;
        padding: 0.42rem 0.85rem !important;
        letter-spacing: 0.01em !important;
    }

    .stButton > button:hover {
        background: #16243d !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
        color: #e2e8f0 !important;
        transform: translateY(-1px) !important;
    }

    /* Primary CTA */
    .stButton > button[data-testid="baseButton-primary"],
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
        border: none !important;
        color: #fff !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 16px rgba(29, 78, 216, 0.28) !important;
    }

    .stButton > button[data-testid="baseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1e40af, #5b21b6) !important;
        box-shadow: 0 6px 24px rgba(29, 78, 216, 0.45) !important;
        transform: translateY(-2px) !important;
    }

    /* Download button */
    .stDownloadButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        background: #0f1829 !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        color: #38bdf8 !important;
    }

    .stDownloadButton > button:hover {
        background: rgba(56, 189, 248, 0.08) !important;
        border-color: rgba(56, 189, 248, 0.45) !important;
    }

    /* ── Tabs ────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #0a1120;
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 10px;
        padding: 4px;
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        padding: 0.5rem 1.1rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.875rem;
        color: #475569;
        border: none;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: #162033 !important;
        color: #e2e8f0 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3) !important;
    }

    /* ── Expanders ───────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: #0d1628 !important;
        border: 1px solid rgba(148, 163, 184, 0.12) !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    .streamlit-expanderContent {
        background: #090f1e !important;
        border: 1px solid rgba(148, 163, 184, 0.08) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 1rem !important;
    }

    /* ── Misc ────────────────────────────────────────────── */
    hr { border-color: rgba(148, 163, 184, 0.07) !important; }

    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        overflow: hidden !important;
    }

    [data-testid="stAlert"] {
        border-radius: 8px !important;
        font-size: 0.875rem !important;
    }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #080e1a; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }

    /* ══════════════════════════════════════════════════════
       COMPONENT CLASSES
    ══════════════════════════════════════════════════════ */

    /* Sidebar brand */
    .vs-brand {
        padding: 1.1rem 1rem;
        border-bottom: 1px solid rgba(56, 189, 248, 0.08);
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .vs-brand-mark {
        width: 34px; height: 34px;
        background: linear-gradient(135deg, #1d4ed8, #7c3aed);
        border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        box-shadow: 0 3px 12px rgba(29,78,216,0.4);
        flex-shrink: 0;
    }
    .vs-brand-name {
        font-family: 'Syne', sans-serif;
        font-size: 1rem; font-weight: 800;
        color: #f1f5f9; letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .vs-brand-sub {
        font-size: 0.62rem; color: #334155; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.07em;
    }

    /* Sidebar section label */
    .vs-section-label {
        padding: 0.7rem 0.85rem 0.3rem;
        font-size: 0.62rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em;
        color: #1e3a5f;
    }

    /* Model spec card */
    .vs-model-card {
        margin: 0.35rem 0.75rem 0.6rem;
        padding: 0.8rem 0.9rem;
        background: #0d1628;
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 10px;
    }
    .vs-mc-header {
        display: flex; justify-content: space-between;
        align-items: center; margin-bottom: 0.45rem;
    }
    .vs-mc-label { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #334155; }
    .vs-mc-live {
        font-size: 0.62rem; font-weight: 600;
        padding: 0.12rem 0.45rem; border-radius: 999px;
        background: rgba(16,185,129,0.1); color: #34d399;
        border: 1px solid rgba(16,185,129,0.18);
    }
    .vs-mc-name { font-size: 0.88rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.15rem; }
    .vs-mc-type { font-size: 0.7rem; color: #475569; margin-bottom: 0.65rem; line-height: 1.35; }
    .vs-mc-metrics {
        display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;
        padding-top: 0.6rem;
        border-top: 1px solid rgba(148,163,184,0.07);
    }
    .vs-mc-metric {
        text-align: center; padding: 0.35rem;
        background: rgba(0,0,0,0.18); border-radius: 6px;
    }
    .vs-mc-val { font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 700; }
    .vs-mc-mlabel { font-size: 0.58rem; color: #334155; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.08rem; }

    /* Sidebar workflow */
    .vs-how {
        margin: 0 0.75rem 0.6rem;
        padding: 0.8rem 0.9rem;
        background: rgba(29,78,216,0.04);
        border: 1px solid rgba(29,78,216,0.1);
        border-radius: 10px;
    }
    .vs-how-title { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em; color: #1e3a5f; margin-bottom: 0.55rem; }
    .vs-how-step { display: flex; align-items: flex-start; gap: 0.55rem; margin-bottom: 0.38rem; font-size: 0.75rem; color: #64748b; line-height: 1.4; }
    .vs-step-n {
        width: 17px; height: 17px; flex-shrink: 0;
        background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.18);
        border-radius: 50%; font-size: 0.58rem; font-weight: 700;
        color: #38bdf8; display: flex; align-items: center; justify-content: center;
        margin-top: 1px;
    }

    /* Sidebar disclaimer */
    .vs-disclaimer {
        margin: 0 0.75rem 0.75rem;
        padding: 0.6rem 0.8rem;
        background: rgba(120,53,15,0.06);
        border: 1px solid rgba(120,53,15,0.15);
        border-radius: 8px;
        font-size: 0.68rem; color: #57534e; line-height: 1.5;
    }
    .vs-disclaimer strong { color: #78350f; }

    /* Page header */
    .vs-header {
        padding: 1.25rem 0 1rem;
        border-bottom: 1px solid rgba(148,163,184,0.07);
        margin-bottom: 1.5rem;
    }
    .vs-breadcrumb {
        display: flex; align-items: center; gap: 0.35rem;
        font-size: 0.72rem; color: #1e3a5f; font-weight: 500;
        margin-bottom: 0.55rem;
    }
    .vs-bc-active { color: #64748b; }
    .vs-header-body {
        display: flex; justify-content: space-between;
        align-items: flex-end; flex-wrap: wrap; gap: 1rem;
    }
    .vs-header-title {
        margin: 0 !important; font-size: clamp(1.5rem, 2.6vw, 2.2rem) !important;
        font-weight: 800 !important; line-height: 1.1 !important;
        letter-spacing: -0.04em !important;
    }
    .vs-header-desc { font-size: 0.84rem; color: #475569; margin-top: 0.35rem; line-height: 1.6; max-width: 660px; }

    /* KPI strip */
    .vs-kpis { display: flex; gap: 0.45rem; }
    .vs-kpi {
        text-align: center; padding: 0.55rem 0.9rem;
        background: #0a1120;
        border: 1px solid rgba(148,163,184,0.1);
        border-radius: 9px; min-width: 82px;
    }
    .vs-kpi-val { font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 700; line-height: 1.1; }
    .vs-kpi-lbl { font-size: 0.6rem; color: #334155; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.15rem; font-weight: 600; }

    /* Toolbar heading above inputs */
    .vs-toolbar-label {
        font-size: 0.62rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em;
        color: #1e3a5f; margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(148,163,184,0.05);
    }

    /* Pending state */
    .vs-pending {
        text-align: center; padding: 3.5rem 2rem;
        background: #0a1120;
        border: 1px dashed rgba(148,163,184,0.12);
        border-radius: 12px;
    }
    .vs-pending-ico {
        width: 52px; height: 52px; margin: 0 auto 1.1rem;
        background: rgba(29,78,216,0.07);
        border: 1px solid rgba(29,78,216,0.15);
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
    }
    .vs-pending h3 { font-size: 1rem !important; font-weight: 700 !important; color: #f1f5f9 !important; margin: 0 0 0.35rem !important; }
    .vs-pending p { font-size: 0.84rem; color: #334155; max-width: 400px; margin: 0 auto; line-height: 1.6; }
    .vs-pending-tag {
        display: inline-flex; align-items: center; gap: 0.35rem;
        margin-top: 1.1rem; padding: 0.3rem 0.8rem;
        background: rgba(56,189,248,0.06); border: 1px solid rgba(56,189,248,0.14);
        border-radius: 999px; font-size: 0.72rem; font-weight: 600; color: #38bdf8;
    }

    /* Result card */
    .vs-result {
        display: flex; align-items: flex-start; gap: 1.1rem;
        padding: 1.2rem 1.4rem; border-radius: 12px;
        border: 1px solid; margin-bottom: 1.1rem;
    }
    .vs-result.critical { background: rgba(239,68,68,0.05); border-color: rgba(239,68,68,0.28); }
    .vs-result.elevated { background: rgba(249,115,22,0.05); border-color: rgba(249,115,22,0.28); }
    .vs-result.moderate { background: rgba(234,179,8,0.05); border-color: rgba(234,179,8,0.28); }
    .vs-result.low      { background: rgba(16,185,129,0.05); border-color: rgba(16,185,129,0.28); }
    .vs-result-ico {
        width: 46px; height: 46px; border-radius: 11px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
    }
    .vs-result-ico.critical { background: rgba(239,68,68,0.1); }
    .vs-result-ico.elevated { background: rgba(249,115,22,0.1); }
    .vs-result-ico.moderate { background: rgba(234,179,8,0.1); }
    .vs-result-ico.low      { background: rgba(16,185,129,0.1); }
    .vs-result-meta { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.25rem; }
    .vs-result-title { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; color: #f1f5f9; margin-bottom: 0.28rem; }
    .vs-decision {
        display: inline-flex; align-items: center; gap: 0.3rem;
        font-size: 0.74rem; font-weight: 700;
        padding: 0.18rem 0.6rem; border-radius: 6px; margin-bottom: 0.45rem;
    }
    .vs-decision.pos { background: rgba(239,68,68,0.09); color: #fca5a5; border: 1px solid rgba(239,68,68,0.2); }
    .vs-decision.neg { background: rgba(16,185,129,0.09); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.2); }
    .vs-result-action { font-size: 0.84rem; color: #64748b; line-height: 1.6; }

    /* 4-metric stat row */
    .vs-stats {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 0.65rem; margin-bottom: 0.85rem;
    }
    .vs-stat {
        padding: 0.9rem 1rem; background: #0a1120;
        border: 1px solid rgba(148,163,184,0.1); border-radius: 10px;
        transition: border-color 0.2s;
    }
    .vs-stat:hover { border-color: rgba(148,163,184,0.2); }
    .vs-stat-label { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #334155; margin-bottom: 0.28rem; }
    .vs-stat-val { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; line-height: 1; }
    .vs-stat-sub { font-size: 0.68rem; color: #334155; margin-top: 0.18rem; }

    /* Risk meter */
    .vs-meter {
        padding: 1rem 1.2rem; background: #0a1120;
        border: 1px solid rgba(148,163,184,0.1); border-radius: 11px;
        margin-bottom: 0.85rem;
    }
    .vs-meter-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.7rem; }
    .vs-meter-title { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em; color: #334155; }
    .vs-meter-val { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; }
    .vs-track {
        height: 9px; border-radius: 5px;
        background: linear-gradient(90deg, #10b981 0%, #84cc16 22%, #eab308 42%, #f97316 66%, #ef4444 100%);
        position: relative; margin: 0.4rem 0;
        box-shadow: 0 1px 5px rgba(0,0,0,0.4);
    }
    .vs-thumb {
        position: absolute; top: -5.5px;
        width: 20px; height: 20px; background: #fff;
        border: 3px solid #080e1a; border-radius: 50%;
        transform: translateX(-50%);
        box-shadow: 0 0 10px rgba(255,255,255,0.5);
    }
    .vs-track-labels {
        display: flex; justify-content: space-between;
        font-size: 0.6rem; font-weight: 600; color: #1e3a5f;
        letter-spacing: 0.04em; text-transform: uppercase; margin-top: 0.4rem;
    }

    /* Biomarker flags */
    .vs-flag {
        display: flex; align-items: flex-start; gap: 0.55rem;
        padding: 0.55rem 0.7rem; border-radius: 8px; margin-bottom: 0.35rem;
    }
    .vs-flag.abn { background: rgba(239,68,68,0.05); border: 1px solid rgba(239,68,68,0.16); }
    .vs-flag.nrm { background: rgba(16,185,129,0.04); border: 1px solid rgba(16,185,129,0.13); }
    .vs-flag-name { font-size: 0.8rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.08rem; }
    .vs-flag-detail { font-size: 0.7rem; color: #475569; }

    /* Protocol items */
    .vs-proto {
        display: flex; align-items: flex-start; gap: 0.7rem;
        padding: 0.6rem 0.85rem; background: #0a1120;
        border: 1px solid rgba(148,163,184,0.08);
        border-radius: 8px; margin-bottom: 0.35rem;
        font-size: 0.8rem; color: #64748b; line-height: 1.4;
    }
    .vs-proto strong { color: #94a3b8; display: block; font-size: 0.78rem; margin-bottom: 0.1rem; }

    /* Analytics card */
    .vs-acard {
        padding: 1rem 1.1rem; background: #0a1120;
        border: 1px solid rgba(148,163,184,0.1); border-radius: 11px;
    }
    .vs-acard-title {
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: #334155;
        margin-bottom: 0.75rem; padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(148,163,184,0.06);
    }

    /* Confusion matrix */
    .vs-cm { display: grid; grid-template-columns: 1fr 1fr; gap: 0.55rem; }
    .vs-cm-cell { padding: 0.9rem; border-radius: 9px; text-align: center; border: 1px solid; }
    .vs-cm-label { font-size: 0.6rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.35rem; }
    .vs-cm-val { font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 700; line-height: 1; }
    .vs-cm-desc { font-size: 0.65rem; margin-top: 0.25rem; opacity: 0.55; }
    .vs-cm-cell.tp { background: rgba(16,185,129,0.05); border-color: rgba(16,185,129,0.18); color: #34d399; }
    .vs-cm-cell.fp { background: rgba(239,68,68,0.05); border-color: rgba(239,68,68,0.18); color: #f87171; }
    .vs-cm-cell.tn { background: rgba(56,189,248,0.05); border-color: rgba(56,189,248,0.18); color: #38bdf8; }
    .vs-cm-cell.fn { background: rgba(245,158,11,0.05); border-color: rgba(245,158,11,0.18); color: #fbbf24; }

    </style>
    """,
    unsafe_allow_html=True,
)

DISEASE_META = {
    "Diabetes": {
        "icon": "🩸",
        "accent": "#ef4444",
        "category": "Endocrinology",
        "description": "Screens type 2 diabetic risk utilizing fasting glucose, insulin kinetics, body mass index, and pedigree risk factors.",
        "model_type": "Tuned Ensemble (ExtraTrees + Random Forest)",
    },
    "Heart Disease": {
        "icon": "❤️",
        "accent": "#f43f5e",
        "category": "Cardiology",
        "description": "Screens coronary artery disease and cardiovascular risk using hemodynamic markers, ST wave response, and fluoroscopy signals.",
        "model_type": "Calibrated Soft-Voting Ensemble (ExtraTrees + Random Forest)",
    },
    "Breast Cancer": {
        "icon": "🎗️",
        "accent": "#ec4899",
        "category": "Oncology",
        "description": "High-precision malignant vs benign classification using digitized fine needle aspirate (FNA) cell nuclei morphometry.",
        "model_type": "Tuned ExtraTrees High-Dimensional Classifier",
    },
    "Kidney Disease": {
        "icon": "🫧",
        "accent": "#0ea5e9",
        "category": "Nephrology",
        "description": "Screens chronic kidney disease (CKD) using hematology, urinary specific gravity/albumin, electrolytes, and renal biochemistry.",
        "model_type": "Tuned Clinical ExtraTrees Classifier",
    },
    "Liver Disease": {
        "icon": "🧬",
        "accent": "#f59e0b",
        "category": "Hepatology",
        "description": "Identifies hepatic impairment and liver inflammation through total/direct bilirubin, aminotransferases, and A/G ratio.",
        "model_type": "Balanced ExtraTrees Classifier",
    },
    "Stroke": {
        "icon": "🧠",
        "accent": "#8b5cf6",
        "category": "Neurology",
        "description": "Predicts acute cerebrovascular accident risk through systemic cardiovascular markers, lifestyle factors, and metabolic indicators.",
        "model_type": "Balanced Ensemble (Random Forest + HistGradientBoosting)",
    },
}


@st.cache_resource
def load_disease_artifacts(disease: str) -> dict[str, Any]:
    safe_name = disease.lower().replace(" ", "_")
    return {
        "model": joblib.load(MODELS_DIR / f"{safe_name}_model.pkl"),
        "scaler": joblib.load(MODELS_DIR / f"{safe_name}_scaler.pkl"),
        "features": joblib.load(MODELS_DIR / f"{safe_name}_features.pkl"),
        "encoders": joblib.load(MODELS_DIR / f"{safe_name}_encoders.pkl"),
        "metrics": json.load(open(MODELS_DIR / f"{safe_name}_metrics.json", encoding="utf-8")),
    }


@st.cache_data
def load_all_metrics() -> dict[str, Any]:
    with open(MODELS_DIR / "metrics_summary.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_clinical_meta() -> dict[str, Any]:
    with open(MODELS_DIR / "clinical_meta.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_sample_profiles() -> dict[str, Any]:
    with open(MODELS_DIR / "sample_profiles.json", encoding="utf-8") as f:
        return json.load(f)


def calculate_risk(
    disease: str,
    raw_inputs: dict[str, Any],
    artifacts: dict[str, Any],
    clinical_meta: dict[str, Any],
) -> dict[str, Any]:
    """Execute complete clinical calculation pipeline with calibrated probabilities and biomarker checks."""
    encoded_vals = {}
    encoders = artifacts["encoders"]

    for feat in artifacts["features"]:
        val = raw_inputs[feat]
        if feat in encoders:
            # Handle categorical encoding safely
            enc = encoders[feat]
            str_val = str(val).strip()
            if str_val in enc.classes_:
                encoded_vals[feat] = enc.transform([str_val])[0]
            else:
                encoded_vals[feat] = 0
        else:
            try:
                encoded_vals[feat] = float(val)
            except (ValueError, TypeError):
                encoded_vals[feat] = 0.0

    # Build input tensor
    feature_df = pd.DataFrame([[encoded_vals[f] for f in artifacts["features"]]], columns=artifacts["features"])
    scaled_tensor = artifacts["scaler"].transform(feature_df)

    # Model inference
    model = artifacts["model"]
    pred = int(model.predict(scaled_tensor)[0])
    probabilities = model.predict_proba(scaled_tensor)[0]

    # Positive class probability
    pos_prob = float(probabilities[1]) if len(probabilities) > 1 else float(pred)

    # Classify clinical risk tier
    if pos_prob >= 0.75:
        tier = "Critical Risk"
        tier_color = "#ef4444"
        tier_class = "critical"
        action_summary = "High probability of positive clinical pattern. Immediate specialized physician diagnostic evaluation is strongly recommended."
    elif pos_prob >= 0.50:
        tier = "Elevated Risk"
        tier_color = "#f97316"
        tier_class = "elevated"
        action_summary = "Elevated screening indicator detected. Schedule confirmatory clinical lab tests and follow-up consultation."
    elif pos_prob >= 0.25:
        tier = "Moderate Risk"
        tier_color = "#eab308"
        tier_class = "moderate"
        action_summary = "Borderline physiological signals observed. Recommend lifestyle optimization and monitoring during routine checkups."
    else:
        tier = "Low Risk"
        tier_color = "#10b981"
        tier_class = "low"
        action_summary = "Biomarkers align with negative clinical screening reference cohort. Maintain standard preventive wellness protocols."

    # Identify biomarker anomalies against physiological normal bounds
    anomalies: list[dict[str, Any]] = []
    normal_signals: list[dict[str, Any]] = []

    for feat in artifacts["features"]:
        if feat in clinical_meta and "normal" in clinical_meta[feat]:
            val = raw_inputs.get(feat)
            if isinstance(val, (int, float)):
                low, high = clinical_meta[feat]["normal"]
                unit = clinical_meta[feat].get("unit", "")
                if val < low:
                    anomalies.append({
                        "feature": feat,
                        "value": val,
                        "normal": f"{low} - {high} {unit}",
                        "status": "Below Normal",
                    })
                elif val > high:
                    anomalies.append({
                        "feature": feat,
                        "value": val,
                        "normal": f"{low} - {high} {unit}",
                        "status": "Above Normal",
                    })
                else:
                    normal_signals.append({
                        "feature": feat,
                        "value": val,
                        "normal": f"{low} - {high} {unit}",
                    })

    return {
        "disease": disease,
        "prediction": pred,
        "probability_positive": pos_prob,
        "probability_negative": float(probabilities[0]) if len(probabilities) > 1 else float(1.0 - pred),
        "risk_tier": tier,
        "tier_color": tier_color,
        "tier_class": tier_class,
        "action_summary": action_summary,
        "anomalies": anomalies,
        "normal_signals": normal_signals,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_inputs": raw_inputs,
    }


def render_sidebar(all_metrics: dict[str, Any]) -> str:
    with st.sidebar:
        # Brand block
        st.markdown(
            """
            <div class="vs-brand">
                <div class="vs-brand-mark">🩺</div>
                <div>
                    <div class="vs-brand-name">VitaScan <span style="color:#38bdf8;">AI</span></div>
                    <div class="vs-brand-sub">Clinical Intelligence Platform</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Disease selector
        st.markdown('<div class="vs-section-label">Screening Module</div>', unsafe_allow_html=True)
        disease_names = list(DISEASE_META.keys())
        selected_disease = st.selectbox(
            "Disease",
            disease_names,
            format_func=lambda d: f"{DISEASE_META[d]['icon']}  {d}",
            label_visibility="collapsed",
        )

        d_meta = DISEASE_META[selected_disease]
        d_metrics = all_metrics.get(selected_disease, {})

        # Model spec card
        st.markdown(
            f"""
            <div class="vs-model-card">
                <div class="vs-mc-header">
                    <span class="vs-mc-label">Active Model</span>
                    <span class="vs-mc-live">● Live</span>
                </div>
                <div class="vs-mc-name">{selected_disease}</div>
                <div class="vs-mc-type">{d_meta['model_type']}</div>
                <div class="vs-mc-metrics">
                    <div class="vs-mc-metric">
                        <div class="vs-mc-val" style="color:#10b981;">{d_metrics.get('accuracy',0)*100:.1f}%</div>
                        <div class="vs-mc-mlabel">Accuracy</div>
                    </div>
                    <div class="vs-mc-metric">
                        <div class="vs-mc-val" style="color:#38bdf8;">{d_metrics.get('roc_auc',0):.3f}</div>
                        <div class="vs-mc-mlabel">ROC-AUC</div>
                    </div>
                    <div class="vs-mc-metric">
                        <div class="vs-mc-val" style="color:#a78bfa;">{d_metrics.get('f1_score',0):.3f}</div>
                        <div class="vs-mc-mlabel">F1 Score</div>
                    </div>
                    <div class="vs-mc-metric">
                        <div class="vs-mc-val" style="color:#fb923c;">{d_metrics.get('cv_accuracy_mean',0)*100:.1f}%</div>
                        <div class="vs-mc-mlabel">5-Fold CV</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # How-to guide
        st.markdown(
            """
            <div class="vs-how">
                <div class="vs-how-title">How to use</div>
                <div class="vs-how-step"><div class="vs-step-n">1</div><span>Select a disease module above</span></div>
                <div class="vs-how-step"><div class="vs-step-n">2</div><span>Enter patient biomarkers or load a preset</span></div>
                <div class="vs-how-step"><div class="vs-step-n">3</div><span>Click Run Diagnostic Analysis</span></div>
                <div class="vs-how-step"><div class="vs-step-n">4</div><span>Review the risk result and export report</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Disclaimer
        st.markdown(
            """
            <div class="vs-disclaimer">
                <strong>⚠ Clinical Notice</strong><br>
                Educational ML prototype only. Does not constitute medical advice. Always consult a licensed practitioner for diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_disease


def main() -> None:
    all_metrics = load_all_metrics()
    clinical_meta = load_clinical_meta()
    sample_profiles = load_sample_profiles()

    selected_disease = render_sidebar(all_metrics)
    meta = DISEASE_META[selected_disease]
    artifacts = load_disease_artifacts(selected_disease)
    disease_metrics = artifacts["metrics"]

    # ── Page Header ───────────────────────────────────────────────────────────
    acc_val = disease_metrics.get("accuracy", 0) * 100
    auc_val = disease_metrics.get("roc_auc", 0)
    cv_val  = disease_metrics.get("cv_accuracy_mean", 0) * 100

    st.markdown(
        f"""
        <div class="vs-header">
            <div class="vs-breadcrumb">
                <span>VitaScan AI</span>
                <span style="color:#1e3a5f;">&#8250;</span>
                <span>{meta['category']}</span>
                <span style="color:#1e3a5f;">&#8250;</span>
                <span class="vs-bc-active">{selected_disease} Screening</span>
            </div>
            <div class="vs-header-body">
                <div>
                    <h1 class="vs-header-title">{meta['icon']} {selected_disease} <span style="color:{meta['accent']};">Screening</span></h1>
                    <div class="vs-header-desc">{meta['description']}</div>
                </div>
                <div class="vs-kpis">
                    <div class="vs-kpi">
                        <div class="vs-kpi-val" style="color:#10b981;">{acc_val:.1f}%</div>
                        <div class="vs-kpi-lbl">Accuracy</div>
                    </div>
                    <div class="vs-kpi">
                        <div class="vs-kpi-val" style="color:#38bdf8;">{auc_val:.3f}</div>
                        <div class="vs-kpi-lbl">ROC-AUC</div>
                    </div>
                    <div class="vs-kpi">
                        <div class="vs-kpi-val" style="color:#a78bfa;">{cv_val:.1f}%</div>
                        <div class="vs-kpi-lbl">5-Fold CV</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_screening, tab_analytics, tab_biomarkers = st.tabs([
        "  Patient Assessment  ",
        "  Model Analytics  ",
        "  Biomarker Reference  ",
    ])

    # =========================================================================
    # TAB 1 — PATIENT ASSESSMENT
    # =========================================================================
    with tab_screening:
        features = artifacts["features"]
        encoders = artifacts["encoders"]

        categorized_features: dict[str, list[str]] = {}
        for feat in features:
            f_meta = clinical_meta.get(feat, {})
            cat = f_meta.get("category", "Clinical Signals")
            categorized_features.setdefault(cat, []).append(feat)

        # Preset toolbar
        st.markdown('<div class="vs-toolbar-label">Patient Data Entry</div>', unsafe_allow_html=True)
        preset_cols = st.columns([1, 1, 1, 1, 2])

        with preset_cols[0]:
            if st.button("Load High-Risk Case", use_container_width=True):
                prof = sample_profiles.get(selected_disease, {}).get("high_risk", {})
                for k, v in prof.items():
                    st.session_state[f"inp_{selected_disease}_{k}"] = v
                st.session_state[f"{selected_disease}_calc"] = False
                st.session_state["_toast"] = "High-risk case loaded"
                st.rerun()

        with preset_cols[1]:
            if st.button("Load Low-Risk Case", use_container_width=True):
                prof = sample_profiles.get(selected_disease, {}).get("low_risk", {})
                for k, v in prof.items():
                    st.session_state[f"inp_{selected_disease}_{k}"] = v
                st.session_state[f"{selected_disease}_calc"] = False
                st.session_state["_toast"] = "Low-risk case loaded"
                st.rerun()

        with preset_cols[2]:
            if st.button("Reset to Defaults", use_container_width=True):
                for feat in features:
                    st.session_state.pop(f"inp_{selected_disease}_{feat}", None)
                st.session_state[f"{selected_disease}_calc"] = False
                st.rerun()

        with preset_cols[3]:
            if st.button("Clear Results", use_container_width=True):
                st.session_state[f"{selected_disease}_calc"] = False
                st.session_state.pop(f"{selected_disease}_result", None)
                st.rerun()

        with preset_cols[4]:
            if "_toast" in st.session_state:
                st.success(st.session_state.pop("_toast"))

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        # Input form
        raw_inputs: dict[str, Any] = {}
        for category_name, cat_feats in categorized_features.items():
            with st.expander(f"{category_name}  ·  {len(cat_feats)} parameters", expanded=True):
                n_cols = min(3, len(cat_feats))
                grid = st.columns(n_cols)
                for idx, feat in enumerate(cat_feats):
                    col = grid[idx % n_cols]
                    with col:
                        state_key = f"inp_{selected_disease}_{feat}"
                        f_meta = clinical_meta.get(feat, {})
                        unit_lbl = f" ({f_meta['unit']})" if "unit" in f_meta else ""
                        label = f"{feat.replace('_', ' ').title()}{unit_lbl}"
                        tip = f_meta.get("desc", feat)
                        if "normal" in f_meta:
                            tip += f"\n\nNormal: {f_meta['normal'][0]} - {f_meta['normal'][1]} {f_meta.get('unit','')}"
                        if feat in encoders:
                            enc = encoders[feat]
                            options = list(enc.classes_)
                            cur = st.session_state.get(state_key, options[0])
                            idx_sel = options.index(str(cur)) if str(cur) in options else 0
                            raw_inputs[feat] = st.selectbox(label, options, index=idx_sel, key=state_key, help=tip)
                        else:
                            nr = f_meta.get("normal", (0.0, 100.0))
                            default = float(st.session_state.get(state_key, nr[0] if nr else 1.0))
                            step = 0.01 if (abs(default) < 10 and default != int(default)) else 1.0
                            raw_inputs[feat] = st.number_input(label, value=default, step=step, key=state_key, help=tip)

        # Primary CTA
        st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
        run_clicked = st.button("Run Diagnostic Analysis", type="primary", use_container_width=True)

        if run_clicked:
            with st.status("Running clinical assessment...", expanded=True) as status:
                st.write("Validating biometric bounds...")
                time.sleep(0.15)
                st.write("Standardising input tensor with trained scaler...")
                time.sleep(0.15)
                st.write(f"Executing {meta['model_type']} inference...")
                time.sleep(0.2)
                st.write("Calibrating risk probability and detecting anomalies...")
                time.sleep(0.12)
                result = calculate_risk(selected_disease, raw_inputs, artifacts, clinical_meta)
                st.session_state[f"{selected_disease}_calc"] = True
                st.session_state[f"{selected_disease}_result"] = result
                status.update(label="Assessment complete", state="complete", expanded=False)

        st.divider()

        # Results
        is_done = st.session_state.get(f"{selected_disease}_calc", False)
        result  = st.session_state.get(f"{selected_disease}_result")

        if not is_done or not result or result.get("disease") != selected_disease:
            st.markdown(
                """
                <div class="vs-pending">
                    <div class="vs-pending-ico">&#9889;</div>
                    <h3>No assessment results yet</h3>
                    <p>Enter patient biomarker values or load a preset, then click <strong>Run Diagnostic Analysis</strong>.</p>
                    <div class="vs-pending-tag">ML inference pipeline ready</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            prob       = result["probability_positive"] * 100.0
            tier_color = result["tier_color"]
            tier_name  = result["risk_tier"]
            tier_cls   = result["tier_class"]
            is_pos     = result["prediction"] == 1
            dec_cls    = "pos" if is_pos else "neg"
            dec_label  = "Positive — Disease Indicator Detected" if is_pos else "Negative — No Disease Indicator"

            st.markdown(
                f"""
                <div class="vs-result {tier_cls}">
                    <div class="vs-result-ico {tier_cls}">{'&#128680;' if prob >= 50 else '&#128737;'}</div>
                    <div style="flex:1;min-width:0;">
                        <div class="vs-result-meta" style="color:{tier_color};">
                            {meta['category']} Screening &nbsp;&#183;&nbsp; {result['timestamp']}
                        </div>
                        <div class="vs-result-title">
                            {tier_name} &mdash; <span style="color:{tier_color};">{prob:.1f}%</span>
                        </div>
                        <div class="vs-decision {dec_cls}">{dec_label}</div>
                        <div class="vs-result-action">{result['action_summary']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="vs-stats">
                    <div class="vs-stat">
                        <div class="vs-stat-label">Positive Probability</div>
                        <div class="vs-stat-val" style="color:{tier_color};">{prob:.1f}%</div>
                        <div class="vs-stat-sub">Disease risk score</div>
                    </div>
                    <div class="vs-stat">
                        <div class="vs-stat-label">Negative Probability</div>
                        <div class="vs-stat-val" style="color:#10b981;">{result['probability_negative']*100:.1f}%</div>
                        <div class="vs-stat-sub">Disease absence score</div>
                    </div>
                    <div class="vs-stat">
                        <div class="vs-stat-label">Risk Classification</div>
                        <div class="vs-stat-val" style="color:{tier_color};font-size:1rem;">{tier_name}</div>
                        <div class="vs-stat-sub">Stratified risk tier</div>
                    </div>
                    <div class="vs-stat">
                        <div class="vs-stat-label">Anomalous Signals</div>
                        <div class="vs-stat-val" style="color:#f59e0b;">{len(result['anomalies'])}</div>
                        <div class="vs-stat-sub">Out-of-range biomarkers</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            ptr = min(max(prob, 1.5), 98.5)
            st.markdown(
                f"""
                <div class="vs-meter">
                    <div class="vs-meter-head">
                        <span class="vs-meter-title">Risk Probability Spectrum</span>
                        <span class="vs-meter-val" style="color:{tier_color};">{prob:.1f}%</span>
                    </div>
                    <div class="vs-track">
                        <div class="vs-thumb" style="left:{ptr}%;"></div>
                    </div>
                    <div class="vs-track-labels">
                        <span>Low &lt;25%</span>
                        <span>Moderate 25&ndash;50%</span>
                        <span>Elevated 50&ndash;75%</span>
                        <span>Critical &gt;75%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_anom, col_proto = st.columns(2)

            with col_anom:
                st.markdown('<div class="vs-acard-title">Biomarker Anomaly Report</div>', unsafe_allow_html=True)
                if result["anomalies"]:
                    for a in result["anomalies"]:
                        st.markdown(
                            f"""
                            <div class="vs-flag abn">
                                <div>
                                    <div class="vs-flag-name">{a['feature'].replace('_',' ').title()}</div>
                                    <div class="vs-flag-detail">Value: <strong style="color:#fca5a5;">{a['value']}</strong> &middot; {a['status']} &middot; Reference: {a['normal']}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("All signals within normal physiological reference ranges.", icon="✓")

                if result["normal_signals"]:
                    with st.expander(f"Normal signals ({len(result['normal_signals'])} in range)"):
                        for n in result["normal_signals"]:
                            st.markdown(
                                f"""
                                <div class="vs-flag nrm">
                                    <div>
                                        <div class="vs-flag-name">{n['feature'].replace('_',' ').title()}</div>
                                        <div class="vs-flag-detail">Value: {n['value']} &middot; Ref: {n['normal']}</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            with col_proto:
                st.markdown('<div class="vs-acard-title">Recommended Clinical Protocol</div>', unsafe_allow_html=True)
                if prob >= 50.0:
                    protocols = [
                        ("🔴", "Immediate Physician Referral", "Urgent specialist evaluation strongly recommended."),
                        ("🧪", "Confirmatory Lab Panel", "Diagnostic bloodwork and imaging to corroborate the screening pattern."),
                        ("📋", "Symptom Documentation", "Track acute changes, cardiovascular events, or discomfort."),
                        ("⚠", "Activity Precautions", "Avoid strenuous exertion or unprescribed medications until cleared."),
                    ]
                else:
                    protocols = [
                        ("✅", "Routine Wellness Surveillance", "Continue standard annual exams per primary care schedule."),
                        ("🥗", "Lifestyle Maintenance", "Sustain balanced nutrition and regular cardiovascular exercise."),
                        ("📅", "Periodic Rescreening", "Repeat screening in 6–12 months as recommended by your provider."),
                        ("💊", "Preventive Measures", "Adhere to prescribed preventive medications and supplements."),
                    ]
                for icon, title, desc in protocols:
                    st.markdown(
                        f"""
                        <div class="vs-proto">
                            <span style="font-size:1rem;flex-shrink:0;">{icon}</span>
                            <div><strong>{title}</strong>{desc}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
            report = {
                "Screening_Target": selected_disease,
                "Timestamp": result["timestamp"],
                "Risk_Probability_Pct": f"{prob:.4f}%",
                "Classification": "Positive" if is_pos else "Negative",
                "Risk_Tier": tier_name,
                "Model": meta["model_type"],
                **result["raw_inputs"],
            }
            report_df = pd.DataFrame([report])
            ec1, ec2, _ = st.columns([1.5, 1, 2.5])
            with ec1:
                st.download_button(
                    "Download Report (CSV)",
                    data=report_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_disease.lower().replace(' ','_')}_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with ec2:
                st.download_button(
                    "Export JSON",
                    data=json.dumps(report, indent=2).encode("utf-8"),
                    file_name=f"{selected_disease.lower().replace(' ','_')}_report.json",
                    mime="application/json",
                    use_container_width=True,
                )

    # =========================================================================
    # TAB 2 — MODEL ANALYTICS
    # =========================================================================
    with tab_analytics:
        st.markdown('<div class="vs-toolbar-label">Multi-Disease Model Benchmarks</div>', unsafe_allow_html=True)
        st.caption("Performance metrics across all six trained models on stratified holdout test partitions.")

        rows = []
        for d_name, d_met in all_metrics.items():
            rows.append({
                "Disease": f"{DISEASE_META[d_name]['icon']}  {d_name}",
                "Accuracy": f"{d_met['accuracy']*100:.2f}%",
                "Precision": f"{d_met['precision']*100:.2f}%",
                "Recall": f"{d_met['recall']*100:.2f}%",
                "Specificity": f"{d_met['specificity']*100:.2f}%",
                "F1 Score": f"{d_met['f1_score']:.4f}",
                "ROC-AUC": f"{d_met['roc_auc']:.4f}",
                "5-Fold CV": f"{d_met['cv_accuracy_mean']*100:.2f}% +/- {d_met['cv_accuracy_std']*100:.2f}%",
                "Brier Score": f"{d_met['brier_score']:.4f}",
                "Test Samples": d_met["test_samples"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="vs-toolbar-label">Deep Analysis &mdash; {meta["icon"]} {selected_disease}</div>',
            unsafe_allow_html=True,
        )

        ac1, ac2 = st.columns(2)

        with ac1:
            st.markdown('<div class="vs-acard"><div class="vs-acard-title">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = disease_metrics.get("confusion_matrix", {})
            tp, fp, tn, fn = cm.get("tp", 0), cm.get("fp", 0), cm.get("tn", 0), cm.get("fn", 0)
            total = tp + fp + tn + fn
            st.markdown(
                f"""
                <div class="vs-cm">
                    <div class="vs-cm-cell tp">
                        <div class="vs-cm-label">True Positive</div>
                        <div class="vs-cm-val">{tp}</div>
                        <div class="vs-cm-desc">Correctly identified disease</div>
                    </div>
                    <div class="vs-cm-cell fp">
                        <div class="vs-cm-label">False Positive</div>
                        <div class="vs-cm-val">{fp}</div>
                        <div class="vs-cm-desc">Healthy flagged as disease</div>
                    </div>
                    <div class="vs-cm-cell tn">
                        <div class="vs-cm-label">True Negative</div>
                        <div class="vs-cm-val">{tn}</div>
                        <div class="vs-cm-desc">Correctly identified healthy</div>
                    </div>
                    <div class="vs-cm-cell fn">
                        <div class="vs-cm-label">False Negative</div>
                        <div class="vs-cm-val">{fn}</div>
                        <div class="vs-cm-desc">Missed positive case</div>
                    </div>
                </div>
                <div style="margin-top:0.8rem;font-size:0.72rem;color:#334155;">
                    Test cohort: <strong style="color:#64748b;">{total} patients</strong>
                </div></div>
                """,
                unsafe_allow_html=True,
            )

        with ac2:
            st.markdown('<div class="vs-acard"><div class="vs-acard-title">Feature Importances</div>', unsafe_allow_html=True)
            st.caption("Gini-impurity based ensemble tree contribution weights:")
            importances = disease_metrics.get("feature_importances", {})
            if importances:
                top = list(importances.items())[:8]
                imp_df = pd.DataFrame(top, columns=["Biomarker Signal", "Importance"])
                imp_df["Importance"] = imp_df["Importance"].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(imp_df, use_container_width=True, hide_index=True)
            else:
                st.info("Scores computed from ensemble tree node impurity splits.")
            st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 3 — BIOMARKER REFERENCE
    # =========================================================================
    with tab_biomarkers:
        st.markdown(
            f'<div class="vs-toolbar-label">{meta["icon"]} {selected_disease} &mdash; Biomarker Reference</div>',
            unsafe_allow_html=True,
        )
        st.caption("Physiological definitions, standard units, and normal diagnostic reference intervals for all model inputs.")
        bio_rows = []
        for feat in artifacts["features"]:
            f_meta = clinical_meta.get(feat, {})
            norm = f_meta.get("normal")
            norm_str = f"{norm[0]} - {norm[1]} {f_meta.get('unit','')}" if norm else "N/A"
            bio_rows.append({
                "Biomarker": feat.replace("_", " ").title(),
                "Category": f_meta.get("category", "General"),
                "Unit": f_meta.get("unit", "—"),
                "Normal Reference Range": norm_str.strip(),
                "Description": f_meta.get("desc", f"Diagnostic signal: {feat}"),
            })
        st.dataframe(pd.DataFrame(bio_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
