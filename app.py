"""
ROSCAN — Remote Optical & Spectral Claim Adjudication Node
Autonomous Edge Diagnostic & Lifecycle Adjudication Platform
Engineered by Ronak Agarwal | github.com/RonakAgarwal18
"""
from __future__ import annotations

import hashlib
import random
import string
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

from modules.agent import build_claim_graph
from modules.mcp_tools import BRANDS, get_models_for_brand, get_device_spec
from modules.vision import (
    analyze_bezel_symmetry,
    calculate_moire_spoof_score,
    detect_oled_line_defect,
    segment_screen_defects,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ROSCAN | Optical Claim Adjudication",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System CSS  — Zero-Emoji, Luxury Glassmorphic Enterprise Dark Mode
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Canvas ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #08090E !important;
    background-image:
        radial-gradient(circle at 86% 7%,  rgba(255,130,40,0.10) 0%, transparent 40%),
        radial-gradient(circle at 10% 90%, rgba(90,50,240,0.07)  0%, transparent 46%);
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: #DDE3ED !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(9,10,17,0.94) !important;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] * { color: #8898AA !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] label { color: #DDE3ED !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #C0C8D8 !important; }
[data-testid="stSidebar"] hr {
    border: none; border-top: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Global heading overrides ── */
h1,h2,h3,h4 { color: #FFFFFF !important; }

/* ── Page header ── */
.rh {
    background: linear-gradient(135deg,
        rgba(14,16,26,0.96) 0%,
        rgba(18,12,34,0.90) 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 26px 34px 20px;
    margin-bottom: 24px;
    backdrop-filter: blur(20px);
    box-shadow: 0 14px 48px rgba(0,0,0,0.55);
}
.rh-title {
    font-size: 1.80rem; font-weight: 800;
    color: #FFFFFF; margin: 0; letter-spacing: -0.4px;
    font-family: 'Inter', sans-serif;
}
.rh-sub {
    font-size: 0.82rem; color: #4B5A70;
    margin: 6px 0 0; letter-spacing: 0.2px;
}

/* ── Glass card ── */
.gc {
    background: rgba(18,22,32,0.75);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.40);
    margin-bottom: 14px;
}

/* ── KPI tiles ── */
.kpi {
    border-radius: 16px; padding: 16px 18px;
    margin-bottom: 12px; position: relative; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
}
.kpi .lbl {
    font-size: .67rem; font-weight: 700; letter-spacing: 1.0px;
    text-transform: uppercase; color: #4B5A70; margin-bottom: 6px;
}
.kpi .val { font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px; }
.kpi .sub { font-size: .73rem; color: #4B5A70; }
.kpi .tag {
    display: inline-block; font-size: .65rem; font-weight: 700;
    letter-spacing: 0.8px; text-transform: uppercase;
    padding: 2px 8px; border-radius: 4px; margin-top: 6px;
}

/* KPI colour variants */
.kpi-auth-ok  { background: linear-gradient(135deg,rgba(16,185,129,.16),rgba(5,150,105,.09)); border-color: rgba(16,185,129,.26)!important; }
.kpi-auth-err { background: linear-gradient(135deg,rgba(239,68,68,.20),rgba(153,27,27,.12)); border-color: rgba(239,68,68,.30)!important; }
.kpi-ga       { background: linear-gradient(135deg,rgba(16,185,129,.16),rgba(4,120,87,.09)); border-color: rgba(16,185,129,.25)!important; }
.kpi-gb       { background: linear-gradient(135deg,rgba(245,158,11,.18),rgba(180,83,9,.10)); border-color: rgba(245,158,11,.28)!important; }
.kpi-gc       { background: linear-gradient(135deg,rgba(239,68,68,.18),rgba(153,27,27,.10)); border-color: rgba(239,68,68,.26)!important; }
.kpi-p-oem    { background: linear-gradient(135deg,rgba(139,92,246,.16),rgba(99,102,241,.09)); border-color: rgba(139,92,246,.26)!important; }
.kpi-p-bad    { background: linear-gradient(135deg,rgba(251,146,60,.18),rgba(194,65,12,.10)); border-color: rgba(251,146,60,.28)!important; }
.kpi-d-clr    { background: linear-gradient(135deg,rgba(6,182,212,.14),rgba(14,116,144,.08)); border-color: rgba(6,182,212,.22)!important; }
.kpi-d-flt    { background: linear-gradient(135deg,rgba(6,182,212,.24),rgba(16,185,129,.14)); border-color: rgba(6,182,212,.36)!important; }

/* Tag colour variants */
.tag-ok     { background: rgba(16,185,129,.18); color: #6EE7B7; }
.tag-warn   { background: rgba(245,158,11,.18); color: #FCD34D; }
.tag-err    { background: rgba(239,68,68,.20);  color: #FCA5A5; }
.tag-info   { background: rgba(139,92,246,.18); color: #C4B5FD; }
.tag-cyan   { background: rgba(6,182,212,.18);  color: #67E8F9; }
.tag-blue   { background: rgba(59,130,246,.18); color: #93C5FD; }
.tag-orange { background: rgba(251,146,60,.18); color: #FED7AA; }

/* ── Pills (verdict badges) ── */
.pill {
    display: inline-block; border-radius: 5px;
    font-weight: 700; letter-spacing: 0.6px;
    padding: 4px 12px; font-size: .76rem; text-transform: uppercase;
    font-family: 'Inter', monospace;
}
.p-slate  { background: rgba(100,116,139,.20); color: #94A3B8; border: 1px solid rgba(100,116,139,.30); }
.p-amber  { background: rgba(245,158,11,.18);  color: #FCD34D; border: 1px solid rgba(245,158,11,.30); }
.p-green  { background: rgba(16,185,129,.18);  color: #6EE7B7; border: 1px solid rgba(16,185,129,.30); }
.p-blue   { background: rgba(59,130,246,.18);  color: #93C5FD; border: 1px solid rgba(59,130,246,.30); }
.p-cyan   { background: rgba(6,182,212,.18);   color: #67E8F9; border: 1px solid rgba(6,182,212,.30); }
.p-red    { background: rgba(239,68,68,.20);   color: #FCA5A5; border: 1px solid rgba(239,68,68,.34); }
.p-orange { background: rgba(251,146,60,.18);  color: #FED7AA; border: 1px solid rgba(251,146,60,.30); }

/* ── Verdict banners ── */
.vb {
    border-radius: 16px; padding: 24px 30px; text-align: center;
    font-size: 1.65rem; font-weight: 900; letter-spacing: 1.2px;
    text-transform: uppercase; backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px); margin-bottom: 20px;
    font-family: 'Inter', sans-serif;
}
.vb-slate  { background: linear-gradient(135deg,rgba(71,85,105,.28),rgba(51,65,85,.18));   color: #94A3B8; border: 1px solid rgba(100,116,139,.28); }
.vb-amber  { background: linear-gradient(135deg,rgba(245,158,11,.20),rgba(180,83,9,.16));  color: #FCD34D; border: 1px solid rgba(245,158,11,.32); }
.vb-green  { background: linear-gradient(135deg,rgba(16,185,129,.20),rgba(5,150,105,.14)); color: #6EE7B7; border: 1px solid rgba(16,185,129,.32); }
.vb-blue   { background: linear-gradient(135deg,rgba(59,130,246,.20),rgba(37,99,235,.14)); color: #93C5FD; border: 1px solid rgba(59,130,246,.32); }
.vb-cyan   { background: linear-gradient(135deg,rgba(6,182,212,.20),rgba(14,116,144,.14)); color: #67E8F9; border: 1px solid rgba(6,182,212,.32); }
.vb-red    { background: linear-gradient(135deg,rgba(239,68,68,.24),rgba(153,27,27,.18));  color: #FCA5A5; border: 1px solid rgba(239,68,68,.38); }
.vb-orange { background: linear-gradient(135deg,rgba(251,146,60,.20),rgba(194,65,12,.14)); color: #FED7AA; border: 1px solid rgba(251,146,60,.32); }

/* ── Workflow tracker ── */
.tr-row  { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 12px 0; }
.tr-step { background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.26);
           border-radius: 4px; padding: 4px 12px; font-size: .73rem;
           color: #6EE7B7; font-weight: 700; letter-spacing: .4px; text-transform: uppercase; }
.tr-arr  { color: #2D3748; font-size: .9rem; }

/* ── Finance table ── */
.ftab { width: 100%; border-collapse: collapse; margin-top: 8px; }
.ftab th {
    text-align: left; padding: 9px 13px; font-size: .66rem;
    font-weight: 700; text-transform: uppercase; letter-spacing: .9px;
    color: #3D4F63; border-bottom: 1px solid rgba(255,255,255,.05);
}
.ftab td { padding: 11px 13px; font-size: .84rem; color: #C0C8D8;
           border-bottom: 1px solid rgba(255,255,255,.04); }
.ftab .hl { color: #FFFFFF; font-weight: 700; }

/* ── Section label ── */
.sl {
    font-size: .65rem; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #3D4F63; margin-bottom: 10px;
}

/* ── Inline economics row ── */
.eco-row { display: flex; justify-content: space-between; padding: 5px 0; }
.eco-lbl { color: #4B5A70; font-size: .81rem; }
.eco-val { font-weight: 600; font-size: .84rem; color: #DDE3ED; }
.eco-total-lbl { color: #FFFFFF; font-weight: 700; font-size: .88rem; }
.eco-total-val { font-size: 1.08rem; font-weight: 800; }
.eco-sep { border: none; border-top: 1px solid rgba(255,255,255,.06); margin: 8px 0; }

/* ── Sidebar device card ── */
.dev-card {
    background: rgba(18,22,32,.60); border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
}
.dev-card .dk { color: #3D4F63; font-size: .72rem; margin-bottom: 2px; }
.dev-card .dv { color: #DDE3ED; font-weight: 600; font-size: .82rem; }
.dev-card .dh { color: #6EE7B7; font-weight: 700; font-size: .80rem; }
.dev-card .dw { color: #FCD34D; font-weight: 700; font-size: .80rem; }

/* ── Footer ── */
.footer {
    font-size: .68rem; color: #1E2A38; margin-top: 8px; letter-spacing: .2px;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-size: .79rem!important; font-weight: 600!important; color: #3D4F63!important;
    border-radius: 10px!important; padding: 7px 16px!important;
    background: transparent!important; border: none!important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #FFFFFF!important;
    background: rgba(255,255,255,.07)!important;
    border: 1px solid rgba(255,255,255,.10)!important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg,rgba(16,185,129,.18),rgba(6,182,212,.12))!important;
    border: 1px solid rgba(16,185,129,.32)!important;
    color: #6EE7B7!important; border-radius: 6px!important;
    font-weight: 700!important; font-size: .80rem!important;
    padding: 8px 22px!important; letter-spacing: .3px!important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: rgba(18,22,32,.70)!important;
    border: 1px solid rgba(255,255,255,.07)!important;
    border-radius: 12px!important; color: #C0C8D8!important;
    backdrop-filter: blur(10px);
}

/* Hide top chrome */
header[data-testid="stHeader"] { background: transparent!important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TEST_DIR = Path(__file__).parent.resolve() / "test_images"
EXTS     = {".jpg", ".jpeg", ".png", ".webp"}

# Neutral benchmark labels — do not hint at defect type
FIELD_STEMS = ["pristine", "shattered", "scratch", "greenline", "moire", "counterfeit"]
NEUTRAL_LABELS = {
    "pristine":    "Sample Hardware Capture 1",
    "shattered":   "Sample Hardware Capture 2",
    "scratch":     "Sample Hardware Capture 3",
    "greenline":   "Sample Hardware Capture 4",
    "moire":       "Sample Hardware Capture 5",
    "counterfeit": "Sample Hardware Capture 6",
}
SYNTH_LABELS = {s: NEUTRAL_LABELS[s] for s in FIELD_STEMS}

_VERDICT_VB = {
    "NO_FAULT_FOUND":               "vb-slate",
    "CLAIM_DISMISSED_WEAR_AND_TEAR":"vb-amber",
    "APPROVED_REPAIR":              "vb-green",
    "APPROVED_REPLACEMENT":         "vb-blue",
    "APPROVED_OEM_RECALL_WARRANTY": "vb-cyan",
    "REJECT_FRAUD":                 "vb-red",
    "REJECT_UNAUTHORIZED_MOD":      "vb-orange",
}
_VERDICT_PILL = {
    "NO_FAULT_FOUND":               "p-slate",
    "CLAIM_DISMISSED_WEAR_AND_TEAR":"p-amber",
    "APPROVED_REPAIR":              "p-green",
    "APPROVED_REPLACEMENT":         "p-blue",
    "APPROVED_OEM_RECALL_WARRANTY": "p-cyan",
    "REJECT_FRAUD":                 "p-red",
    "REJECT_UNAUTHORIZED_MOD":      "p-orange",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_file_image(stem: str) -> Optional[np.ndarray]:
    for ext in EXTS:
        p = TEST_DIR / f"{stem}{ext}"
        if p.exists():
            raw = np.frombuffer(p.read_bytes(), np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return None


def _make_synthetic(name: str) -> np.ndarray:
    img = np.full((380, 260, 3), 18, dtype=np.uint8)
    cv2.rectangle(img, (14, 8),  (246, 372), (42, 44, 54), -1)
    cv2.rectangle(img, (14, 8),  (246, 372), (62, 65, 78),  2)
    screen_b = 310 if "counterfeit" not in name else 264
    cv2.rectangle(img, (26, 26), (234, screen_b), (8, 8, 14), -1)
    if "shattered" in name:
        cx, cy = 130, 180
        for i in range(36):
            import math
            a = i * (2 * math.pi / 36)
            cv2.line(img, (cx, cy),
                     (int(cx+100*math.cos(a)), int(cy+95*math.sin(a))),
                     (185, 187, 195), 2)
        for r in (14, 30, 52, 78, 100):
            cv2.ellipse(img, (cx, cy), (r, int(r*.9)), 10, 0, 360, (138, 140, 148), 1)
    elif "scratch" in name:
        for i in range(8):
            x0, y0 = 54 + i*14, 140 + i*9
            cv2.line(img, (x0, y0), (x0+40, y0+7), (155, 157, 163), 1)
    elif "greenline" in name:
        cv2.line(img, (152, 28), (152, 308), (0, 246, 60), 3)
    elif "moire" in name:
        for x in range(26, 235, 5):
            cv2.line(img, (x, 26), (x, screen_b), (162, 164, 168), 1)
        for y in range(26, screen_b, 5):
            cv2.line(img, (26, y), (234, y), (140, 142, 146), 1)
    elif "counterfeit" in name:
        cv2.rectangle(img, (26, 264), (234, 372), (28, 30, 38), -1)
    return img


def _run_optical(img: np.ndarray) -> tuple:
    spoof, hmap, sr       = calculate_moire_spoof_score(img)
    is_cf, asym, bexp     = analyze_bezel_symmetry(img)
    grade, dpct, overlay  = segment_screen_defects(img)
    has_gl, gl_ls, ann    = detect_oled_line_defect(img)
    return spoof, hmap, sr, is_cf, asym, bexp, grade, dpct, overlay, has_gl, gl_ls, ann


def _kpi(label: str, val: str, sub: str, tag: str, tag_cls: str, css: str) -> str:
    return f"""
<div class="kpi {css}">
  <div class="lbl">{label}</div>
  <div class="val">{val}</div>
  <div class="sub">{sub}</div>
  <span class="tag {tag_cls}">{tag}</span>
</div>"""


def _claim_id() -> str:
    chars = string.ascii_uppercase + string.digits
    return "ROS-IND-" + "".join(random.choices(chars, k=6))


def _gen_report(state: dict, ref: str, ts: str, brand: str, model: str) -> str:
    spoof   = float(state.get("spoof_score") or 0)
    asym    = float(state.get("bezel_asymmetry_pct") or 0)
    grade   = state.get("defect_grade", "N/A")
    dpct    = float(state.get("defect_percentage") or 0)
    has_gl  = bool(state.get("has_green_line", False))
    verdict = state.get("final_verdict", "N/A")
    rat     = state.get("decision_reasoning", "N/A")
    ded     = float(state.get("deductible") or 0)
    liab    = float(state.get("insurer_liability") or 0)
    rep     = float(state.get("repair_cost") or 0)
    pol     = state.get("policy_coverage") or {}
    lim     = float(pol.get("aggregate_limit_inr", 150000))
    spec    = get_device_spec(brand, model)
    note    = spec.get("pricing_note", "OEM certified service rate card")
    chk     = hashlib.sha256(f"{ref}{ts}{verdict}".encode()).hexdigest()[:24].upper()

    return f"""ENTERPRISE CONSUMER ELECTRONICS PROTECTION - CLAIM ADJUDICATION REPORT
{'='*78}

System Reference   : ROSCAN Core Engine v3.4 | Author: Ronak Agarwal
Claim ID           : {ref}
Timestamp (ISO)    : {ts}
Device             : {brand} {model}
Policy ID          : {state.get('policy_id', 'UNKNOWN')}
Pricing Authority  : {note}

{'─'*78}
SECTION A  --  HARDWARE & OPTICAL ANALYSIS
{'─'*78}
  2D-FFT PAPR Score        : {spoof:.4f}  {'[FLAG: FRAUD_SIGNAL]' if spoof > .65 else '[STATUS: HARDWARE_VERIFIED]'}
  Chin Variance            : {asym:.1f}%  {'[FLAG: NON_OEM_PANEL]' if asym > 20 else '[TIER: OEM_AUTHORIZED]'}
  Surface Defect Grade     : [GRADE: {grade.split()[1] if len(grade.split()) > 1 else grade}]  ({dpct:.1f}% defect coverage)
  AMOLED Column Line Fault : {'[FAULT: LINE_DETECTED]' if has_gl else '[STATUS: PANEL_CLEAR]'}

{'─'*78}
SECTION B  --  POLICY DISPOSITION & SUBROGATION
{'─'*78}
  Verdict                  : [{verdict.replace('_', ' ')}]
  Rationale:
  {rat}

  Policy Reference         : Section 3.1 / 4.2 -- ROSCAN India CEP
  Subrogation Applicable   : {'YES -- OEM Warranty Channel' if 'OEM_RECALL' in verdict else 'N/A'}

{'─'*78}
SECTION C  --  ITEMIZED FINANCIAL SETTLEMENT (INR)
{'─'*78}
  Aggregate Policy Limit   : Rs. {lim:>12,.2f}
  OEM Repair Authorization : Rs. {rep:>12,.2f}
  Customer Deductible      : Rs. {ded:>12,.2f}
  Net Underwriter Liability: Rs. {liab:>12,.2f}

{'─'*78}
SECTION D  --  CRYPTOGRAPHIC VALIDATION
{'─'*78}
  SHA-256 Integrity Hash   : {chk}
  Issued By                : ROSCAN Optical Adjudication Engine v3.4
  Engineer                 : Ronak Agarwal  |  github.com/RonakAgarwal18
  Digital Seal             : [AUTHORIZED -- ROSCAN ENTERPRISE NODE]

{'='*78}
  This document is system-generated and constitutes a formal adjudication
  determination under the ROSCAN India Consumer Electronics Protection
  framework. Retain for your records.
{'='*78}""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("img_bgr", None), ("optical", None), ("workflow_state", None),
             ("sel_brand", "Samsung"), ("sel_model", "Galaxy S22")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# Page Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rh">
  <div class="rh-title">ROSCAN &mdash; Remote Optical &amp; Spectral Claim Adjudication Node</div>
  <div class="rh-sub">Autonomous Edge Diagnostic &amp; Lifecycle Adjudication Platform</div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Device Under Inspection")

    brand = st.selectbox(
        "Brand",
        BRANDS,
        index=BRANDS.index(st.session_state.sel_brand)
              if st.session_state.sel_brand in BRANDS else 1,
    )
    st.session_state.sel_brand = brand

    models      = get_models_for_brand(brand)
    default_m   = st.session_state.sel_model if st.session_state.sel_model in models else models[0]
    model       = st.selectbox("Model", models, index=models.index(default_m))
    st.session_state.sel_model = model

    spec      = get_device_spec(brand, model)
    total_r   = spec.get("oem_display_inr", 0) + spec.get("labor_inr", 0)
    resale_v  = spec.get("cashify_resale_inr", 0)
    scrap_v   = spec.get("scrap_inr", 0)
    deduct_v  = spec.get("base_deductible_inr", 0)
    recall_ok = spec.get("oem_recall_eligible", False)
    note_v    = spec.get("pricing_note", "OEM certified service rate card")

    st.markdown(f"""
<div class="dev-card">
  <div class="dk">OEM Repair Total</div><div class="dh">Rs. {total_r:,.0f}</div>
  <div class="dk" style="margin-top:6px;">Cashify Resale</div><div class="dv">Rs. {resale_v:,.0f}</div>
  <div class="dk" style="margin-top:6px;">Scrap Recovery</div><div class="dv">Rs. {scrap_v:,.0f}</div>
  <div class="dk" style="margin-top:6px;">Base Deductible</div><div class="dw">Rs. {deduct_v:,.0f}</div>
  {'<div class="dk" style="margin-top:8px;">OEM Recall Status</div><div style="color:#67E8F9;font-size:.74rem;font-weight:700;">RECALL_ELIGIBLE</div>' if recall_ok else ''}
  <div class="dk" style="margin-top:8px;font-size:.64rem;line-height:1.4;">{note_v}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Inspection Source")
    source = st.radio("", [
        "Real Device Field Samples",
        "Procedural Synthetic Benchmarks",
        "Live Camera / Custom Upload",
    ], label_visibility="collapsed")

    new_img: Optional[np.ndarray] = None

    if source == "Real Device Field Samples":
        avail = {s: NEUTRAL_LABELS[s] for s in FIELD_STEMS if _load_file_image(s) is not None}
        if not avail:
            st.warning("No images found in test_images/.")
        else:
            ch = st.selectbox("Select Sample", list(avail.keys()),
                              format_func=lambda k: avail[k])
            if st.button("Load and Inspect", use_container_width=True):
                new_img = _load_file_image(ch)

    elif source == "Procedural Synthetic Benchmarks":
        ch = st.selectbox("Select Benchmark", list(SYNTH_LABELS.keys()),
                          format_func=lambda k: SYNTH_LABELS[k])
        if st.button("Generate and Inspect", use_container_width=True):
            new_img = _make_synthetic(ch)

    else:
        cam = st.camera_input("Capture Device")
        up  = st.file_uploader("Or upload image", type=["jpg", "jpeg", "png", "webp"])
        src = cam or up
        if src:
            fb = np.frombuffer(src.read(), np.uint8)
            d  = cv2.imdecode(fb, cv2.IMREAD_COLOR)
            if d is not None:
                new_img = d

    st.markdown("---")
    st.markdown(
        '<div class="footer">'
        'Engineered by Ronak Agarwal<br>'
        'github.com/RonakAgarwal18'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Auto-pipeline trigger
# ─────────────────────────────────────────────────────────────────────────────
if new_img is not None:
    with st.spinner("Running optical diagnostics and adjudication pipeline..."):
        st.session_state.img_bgr = new_img
        st.session_state.optical = _run_optical(new_img)
        st.session_state.workflow_state = build_claim_graph().invoke({
            "device_brand": st.session_state.sel_brand,
            "device_model": st.session_state.sel_model,
            "policy_id":    "POL-IND-2026-9912",
            "raw_image":    new_img,
        })

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
_EMPTY = ("<div class='gc' style='text-align:center;color:#1E2A38;padding:52px;'>"
          "Select an inspection source in the sidebar to begin."
          "</div>")

tab1, tab2, tab3 = st.tabs([
    "Stage 1 — Optical Hardware Diagnostic",
    "Stage 2 — Agent Adjudication Pipeline",
    "Stage 3 — Executive Settlement",
])

# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    if st.session_state.img_bgr is None:
        st.markdown(_EMPTY, unsafe_allow_html=True)
    else:
        img = st.session_state.img_bgr
        (spoof, hmap, sr,
         is_cf, asym, bexp,
         grade, dpct, overlay,
         has_gl, gl_ls, ann) = st.session_state.optical

        st.markdown("<div class='sl'>Sensor Triptych</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption="Raw Sensor Capture", use_container_width=True)
        c2.image(hmap, caption="2D-FFT Spectrum (INFERNO)", use_container_width=True)
        c3.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                 caption="Hardware Anomaly Overlay", use_container_width=True)

        st.markdown("<div class='sl' style='margin-top:18px;'>Optical Diagnostics — KPI Matrix</div>",
                    unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)

        a_css = "kpi-auth-err" if spoof > 0.65 else "kpi-auth-ok"
        a_val = "Replay Spoof" if spoof > 0.65 else "Hardware Verified"
        a_tag = "[FLAG: SPOOF_DETECTED]" if spoof > 0.65 else "[STATUS: HARDWARE_VERIFIED]"
        a_tc  = "tag-err" if spoof > 0.65 else "tag-ok"
        k1.markdown(_kpi("Authenticity", a_val, f"PAPR Score: {spoof:.4f}", a_tag, a_tc, a_css),
                    unsafe_allow_html=True)

        g_css = {"Grade A":"kpi-ga","Grade B":"kpi-gb","Grade C":"kpi-gc"}.get(grade,"kpi-gb")
        g_tag = f"[GRADE: {grade.split()[-1]}]"
        g_tc  = {"Grade A":"tag-ok","Grade B":"tag-warn","Grade C":"tag-err"}.get(grade,"tag-warn")
        k2.markdown(_kpi("Display Grade", grade, f"Defect Coverage: {dpct:.1f}%",
                         g_tag, g_tc, g_css), unsafe_allow_html=True)

        p_css = "kpi-p-bad" if is_cf else "kpi-p-oem"
        p_val = "Non-OEM Panel" if is_cf else "OEM Certified"
        p_tag = "[FLAG: NON_OEM_PANEL]" if is_cf else "[TIER: OEM_AUTHORIZED]"
        p_tc  = "tag-err" if is_cf else "tag-info"
        k3.markdown(_kpi("Panel Integrity", p_val, f"Chin Variance: {asym:.1f}%",
                         p_tag, p_tc, p_css), unsafe_allow_html=True)

        d_css = "kpi-d-flt" if has_gl else "kpi-d-clr"
        d_val = "Line Detected" if has_gl else "Panel Clear"
        d_tag = "[FAULT: AMOLED_LINE]" if has_gl else "[STATUS: PANEL_CLEAR]"
        d_tc  = "tag-cyan" if has_gl else "tag-cyan"
        k4.markdown(_kpi("Panel Defect", d_val, "OLED Column Fault Check",
                         d_tag, d_tc, d_css), unsafe_allow_html=True)

        # Diagnostic summary
        if spoof > 0.65:
            insight = (
                "2D-FFT PAPR analysis detected isolated harmonic delta peaks exceeding 4.5 sigma "
                "in the ultra-high frequency band (>0.85 Nyquist) with confirmed geometric symmetry. "
                "Definitive signature of a secondary monitor pixel grid. Claim flagged for SIU escalation."
            )
        elif is_cf and asym > 20.0:
            insight = (
                f"Chin-to-bezel asymmetry of {asym:.1f}% recorded on a flat, perspective-invariant capture. "
                "Exceeds 20% OEM factory tolerance. Display panel identified as non-OEM aftermarket assembly. "
                "Coverage voided under Section 4.2."
            )
        elif has_gl:
            gl_note = (
                "OEM subrogation channel potentially eligible."
                if grade != "Grade C"
                else "Concurrent physical fractures detected. Standard accidental damage terms apply."
            )
            insight = (
                f"Vertical AMOLED column line fault confirmed ({grade} glass surface). "
                f"HSV multi-channel masking detected continuous vertical anomaly. {gl_note}"
            )
        elif grade == "Grade C":
            insight = (
                f"Spiderweb fracture topology detected across {dpct:.1f}% of the display surface. "
                "Bilateral-filter segmentation confirmed non-geometric branching crack network. "
                "Authentic physical impact event verified. OEM repair authorized."
            )
        elif grade == "Grade B":
            insight = (
                f"Superficial abrasions across {dpct:.1f}% of the surface glass. "
                "No structural fractures or AMOLED anomalies detected. "
                "Minor cosmetic wear is non-indemnifiable under Section 3.1."
            )
        else:
            insight = (
                "Device surface in pristine condition. Zero structural fractures, AMOLED anomalies, "
                "or spoofing signals detected. Claim closed with no underwriter liability."
            )

        st.info(f"Optical Diagnostic Summary: {insight}")

# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    state = st.session_state.workflow_state
    if state is None:
        st.markdown(_EMPTY, unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="tr-row">
  <span class="tr-step">Optical Fingerprint</span>
  <span class="tr-arr">-&gt;</span>
  <span class="tr-step">Cashify Salvage API</span>
  <span class="tr-arr">-&gt;</span>
  <span class="tr-step">OEM Parts Matrix</span>
  <span class="tr-arr">-&gt;</span>
  <span class="tr-step">Policy Applied</span>
</div>""", unsafe_allow_html=True)

        oem = state.get("oem_quote") or {}
        sal = state.get("salvage_quote") or {}
        disp   = oem.get("certified_display_assembly_inr", 0)
        labor  = oem.get("authorized_labor_inr", 0)
        total  = oem.get("total_repair_cost_inr", 0)
        days   = oem.get("dispatch_turnaround_days", 3)
        part   = oem.get("service_partner", "OEM Service Center")
        pnote  = oem.get("pricing_note", "OEM certified service rate card")
        resale = sal.get("cashify_fair_market_resale_inr", 0)
        scrap  = sal.get("salvage_scrap_recovery_inr", 0)
        netloss= sal.get("net_replacement_loss_inr", 0)
        dsrc   = sal.get("data_source", "Cashify Index")

        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown(f"""
<div class="gc">
  <div class="sl">OEM Authorized Repair Estimate</div>
  <div class="eco-row"><span class="eco-lbl">Display Assembly</span><span class="eco-val">Rs. {disp:,.2f}</span></div>
  <div class="eco-row"><span class="eco-lbl">Authorized Labor</span><span class="eco-val">Rs. {labor:,.2f}</span></div>
  <hr class="eco-sep">
  <div class="eco-row">
    <span class="eco-total-lbl">Total Authorization</span>
    <span class="eco-total-val" style="color:#6EE7B7;">Rs. {total:,.2f}</span>
  </div>
  <div style="font-size:.68rem;color:#2D3748;margin-top:10px;">
    Turnaround: {days} business days &nbsp;&middot;&nbsp; {part}<br>
    <span style="font-style:italic;">{pnote}</span>
  </div>
</div>""", unsafe_allow_html=True)

        with ec2:
            st.markdown(f"""
<div class="gc">
  <div class="sl">Salvage &amp; Circular Economics</div>
  <div class="eco-row"><span class="eco-lbl">Fair Market Resale</span><span class="eco-val">Rs. {resale:,.2f}</span></div>
  <div class="eco-row"><span class="eco-lbl">Salvage Scrap Recovery</span><span class="eco-val">Rs. {scrap:,.2f}</span></div>
  <hr class="eco-sep">
  <div class="eco-row">
    <span class="eco-total-lbl">Net Replacement Loss</span>
    <span class="eco-total-val" style="color:#FCD34D;">Rs. {netloss:,.2f}</span>
  </div>
  <div style="font-size:.68rem;color:#2D3748;margin-top:10px;">Source: {dsrc}</div>
</div>""", unsafe_allow_html=True)

        # OEM subrogation panel (shown when recall-eligible)
        if oem.get("oem_recall_eligible"):
            rpol = oem.get("recall_policy", "OEM Warranty")
            rded = oem.get("recall_deductible_inr", 0)
            st.markdown(f"""
<div class="gc" style="border-color:rgba(6,182,212,.28);background:rgba(6,182,212,.06);">
  <div class="sl" style="color:#0891B2;">OEM Warranty Subrogation Protocol</div>
  <div style="color:#CBD5E1;font-size:.84rem;line-height:1.6;">{rpol}<br>
  Customer nominal charge: <b style="color:#67E8F9;">Rs. {rded:,.0f}</b>
  &nbsp;&middot;&nbsp; Net underwriter exposure: <b style="color:#6EE7B7;">Rs. 0</b></div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    state = st.session_state.workflow_state
    if state is None:
        st.markdown(_EMPTY, unsafe_allow_html=True)
    else:
        verdict  = state.get("final_verdict") or "PENDING"
        vb_cls   = _VERDICT_VB.get(verdict, "vb-slate")
        pill_cls = _VERDICT_PILL.get(verdict, "p-slate")
        label    = verdict.replace("_", " ")
        brand_s  = state.get("device_brand", st.session_state.sel_brand)
        model_s  = state.get("device_model", st.session_state.sel_model)

        st.markdown(f"<div class='vb {vb_cls}'>{label}</div>", unsafe_allow_html=True)

        # Rationale card
        st.markdown("<div class='gc'>", unsafe_allow_html=True)
        st.markdown("<div class='sl'>Executive Decision Rationale</div>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:#C0C8D8;line-height:1.75;font-size:.88rem;'>"
            f"{state.get('decision_reasoning','N/A')}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Settlement table
        ded  = float(state.get("deductible")         or 0)
        liab = float(state.get("insurer_liability")   or 0)
        rep  = float(state.get("repair_cost")         or 0)
        pol  = state.get("policy_coverage")           or {}
        lim  = float(pol.get("aggregate_limit_inr",    150000))
        spec = get_device_spec(brand_s, model_s)
        note = spec.get("pricing_note", "OEM certified service rate card")

        st.markdown(f"""
<div class="gc">
  <div class="sl">Financial Settlement &mdash; INR</div>
  <table class="ftab">
    <thead><tr>
      <th>Claim Type</th><th>Device</th><th>Policy Limit</th>
      <th>Deductible (Customer)</th><th>Net Underwriter Liability</th>
    </tr></thead>
    <tbody><tr>
      <td><span class="pill {pill_cls}">{label}</span></td>
      <td style="color:#4B5A70;font-size:.79rem;">{brand_s}<br>{model_s}</td>
      <td>Rs. {lim:,.2f}</td>
      <td class="hl">Rs. {ded:,.2f}</td>
      <td class="hl">Rs. {liab:,.2f}</td>
    </tr></tbody>
  </table>
  <div style="font-size:.67rem;color:#1E2A38;margin-top:8px;font-style:italic;">{note}</div>
</div>""", unsafe_allow_html=True)

        # Download
        ref     = _claim_id()
        ts      = datetime.now().isoformat()
        report  = _gen_report(state, ref, ts, brand_s, model_s)
        st.download_button(
            "Download Claim Determination Summary",
            data=report, file_name=f"{ref}_Determination.txt", mime="text/plain",
        )

        # Debug expander — all raw payloads isolated here
        with st.expander("System Protocol Payloads and State Graph Inspector"):
            st.json({k: v for k, v in state.items()
                     if k not in ("raw_image", "defect_overlay")})
