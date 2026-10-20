# Captain ROScan

**ROSCAN — Remote Optical & Spectral Claim Adjudication Node**

A phone insurance claim checker that works from a single photo. Upload a picture of a damaged phone and Captain ROScan grades the damage, checks the photo is genuine, looks up what the repair should cost, and gives a clear claim decision — approve repair, approve replacement, route it to a free OEM warranty programme, or reject it — with the money worked out for each case.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)

**Live demo:** https://roscan.streamlit.app

---

## Why this exists

Device insurance claims are slow because every claim needs a human to look at the phone, judge the damage, and check prices before anything is approved. Captain ROScan does that first pass automatically:

- Is the photo real, or a picture of another screen (a common fraud)?
- Is the display a genuine OEM panel or a cheap replacement?
- How bad is the damage, really?
- Does the phone qualify for a free OEM screen programme (many AMOLED line defects do)?
- Is repair cheaper than replacement for this exact model?

## How it works

```
Photo (upload / camera / sample)
        │
        ▼
┌─ Vision engine (modules/vision.py) ──────────────────┐
│ 1. Moiré anti-spoofing   – is this photo genuine?     │
│ 2. Bezel symmetry        – is the panel OEM?          │
│ 3. Surface grading       – Grade A / B / C damage     │
│ 4. OLED line detection   – green / pink / white lines │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌─ LangGraph agent (modules/agent.py) ─────────────────┐
│ verify_authenticity → segment_defects →              │
│ fetch_mcp_economics → synthesize_adjudication        │
└──────────────────────┬───────────────────────────────┘
                       ▼
        Verdict + deductible + insurer liability
```

### The four vision checks

| Check | What it does |
|---|---|
| **Moiré anti-spoofing** | Converts the photo to the frequency domain (2D-FFT) and looks for the regular pixel-grid pattern a screen replay leaves behind, using a PAPR peak test plus a geometric symmetry check, so normal on-screen content is not flagged by mistake. |
| **Bezel symmetry** | Measures chin-to-top bezel balance to spot non-OEM panels. It is tilt-aware: if the phone was photographed at an angle (over 4°), the check is skipped instead of guessing. Flat photos with more than 20% asymmetry are flagged. |
| **Surface grading** | Grades glass damage A (clean), B (light wear) or C (fracture). A bilateral filter removes app-icon edges first, then a branching test separates real spiderweb cracks from straight UI lines. Grade C needs both heavy coverage and real crack branching. |
| **OLED line detection** | Scans green, magenta and white colour ranges in HSV, then uses Hough line detection for vertical lines spanning at least 45% of the screen — the classic AMOLED column failure. |

### The decision agent

A LangGraph state machine runs the checks, then pulls prices from the economics registry and applies a 7-case decision matrix:

| Verdict | When |
|---|---|
| `REJECT_FRAUD` | Screen replay / spoofed photo confirmed |
| `REJECT_UNAUTHORIZED_MOD` | Non-OEM panel fitted |
| `NO_FAULT_FOUND` | Phone is in pristine condition |
| `CLAIM_DISMISSED_WEAR_AND_TEAR` | Only light cosmetic wear (Grade B) |
| `APPROVED_OEM_RECALL_WARRANTY` | AMOLED line defect covered by a free OEM programme — Samsung S-series one-time free screen replacement, or OnePlus lifetime screen warranty |
| `APPROVED_REPAIR` | Real damage, and repair costs less than replacement |
| `APPROVED_REPLACEMENT` | Repair costs more than the net replacement loss |

Every verdict comes with written reasoning, the customer deductible, and the insurer's liability in ₹.

### Economics registry (modules/mcp_tools.py)

A built-in price registry for **12 phone brands**, with OEM display + labour prices, resale values and scrap values in INR (based on Cashify Q3-2026 figures and OEM India service rate cards), plus per-model deductibles and recall eligibility. Models without exact figures fall back to price tiers (budget / mid-range / premium / ultra-premium).

## App features

- Pick the brand and model in the sidebar and see its repair cost, resale value, scrap value and deductible before you even run a check
- Three ways to inspect: bundled field samples, generated synthetic benchmarks, or your own photo / live camera
- Results dashboard with a KPI matrix, the FFT heatmap, the defect overlay, and a plain-language decision explanation
- A downloadable claim report for every inspection
- Capture guidance prompts in English and Hinglish

## Quickstart

```bash
git clone https://github.com/RonakAgarwal18/ROSCAN-Multimodal-Diagnostic-Agent.git
cd ROSCAN-Multimodal-Diagnostic-Agent
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501, choose a brand and model, and load one of the sample inspections to see a full decision.

## Run the tests

```bash
python test_pipeline.py
```

This checks the imports, verifies the price registry for all 12 brands, and runs the vision + agent pipeline end to end.

## Project structure

```
app.py                  Streamlit app (UI, dashboards, claim report)
modules/
  vision.py             The four OpenCV checks
  agent.py              LangGraph claim decision graph (7-case matrix)
  mcp_tools.py          Price & policy registry (12 brands, INR)
  voice_guide.py        English / Hinglish capture guidance
test_pipeline.py        End-to-end pipeline checks
test_images/            Field samples: pristine, shattered, scratch,
                        green line, moiré, counterfeit
requirements.txt
```

## Tech stack

Python · Streamlit · OpenCV · NumPy · SciPy · LangGraph · Pydantic · Matplotlib

## Note on the data

The price registry ships inside the repo so the demo runs anywhere with no setup. Figures are indicative (Cashify and OEM rate-card based, Q3 2026) — plug in a live pricing source before using this for real claims.

---

Built by [Ronak Agarwal](https://github.com/RonakAgarwal18)
