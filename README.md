# ROSCAN: Remote Optical & Spectral Claim Adjudication Node

![Status](https://img.shields.io/badge/Status-Production-green)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Architecture](https://img.shields.io/badge/Architecture-LangGraph_Agentic-orange)

ROSCAN is an enterprise multimodal edge diagnostic and claims adjudication system designed for consumer electronics warranty and device insurance lifecycle operations. It aligns with enterprise tech stacks to automate hardware diagnostics visually and acoustically.

## Architecture

```
[ Camera / Audio Input ] 
       │
       ▼
┌──────────────────────┐      ┌────────────────────────┐
│ Visual Engine        │      │ Acoustic Engine        │
│ - 2D-FFT Moiré       │      │ - Sine-Sweep Gen       │
│ - Bezel Contour Math │      │ - STFT Spectrogram     │
│ - Morphological Edge │      │ - High-Band Roll-off   │
└──────────┬───────────┘      └───────────┬────────────┘
           │                              │
           ▼                              ▼
    ┌───────────────────────────────────────────┐
    │ LangGraph Agent State Machine             │
    │ (ClaimAdjudicationGraph)                  │
    └────────────────────┬──────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ MCP Tools (Economics / Policy)  │
        │ - OEM Parts & Labor Index       │
        │ - Salvage Market Scrape         │
        └────────────────┬────────────────┘
                         │
                         ▼
               [ Final Verdict ]
        APPROVE_REPAIR | APPROVE_REPLACEMENT | REJECT_FRAUD
```

## Setup & Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run tests:
   ```bash
   python test_pipeline.py
   ```
3. Run Streamlit Application:
   ```bash
   streamlit run app.py
   ```

## Key Mathematical Underpinnings
- **2D-FFT Moiré Spoof Detection**: Converts grayscale images to the frequency domain to identify periodic high-frequency noise spikes characteristic of screen-in-screen replay fraud.
- **STFT (Short-Time Fourier Transform) Diagnostics**: Measures energy roll-off in frequencies above 8 kHz to detect muffled sound (liquid ingress) or excessive harmonic distortion (blown diaphragm).
