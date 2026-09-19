import streamlit as st
import numpy as np
import cv2
import base64
import json
from modules.vision import calculate_moire_spoof_score, analyze_bezel_symmetry, segment_screen_defects
from modules.acoustics import run_acoustic_sweep, compute_stft_diagnostics
from modules.agent import build_claim_graph
from modules.voice_guide import get_visual_guidance_prompt

st.set_page_config(page_title="ROSCAN Edge Diagnostics", page_icon="🔍", layout="wide")

# Custom CSS for dark mode styling and badges
st.markdown("""
<style>
    .badge {
        padding: 5px 10px; border-radius: 5px; font-weight: bold; font-size: 14px;
    }
    .badge-success { background-color: #28a745; color: white; }
    .badge-warning { background-color: #ffc107; color: black; }
    .badge-danger { background-color: #dc3545; color: white; }
    .metric-card {
        background-color: #1e1e1e; padding: 20px; border-radius: 10px; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("ROSCAN: Remote Optical & Spectral Claim Adjudication Node")
st.markdown("Enterprise multimodal edge diagnostic system for consumer electronics.")

# Initialize session state variables
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = None

tab1, tab2, tab3, tab4 = st.tabs(["Step 1: Visual Inspection", "Step 2: Acoustics Check", "Step 3: Agentic Adjudication", "Step 4: Claim Verdict"])

with tab1:
    st.header("Optical Analysis")
    img_file = st.file_uploader("Upload Device Image", type=['png', 'jpg', 'jpeg'])
    camera_input = st.camera_input("Or Capture Device Image")
    
    input_image = img_file or camera_input
    
    if input_image is not None:
        file_bytes = np.asarray(bytearray(input_image.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.session_state.raw_image = img_bgr
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Raw Capture", use_column_width=True)
            
            prompts = get_visual_guidance_prompt(0.8, 0.7)
            st.info(f"Voice Guidance: {prompts['english']} / {prompts['hinglish']}")
            
        with col2:
            st.write("### Running Vision Diagnostics...")
            spoof_score, heatmap = calculate_moire_spoof_score(img_bgr)
            is_counterfeit, asymmetry = analyze_bezel_symmetry(img_bgr)
            grade, defect_pct, blended = segment_screen_defects(img_bgr)
            
            st.image(heatmap, caption="2D-FFT Moiré Frequency Spectrum", use_column_width=True)
            st.metric("Spoof Score", f"{spoof_score:.2f}")
            st.metric("Bezel Asymmetry", f"{asymmetry:.1f}%")
            st.metric("Cosmetic Grade", grade)

with tab2:
    st.header("Acoustic Transducer Check")
    if st.button("Run Frequency Sweep"):
        with st.spinner("Running 200Hz - 12kHz audio sweep..."):
            sample_rate = 44100
            audio_buffer = run_acoustic_sweep(sample_rate=sample_rate, duration=2.5)
            st.session_state.audio_buffer = audio_buffer
            st.session_state.audio_sample_rate = sample_rate
            
            st.audio(audio_buffer, sample_rate=sample_rate)
            
            status, fig_bytes = compute_stft_diagnostics(audio_buffer, sample_rate)
            
            st.image(fig_bytes, caption="STFT Spectrogram", use_column_width=True)
            
            if "PASS" in status:
                st.markdown(f"<span class='badge badge-success'>{status}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='badge badge-danger'>{status}</span>", unsafe_allow_html=True)

with tab3:
    st.header("LangGraph Agent & MCP Execution")
    if st.button("Execute Agent Adjudication Workflow"):
        if 'raw_image' not in st.session_state or 'audio_buffer' not in st.session_state:
            st.error("Please complete Step 1 (Visual) and Step 2 (Acoustic) first.")
        else:
            with st.spinner("Running Adjudication Graph..."):
                graph = build_claim_graph()
                
                initial_state = {
                    "device_model": "IPHONE-13-PRO",
                    "policy_id": "POL-991238",
                    "raw_image": st.session_state.raw_image,
                    "audio_buffer": st.session_state.audio_buffer,
                    "audio_sample_rate": st.session_state.audio_sample_rate
                }
                
                final_state = graph.invoke(initial_state)
                st.session_state.workflow_state = final_state
                
                st.success("Graph execution complete!")
                st.json({k: v for k, v in final_state.items() if k not in ['raw_image', 'audio_buffer']})

with tab4:
    st.header("Executive Claim Adjudication Card")
    if st.session_state.workflow_state is not None:
        state = st.session_state.workflow_state
        verdict = state.get("final_verdict", "PENDING")
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        if verdict == "APPROVE_REPAIR":
            badge_class = "badge-warning"
        elif verdict == "APPROVE_REPLACEMENT":
            badge_class = "badge-success"
        else:
            badge_class = "badge-danger"
            
        st.markdown(f"<span class='badge {badge_class}' style='font-size:24px;'>{verdict}</span>", unsafe_allow_html=True)
        st.write("### Decision Reasoning")
        st.info(state.get("decision_reasoning"))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Economics**")
            st.json(state.get("oem_quote", {}))
        with col2:
            st.write("**Salvage / Policy**")
            st.json(state.get("salvage_quote", {}))
            st.json(state.get("policy_coverage", {}))
            
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Please run the Agentic Adjudication workflow in Step 3.")
