import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import io
from typing import Tuple

def run_acoustic_sweep(sample_rate: int = 44100, duration: float = 2.5) -> np.ndarray:
    """
    Generate synthetic 200 Hz - 12,000 Hz stepped sine-sweep audio buffer.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Sine sweep from 200 Hz to 12000 Hz
    sweep = signal.chirp(t, 200, duration, 12000, method='logarithmic')
    # Add a bit of noise to simulate real-world recording
    noise = np.random.normal(0, 0.01, sweep.shape)
    buffer = sweep + noise
    
    # Normalize to -1.0 to 1.0 range
    buffer = np.clip(buffer, -1.0, 1.0)
    return buffer

def compute_stft_diagnostics(audio_buffer: np.ndarray, sample_rate: int) -> Tuple[str, bytes]:
    """
    Compute Short-Time Fourier Transform using scipy.signal.stft.
    Measure spectral roll-off and harmonic distortion across high bands to flag 
    speaker diaphragm damage or microphone liquid ingress.
    Returns diagnostic status and spectrogram visualization figure (as PNG bytes).
    """
    f, t, Zxx = signal.stft(audio_buffer, fs=sample_rate, nperseg=1024)
    magnitude = np.abs(Zxx)
    
    # Measure high-frequency energy (approx above 8kHz)
    freq_indices = np.where(f > 8000)[0]
    if len(freq_indices) > 0:
        high_band_energy = np.sum(magnitude[freq_indices, :])
        total_energy = np.sum(magnitude)
        
        # Simple heuristic for diaphragm damage / liquid ingress
        if total_energy > 0:
            high_band_ratio = high_band_energy / total_energy
            if high_band_ratio < 0.05:
                status = "FAIL: Liquid Ingress / Muffled Highs Detected"
            elif high_band_ratio > 0.4:
                status = "FAIL: Speaker Diaphragm Distortion Detected"
            else:
                status = "PASS: Acoustics Normal"
        else:
            status = "FAIL: No audio signal detected"
    else:
        status = "FAIL: Sample rate too low for high band detection"

    # Generate spectrogram plot
    plt.figure(figsize=(6, 4))
    plt.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx) + 1e-10), shading='gouraud', cmap='viridis')
    plt.title('STFT Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude [dB]')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return status, buf.read()
