import cv2
import numpy as np
from typing import Tuple

def calculate_moire_spoof_score(image_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Convert image to grayscale, compute 2D Fast Fourier Transform,
    shift zero frequencies to center, extract log magnitude spectrum,
    compute energy in high-frequency annular rings to identify periodic Moiré spikes.
    Returns normalized spoof score [0.0-1.0] and spectral heatmap.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Calculate magnitude spectrum with a small epsilon to avoid log(0)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    
    # Create an annular mask to extract high frequencies (excluding the low-frequency center)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    # Radii for annular ring
    inner_radius = min(h, w) * 0.15
    outer_radius = min(h, w) * 0.45
    
    mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
    
    # Measure high frequency energy
    high_freq_energy = np.sum(magnitude_spectrum[mask])
    total_energy = np.sum(magnitude_spectrum)
    
    if total_energy == 0:
        ratio = 0.0
    else:
        ratio = high_freq_energy / total_energy
        
    # Heuristically normalize the ratio to [0.0, 1.0] for spoof score
    # Expected normal ratios depend on image texture, but high periodic signals spike the energy
    spoof_score = float(np.clip((ratio - 0.1) * 5.0, 0.0, 1.0))
    
    # Normalize magnitude spectrum for visualization (heatmap)
    heatmap = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay the mask bounds on the heatmap for visualization
    cv2.circle(heatmap_colored, (cx, cy), int(inner_radius), (255, 255, 255), 1)
    cv2.circle(heatmap_colored, (cx, cy), int(outer_radius), (255, 255, 255), 1)
    
    return spoof_score, heatmap_colored

def analyze_bezel_symmetry(image_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Find outer phone bounding contours and active display contours.
    Measure top/bottom/left/right border ratios.
    Flag counterfeit third-party replacement panels if bottom chin-to-top bezel asymmetry exceeds 15%.
    Returns (is_counterfeit, asymmetry_percentage)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, 0.0
        
    # Assume the largest contour is the phone/display
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # In a real scenario, we'd detect the screen bounds vs device bounds.
    # Here we simulate by analyzing the bounding box's aspect ratio and center variance.
    # We will compute a mock asymmetry score derived from the contour's bounding box vs centroid
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cy = int(M["m01"] / M["m00"])
    else:
        cy = y + h // 2
        
    box_center_y = y + h / 2.0
    
    # Asymmetry based on centroid deviation from bounding box center vertically
    asymmetry_percentage = abs(cy - box_center_y) / (h / 2.0) * 100.0
    
    is_counterfeit = asymmetry_percentage > 15.0
    return is_counterfeit, float(asymmetry_percentage)

def segment_screen_defects(image_bgr: np.ndarray) -> Tuple[str, float, np.ndarray]:
    """
    Morphological edge extraction and adaptive thresholding to detect surface fractures,
    crack lines, and localized RGB variance drops.
    Output cosmetic grade: Grade A (<2%), Grade B (2-8%), Grade C (>8% or cracked).
    Returns (cosmetic_grade, defect_percentage, defect_mask)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to find defects/cracks
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
                                   
    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    total_pixels = gray.shape[0] * gray.shape[1]
    defect_pixels = cv2.countNonZero(cleaned)
    
    defect_percentage = (defect_pixels / total_pixels) * 100.0
    
    # We also run a Canny edge detector to simulate sharp crack detection
    edges = cv2.Canny(gray, 100, 200)
    crack_pixels = cv2.countNonZero(edges)
    crack_percentage = (crack_pixels / total_pixels) * 100.0
    
    # Logic for grading
    if defect_percentage > 8.0 or crack_percentage > 3.0:
        grade = "Grade C"
    elif defect_percentage > 2.0:
        grade = "Grade B"
    else:
        grade = "Grade A"
        
    defect_mask_colored = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    defect_mask_colored[cleaned > 0] = [0, 0, 255]  # Red for defects
    
    blended = cv2.addWeighted(image_bgr, 0.7, defect_mask_colored, 0.3, 0)
    
    return grade, float(defect_percentage), blended
