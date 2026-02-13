import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pyfftw
import time # Import time module

# --- 1. FFTW Setup ---
pyfftw.interfaces.cache.enable()
fft_lib = pyfftw.interfaces.numpy_fft

# --- 2. Load and Prepare Image (LOCAL FILE) ---
file_path = r"C:\Users\samer\OneDrive\Desktop\NANO\ANT-Quantum-Imaging\QIUP-APP\test-images\Flower.jpg"

try:
    img = Image.open(file_path)
    img_gray = np.array(img.convert('L'))
except FileNotFoundError:
    print(f"Error: Could not find the file at {file_path}")
    exit()

# OPTIONAL PERFORMANCE BOOST: Align memory for FFTW
img_aligned = pyfftw.empty_aligned(img_gray.shape, dtype='float32')
img_aligned[:] = img_gray

# --- START TIMER ---
start_time = time.time()

# --- 3. Compute FFT (Using FFTW) ---
rows, cols = img_gray.shape
crow, ccol = rows // 2, cols // 2

f = fft_lib.fft2(img_aligned)
fshift = fft_lib.fftshift(f)

# --- 4. Generate Variations ---

# A. Magnitude Spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# B. Low-Pass Filter
mask_lp = np.zeros((rows, cols), np.uint8)
mask_lp[crow-30:crow+30, ccol-30:ccol+30] = 1
fshift_lp = fshift * mask_lp
img_lp = np.abs(fft_lib.ifft2(fft_lib.ifftshift(fshift_lp)))

# C. High-Pass Filter
mask_hp = np.ones((rows, cols), np.uint8)
mask_hp[crow-30:crow+30, ccol-30:ccol+30] = 0
fshift_hp = fshift * mask_hp
img_hp = np.abs(fft_lib.ifft2(fft_lib.ifftshift(fshift_hp)))

# D. CORRECTED COMPRESSION (Percentile Method)
keep_fraction = 0.1  # Keep top 10%

# 1. Sort absolute values
f_abs_flatten = np.abs(fshift).flatten()
threshold_val = np.percentile(f_abs_flatten, (1 - keep_fraction) * 100)

# 2. Create mask
mask_compressed = np.abs(fshift) > threshold_val

# 3. Apply mask and reconstruct
fshift_compressed = fshift * mask_compressed
img_compressed = np.abs(fft_lib.ifft2(fft_lib.ifftshift(fshift_compressed)))

# --- END TIMER ---
end_time = time.time()
execution_time = end_time - start_time

print(f"FFTW Processing Time: {execution_time:.4f} seconds")

# --- 5. Display All in One Window ---
plt.figure(figsize=(15, 8))

# Original
plt.subplot(2, 3, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Magnitude Spectrum
plt.subplot(2, 3, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title(f'Magnitude Spectrum (FFTW)\nTime: {execution_time:.4f}s')
plt.axis('off')

# Low-Pass
plt.subplot(2, 3, 4)
plt.imshow(img_lp, cmap='gray')
plt.title('Low-Pass Filter')
plt.axis('off')

# High-Pass
plt.subplot(2, 3, 5)
plt.imshow(img_hp, cmap='gray')
plt.title('High-Pass Filter')
plt.axis('off')

# Compressed
plt.subplot(2, 3, 6)
plt.imshow(img_compressed, cmap='gray')
plt.title(f'Compressed (Top {keep_fraction*100}% Kept)')
plt.axis('off')

# Show the compression mask itself
plt.subplot(2, 3, 3)
plt.imshow(mask_compressed, cmap='gray')
plt.title('Compression Mask (White = Kept)')
plt.axis('off')

plt.tight_layout()
plt.show()