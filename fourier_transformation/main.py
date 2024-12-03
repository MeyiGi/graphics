import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread("demoImages/flower.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("demoImages/A.jpg", cv2.IMREAD_GRAYSCALE)

# Perform Fourier Transform for img1
f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
magnitude_spectrum1 = 20 * np.log(np.abs(fshift1) + 1e-8)  # Add small constant to avoid log(0)
magnitude_spectrum1 = np.asarray(magnitude_spectrum1, dtype=np.uint8)

# Perform Fourier Transform for img2
f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)
magnitude_spectrum2 = 20 * np.log(np.abs(fshift2) + 1e-8)  # Add small constant to avoid log(0)
magnitude_spectrum2 = np.asarray(magnitude_spectrum2, dtype=np.uint8)

# Create a single figure with 4 subplots
plt.figure(figsize=(12, 6))

# Original Image 1
plt.subplot(2, 2, 1)
plt.title("Original Image 1")
plt.imshow(img1, cmap='gray')
plt.axis("off")

# Magnitude Spectrum 1
plt.subplot(2, 2, 2)
plt.title("Magnitude Spectrum 1")
plt.imshow(magnitude_spectrum1, cmap='gray')
plt.axis("off")

# Original Image 2
plt.subplot(2, 2, 3)
plt.title("Original Image 2")
plt.imshow(img2, cmap='gray')
plt.axis("off")

# Magnitude Spectrum 2
plt.subplot(2, 2, 4)
plt.title("Magnitude Spectrum 2")
plt.imshow(magnitude_spectrum2, cmap='gray')
plt.axis("off")

# Adjust layout and display
plt.tight_layout()
plt.show()
