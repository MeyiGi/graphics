import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Load the image
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 3: Remove noise using morphological operations
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 4: Find sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Step 5: Find sure foreground area using distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Step 6: Find unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Step 7: Label markers for watershed
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Step 8: Apply the watershed algorithm
markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]  # Mark boundaries in red

# Display all steps in one screen using subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(gray, cmap='gray')
axs[0, 1].set_title('Gray Image')
axs[0, 1].axis('off')

axs[0, 2].imshow(thresh, cmap='gray')
axs[0, 2].set_title('Thresholded Image')
axs[0, 2].axis('off')

axs[1, 0].imshow(opening, cmap='gray')
axs[1, 0].set_title('Noise Removed Image')
axs[1, 0].axis('off')

axs[1, 1].imshow(sure_bg, cmap='gray')
axs[1, 1].set_title('Sure Background')
axs[1, 1].axis('off')

axs[1, 2].imshow(dist_transform, cmap='jet')
axs[1, 2].set_title('Distance Transform')
axs[1, 2].axis('off')

axs[2, 0].imshow(sure_fg, cmap='gray')
axs[2, 0].set_title('Sure Foreground')
axs[2, 0].axis('off')

axs[2, 1].imshow(unknown, cmap='gray')
axs[2, 1].set_title('Unknown Region')
axs[2, 1].axis('off')

axs[2, 2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[2, 2].set_title('Final Image with Watershed')
axs[2, 2].axis('off')

plt.tight_layout()
plt.show()
