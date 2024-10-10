import cv2
import numpy as np

# Load the image
image = cv2.imread('image1.jpg')

# Convert the image into a 2D array of pixels
pixels = image.reshape((-1, 3))
pixels = np.float32(pixels)

# Define criteria for K-Means: (type, max_iter, epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Number of clusters (K)
K = 6  # You can change this value depending on how many clusters you want

# Apply KMeans
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert the centers to uint8 (since they are float32)
centers = np.uint8(centers)

# Map labels back to the colors
segmented_image = centers[labels.flatten()]

# Reshape the segmented image back to the original image dimensions
segmented_image = segmented_image.reshape(image.shape)

# Display the image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
