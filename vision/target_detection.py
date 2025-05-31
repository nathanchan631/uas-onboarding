import pandas as pd
import numpy as np
from glob import glob
import cv2
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from sklearn.cluster import KMeans

def segment_full_image_from_center_kmeans(image_bgr):
    assert image_bgr.shape[:2] == (128, 128), "Image must be 128x128"

    # Extract center 32x32 patch
    cx, cy = 64, 64
    half = 16
    center_patch = image_bgr[cy - half:cy + half, cx - half:cx + half]

    # Convert entire image and center patch to HSV
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    patch_hsv = cv2.cvtColor(center_patch, cv2.COLOR_BGR2HSV)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    patch_rgb = cv2.cvtColor(center_patch, cv2.COLOR_BGR2RGB)

    # Flatten center patch and ignore black pixels
    patch_rgb_flat = patch_rgb.reshape(-1, 3)
    non_black_mask = ~(np.all(patch_rgb_flat == [0, 0, 0], axis=1))

    # Extract H, S, V channels from the center patch
    patch_h_flat = patch_hsv[:, :, 0].flatten()
    patch_s_flat = patch_hsv[:, :, 1].flatten()
    patch_v_flat = patch_hsv[:, :, 2].flatten()

    # Filter non-black pixels
    patch_h_filtered = patch_h_flat[non_black_mask].reshape(-1, 1)
    patch_s_filtered = patch_s_flat[non_black_mask].reshape(-1, 1)
    patch_v_filtered = patch_v_flat[non_black_mask].reshape(-1, 1)

    # Perform KMeans on each channel
    kmeans_h = KMeans(n_clusters=2, random_state=42, n_init='auto')
    kmeans_s = KMeans(n_clusters=2, random_state=42, n_init='auto')
    kmeans_v = KMeans(n_clusters=2, random_state=42, n_init='auto')

    if len(patch_h_filtered) < 2 or len(patch_s_filtered) < 2 or len(patch_v_filtered) < 2:
        raise ValueError("Not enough non-black pixels in center patch for KMeans.")

    kmeans_h.fit(patch_h_filtered)
    kmeans_s.fit(patch_s_filtered)
    kmeans_v.fit(patch_v_filtered)

    # Apply cluster thresholds to the full image for H, S, V
    full_h = image_hsv[:, :, 0]
    full_s = image_hsv[:, :, 1]
    full_v = image_hsv[:, :, 2]

    # Apply clustering for each channel
    label_map_h = np.zeros_like(full_h, dtype=np.int8)
    label_map_s = np.zeros_like(full_s, dtype=np.int8)
    label_map_v = np.zeros_like(full_v, dtype=np.int8)

    # Hue (H) clustering
    centers_h = np.sort(kmeans_h.cluster_centers_.flatten())
    mid_threshold_h = np.mean(centers_h)
    label_map_h[full_h >= mid_threshold_h] = 1

    # Saturation (S) clustering
    centers_s = np.sort(kmeans_s.cluster_centers_.flatten())
    mid_threshold_s = np.mean(centers_s)
    label_map_s[full_s >= mid_threshold_s] = 1

    # Value (V) clustering
    centers_v = np.sort(kmeans_v.cluster_centers_.flatten())
    mid_threshold_v = np.mean(centers_v)
    label_map_v[full_v >= mid_threshold_v] = 1

    # Mask out black pixels in original image
    black_mask = np.all(image_rgb == [0, 0, 0], axis=-1)
    label_map_h[black_mask] = -1
    label_map_s[black_mask] = -1
    label_map_v[black_mask] = -1

    # Visualization
    vis_h = np.zeros((128, 128, 3), dtype=np.uint8)
    vis_s = np.zeros((128, 128, 3), dtype=np.uint8)
    vis_v = np.zeros((128, 128, 3), dtype=np.uint8)

    vis_h[label_map_h == 0] = [255, 0, 0]    # Red for cluster 0 (Hue)
    vis_h[label_map_h == 1] = [0, 0, 255]    # Blue for cluster 1 (Hue)

    vis_s[label_map_s == 0] = [255, 0, 0]    # Red for cluster 0 (Saturation)
    vis_s[label_map_s == 1] = [0, 0, 255]    # Blue for cluster 1 (Saturation)

    vis_v[label_map_v == 0] = [255, 0, 0]    # Red for cluster 0 (Value)
    vis_v[label_map_v == 1] = [0, 0, 255]    # Blue for cluster 1 (Value)

    vis_h[label_map_h == -1] = [0, 0, 0]     # Black for ignored
    vis_s[label_map_s == -1] = [0, 0, 0]     # Black for ignored
    vis_v[label_map_v == -1] = [0, 0, 0]     # Black for ignored
    return label_map_h, label_map_s, label_map_v

def convert_colors(img):
    # Count occurrences of -1, 0, and 1
    unique, counts = np.unique(img, return_counts=True)
    value_counts = dict(zip(unique, counts))

    # Find the least common of the possible values
    candidates = {-1: float('inf'), 0: float('inf'), 1: float('inf')}
    candidates.update(value_counts)
    least_common = min(candidates, key=candidates.get)

    # Prepare output image
    h, w = img.shape
    output = np.ones((h, w, 3), dtype=np.uint8) * 255  # Start all white
    output[img == least_common] = [0, 0, 0]  # Least common value -> black

    return output

def get_shape_text_masks(img):
    filtered_image = cv2.medianBlur(image, ksize=11)
    hsv_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 1]

    # Blur to reduce noise
    v_blurred = cv2.GaussianBlur(v_channel, (3, 3), 0)

    # Sobel edge detection
    sobel_x = cv2.Sobel(v_blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(v_blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Threshold to get edges
    edge_threshold = 20  # Adjust if needed
    edges = np.uint8(gradient_magnitude > edge_threshold)

    # Normalize and binarize
    normalized_gradient = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    _, binary_image = cv2.threshold(normalized_gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_mask = np.zeros_like(binary_image)
    centroids = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Skip tiny noisy regions
            continue

        # Convex hull + solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        if solidity < 0.4:
            continue

        # Add aspect ratio check (cracks are often long and narrow)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)  # avoid division by zero
        if aspect_ratio > 2:  # very long and thin => likely a crack
            continue

        # Optionally: filter contours that are too "skinny"
        if area / (w * h + 1e-5) < 0.3:  # not occupying enough of its bbox
            continue

        # Fill the contour into a mask to measure white pixel ratio
        temp_mask = np.zeros_like(binary_image)
        cv2.drawContours(temp_mask, [contour], -1, 255, thickness=cv2.FILLED)
        white_pixel_count = np.sum(temp_mask == 255)
        fill_ratio = white_pixel_count / area

        if fill_ratio < 0.95:
            continue

        # Optional: Reject overly complex contours (optional)
        '''epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) > 15:
            continue'''

        # Passed all filters – draw to final output
        cv2.drawContours(output_mask, [contour], -1, 255, thickness=cv2.FILLED)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
    kernel = np.ones((9, 9), np.uint8)  # You can tweak the size (3x3, 5x5, etc.)
    eroded_mask = cv2.erode(output_mask, kernel, iterations=1)
    masked_image = cv2.bitwise_and(image, image, mask=eroded_mask)

    # Step 2: Crop 64x64 patches around each centroid
    crop_size = 128
    half_crop = crop_size // 2
    masked_crops = []
    shape_crops = []
    text_crops = []

    for idx, (cX, cY) in enumerate(centroids):
        x1 = max(cX - half_crop, 0)
        y1 = max(cY - half_crop, 0)
        x2 = min(cX + half_crop, masked_image.shape[1])
        y2 = min(cY + half_crop, masked_image.shape[0])
        crop = masked_image[y1:y2, x1:x2]

        # Pad if the crop is smaller than 64x64
        h, w = crop.shape[:2]
        if h < crop_size or w < crop_size:
            top = (crop_size - h) // 2
            bottom = crop_size - h - top
            left = (crop_size - w) // 2
            right = crop_size - w - left
            crop = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
        masked_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        segmentation_labels_h, segmentation_labels_s, segmentation_labels_v = segment_full_image_from_center_kmeans(crop)
        vis_h = convert_colors(segmentation_labels_h)
        vis_s = convert_colors(segmentation_labels_s)
        vis_v = convert_colors(segmentation_labels_v)    
        text_crops.append(cv2.cvtColor(vis_h, cv2.COLOR_BGR2RGB))    
        non_black_mask = ~(np.all(crop == [0, 0, 0], axis=-1))
        crop[non_black_mask] = [255, 255, 255]
        shape_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return shape_crops, text_crops

# Test Case
'''
image = cv2.imread('./IMG_6823.png')
shape_crops, text_crops = get_shape_text_masks(image)
print(text_crops[2])
'''