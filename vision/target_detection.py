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
import time

def segment_text_by_pixel_count(image_bgr, channel='h'):
    """
    Segments text by applying KMeans clustering on the specified HSV channel.
    The text region is kept in its original color, and the background is set to black.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.
        channel (str): HSV channel to use ('h', 's', or 'v').

    Returns:
        np.ndarray: RGB image with black background and original-color text.
    """
    # Convert to HSV and RGB
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    channel_map = {'h': 0, 's': 1, 'v': 2}
    selected_channel = image_hsv[:, :, channel_map[channel]]

    H, W = image_bgr.shape[:2]

    # Define center 50% region
    x_start = W // 4
    x_end = x_start + W // 2
    y_start = H // 4
    y_end = y_start + H // 2

    patch = selected_channel[y_start:y_end, x_start:x_end]
    patch_rgb = image_rgb[y_start:y_end, x_start:x_end]

    # Remove black pixels
    non_black_mask = ~(np.all(patch_rgb == [0, 0, 0], axis=-1))
    center_vals = patch[non_black_mask].reshape(-1, 1)

    # Prepare black output canvas
    output = np.zeros_like(image_rgb, dtype=np.uint8)

    if len(center_vals) < 2:
        return output  # Not enough data for clustering

    # KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    kmeans.fit(center_vals)
    centers = np.sort(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)

    # Generate label map
    label_map = np.zeros_like(selected_channel, dtype=np.uint8)
    label_map[selected_channel >= threshold] = 1

    # Mask out black background
    black_mask = np.all(image_rgb == [0, 0, 0], axis=-1)
    label_map[black_mask] = 255  # ignore flag

    # Choose smaller cluster as text
    count_0 = np.sum(label_map == 0)
    count_1 = np.sum(label_map == 1)
    text_label = 0 if count_0 < count_1 else 1

    # Create binary text mask
    text_mask = np.zeros_like(label_map, dtype=np.uint8)
    text_mask[label_map == text_label] = 255

    # Draw the largest contour as a mask
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        contour_mask = np.zeros_like(text_mask)
        cv2.drawContours(contour_mask, [largest], -1, 255, thickness=cv2.FILLED)

        # Apply mask to original RGB image
        output[contour_mask == 255] = image_rgb[contour_mask == 255]

    return output


def get_shape_text_masks(img):
    height, width = img.shape[:2]
    img_area = height * width
    filtered_image = cv2.medianBlur(img, ksize=11)
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
    bounding_boxes = []
    percent = 1500 / 18019728

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < percent * img_area:  # Skip tiny noisy regions
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

        # Passed all filters – draw to final output
        cv2.drawContours(output_mask, [contour], -1, 255, thickness=cv2.FILLED)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

            left = tuple(contour[contour[:, :, 0].argmin()][0])
            right = tuple(contour[contour[:, :, 0].argmax()][0])
            top = tuple(contour[contour[:, :, 1].argmin()][0])
            bottom = tuple(contour[contour[:, :, 1].argmax()][0])

            bounding_boxes.append((left, right, top, bottom))


    masked_crops = []
    bounding_box_coords = []

    # Percent of bounding box size to erode (e.g., 10%)
    erosion_percent = 0.15

    for idx, box in enumerate(bounding_boxes):
        left, right, top, bottom = box

        # Clamp bounding box to image boundaries
        x1 = max(left[0], 0)
        x2 = min(right[0], img.shape[1])
        y1 = max(top[1], 0)
        y2 = min(bottom[1], img.shape[0])

        # Extract region from original mask and image
        bbox_mask = output_mask[y1:y2, x1:x2]
        bbox_image = img[y1:y2, x1:x2]

        # Compute kernel size as a percentage of bounding box size
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        kx = max(1, int(bw * erosion_percent))
        ky = max(1, int(bh * erosion_percent))

        # Ensure kernel size is odd and >= 3
        kx = kx + 1 if kx % 2 == 0 else kx
        ky = ky + 1 if ky % 2 == 0 else ky
        kx = max(kx, 3)
        ky = max(ky, 3)

        # Apply erosion to the mask region
        kernel = np.ones((ky, kx), np.uint8)
        eroded = cv2.erode(bbox_mask, kernel, iterations=1)

        # Mask the image patch
        masked_crop = cv2.bitwise_and(bbox_image, bbox_image, mask=eroded)

        # Save bounding box and masked crop
        bounding_box_coords.append(((x1, y1), (x2, y2)))

        # For display
        masked_crops.append(cv2.cvtColor(masked_crop, cv2.COLOR_BGR2RGB))
    shape_crops = []
    text_crops = []
    crop_areas = [crop.shape[0] * crop.shape[1] for crop in masked_crops]
    sorted_indices = sorted(range(len(masked_crops)), key=lambda i: crop_areas[i], reverse=True)
    masked_crops_sorted = [masked_crops[i] for i in sorted_indices]
    for crop in masked_crops_sorted:
        text = segment_text_by_pixel_count(crop, channel='h')
        text_crops.append(text)
        #non_black_mask = ~(np.all(crop == [0, 0, 0], axis=-1))
        #crop[non_black_mask] = [255, 255, 255]
        shape_crops.append(crop)
    return centroids[0], shape_crops[0], text_crops[0]

# Test Case
#image = cv2.imread('./IMG_6823.png')
#centroids, shape_crops, text_crops = get_shape_text_masks(image)
#print(centroids)
#plt.imshow(shape_crops)