import numpy as np
import cv2
from collections import Counter

# Predefined custom colors in HSV
custom_colors_hsv = {
    'white':  (0,   0,   255),
    'black':  (0,   0,   0),
    'red':    (0,   200, 200),
    'orange': (20,  180, 240),
    'brown':  (20,  150, 100),
    'yellow': (30,  200, 250),
    'green':  (60,  200, 200),
    'blue':   (120, 200, 200),
    'purple': (150, 180, 180)
}

def closest_hsv(requested_hsv):
    requested_h = int(requested_hsv[0])
    requested_s = int(requested_hsv[1])
    requested_v = int(requested_hsv[2])

    distances = {}
    for name, hsv in custom_colors_hsv.items():
        h = int(hsv[0])
        s = int(hsv[1])
        v = int(hsv[2])

        h_diff = min(abs(h - requested_h), 180 - abs(h - requested_h))  # Circular hue
        s_diff = s - requested_s
        v_diff = v - requested_v

        distance = np.sqrt((3 * h_diff) ** 2 + (2 * s_diff) ** 2 + (1 * v_diff) ** 2)
        distances[name] = distance
        print(f"{name}: distance = {distance:.2f}")

    return min(distances, key=distances.get)

def detect_color_from_array(image_array):
    if image_array is None or image_array.size == 0:
        raise ValueError("Empty or invalid image array.")

    # Convert BGR to RGB and find most common RGB
    rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    rgb_reshaped = rgb_image.reshape(-1, 3)
    rgb_counts = Counter(map(tuple, rgb_reshaped))
    most_common_rgb = rgb_counts.most_common(1)[0][0]
    most_common_rgb = tuple(int(x) for x in most_common_rgb)
    print("Most common RGB:", most_common_rgb)

    # Convert BGR to HSV and find most common HSV
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    hsv_reshaped = hsv_image.reshape(-1, 3)
    hsv_counts = Counter(map(tuple, hsv_reshaped))
    most_common_hsv = hsv_counts.most_common(1)[0][0]
    most_common_hsv = tuple(int(x) for x in most_common_hsv)
    print("Most common HSV:", most_common_hsv)

    color_name = closest_hsv(most_common_hsv)
    print("Closest color name:", color_name)
    return color_name

'''
image = cv2.imread('Downloads/t1.jpg')
detect_color_from_array(image)

image = cv2.imread('Downloads/t2.JPG')
detect_color_from_array(image)

image = cv2.imread('Downloads/t10.jpg')
detect_color_from_array(image)
'''