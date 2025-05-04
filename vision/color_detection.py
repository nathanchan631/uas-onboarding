import numpy as np
import cv2
from collections import Counter

# Predefined custom colors in HSV
custom_colors_hsv = {
    'white':  (0,   0,   255),
    'black':  (0,   0,   0),
    'red_low': (0,   200, 200),
    'orange': (11,  180, 240),
    'brown':  (5,  150, 100),
    'yellow': (30,  200, 250),
    'green':  (60,  200, 200),
    'blue':   (120, 200, 200),
    'purple': (140, 180, 180),
    'red_high': (160, 200, 200)
}

def closest_hsv(requested_hsv):
    requested_h = int(requested_hsv[0])
    # print(f"Requested hue: {requested_h}")
    distances = {}
    for name, hsv in custom_colors_hsv.items():
        h = int(hsv[0])

        h_diff = min(abs(h - requested_h), 180 - abs(h - requested_h)) # this should handle red wrap around
        distances[name] = h_diff
        # print(f"{name}: hue diff = {h_diff}")
    # Find the minimum hue difference
    min_diff = min(distances.values())
    candidates = [name for name, diff in distances.items() if diff == min_diff]
    
    # If multiple colors tie, prioritize 'red_low' or 'red_high' for hues near red (0-10 or 160-180)
    if len(candidates) > 1 and requested_h >= 160 or requested_h <= 10:
        for red_name in ['red_low', 'red_high']:
            if red_name in candidates:
                return red_name
    return min(distances, key=distances.get)


def detect_color_from_array(image_array):
    if image_array is None or image_array.size == 0:
        raise ValueError("Empty or invalid image array.")

    # Convert BGR to HSV and find most common HSV
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    hsv_reshaped = hsv_image.reshape(-1, 3)

    remove_black = [pixel for pixel in hsv_reshaped if not (pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0)]
    if not remove_black:
        print("ERROR: No non-black pixels found.")
        return 'black'

    mean_hsv = np.mean(remove_black, axis=0)
    color_name = closest_hsv(mean_hsv)

    if color_name in ('yellow', 'orange', 'brown'):
        h, s, v = mean_hsv
        if v < 140:
            color_name = 'brown'
        elif h >= 20 and h <= 35:
            if v >= 200 and s >= 150:
                color_name = 'yellow'
            elif v >= 150:
                color_name = 'orange'
            else:
                color_name = 'brown'
    if color_name in ('red_low', 'red_high'):
        color_name = 'red'
    # print("Closest color name:", color_name)
    return color_name