"""
Detect text and shape color
"""

import cv2
import numpy as np

# (color, hue lower bound, hue upper bound)
COLOR_RANGES = [
    ('red', 0, 6), ('orange', 7, 22), ('yellow', 23, 32), ('green', 33, 82),
    ('blue', 83, 126), ('purple', 127, 155), ('red', 156, 180)
]


# Match an rgb code to a color name
def color_name(rgb):
    # h ranges from 0 to 180, l and s range from 0 to 255
    hls = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HLS)[0][0]

    # l, a, and b range from 0 to 255
    lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0]

    # black detection: lightness <= 40
    if hls[1] <= 40:
        return 'black'

    # white detection: lightness >= 215
    if hls[1] >= 215:
        return 'white'

    # brown detection: values obtained through testing
    b_lower_bound = max([1192-8*lab[1], 132, lab[1]-20])
    if (lab[0] <= 127.5) and (lab[1] <= 160.5) and (lab[2] >= b_lower_bound):
        return "brown"

    # gray detection: a* and b* are close to neutral(128)
    if 118 <= lab[1] <= 138 and 118 <= lab[2] <= 138:
        return 'gray'

    # general color detection, looping through color ranges
    for color in COLOR_RANGES:
        if color[1] <= hls[0] <= color[2]:
            return color[0]


# Extract the median color and rnn a color detection algorithm
# to detect the color
def get_color(image, mask):
    # extract the pixels
    pixels = cv2.bitwise_and(image, image, mask=mask)
    pixels = pixels.reshape(-1, 3)

    # get median color
    pixels = pixels[np.any(image, axis=1)]
    rgb = np.median(pixels, axis=0)

    return color_name(rgb)
