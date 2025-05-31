import cv2
import numpy as np

from vision.color_detection import detect_color_from_array
from vision.target_detection import get_shape_text_masks
from vision.text_detection import predict_text

def detect(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # get masks
    shape_mask, text_mask = get_shape_text_masks(img)

    # text detection
    text_pred = predict_text(text_mask, "model/text.pth", 3)
    for i, (label, prob) in enumerate(text_pred, 1):
        print(f"{i}. {label}: {prob:.4f}") 

    # color detection
    shape_pixels = cv2.bitwise_and(img, img, mask=shape_mask)
    text_pixels = cv2.bitwise_and(img, img, mask=text_mask)
    shape_color = detect_color_from_array(shape_pixels)
    text_color = detect_color_from_array(text_pixels)

    print(shape_color, text_color)

    # calculate mask center using contour moments
    M = cv2.moments(np.uint8(shape_mask))
    if M["m00"] == 0:
        return None

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # calculate offsets. note (0, 0) is the top left of the image
    height, width = img.shape[:2]
    offsetX = cX - width/2
    offsetY = -(cY - height/2)

    return (offsetX, offsetY)
