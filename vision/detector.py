import cv2
import numpy as np

from vision.color_detection import detect_color_from_array
from vision.target_detection import get_shape_text_masks
from vision.text_detection import predict_text

def detect(img_path):
    img = cv2.imread(img_path)
    # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # get masks
    centroid, shape_mask, text_mask = get_shape_text_masks(img)

    cv2.imwrite(f"{img_path}.shape_mask.png", cv2.cvtColor(shape_mask, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{img_path}.text_mask.png", cv2.cvtColor(text_mask, cv2.COLOR_RGB2BGR))

    # text detection
    text_nonblack_mask = ~(np.all(text_mask == [0, 0, 0], axis=-1))
    text_white = text_mask.copy()
    text_white[text_nonblack_mask] = [255, 255, 255]

    text_pred = predict_text(text_white, "model/text.pth", 3)
    for i, (label, prob) in enumerate(text_pred, 1):
        print(f"{i}. {label}: {prob:.4f}") 

    # color detection
    shape_color = detect_color_from_array(shape_mask)
    text_color = detect_color_from_array(text_mask)

    print(f"{shape_color=}, {text_color=}")

    # calculate mask center using contour moments
    #M = cv2.moments(np.uint8(shape_mask))
    #if M["m00"] == 0:
        #    return None

    #cX = int(M["m10"] / M["m00"])
    #cY = int(M["m01"] / M["m00"])

    # calculate offsets. note (0, 0) is the top left of the image
    height, width = img.shape[:2]
    cX, cY = centroid
    offsetX = cX - width/2
    offsetY = -(cY - height/2)

    return (offsetX, offsetY)
