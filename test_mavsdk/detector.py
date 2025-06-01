import cv2
import numpy as np

RED_LOWER_1 = (165, 10, 0)
RED_UPPER_1 = (180, 255, 255)
RED_LOWER_2 = (0, 10, 0)
RED_UPPER_2 = (10, 255, 255)

DETECTION_SIZE_THRESHOLD = 0.005


def detect(img_path):
    # read in image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # blur the image
    blurred = cv2.medianBlur(img, ksize=11)

    # extract red and erode
    red_mask_1 = cv2.inRange(blurred, RED_LOWER_1, RED_UPPER_1)
    red_mask_2 = cv2.inRange(blurred, RED_LOWER_2, RED_UPPER_2)

    red_mask = red_mask_1 ^ red_mask_2
    red_mask = cv2.erode(red_mask, np.ones((3, 3)))

    # get largest red contour
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No red contours found")
        return None

    largest = max(contours, key=cv2.contourArea)

    # return if the largest contour is too small
    height, width = img.shape[:2]
    if cv2.contourArea(largest) < DETECTION_SIZE_THRESHOLD * height * width:
        print("Contour Area too small")
        return None

    # get center
    target_mask = cv2.drawContours(np.zeros(img.shape[0:2], dtype=np.uint8),
                                   [largest], -1, 255, -1)

    # calculate mask center using contour moments
    M = cv2.moments(np.uint8(target_mask))
    if M["m00"] == 0:
        return None

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # calculate offsets. note (0, 0) is the top left of the image
    offsetX = cX - width/2
    offsetY = -(cY - height/2)

    return (offsetX, offsetY)

if __name__ == "__main__":
    print(detect("../images/image.jpg"))
