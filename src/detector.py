import cv2
import json
import redis
import numpy as np

r = redis.Redis(host='redis', port=6379, db=0)

RED_LOWER = (165, 10, 0)
RED_UPPER = (180, 255, 255)
DETECTION_SIZE_THRESHOLD = 0.05

def detect(img_path, telemetry):
    # read in image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # blur the image
    blurred = cv2.medianBlur(img, ksize=11)

    # extract red and erode
    red_mask = cv2.inRange(blurred, RED_LOWER, RED_UPPER)
    red_mask = cv2.erode(red_mask, np.ones((3, 3)))

    # get largest red contour
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No red contours found")
        return
    
    largest = max(contours, key=cv2.contourArea)

    # return if the largest contour is too small
    height, width = img.shape[:2]
    if cv2.contourArea(largest) < DETECTION_SIZE_THRESHOLD * height * width:
        return
    
    # get center
    target_mask = cv2.drawContours(np.zeros(img.shape[0:2], dtype=np.uint8),
                                   [largest], -1, 255, -1)

    # Calculate mask center using contour moments
    M = cv2.moments(np.uint8(target_mask))
    if M["m00"] == 0:
        return
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Calculate offsets
    offsetX = cX - height // 2
    offsetY = cY - width // 2

    # Calculate new GPS coordinate
    lat = telemetry['latitude']
    lon = telemetry['longitude']
    alt = telemetry['altitude']

    