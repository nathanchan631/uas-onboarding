#include "detector.h"

// HSV ranges for red (two halves of the hue wheel)
static const cv::Scalar RED_LOWER1(165, 10, 0);
static const cv::Scalar RED_UPPER1(180,255,255);
static const cv::Scalar RED_LOWER2(  0, 10, 0);
static const cv::Scalar RED_UPPER2( 10,255,255);

// Minimum fraction of image area for a valid detection
static constexpr double DETECTION_SIZE_THRESHOLD = 0.00005;

std::optional<cv::Point2d> get_offset(const cv::Mat& bgrImage)
{
    if (bgrImage.empty()) {
        return std::nullopt;
    }

    // 1. Convert to HSV and blur
    cv::Mat hsv, blurred;
    cv::cvtColor(bgrImage, hsv, cv::COLOR_BGR2HSV);
    cv::medianBlur(hsv, blurred, 11);

    // 2. Threshold for red in two hue ranges
    cv::Mat mask1, mask2;
    cv::inRange(blurred, RED_LOWER1, RED_UPPER1, mask1);
    cv::inRange(blurred, RED_LOWER2, RED_UPPER2, mask2);
    cv::Mat redMask = mask1 ^ mask2;

    // 3. Erode to clean up noise
    cv::erode(redMask, redMask, cv::Mat::ones(3,3,CV_8U));

    // 4. Find all external contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(redMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return std::nullopt;
    }

    // 5. Pick the largest contour by area
    auto largest = std::max_element(
        contours.begin(), contours.end(),
        [](auto& a, auto& b){ return cv::contourArea(a) < cv::contourArea(b); }
    );
    double area = cv::contourArea(*largest);
    int H = bgrImage.rows, W = bgrImage.cols;
    if (area < DETECTION_SIZE_THRESHOLD * H * W) {
        // too small to be a real detection
        return std::nullopt;
    }

    // 6. Compute centroid via moments
    cv::Moments M = cv::moments(*largest);
    if (M.m00 == 0) {
        return std::nullopt;
    }
    double cX = M.m10 / M.m00;
    double cY = M.m01 / M.m00;

    // 7. Offsets: +X to the right, +Y upwards
    double offsetX =  cX - (W  / 2.0);
    double offsetY = -(cY - (H  / 2.0));

    return cv::Point2d{offsetX, offsetY};
}

