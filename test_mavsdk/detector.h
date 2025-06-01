// red_dot_detector.h
#pragma once

#include <opencv2/opencv.hpp>
#include <optional>

/// Detects the largest red contour in `bgrImage` and computes its offset
/// from the image center.  Offsets are in pixels, with +X rightwards and
/// +Y upwards.  Returns std::nullopt if no valid red dot is found.
std::optional<cv::Point2d> get_offset(const cv::Mat& bgrImage);
