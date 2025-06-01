#ifndef CAMERA_H
#define CAMERA_H

#include <string>
#include <opencv2/opencv.hpp>

/// Pull a single H.264 frame from the UDP stream on port 5600,
/// decode it, and save it to the given filesystem path (e.g. "/tmp/img.png").
/// Returns true on success, false on any failure.
cv::Mat take_photo();

#endif // UDP_FRAME_SAVER_H
