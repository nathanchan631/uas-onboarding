#include "camera.h"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat take_photo()
{
    // 1) Initialize GStreamer (only once per process)
    static bool gst_inited = false;
    if (!gst_inited) {
        gst_init(nullptr, nullptr);
        gst_inited = true;
    }

    // 2) Build pipeline
    GstElement *pipeline   = gst_pipeline_new("udp-to-cv-saver");
    GstElement *src        = gst_element_factory_make("udpsrc",       "src");
    GstElement *capsfilter = gst_element_factory_make("capsfilter",   "caps");
    GstElement *depay      = gst_element_factory_make("rtph264depay", "depay");
    GstElement *parse      = gst_element_factory_make("h264parse",    "parse");
    GstElement *dec        = gst_element_factory_make("avdec_h264",   "dec");
    GstElement *convert    = gst_element_factory_make("videoconvert","convert");
    GstElement *rgbcaps    = gst_element_factory_make("capsfilter",   "rgbcaps");
    GstElement *sink_el    = gst_element_factory_make("appsink",      "sink");

    if (!pipeline || !src || !capsfilter || !depay || !parse ||
        !dec || !convert || !rgbcaps || !sink_el) {
        std::cerr << "[saveDroneFrame] Failed to create GStreamer elements\n";
        exit(1);
    }

    // 3) Configure elements
    g_object_set(src, "port", 5600, NULL);

    {
        auto *c = gst_caps_from_string("application/x-rtp, payload=96");
        g_object_set(capsfilter, "caps", c, NULL);
        gst_caps_unref(c);
    }
    {
        auto *c = gst_caps_from_string("video/x-raw,format=RGB");
        g_object_set(rgbcaps, "caps", c, NULL);
        gst_caps_unref(c);
    }
    // We will pull manually, so disable signals
    g_object_set(sink_el,
                 "emit-signals", FALSE,
                 "sync",         FALSE,
                 "max-buffers",  1,
                 "drop",         TRUE,
                 NULL);

    // 4) Wire up
    gst_bin_add_many(GST_BIN(pipeline),
                     src, capsfilter, depay, parse,
                     dec, convert, rgbcaps, sink_el, NULL);
    if (!gst_element_link_many(src, capsfilter, depay, parse,
                               dec, convert, rgbcaps, sink_el, NULL))
    {
        std::cerr << "[saveDroneFrame] Link error\n";
        gst_object_unref(pipeline);
        exit(1);
    }

    // 5) Start
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // 6) Pull one frame (with a short timeout)
    GstAppSink *appsink = GST_APP_SINK(sink_el);
    GstSample *sample = gst_app_sink_try_pull_sample(appsink, 500 * GST_MSECOND);
    if (!sample) {
        std::cerr << "[saveDroneFrame] No sample received\n";
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        exit(1);
    }

    // 7) Convert buffer → OpenCV Mat
    GstBuffer *buf = gst_sample_get_buffer(sample);
    GstMapInfo map;
    gst_buffer_map(buf, &map, GST_MAP_READ);

    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *str = gst_caps_get_structure(caps, 0);
    int width = 0, height = 0;
    gst_structure_get_int(str, "width", &width);
    gst_structure_get_int(str, "height", &height);

    cv::Mat rgb(height, width, CV_8UC3, (guchar*)map.data, cv::Mat::AUTO_STEP);
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

    gst_buffer_unmap(buf, &map);
    gst_sample_unref(sample);

    // 8) Tear down
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return bgr;
}
