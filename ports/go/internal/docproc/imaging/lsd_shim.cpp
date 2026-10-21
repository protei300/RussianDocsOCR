#include "lsd_shim.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

extern "C" int rdocs_lsd_detect(void* mat, float** out) {
    *out = nullptr;
    try {
        cv::Mat* gray = static_cast<cv::Mat*>(mat);
        // A detector per call: nothing is shared between goroutines, and construction is cheap
        // next to the detection itself.
        cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
        std::vector<cv::Vec4f> lines;
        lsd->detect(*gray, lines);
        const int n = static_cast<int>(lines.size());
        if (n == 0) {
            return 0;
        }
        float* buf = static_cast<float*>(std::malloc(sizeof(float) * 4 * lines.size()));
        if (buf == nullptr) {
            return -1;
        }
        for (int i = 0; i < n; i++) {
            std::memcpy(buf + 4 * i, lines[i].val, sizeof(float) * 4);
        }
        *out = buf;
        return n;
    } catch (const cv::Exception&) {
        return -1;
    }
}

extern "C" void rdocs_lsd_free(float* p) { std::free(p); }

// The two OpenCV calls of the oriented-box path that gocv does not bind the way the reference
// calls them. Both run the same library code the Python wheel runs.

extern "C" int rdocs_get_rect_sub_pix(void* src, int w, int h, double cx, double cy, void* dst) {
    try {
        cv::Mat* in = static_cast<cv::Mat*>(src);
        cv::Mat* out = static_cast<cv::Mat*>(dst);
        // The centre is a Point2f in cv2 as well: the double is narrowed to float here.
        cv::getRectSubPix(*in, cv::Size(w, h),
                          cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)), *out);
        return 0;
    } catch (const cv::Exception&) {
        return -1;
    }
}

extern "C" double rdocs_rotated_intersection_area(const float* a, const float* b) {
    try {
        cv::RotatedRect ra(cv::Point2f(a[0], a[1]), cv::Size2f(a[2], a[3]), a[4]);
        cv::RotatedRect rb(cv::Point2f(b[0], b[1]), cv::Size2f(b[2], b[3]), b[4]);
        std::vector<cv::Point2f> region;
        int kind = cv::rotatedRectangleIntersection(ra, rb, region);
        if (kind == cv::INTERSECT_NONE || region.empty()) {
            return 0.0;
        }
        return cv::contourArea(region);
    } catch (const cv::Exception&) {
        return -1.0;
    }
}

// In-memory recognition: see rdocs_imdecode_rgb_scrubbed in lsd_shim.h. The flag is the one
// cv2.imdecode(..., IMREAD_COLOR) uses, so the pixels are the ones DecodeRGB produces.
extern "C" int rdocs_imdecode_rgb_scrubbed(const unsigned char* data, int len, void* dst) {
    cv::Mat bgr;
    try {
        // A header over the caller's bytes: imdecode reads them where they are.
        const cv::Mat encoded(1, len, CV_8UC1, const_cast<unsigned char*>(data));
        bgr = cv::imdecode(encoded, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            return 1;
        }
        cv::cvtColor(bgr, *static_cast<cv::Mat*>(dst), cv::COLOR_BGR2RGB);
        bgr.setTo(cv::Scalar::all(0));
        return 0;
    } catch (const cv::Exception&) {
        if (!bgr.empty()) {
            bgr.setTo(cv::Scalar::all(0));
        }
        return -1;
    }
}
