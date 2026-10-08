#include "lsd_shim.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#include <opencv2/core.hpp>
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
