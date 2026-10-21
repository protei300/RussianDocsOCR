#ifndef RDOCS_LSD_SHIM_H
#define RDOCS_LSD_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* cv::createLineSegmentDetector(cv::LSD_REFINE_STD)->detect(gray, lines) with OpenCV's own
   defaults - what cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0] runs.

   mat is the cv::Mat* behind a gocv.Mat (CV_8UC1). On success returns the number of segments
   and sets *out to a malloc'd array of 4 floats per segment (x1, y1, x2, y2), which the
   caller frees with rdocs_lsd_free; a negative value is an OpenCV error. */
int rdocs_lsd_detect(void* mat, float** out);
void rdocs_lsd_free(float* p);

/* cv::getRectSubPix(*src, Size(w, h), Point2f(cx, cy), *dst) for the cv::Mat* behind two
   gocv.Mats (CV_8UC3). gocv's own binding takes an INTEGER centre; the reference cuts an
   address line about a fractional one. Returns 0, or a negative value on an OpenCV error. */
int rdocs_get_rect_sub_pix(void* src, int w, int h, double cx, double cy, void* dst);

/* Area of the intersection of two rotated rectangles, each (cx, cy, w, h, angle in DEGREES)
   as five float32 - cv2.contourArea(cv2.rotatedRectangleIntersection(a, b)[1]). 0 when they
   do not meet; a negative value is an OpenCV error. */
double rdocs_rotated_intersection_area(const float* a, const float* b);


/* In-memory recognition (ports/MEMORY.md §5): decode an encoded image to RGB with NO copy of the
   encoded bytes and with the intermediate BGR zeroed before it is released. gocv's IMDecode first
   copies the whole buffer into a std::vector it frees without zeroing - a copy of a passport photo
   left in the native heap; this one wraps the caller's bytes in a cv::Mat header instead.

   data/len are the caller's bytes (not retained); dst is the cv::Mat* behind a gocv.Mat that
   receives CV_8UC3 RGB. Returns 0; 1 when the decoder produced nothing; a negative value on an
   OpenCV error. */
int rdocs_imdecode_rgb_scrubbed(const unsigned char* data, int len, void* dst);

#ifdef __cplusplus
}
#endif

#endif
