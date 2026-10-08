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

#ifdef __cplusplus
}
#endif

#endif
