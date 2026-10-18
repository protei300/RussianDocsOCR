"""Quality checks: glare, blur, print and screen spoofing."""
from concurrent.futures import ThreadPoolExecutor
from time import time


class QualityStage:
    """Pipeline methods: quality checks: glare, blur, print and screen spoofing.

    A part of Pipeline, not usable on its own: the methods read the
    models and the results that Pipeline.__init__ and process_img set up.
    """

    def _glare(self, img):
        """Check for glare quality"""
        qual, coef = self.glare.predict(img)[self.glare.model_name]
        self.results._meta_results['Quality']['Glare'] = qual
        return qual

    def _blur(self, img):
        """Check for blur quality."""
        qual, coef = self.blur.predict(img)[self.blur.model_name]
        self.results._meta_results['Quality']['Blur'] = qual
        return qual

    def _print_spoofing(self, img):
        """Check for print spoofing."""
        qual, coef = self.print_spoofing.predict(img)[self.print_spoofing.model_name]
        self.results._meta_results['Quality']['PrintSpoofing'] = qual
        return qual

    def _lcd_spoofing(self, img):
        """Check for LCD spoofing."""
        qual, coef = self.lcd_spoofing.predict(img)[self.lcd_spoofing.model_name]
        self.results._meta_results['Quality']['LCDSpoofing'] = qual
        return qual

    def _quality_and_borders_parallel(self, img):
        """Run the 4 quality checks and border detection concurrently.

        All five take the same (rotated) image and are mutually independent
        (different models/sessions, no shared mutable state in the
        preprocessing/inference/postprocessing layers - verified by code
        review, see docs/progress-log.md). Only used when low_quality=True:
        with low_quality=False the quality verdict must be known BEFORE
        deciding whether to run the heavier border detector at all, which
        needs the original sequential order (see process_img).

        Every `self.results` write happens here, on the main thread, after
        gathering all futures - concurrent writes into the shared
        meta_results dict (particularly _doc_detector's top-level `|` rebind)
        are not safe to do from worker threads.
        """
        doc_type = (self.results.doctype or '')
        is_spread = 'intpassport' in doc_type.lower()
        max_pages = 2 if is_spread else 1

        def timed(fn, *args, **kwargs):
            t0 = time()
            r = fn(*args, **kwargs)
            return r, round(time() - t0, 4)

        group_start = time()
        with ThreadPoolExecutor(max_workers=5) as ex:
            f_glare = ex.submit(timed, self.glare.predict, img)
            f_blur = ex.submit(timed, self.blur.predict, img)
            f_print = ex.submit(timed, self.print_spoofing.predict, img)
            f_lcd = ex.submit(timed, self.lcd_spoofing.predict, img)
            f_det = ex.submit(timed, self.doc_detector.predict_transform, img,
                              stack='auto', max_pages=max_pages)

            glare_res, t_glare = f_glare.result()
            blur_res, t_blur = f_blur.result()
            print_res, t_print = f_print.result()
            lcd_res, t_lcd = f_lcd.result()
            det_res, t_det = f_det.result()
        group_elapsed = time() - group_start

        self.results._meta_results['Quality']['Glare'] = glare_res[self.glare.model_name][0]
        self.results._meta_results['Quality']['Blur'] = blur_res[self.blur.model_name][0]
        self.results._meta_results['Quality']['PrintSpoofing'] = print_res[self.print_spoofing.model_name][0]
        self.results._meta_results['Quality']['LCDSpoofing'] = lcd_res[self.lcd_spoofing.model_name][0]
        self.results._meta_results = self.results._meta_results | det_res

        # these five overlap in time, so only the group's own elapsed time may
        # count towards timings['total'] - the per-stage numbers stay for detail
        self.results.add_concurrent_group(
            '_quality_and_borders', group_elapsed,
            {
                '_glare': t_glare, '_blur': t_blur, '_print_spoofing': t_print,
                '_lcd_spoofing': t_lcd, '_doc_detector': t_det,
            },
        )
