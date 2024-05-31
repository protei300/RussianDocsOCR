import pathlib
import cv2
import time
import numpy as np


class QualityChecker(object):
    """Checks image quality by analyzing blur amount.

    Attributes:
        model: Blur detection ML model
        canvas_size: Dimensions for image patch sampling
        colors: Set of colors used when annotating

    """

    def __init__(self, init_model, init_canvas_size):
        """Initializes with blur model and canvas size."""
        self.model = init_model
        # canvas size in blocks
        self.canvas_size = init_canvas_size
        # window size must be 128
        self.window_size = 128
        # colors for drawing
        self.colors = {'c_blue': (260, 80, 80), 'c_white': (255, 255, 255), 'c_gray': (160, 160, 160), 'c_gray_f': (96,
                                                                                                                    96,
                                                                                                                    96),
                       'c_red': (255, 0, 0), 'c_redish': (255, 153, 153), 'c_yellow': (255, 255, 0),
                       'c_green': (173, 255, 47)}
        # list of (class, coordinates, confidence)
        self.quality_result_list = []
        # drawn image
        self.tested_image = np.ndarray([])

    def perform_image(self, image):
        """Analyzes image patches and detects blur.

        Samples patches based on canvas_size, runs model
        inference and populates quality_result_list.

        Args:
           image: Input document image

        """
        self.quality_result_list = []
        canvas_in_pixels = tuple(map(lambda x: x * self.window_size, self.canvas_size))
        self.tested_image = cv2.cvtColor(cv2.resize(image, canvas_in_pixels), cv2.COLOR_BGR2RGB)
        # self.tested_image = cv2.resize(image, canvas_in_pixels)

        for x_step in range(self.canvas_size[0]):
            for y_step in range(self.canvas_size[1]):
                x = self.window_size * x_step
                y = self.window_size * y_step
                frame_image = self.tested_image[y:y + self.window_size, x:x + self.window_size]
                result = self.model.predict(frame_image)
                # print(result)
                result_class = result[0]
                confidence = result[1]
                # class, coordinates, confidence
                self.quality_result_list.append((result_class, ((x, y), (x + self.window_size, y + self.window_size)),
                                                 confidence))

        # print(self.quality_result_list)
        self.tested_image = cv2.cvtColor(self.tested_image, cv2.COLOR_RGB2BGR)
        return True

    def annotate_image(self, image):
        """Annotates image with blur detection results.

        Draws bounding boxes and labels on blurred regions.

        Args:
           image: Input document image

        Returns:
           Annotated version of image
        """
        self.perform_image(image)

        for block in self.quality_result_list:

            cv2.rectangle(self.tested_image, block[1][0], block[1][1], color=self.colors[
                "c_white"], thickness=1)
            result = block[0]
            x = block[1][0][0]
            y = block[1][0][1]

            if result == 'Background':
                cv2.putText(self.tested_image, result, (x + 14, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            self.colors['c_gray']
                            , 2)

            if result == 'Faces':
                cv2.putText(self.tested_image, result, (x + 14, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[
                    "c_gray_f"], 2)

            if result == 'Blur05' or result == 'Blur5':
                cv2.putText(self.tested_image, result, (x + 14, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[
                    "c_redish"], 2)

            if result == 'Blur10':
                cv2.putText(self.tested_image, result, (x + 14, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            self.colors["c_red"]
                            , 2)

            if result == 'Blur0' or result == 'NonBlur':
                cv2.putText(self.tested_image, result, (x + 14, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[
                    "c_white"], 2)

            if result == 'Fingers':
                cv2.putText(self.tested_image, result, (x + 14, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[
                    "c_yellow"], 2)

            if result == 'Glare':
                cv2.putText(self.tested_image, result, (x + 14, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[
                    "c_green"], 2)

        return self.tested_image

    def check_image_quality(self, image):
        """Computes overall quality score based on blur amount.

        Analyzes blur on patches, aggregates scores and
        returns overall document quality metric.

        Args:
           image: Input document image

        Returns:
           Quality score between 0-1
        """
        self.perform_image(image)
        result_list = []
        for block in self.quality_result_list:
            result = block[0]
            if result == 'Blur5':
                result_list.append(0.5)
            if result == 'Blur10':
                result_list.append(1)
            if result == 'NonBlur':
                result_list.append(0)
        max_level_for_normalization = len(result_list)
        quality_level = 0
        for block in result_list:
            quality_level += block
        quality_level = 1 - quality_level / max_level_for_normalization
        return quality_level