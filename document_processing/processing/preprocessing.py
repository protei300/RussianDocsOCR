import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple


class BasePreprocessing(object):
    """Base class for image preprocessing.

    Defines common preprocessing operations like loading,
    normalization and padding. Child classes implement
    the full preprocessing pipeline.

    Attributes:
        image_size (tuple): Target size for input images
        normalization (tuple): Mean and std dev for normalization
        padding_size (tuple): Padding to add around images
        padding_color (tuple): RGB color for padding
        verbose (bool): Print logging messages if True

    """
    def __init__(self,
                 image_size=(224, 224, 3),
                 normalization=(0, 1),
                 padding_size=(0,0),
                 padding_color = (114,114,114),
                 verbose=False):
        """Initializes base preprocessing with parameters."""
        self.image_size = image_size
        self.normalization = normalization
        self.padding_size = padding_size
        self.padding_color = padding_color

        if verbose:
            print(f'[+] {self.__class__.__name__} loaded')


    def __call__(self, image_path: Union[Path, str, np.ndarray]):
        """Loads image from file path or NumPy array.

        Args:
            image_path: Path to image or NumPy array

        Returns:
            Loaded RGB image array
        """
        if isinstance(image_path, Path) or isinstance(image_path, str):
            image = cv2.imread(image_path.as_posix() if isinstance(image_path, Path) else image_path)
            if image is None:
                raise Exception(f"Not an image {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_path, np.ndarray):
            image = image_path
        else:
            raise Exception("Unsupported Image Type")

        return image
    def padding(self, img: np.array):
        """Adds padding of padding_size with padding_color.

        Args:
            img (np.array): Input image

        Returns:
            img (np.array): Image with added padding
            pad (tuple): Dimensions of added padding
        """
        pad_h, pad_v = self.padding_size
        img = cv2.copyMakeBorder(img, pad_v//2, pad_v//2 ,pad_h//2, pad_h//2, cv2.BORDER_CONSTANT, value=self.padding_color)
        return img, (pad_h//2, pad_v//2)

    def normalization(self, img: np.array, normalization: tuple):
        """Normalizes image by mean and standard deviation.

        Applies normalization using parameters in normalization.

        Args:
            img (np.array): Input image

        Returns:
            np.array: Normalized image
        """
        mean, stdev = normalization
        img = img / stdev - mean
        return img


class ClassificationPreprocessing(BasePreprocessing):
    """Preprocessing for image classification models.

    Performs padding, resizing and batching operations to prepare
    images for a classification model.

    """
    def __init__(self,
                 image_size=(224, 224, 3),
                 normalization=(0,1),
                 padding_size=(0,0),
                 padding_color=(0,0,0),
                 verbose=False):
        """Initializes preprocessing for image classification."""
        super().__init__(image_size=image_size,
                         normalization=normalization,
                         padding_size=padding_size,
                         padding_color=padding_color,
                         verbose=verbose)


        # print(f'[+] {self.__class__.__name__} loaded')

    def __call__(self, image_path: Union[Path, str, np.ndarray]):
        """Runs classification preprocessing pipeline.

        1. Loads image using base class call() method
        2. Applies padding
        3. Resizes image to model input shape
        4. Adds batch dimension if needed

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image tensor
        """
        image = super().__call__(image_path)

        image, _ = self.padding(image)

        image = cv2.resize(image, self.image_size[:2])
        if len(image.shape) == 3:
            image = np.expand_dims(image,0)

        return image




class YoloPreprocessing(BasePreprocessing):
    """Preprocessing for YOLO models.

    Performs padding, resizing and formatting operations needed to
    prepare images for YOLO model input.

    """
    def __init__(self,
                 image_size=(640, 640, 3),
                 normalization = (0,1),
                 padding_size=(0,0),
                 padding_color=(114,114,114),
                 verbose=False
                 ):
        """Initializes YOLO preprocessing."""

        super().__init__(image_size=image_size,
                         normalization=normalization,
                         padding_size=padding_size,
                         padding_color=padding_color,
                         verbose=verbose)
        # print(f'[+] {self.__class__.__name__} loaded')


    def __call__(self, image_path: Union[Path, str, np.ndarray]):
        """Runs YOLO image preprocessing pipeline.

        1. Loads image
        2. Applies padding
        3. Resizes using letterbox to keep aspect ratio
        4. Formats image channels and adds batch dim

        Returns:
           Image tensor, padding ratios, other metadata
        """

        image = super().__call__(image_path)

        # YOLO works with BGR color format, so need swap
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image, pad_add_extra = self.padding(image) #extra_padding is padding surrounding image

        padded_image_shape = image.shape[:2] # shape of image with extra padding




        #pad_add_to_size - padding added after resize to fill till size we need
        image, pad_ratio, pad_add_to_size = self.__letterbox(image,
                                                          new_shape=self.image_size,
                                                          color=self.padding_color,
                                                          auto=False,
                                                          scaleFill=False,
                                                          scaleup=True,
                                                          stride=32)



        if len(image.shape) == 3:
            image = np.expand_dims(image,0)

        return image, pad_ratio, pad_add_extra, pad_add_to_size, padded_image_shape


    def __letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """Letterbox resize keeping aspect ratio.

        Args:
            img: Input image
            new_shape: Size to resize to
            color: Padding RGB color
            auto: Make minimum rectangle as possible
            scaleFill: should we use stretch
            scaleup: IF needed should we use scaleup
            stride: size, of squares, default 32

        Returns:
            Padded and resized image, pad ratios, pad sizes
        """

        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)


class OCRPreprocessing(BasePreprocessing):
    """Preprocessing for OCR models.

    Handles padding and resizing for OCR input images.

    """
    def __init__(self,
                 image_size=(31, 200, 1),
                 normalization=(0,1),
                 padding_size=(0,0),
                 padding_color=(0,0,0),
                 verbose=False):
        """Initializes OCR preprocessing."""

        super().__init__(image_size=image_size,
                         normalization=normalization,
                         padding_size=padding_size,
                         padding_color=padding_color,
                         verbose=verbose)

    def __call__(self, image_path: np.ndarray):
        """Runs OCR preprocessing pipeline.

        1. Loads image
        2. Applies padding
        3. Returns processed image

        Args:
           image_path: Path to input image

        Returns:
           Preprocessed image
        """

        image = super().__call__(image_path)

        image = self.padding(image)

        return image

    @staticmethod
    def recalc_image(original_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Recalculates image shape for padding.

        Computes new height and width to preserve aspect
        ratio for padding.

        Args:
            original_shape: Original image shape

        Returns:
            new_h, new_w: Target height and width
        """

        target_h, target_w = [31, 200]
        orig_h, orig_w = original_shape
        new_h = target_h
        ratio = new_h / float(orig_h)
        new_w = int(orig_w * ratio)
        # для длинных лоскутов подгоняем высоту
        if new_w > target_w:
            new_w = target_w
            r = new_w / float(orig_w)
            new_h = int(orig_h * r)
        return new_h, new_w

    def padding(self, image: np.ndarray) -> np.ndarray:
        """Applies padding to resize and fit OCR input.

        Args:
            image: Input image

        Returns:
            Padded image
        """

        target_shape = [31, 200]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(image, self.recalc_image(image.shape)[::-1])
        orig_h, orig_w = resized.shape
        target_h, target_w = target_shape

        color_value = int(image[-1][-1])
        x_offset = 0
        y_offset = 0

        padded = cv2.copyMakeBorder(resized, y_offset, target_h - orig_h - y_offset, x_offset,
                                    target_w - orig_w - x_offset,
                                    borderType=0, value=[color_value, color_value])

        return padded
