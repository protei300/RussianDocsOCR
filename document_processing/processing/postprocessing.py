import string
import numpy as np
from pathlib import Path
import cv2


class BasePostprocessing:
    """Base class for postprocessing steps.

    This class provides a common interface for postprocessing
    steps in a machine learning pipeline. Child classes should
    implement the actual postprocessing logic in the call() method.

    Attributes:
        verbose (bool): Whether to print logging messages during init.
        Defaults to False.

    """

    def __init__(self, verbose: bool = False):
        """Initialize the postprocessing object.

        Args:
            verbose (bool): Whether to print a loaded message on init.
                            Defaults to False.
        """

        if verbose:
            print(f"[+] {self.__class__.__name__} loaded")

    def __call__(self, *args, **kwargs):
        """Runs postprocess logic on provided arguments.

        This method should be implemented by child classes to
        execute the actual postprocessing logic.
        """
        pass


class BinaryClassPostprocessing(BasePostprocessing):
    """Postprocessing for binary classification models.

    Attributes:
        labels (list): Class labels used for model
        threshold (float): Threshold for positive classification.
                           Defaults to 0.5.

    """
    def __init__(self, labels: list, threshold=0.5, verbose=False):
        """Initializes postprocessing object.

        Args:
            labels (list): Class labels
            threshold (float): Threshold for positive classification.
                               Defaults to 0.5.
            verbose (bool): Prints message on init if True.
                            Defaults to False.
        """
        super().__init__(verbose)
        self.labels=labels
        self.threshold = threshold
    def __call__(self, probability: np.ndarray, **kwargs):
        """Transforms network output to class label and confidence.

        Args:
            probability (np.ndarray): Output probability from network

        Returns:
            str: Class label
            float: Confidence probability
        """
        return self.labels[0 if probability[0] < self.threshold else 1], probability[0]


class OCRPostprocessing(BasePostprocessing):
    """Postprocessing for OCR models.

    Transforms model output to decoded text string.
    Supports English and Russian languages.

    Attributes:
        lang (str): Language to use for decoding text.
                    Must be either 'eng' or 'rus'.

    """

    def __init__(self, lang: str, verbose=False):
        """Initializes the OCR postprocessing.

        Args:
            lang (str): Language to decode text into.
                        Must be either 'eng' or 'rus'.
            verbose (bool): Whether to print logging message.
                            Defaults to False.
        """


        super().__init__(verbose)
        assert lang in ['eng', 'rus']
        self.lang = lang

    def __call__(self, output_value: np.ndarray) -> str:
        """Transforms network output values into text string.

        Args:
            output_value (np.ndarray): Output values from network

        Returns:
            str: Decoded text string
        """

        labels = string.digits + string.ascii_uppercase + "%'(),-./:*#" if self.lang == 'eng' \
            else "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.-"


        decoded = ''.join([labels[x] for x in output_value if x > -1 ])

        return decoded


class MetricPostprocessing(BasePostprocessing):
    '''
    Postprocessing for Metric learning. Accept both Euclidean distance and Cosine Distance
    '''
    def __init__(self, centers: Path, metric:str, verbose=False):
        '''
        :param centers: coordinates for counted centers
        :param metric: type of metric to use, can be either 'cosine' or 'euclidean'
        '''
        import pandas as pd

        super().__init__(verbose)
        self.metric = metric
        if self.metric == 'cosine':
            self.radius = 1
        elif self.metric == 'euclidean':
            self.radius = 10
        else:
            raise Exception('Unsupported metric type')
        self.centers = pd.read_pickle(centers)

    def __call__(self, vector, **kwargs):
        '''
        :param vector: embeding from feature vector layer
        :return: tuple of predicted label, distance till center of predicted label, and threshold of that label
        '''
        from sklearn.neighbors import NearestNeighbors

        center_coords = np.stack(self.centers['centers'].values)
        nn = NearestNeighbors(n_neighbors=1, metric=self.metric, radius=self.radius).fit(center_coords)

        vector = vector.reshape(1,-1)

        pred = nn.radius_neighbors(vector, return_distance=True, sort_results=True)
        pred_dist = pred[0][0][0]
        pred_label = self.centers.index[pred[1][0][0]]
        threshold = self.centers.distance[pred_label]
        if pred_dist < threshold:
            return pred_label, pred_dist, threshold
        else:
            return 'NONE', pred_dist, threshold


class MultiClassPostprocessing(BasePostprocessing):
    """Postprocessing for metric learning models.

    Supports both euclidean and cosine distance metrics.
    Classifies input vector against loaded centroid coordinates.

    Attributes:
        centers (Path): File path to serialized centroids
        metric (str): Distance metric to use.
                      Must be either 'euclidean' or 'cosine'.

    """
    def __init__(self, labels: list, verbose=False):
        """Initializes metric postprocessing.

        Args:
            centers (Path): File path to serialized centroids
            metric (str): Distance metric to use. Must be
                          'euclidean' or 'cosine'.
            verbose (bool): Print logging messages if True.
                            Defaults to False.
        """
        super().__init__(verbose)
        self.labels = labels

    def __call__(self, probability: np.ndarray, **kwargs):
        """Classifies vector against loaded centroids.

        Args:
            vector: Input vector from model

        Returns:
            tuple: Label, distance, threshold
        """
        return self.labels[probability.argmax()], probability.max(initial = 0)


class YoloDetectorPostprocessing(BasePostprocessing):
    """
    Class for postprocessing YOLO detector predictions
    and preparing detection results

    Attributes:
        iou (float): IOU threshold for NMS
        cls (float): confidence threshold
        labels (list): list of class labels
        verbose (bool): enable verbose output
    """
    def __init__(self, labels:list, iou=0.2, cls=0.5, verbose=False):
        """
        Initialize class parameters
        Args:
            iou (float): IOU threshold for NMS
            cls (float): confidence threshold
            labels (list): list of class labels
            verbose (bool): enable verbose output
        """
        super().__init__(verbose)
        self.iou = iou
        self.cls = cls
        self.labels = labels


    def __call__(self,
                 vector: np.ndarray,
                 **kwargs,
                 ):
        """
        Run postprocessing on YOLO predictions
        Args:
            vector (np.ndarray): YOLO predictions
            **kwargs: additional metadata:
                prediction: Tensor from yolo exit
                imgsize: size of image, h,w
                conf_thresh: class confidence threshold
                iou_thresh:  intersection over union threshold
        Returns:
            list: detection results
        """

        if kwargs.get('padding_meta') is None:
            padding_meta = {
                'pad_to_size': (0,0),
                'pad_extra': (0,0),
                'ratio': (1, 1),

            }
        else:
            padding_meta = kwargs['padding_meta']

        detect_res = []
        if vector.shape[0] == 0:
            return detect_res

        n_classes = len(self.labels)
        vector = vector[vector[..., 4:4+n_classes].max(axis=1) > self.cls]

        if len(vector) == 0:
            return []

        box, det, seg = np.split(vector, [4, 4+n_classes], axis=1)
        box = self.xywh2xyxy(box)  # creating from x,y height, width -> xy xy coords of box

        conf, j = det.max(axis=1, keepdims=True), det.argmax(axis=1, keepdims=True)
        i = self.nms(box, conf, self.iou)  # calculating non maximum suppression
        detect_res = np.concatenate((box, conf, j, seg), axis=1)[i]


        ind = np.lexsort((detect_res[...,0], detect_res[..., 1]))
        detect_res = detect_res[ind]

        if kwargs.get('resize'): # should we resize to original size
        ### resize to original pic
            detect_res[..., :4:2] = (detect_res[..., :4:2] - padding_meta['pad_to_size'][0]) \
                                    / padding_meta['ratio'][0] - padding_meta['pad_extra'][0]
            detect_res[..., 1:4:2] = (detect_res[..., 1:4:2] - padding_meta['pad_to_size'][1]) \
                                     / padding_meta['ratio'][1] - padding_meta['pad_extra'][1]
            detect_res[detect_res < 0] = 0
            detect_res[..., :4] = np.round(detect_res[:, :4], 0)
            detect_res[..., 4] = np.round(detect_res[:, 4], 3)

        if kwargs.get('numpy') is not None:  # Return result as numpy array without adding labels
            return detect_res

        if self.labels is not None:
            detection_result = []
            for detected in detect_res:
                obj_detected = list(detected)
                obj_detected[:4] = list(map(lambda x: int(x), obj_detected[:4]))
                obj_detected[5] = int(obj_detected[5])
                try:
                    obj_detected.append(self.labels[obj_detected[5]])
                except:
                    obj_detected.append("Unsupported class")

                detection_result.append(obj_detected)
            return detection_result
        else:
            return detect_res.tolist()

    @staticmethod
    def xywh2xyxy(x):
        """
        Convert bounding boxes from XYWH to XYXY format
        Args:
            x (np.ndarray): bounding boxes in XYWH format
        Returns:
            np.ndarray: bounding boxes in XYXY format
        """
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod
    def nms(bounding_boxes: np.array, confidence_score: np.array, threshold: float):
        """
        Apply non-maximum suppression to remove overlapping boxes
        Args:
            bounding_boxes (np.ndarray): bounding box coordinates
            confidence_score (np.ndarray): confidence scores
            threshold (float): IOU threshold
        Returns:
            list: indices of kept bounding boxes
        """

        # If no bounding boxes, return empty list
        if len(bounding_boxes) == 0:
            return [], []

        # Bounding boxes
        boxes = np.array(bounding_boxes)

        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Confidence scores of bounding boxes
        score = np.array(confidence_score)

        # Picked bounding boxes
        picked_boxes_index = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x) * (end_y - start_y)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score, axis=0).reshape(-1)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes_index.append(index)


            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])




            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1)
            h = np.maximum(0.0, y2 - y1)

            intersection = w * h


            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < threshold)
            order = order[left]

        return picked_boxes_index


class YoloSegmentorPostprocessing(BasePostprocessing):
    """Postprocessing for YOLO segmentation models.

    Transforms raw masks into final masks and segments.

    Attributes:
        mask_filter (float): Filtering threshold to apply to masks.

    """

    def __init__(self, mask_filter: float = 0.8, verbose=False):
        """Initializes YOLO segmentation postprocessing.

        Args:
            mask_filter (float): Filtering threshold for masks.
            verbose (bool): Print message on init if True.

        """
        super().__init__(verbose)
        assert 0<=mask_filter<=1, "[!] Filter must be within [0..1]"
        self.mask_filter = mask_filter


    def __call__(self, proto_masks: np.array, masks: np.array, bbox: np.array, extra_padding:tuple, img_shape: list, upsample=True):
        """Transforms raw YOLO masks into final masks & segments.

        Args:
            Various args for processing masks

        Returns:
            Processed masks, generated segments

        """
        imh, imw, chn = proto_masks.shape
        im_orig_h, im_orig_w = img_shape

        masks_matmul = masks @ proto_masks.transpose([-1, 0, 1]).reshape((chn,-1))
        masks = self.sigmoid(masks_matmul).reshape((-1, imh, imw))


        masks_resized = np.zeros((masks.shape[0],im_orig_h, im_orig_w))


        # removing padding
        gain = min(imh / im_orig_h, imw / im_orig_w)  # gain  = old / new
        pad = (imw - im_orig_w * gain) / 2, (imh - im_orig_h * gain) / 2  # wh padding
        top, left = int(pad[1]), int(pad[0])  # y, x
        bottom, right = int(imh - pad[1]), int(imw - pad[0])
        masks = masks[:, top:bottom, left:right]

        for i, mask in enumerate(masks):
            masks_resized[i] = cv2.resize(mask, (im_orig_w, im_orig_h),  cv2.INTER_LINEAR)

        top_extra_padding, left_extra_padding = extra_padding[1], extra_padding[0]
        bottom_extra_padding, right_extra_padding = masks_resized.shape[1]-extra_padding[1], masks_resized.shape[2]-extra_padding[0]
        masks_resized = masks_resized[:, top_extra_padding: bottom_extra_padding, left_extra_padding:right_extra_padding] #removing extra padding

        masks_resized = self.clip_boxes(masks_resized, bbox)

        masks_resized = np.where(masks_resized>self.mask_filter, 1, 0) * 255

        segments = self.get_segments(masks_resized)

        return masks_resized, segments


    @staticmethod
    def get_segments(masks):
        """Generates segments from input masks.

        Args:
            masks (np.ndarray): Array of input masks

        Returns:
            List of generated segments
        """
        segments=[]
        for x in masks.astype('uint8'):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype('float32'))
        return segments

    @staticmethod
    def clip_boxes(masks, boxes):
        """Clips masks to fit within given bounding boxes.

        Removes mask parts outside of the provided boxes.

        Args:
           masks (np.ndarray): Input masks
           boxes (np.ndarray): Bounding box coordinates

        Returns:
           Clipped masks
        """
        n, h, w = masks.shape
        # print(boxes)
        x1, y1, x2, y2 = np.split(boxes, 4, axis=1)
        r = np.arange(w)
        r_mask = (r>x1) * (r<x2)
        r_mask = np.expand_dims(r_mask,1)
        c = np.arange(h)
        c_mask = (c>y1) * (c<y2)
        c_mask = np.expand_dims(c_mask, 2)
        # print(r_mask.shape, c_mask.shape)
        masks = masks * (c_mask * r_mask)



        return masks



    @staticmethod
    def sigmoid(x:np.array) -> np.array:
        """Applies sigmoid function.

        Args:
            x (np.ndarray): Input vector

        Returns:
            Vector after applying sigmoid
        """
        return 1 / (1 + np.exp(-x))
