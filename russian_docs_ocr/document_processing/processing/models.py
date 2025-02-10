import os
from pathlib import Path
from .preprocessing import BasePreprocessing, ClassificationPreprocessing, YoloPreprocessing, OCRPreprocessing
from .postprocessing import BasePostprocessing, MetricPostprocessing, OCRPostprocessing, \
    YoloDetectorPostprocessing, YoloSegmentorPostprocessing, MultiClassPostprocessing, BinaryClassPostprocessing
from .inference import ModelInference
import json
from typing import Union, List
import numpy as np


class ModelLoader:
    """
    Class for loading models from a JSON configuration file.

    Attributes:
    - verbose (bool): enables debug logging
    """
    def __init__(self, verbose=False):
        """
        Initializes the ModelLoader object.

        Arguments:
        - verbose (bool): enables debug logging
        """
        self.verbose = verbose

    def __call__(self, json_file: Path, device='gpu'):
        """
        Loads and returns a model based on the JSON config file.

        Arguments:
        - json_file (Path): path to JSON config file
        - device (str): device to load model on (gpu/cpu)

        Returns: loaded model
        """

        self.json_file = json.loads(json_file.read_text(encoding="utf8"))
        self.working_dir = json_file.parent
        self.device = device
        if self.json_file['Type'] == 'Metric':
            model = self.__load_metric_model()
        elif self.json_file['Type'] == 'YoloDetector':
            model = self.__load_yolo_detector()
        elif self.json_file['Type'] == 'YoloSegmentor':
            model = self.__load_yolo_segmentor()
        elif self.json_file['Type'] == 'BinaryClassification':
            model = self.__load_binary_classificator()
        elif self.json_file['Type'] == 'MultiLabelClassification':
            model = self.__load_multi_label_classificator()
        elif self.json_file['Type'] == 'OCR':
            model = self.__load_ocr(self.json_file['Lang'])
        else:
            raise Exception(f"[!] Not supported model type: {self.json_file['Type']}")


        return model


    def __load_metric_model(self):
        """
        Loads a metric processing model from the JSON file.
        """

        preprocessings = []
        for inp_preprocess in self.json_file['Input']:
            preprocessings.append(
                ClassificationPreprocessing(
                    image_size=inp_preprocess['Shape'],
                    padding_size=inp_preprocess['PaddingSize'],
                    padding_color=inp_preprocess['PaddingColor'],
                    normalization=inp_preprocess['Normalization'],
                    verbose=self.verbose,
                )
            )

        model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
                                         device=self.device,
                                         verbose=self.verbose)

        postprocessings = []
        for outp_postprocess in self.json_file['Output']:
            postprocessings.append(
                MetricPostprocessing(
                    self.working_dir.joinpath(str(self.json_file['Centers']).replace('\\', os.sep)),
                    metric=outp_postprocess['Metric'],
                    verbose=self.verbose
                )
            )

        model = ClassificationModel(
            model_type= self.json_file['Type'],
            preprocessing=preprocessings[0],
            model_inference=model_inference,
            postprocessing=postprocessings[0],
        )

        return model

    def __load_binary_classificator(self):
        preprocessings = []
        for inp_preprocess in self.json_file['Input']:
            preprocessings.append(
                ClassificationPreprocessing(
                    image_size=inp_preprocess['Shape'],
                    padding_size=inp_preprocess['PaddingSize'],
                    padding_color=inp_preprocess['PaddingColor'],
                    normalization=inp_preprocess['Normalization'],
                    verbose=self.verbose
                )
            )

        model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
                                         device=self.device,
                                         verbose=self.verbose)

        postprocessings = []
        for _ in self.json_file['Output']:
            postprocessings.append(
                BinaryClassPostprocessing(
                    self.json_file['Labels'],
                    verbose=self.verbose
                )
            )

        model = ClassificationModel(
            model_type=self.json_file['Type'],
            preprocessing=preprocessings[0],
            model_inference=model_inference,
            postprocessing=postprocessings[0],
        )

        return model

    def __load_ocr(self, lang):
        preprocessings = []
        for inp_preprocess in self.json_file['Input']:
            preprocessings.append(
                OCRPreprocessing(
                    image_size=inp_preprocess['Shape'],
                    padding_size=inp_preprocess['PaddingSize'],
                    padding_color=inp_preprocess['PaddingColor'],
                    normalization=inp_preprocess['Normalization'],
                    verbose=self.verbose
                )
            )

        model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
                                         device=self.device,
                                         verbose=self.verbose)

        model = OCRModel(
            model_type=self.json_file['Type'],
            preprocessing=preprocessings[0],
            model_inference=model_inference,
            postprocessing=OCRPostprocessing(lang=lang, verbose=False),
        )

        return model

    def __load_multi_label_classificator(self):
        preprocessings = []
        for inp_preprocess in self.json_file['Input']:
            preprocessings.append(
                ClassificationPreprocessing(
                    image_size=inp_preprocess['Shape'],
                    padding_size=inp_preprocess['PaddingSize'],
                    padding_color=inp_preprocess['PaddingColor'],
                    normalization=inp_preprocess['Normalization'],
                    verbose=self.verbose
                )
            )

        model_inference = ModelInference(
            self.working_dir.joinpath(self.json_file['File']),
            device=self.device,
            verbose=self.verbose
        )

        postprocessings = []
        for _ in self.json_file['Output']:
            postprocessings.append(
                MultiClassPostprocessing(
                    self.json_file['Labels'],
                    verbose=self.verbose
                )
            )

        model = ClassificationModel(
            model_type=self.json_file['Type'],
            preprocessing=preprocessings[0],
            model_inference=model_inference,
            postprocessing=postprocessings[0],
        )

        return model

    def __load_yolo_detector(self):
        preprocessings = []
        for inp_preprocess in self.json_file['Input']:
            preprocessings.append(
                YoloPreprocessing(
                    image_size=inp_preprocess['Shape'],
                    padding_size=inp_preprocess['PaddingSize'],
                    padding_color=inp_preprocess['PaddingColor'],
                    normalization=inp_preprocess['Normalization'],
                    verbose=self.verbose
                )
            )
        model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
                                         device=self.device,
                                         verbose=self.verbose)

        postprocessings = []
        for _ in self.json_file['Output']:
            postprocessings.append(
                YoloDetectorPostprocessing(
                    iou=self.json_file['IOU'],
                    cls=self.json_file['CLS'],
                    labels=self.json_file['Labels'],
                    verbose=self.verbose
                )
            )

        model = YoloDetectorModel(
            model_type=self.json_file['Type'],
            preprocessing=preprocessings[0],
            model_inference=model_inference,
            postprocessing=postprocessings[0],
        )

        return model

    def __load_yolo_segmentor(self):
        preprocessings = []
        for inp_preprocess in self.json_file['Input']:
            preprocessings.append(
                YoloPreprocessing(
                    image_size=inp_preprocess['Shape'],
                    padding_size=inp_preprocess['PaddingSize'],
                    padding_color=inp_preprocess['PaddingColor'],
                    normalization=inp_preprocess['Normalization'],
                    verbose=self.verbose,
                )
            )
        model_inference = ModelInference(
            self.working_dir.joinpath(self.json_file['File']),
            device=self.device,
            verbose=self.verbose
        )

        postprocessings = [
            YoloDetectorPostprocessing(
                iou=self.json_file['IOU'],
                cls=self.json_file['CLS'],
                labels=self.json_file['Labels'],
                verbose=self.verbose
            ),
            YoloSegmentorPostprocessing(self.json_file['MaskFilter'],verbose=self.verbose),
        ]

        model = YoloSegmentorModel(
            model_type=self.json_file['Type'],
            preprocessing=preprocessings[0],
            model_inference=model_inference,
            postprocessing=postprocessings,
        )

        return model

class Model:
    """Model class for making predictions using a configurable pipeline.

    Attributes:
       model_type (str): Type of the model.
       preprocessing (BasePreprocessing): Preprocessing algorithm.
       inference_model (ModelInference): Model to make actual predictions.
       postprocessing (Union[List[BasePostprocessing], BasePostprocessing]):
           Postprocessing algorithm(s) to apply on raw predictions.

    Methods:
       predict: Runs the prediction pipeline with preprocessing,
           model inference and postprocessing.
       predict_fv: Runs prediction pipeline with preprocessing and model
           inference only, without postprocessing.
       model_type: Returns model type.
    """


    def __init__(self,
                 model_type: str,
                 preprocessing: BasePreprocessing,
                 model_inference: ModelInference,
                 postprocessing: Union[List[BasePostprocessing],BasePostprocessing]):
        """Initialize the Model instance.

        Args:
           model_type (str): Type of the model.
           preprocessing (BasePreprocessing): Preprocessing algorithm.
           model_inference (ModelInference): Model to make predictions.
           postprocessing (Union[List[BasePostprocessing], BasePostprocessing]):
               Postprocessing algorithm(s).
        """
        self.__model_type = model_type
        self.preprocessing = preprocessing
        self.inference_model = model_inference
        self.postprocessing = postprocessing


    def predict(self, img: Union[Path, np.ndarray]):
        """Runs prediction pipeline with preprocessing, inference and postprocessing.

        Args:
            img (Union[Path, np.ndarray]): Image to predict on.

        Returns:
            Result of prediction pipeline.
        """
        pass

    def predict_fv(self, img: Union[Path, np.ndarray]):
        """Runs prediction pipeline with preprocessing and inference only.

        Args:
            img (Union[Path, np.ndarray]): Image to predict on.

        Returns:
            Result of inference without postprocessing.
        """
        pass

    @property
    def model_type(self):
        """Returns model type.

        Returns:
            str: Type of the model.
        """
        return self.__model_type

class ClassificationModel(Model):
    """Classification model implementation.

    Inherits from Model class. Implements classification pipeline with
    predict and predict_fv methods.

    Attributes:
        model_type (str): Type of the model. Inherited from Model
        preprocessing (BasePreprocessing): Preprocessing algorithm. Inherited from Model
        inference_model (ModelInference): Model to make predictions. Inherited from Model
        postprocessing (BasePostprocessing): Postprocessing algorithm. Inherited from Model

    Methods:
        predict: Runs classification pipeline with preprocessing,
            model inference and postprocessing.
        predict_fv: Runs classification pipeline with preprocessing
            and model inference only, without postprocessing.

    """
    def __init__(self, model_type:str,  preprocessing: BasePreprocessing, model_inference: ModelInference,
                 postprocessing: BasePostprocessing):
        """Initialize ClassificationModel by inheriting from Model

        Args:
            model_type (str): Type of the model.
            preprocessing (BasePreprocessing): Preprocessing algorithm.
            model_inference (ModelInference): Model to make predictions.
            postprocessing (BasePostprocessing): Postprocessing algorithm.

        """
        super().__init__(model_type, preprocessing, model_inference, postprocessing)

    def predict(self, img: Union[Path, np.ndarray]):
        """Runs classification pipeline with postprocessing.

        Args:
            img (Union[Path, np.ndarray]): Image to predict on

        Returns:
            Result of classification prediction
        """
        tensor = self.preprocessing(img)
        inf_result = self.inference_model.predict(tensor)[0]
        result = self.postprocessing(inf_result)
        return result

    def predict_fv(self, img: Union[Path, np.ndarray]):
        """Classification pipeline without postprocessing.

        Args:
            img (Union[Path, np.ndarray]): Image to predict on

        Returns:
            Result of inference without postprocessing
        """
        tensor = self.preprocessing(img)
        inf_result = self.inference_model.predict(tensor)[0]
        return inf_result


class OCRModel(Model):
    def __init__(self, model_type:str,  preprocessing: OCRPreprocessing, model_inference: ModelInference,
                 postprocessing: OCRPostprocessing):
        super().__init__(model_type, preprocessing, model_inference, postprocessing)

    def predict(self, img: Union[Path, np.ndarray]):
        tensor = self.preprocessing(img)
        inf_result = self.inference_model.predict(np.expand_dims(np.expand_dims(tensor, -1), 0))[0]
        result = self.postprocessing(inf_result)
        return result


class YoloDetectorModel(Model):
    """OCR model implementation for text recognition.

    Inherits from Model class. Implements text recognition pipeline with
    predict method utilizing OCR specific preprocessing,
    inference and postprocessing.

    Attributes:
        model_type (str): Type of the model. Inherited from Model.
        preprocessing (YoloPreprocessing): YOLO specific preprocessing.
        inference_model (ModelInference): Model to make predictions. Inherited from Model.
        postprocessing (YoloDetectorPostprocessing): YOLO detector specific postprocessing.

    Methods:
        predict: Runs OCR pipeline with preprocessing, model
            inference and postprocessing.

    """

    def __init__(self, model_type:str, preprocessing: YoloPreprocessing, model_inference: ModelInference,
                 postprocessing: YoloDetectorPostprocessing):
        """Initialize OCRModel by inheriting from Model

        Args:
            model_type (str): Type of the model.
            preprocessing (OCRPreprocessing): OCR specific preprocessing.
            model_inference (ModelInference): Model to make predictions.
            postprocessing (OCRPostprocessing): OCR specific postprocessing.

        """
        super().__init__(model_type, preprocessing, model_inference, postprocessing)

    def predict(self, img: Union[Path, np.ndarray]):
        """Runs OCR prediction pipeline.

        Args:
            img (Union[Path, np.ndarray]): Image to recognize text from.

        Returns:
            Recognized text string.
        """
        tensor, pad_ratio, pad_extra, pad_to_size, _  = self.preprocessing(img)

        inf_result = self.inference_model.predict(tensor)

        padding_meta = {
            'pad_to_size': pad_to_size,
            'pad_extra': pad_extra,
            'ratio': pad_ratio,
        }

        bboxes = np.squeeze(inf_result)

        result = self.postprocessing(bboxes, padding_meta=padding_meta, resize=True)
        return result

    def predict_fv(self, img: Union[Path, np.ndarray]):
        """OCR pipeline without postprocessing.

        Args:
            img (Union[Path, np.ndarray]): Image to predict on

        Returns:
            Result of inference without postprocessing
        """

        tensor, pad_ratio, pad_add_extra, pad_add_to_size, _ = self.preprocessing(img)
        inf_result = self.inference_model.predict(tensor)
        bboxes = np.squeeze(inf_result)
        return bboxes


class YoloSegmentorModel(Model):
    """YoloSegmentorModel class for segmentation using YOLO detection.

    Attributes:
        model_type (str): Type of YOLO model used for inference
        preprocessing (YoloPreprocessing): Image preprocessing object
        model_inference (ModelInference): Model inference object
        postprocessing (List[Union[YoloDetectorPostprocessing,
                                   YoloSegmentorPostprocessing]]): List of postprocessing objects

    """
    def __init__(self, model_type: str, preprocessing: YoloPreprocessing, model_inference: ModelInference,
                 postprocessing: List[Union[YoloDetectorPostprocessing, YoloSegmentorPostprocessing]]):
        """Initializes YoloSegmentorModel by inheriting from Model



        """

        super().__init__(model_type, preprocessing, model_inference, postprocessing)

    def predict(self, img: Union[Path, np.ndarray]):
        """Runs segmentation pipeline with postprocessing

        Args:
            img: Input image

        Returns:
            nms_prediction: Processed bboxes
            masks: Predicted masks
            segments: Output image segments
        """

        tensor, pad_ratio, pad_extra, pad_to_size, img_shape  = self.preprocessing(img)
        # print(pad_extra)
        inf_result = self.inference_model.predict(tensor)


        padding_meta = {
            'pad_to_size': pad_to_size,
            'pad_extra': pad_extra,
            'ratio': pad_ratio,
        }

        bboxes, masks = np.squeeze(inf_result[0]), np.squeeze(inf_result[1])

        nms_prediction = self.postprocessing[0](bboxes[:], padding_meta=padding_meta, resize=True, numpy=True)
        if len(nms_prediction) == 0:
            return None, None, None

        masks, segments = self.postprocessing[1](masks, nms_prediction[:, 6:], nms_prediction[:, :4], pad_extra, img_shape, upsample=True)
        return nms_prediction[:, :6], masks, segments

    def predict_fv(self, img: Union[Path, np.ndarray]):
        """Returns raw network output without postprocessing

        Args:
            img: Input image

        Returns:
            bboxes: Raw bboxes from network
            masks: Raw masks from network
        """
        tensor, pad_ratio, pad_add_extra, pad_add_to_size = self.preprocessing(img)
        inf_result = self.inference_model.predict(tensor)

        bboxes, masks = (np.squeeze(inf_result[0]), np.squeeze(inf_result[1])) \
            if isinstance(inf_result, list) else (np.squeeze(inf_result), None)

        # print(bboxes)

        return bboxes, masks