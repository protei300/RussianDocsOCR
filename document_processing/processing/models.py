import os
from pathlib import Path
from .preprocessing import BasePreprocessing, ClassificationPreprocessing, YoloPreprocessing, OCRPreprocessing, OBBPreprocessing, OCRv2Preprocessing
from .postprocessing import *
from .inference import ModelInference
from ..config.alphabets import allowed_charset
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

    def __call__(self, json_file: Path, device='gpu', runtime=None):
        """
        Loads and returns a model based on the JSON config file.

        Arguments:
        - json_file (Path): path to JSON config file
        - device (str): device to load model on (gpu/cpu)
        - runtime (str): execution backend ('ONNX', 'OpenVINO'); overrides the
          config's own 'Runtime' key. Falls back to the file extension.

        Returns: loaded model
        """

        self.json_file = json.loads(json_file.read_text(encoding="utf8"))
        self.working_dir = json_file.parent
        self.device = device
        self.runtime = runtime or self.json_file.get('Runtime')
        model = self.load_model()

        # if self.json_file['Type'] == 'Metric':
        #     model = self.__load_metric_model()
        # elif self.json_file['Type'] == 'YoloDetector':
        #     model = self.__load_yolo_detector()
        # elif self.json_file['Type'] == 'YoloSegmentor':
        #     model = self.__load_yolo_segmentor()
        # elif self.json_file['Type'] == 'BinaryClassification':
        #     model = self.__load_binary_classificator()
        # elif self.json_file['Type'] == 'MultiLabelClassification':
        #     model = self.__load_multi_label_classificator()
        # elif self.json_file['Type'] == 'OCR':
        #     model = self.__load_ocr(self.json_file['Lang'])
        # elif self.json_file['Type'] == 'OCRFV':
        #     model = self.__load_ocr(self.json_file['Lang'])
        # else:
        #     raise Exception(f"[!] Not supported model type: {self.json_file['Type']}")


        return model


    def __load_preprocess(self, input_info:dict):
        input_type = input_info['Type']
        match input_type:
            case 'Classification':
                return ClassificationPreprocessing(
                        image_size=input_info['Shape'],
                        padding_size=input_info['PaddingSize'],
                        padding_color=input_info['PaddingColor'],
                        normalization=input_info['Normalization'],
                        verbose=self.verbose,
                    )
            case 'YOLO':
                return YoloPreprocessing(
                    image_size=input_info['Shape'],
                    padding_size=input_info['PaddingSize'],
                    padding_color=input_info['PaddingColor'],
                    normalization=input_info['Normalization'],
                    verbose=self.verbose
                )

            case 'YOLOOBB':
                return OBBPreprocessing(
                    image_size=input_info['Shape'],
                    padding_size=input_info['PaddingSize'],
                    padding_color=input_info['PaddingColor'],
                    normalization=input_info['Normalization'],
                    verbose=self.verbose
                )

            case 'OCR':
                return OCRPreprocessing(
                    image_size=input_info['Shape'],
                    padding_size=input_info['PaddingSize'],
                    padding_color=input_info['PaddingColor'],
                    normalization=input_info['Normalization'],
                    verbose=self.verbose
                )

            case 'OCRv2':
                return OCRv2Preprocessing(
                    height=input_info.get('Height', 32),
                    color_order=input_info.get('ColorOrder', 'BGR'),
                    dtype=input_info.get('Dtype', 'uint8'),
                    verbose=self.verbose,
                )

    def __load_postprocess(self, output_info:dict):
        output_type = output_info['Type']
        match output_type:
            case "BinaryClassification":
                return BinaryClassPostprocessing(
                    labels=output_info['Labels'],
                    threshold=output_info['Threshold'],
                    verbose=self.verbose,
                )
            case "MultiLabelClassification":
                return MultiClassPostprocessing(
                    labels=output_info['Labels'],
                    verbose=self.verbose,
                )
            case "Metric":
                return MetricPostprocessing(
                    # The shipped model.json writes this path Windows-style
                    # ("resources\\centers.npz"). On POSIX a backslash is an
                    # ordinary filename character, so joining it verbatim yields
                    # `.../ONNX/resources\centers.npz` and DocTypeAngles dies at
                    # construction — only in a Linux container, never here.
                    # Normalised in code rather than in the artifact so both
                    # conventions keep working and no model has to be re-shipped;
                    # `config/__init__.py` does the same for models_path.yaml.
                    centers=self.working_dir / str(output_info['Centers']).replace('\\', os.sep),
                    metric=output_info['Metric'],
                    verbose=self.verbose,
                )
            case "YOLODetector":
                return YOLODetectorPostprocessing(
                    labels=output_info['Labels'],
                    iou=output_info['IOU'],
                    cls=output_info['CLS'],
                    verbose=self.verbose,
                )
            case "PerClassYOLODetector":
                return PerClassYOLODetectorPostprocessing(
                    labels=output_info['Labels'],
                    iou=output_info['IOU'],
                    cls=output_info['CLS'],
                    # optional: raises the NMS threshold for named classes only
                    iou_per_class=output_info.get('IOUPerClass'),
                    verbose=self.verbose,
                )
            case "YOLOOBBDetector":
                return YOLOOBBDetectorPostprocessing(
                    labels=output_info['Labels'],
                    iou=output_info['IOU'],
                    cls=output_info['CLS'],
                    verbose=self.verbose,
                )
            case "YOLOSegmentor":
                return YOLOSegmentorPostprocessing(
                    mask_filter=output_info['MaskFilter'],
                    verbose=self.verbose,
                )

            case "OCR":
                return OCRPostprocessing(
                    lang=output_info['Lang'],
                    verbose=self.verbose,
                )
            case "OCRFV":
                return OCRFVPostprocessing(
                    verbose=self.verbose,
                )

            case "OCRProbs":
                # Full alphabet lives in the model.json; the allowed mask is
                # resolved from the vendored per-country config using the output's
                # Script (+ optional Country, default per-script).
                return OCRProbsPostprocessing(
                    alphabet=output_info['Alphabet'],
                    allowed=allowed_charset(output_info['Script'], output_info.get('Country')),
                    blank_index=output_info.get('BlankIndex', 0),
                    verbose=self.verbose,
                )



    def load_model(self):
        preprocessings = []
        for input_info in self.json_file['Inputs']:
            preprocessings.append(self.__load_preprocess(input_info))

        model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
                                         device=self.device,
                                         verbose=self.verbose,
                                         runtime=self.runtime)

        postprocessings = []
        for output_info in self.json_file['Outputs']:
            postprocessings.append(self.__load_postprocess(output_info))

        ### YOLODetector
        match self.json_file['ModelType']:
            case "YOLODetection":
                model = YOLODetectionModel(
                    model_type=self.json_file['ModelType'],
                    preprocessings=preprocessings,
                    model_inference=model_inference,
                    postprocessings=postprocessings,
                )
        ### YOLOSegmentator
            case "YOLOSegmentation":
                model = YOLOSegmentionModel(
                    model_type=self.json_file['ModelType'],
                    preprocessings=preprocessings,
                    model_inference=model_inference,
                    postprocessings=postprocessings,
                )
        ### YOLO Oriented BBox detector
            case "YOLOOBBDetection":
                model = YOLOOBBDetectionModel(
                    model_type=self.json_file['ModelType'],
                    preprocessings=preprocessings,
                    model_inference=model_inference,
                    postprocessings=postprocessings,
                )
        ### UnifiedModel
            case _:
                model = UnifiedModel(
                    model_type=self.json_file['ModelType'],
                    preprocessings=preprocessings,
                    model_inference=model_inference,
                    postprocessings=postprocessings,
                )
        return model


    # def __load_metric_model(self):
    #     """
    #     Loads a metric processing model from the JSON file.
    #     """
    #
    #     preprocessings = []
    #     for inp_preprocess in self.json_file['Inputs']:
    #         preprocessings.append(
    #             ClassificationPreprocessing(
    #                 image_size=inp_preprocess['Shape'],
    #                 padding_size=inp_preprocess['PaddingSize'],
    #                 padding_color=inp_preprocess['PaddingColor'],
    #                 normalization=inp_preprocess['Normalization'],
    #                 verbose=self.verbose,
    #             )
    #         )
    #
    #     model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
    #                                      device=self.device,
    #                                      verbose=self.verbose)
    #
    #     postprocessings = []
    #     for outp_postprocess in self.json_file['Outputs']:
    #         postprocessings.append(
    #             MetricPostprocessing(
    #                 self.working_dir.joinpath(str(self.json_file['Centers']).replace('\\', os.sep)),
    #                 metric=outp_postprocess['Metric'],
    #                 verbose=self.verbose
    #             )
    #         )
    #
    #     model = ClassificationModel(
    #         model_type= self.json_file['Type'],
    #         preprocessing=preprocessings[0],
    #         model_inference=model_inference,
    #         postprocessing=postprocessings[0],
    #     )
    #
    #     return model
    #
    # def __load_binary_classificator(self):
    #     preprocessings = []
    #     for inp_preprocess in self.json_file['Input']:
    #         preprocessings.append(
    #             ClassificationPreprocessing(
    #                 image_size=inp_preprocess['Shape'],
    #                 padding_size=inp_preprocess['PaddingSize'],
    #                 padding_color=inp_preprocess['PaddingColor'],
    #                 normalization=inp_preprocess['Normalization'],
    #                 verbose=self.verbose
    #             )
    #         )
    #
    #     model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
    #                                      device=self.device,
    #                                      verbose=self.verbose)
    #
    #     postprocessings = []
    #     for _ in self.json_file['Output']:
    #         postprocessings.append(
    #             BinaryClassPostprocessing(
    #                 self.json_file['Labels'],
    #                 verbose=self.verbose
    #             )
    #         )
    #
    #     model = ClassificationModel(
    #         model_type=self.json_file['Type'],
    #         preprocessing=preprocessings[0],
    #         model_inference=model_inference,
    #         postprocessing=postprocessings[0],
    #     )
    #
    #     return model
    #
    # def __load_ocr(self, lang):
    #     preprocessings = []
    #     for inp_preprocess in self.json_file['Input']:
    #         preprocessings.append(
    #             OCRPreprocessing(
    #                 image_size=inp_preprocess['Shape'],
    #                 padding_size=inp_preprocess['PaddingSize'],
    #                 padding_color=inp_preprocess['PaddingColor'],
    #                 normalization=inp_preprocess['Normalization'],
    #                 verbose=self.verbose
    #             )
    #         )
    #
    #     model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
    #                                      device=self.device,
    #                                      verbose=self.verbose)
    #
    #     model = OCRModel(
    #         model_type=self.json_file['Type'],
    #         preprocessing=preprocessings[0],
    #         model_inference=model_inference,
    #         postprocessing=OCRPostprocessing(lang=lang, verbose=False),
    #     )
    #
    #     return model
    #
    # def __load_multi_label_classificator(self):
    #     preprocessings = []
    #     for inp_preprocess in self.json_file['Input']:
    #         preprocessings.append(
    #             ClassificationPreprocessing(
    #                 image_size=inp_preprocess['Shape'],
    #                 padding_size=inp_preprocess['PaddingSize'],
    #                 padding_color=inp_preprocess['PaddingColor'],
    #                 normalization=inp_preprocess['Normalization'],
    #                 verbose=self.verbose
    #             )
    #         )
    #
    #     model_inference = ModelInference(
    #         self.working_dir.joinpath(self.json_file['File']),
    #         device=self.device,
    #         verbose=self.verbose
    #     )
    #
    #     postprocessings = []
    #     for _ in self.json_file['Output']:
    #         postprocessings.append(
    #             MultiClassPostprocessing(
    #                 self.json_file['Labels'],
    #                 verbose=self.verbose
    #             )
    #         )
    #
    #     model = ClassificationModel(
    #         model_type=self.json_file['Type'],
    #         preprocessing=preprocessings[0],
    #         model_inference=model_inference,
    #         postprocessing=postprocessings[0],
    #     )
    #
    #     return model
    #
    # def __load_yolo_detector(self):
    #     preprocessings = []
    #     for inp_preprocess in self.json_file['Input']:
    #         preprocessings.append(
    #             YoloPreprocessing(
    #                 image_size=inp_preprocess['Shape'],
    #                 padding_size=inp_preprocess['PaddingSize'],
    #                 padding_color=inp_preprocess['PaddingColor'],
    #                 normalization=inp_preprocess['Normalization'],
    #                 verbose=self.verbose
    #             )
    #         )
    #     model_inference = ModelInference(self.working_dir.joinpath(self.json_file['File']),
    #                                      device=self.device,
    #                                      verbose=self.verbose)
    #
    #     postprocessings = []
    #     for _ in self.json_file['Output']:
    #         postprocessings.append(
    #             YoloDetectorPostprocessing(
    #                 iou=self.json_file['IOU'],
    #                 cls=self.json_file['CLS'],
    #                 labels=self.json_file['Labels'],
    #                 verbose=self.verbose
    #             )
    #         )
    #
    #     model = YoloDetectorModel(
    #         model_type=self.json_file['Type'],
    #         preprocessing=preprocessings[0],
    #         model_inference=model_inference,
    #         postprocessing=postprocessings[0],
    #     )
    #
    #     return model
    #
    # def __load_yolo_segmentor(self):
    #     preprocessings = []
    #     for inp_preprocess in self.json_file['Input']:
    #         preprocessings.append(
    #             YoloPreprocessing(
    #                 image_size=inp_preprocess['Shape'],
    #                 padding_size=inp_preprocess['PaddingSize'],
    #                 padding_color=inp_preprocess['PaddingColor'],
    #                 normalization=inp_preprocess['Normalization'],
    #                 verbose=self.verbose,
    #             )
    #         )
    #     model_inference = ModelInference(
    #         self.working_dir.joinpath(self.json_file['File']),
    #         device=self.device,
    #         verbose=self.verbose
    #     )
    #
    #     postprocessings = [
    #         YoloDetectorPostprocessing(
    #             iou=self.json_file['IOU'],
    #             cls=self.json_file['CLS'],
    #             labels=self.json_file['Labels'],
    #             verbose=self.verbose
    #         ),
    #         YoloSegmentorPostprocessing(self.json_file['MaskFilter'],verbose=self.verbose),
    #     ]
    #
    #     model = YoloSegmentorModel(
    #         preprocessing=preprocessings[0],
    #         model_inference=model_inference,
    #         postprocessing=postprocessings,
    #     )
    #
    #     return model

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
                 preprocessings: List[BasePreprocessing],
                 model_inference: ModelInference,
                 postprocessings: List[BasePostprocessing]):
        """Initialize the Model instance.

        Args:
           preprocessings (BasePreprocessing): Preprocessing algorithm.
           model_inference (ModelInference): Model to make predictions.
           postprocessings (Union[List[BasePostprocessing], BasePostprocessing]):
               Postprocessing algorithm(s).
        """
        self.__model_type = model_type
        self.preprocessings = preprocessings
        self.inference_model = model_inference
        self.postprocessings = postprocessings


    def predict(self, img: Union[Path, np.ndarray, List[Union[Path, np.ndarray]]]):
        """Runs prediction pipeline with preprocessing, inference and postprocessing.

        Args:
            img (Union[Path, np.ndarray]): Image to predict on.

        Returns:
            Result of prediction pipeline.
        """
        pass

    def predict_fv(self, img: Union[Path, np.ndarray, List[Union[Path, np.ndarray]]]):
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



class UnifiedModel(Model):
    """Model class for making predictions using a configurable pipeline."""

    def __init__(
            self,
            model_type:str,
            preprocessings: List[BasePreprocessing],
            model_inference: ModelInference,
            postprocessings: List[BasePostprocessing]
    ):
        """Initialize ClassificationModel by inheriting from Model

        Args:
            model_type (str): Type of the model.
            preprocessing (BasePreprocessing): Preprocessing algorithm.
            model_inference (ModelInference): Model to make predictions.
            postprocessing (BasePostprocessing): Postprocessing algorithm.

        """
        super().__init__(model_type, preprocessings, model_inference, postprocessings)

    def predict(self, img: Union[Path, np.ndarray, List[Union[Path, np.ndarray]]]):
        """Runs classification pipeline with postprocessing.

        Args:
            img (Union[Path, np.ndarray]): Image to predict on

        Returns:
            Result of classification prediction
        """

        inf_results = self.predict_fv(img)

        if len(self.postprocessings) == 0:
            return inf_results

        results = []
        for i, postprocessing in enumerate(self.postprocessings):
            result = postprocessing(inf_results[i])
            results.append(result)
        return results

    def predict_fv(self, img: Union[Path, np.ndarray, List[Union[Path, np.ndarray]]]):
        """Classification pipeline without postprocessing.

        Args:
            img (Union[Path, np.ndarray]): Image to predict on

        Returns:
            Result of inference without postprocessing
        """
        if isinstance(img, list):
            tensors = []
            for i, preprocessing in enumerate(self.preprocessings):
                tensors.append(preprocessing(img[i]))
        else:
            tensors = []
            tensors.append(self.preprocessings[0](img))
        inf_result = self.inference_model.predict(tensors)
        return inf_result


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
    def __init__(self, model_type:str,  preprocessings: List[OCRPreprocessing], model_inference: ModelInference,
                 postprocessings: List[OCRPostprocessing]):
        super().__init__(model_type, preprocessings, model_inference, postprocessings)

    def predict(self, img: Union[Path, np.ndarray]):
        tensor = self.preprocessings[0](img)
        inf_result = self.inference_model.predict(tensor)[0]
        result = self.postprocessings[0](inf_result)
        return result

    def predict_fv(self, img: Union[Path, np.ndarray]):
        """
        ctc vector
        """
        tensor = self.preprocessing(img)
        inf_result = self.inference_model.predict(np.expand_dims(np.expand_dims(tensor, -1), 0))[0]
        return inf_result


class OCRFVModel(Model):
    def __init__(self, model_type:str,  preprocessing: OCRPreprocessing, model_inference: ModelInference,
                 postprocessing: OCRPostprocessing):
        super().__init__(model_type, preprocessing, model_inference, postprocessing)

    def predict(self, img: Union[Path, np.ndarray]):
        pass

    def predict_fv(self, img: Union[Path, np.ndarray]):
        """
        ctc vector
        """
        tensor = self.preprocessing(img)
        inf_result = self.inference_model.predict(np.expand_dims(np.expand_dims(tensor, -1), 0))[0]
        return inf_result


class YOLODetectionModel(Model):
    """YOLO object-detection model implementation (e.g. TextFields/Words/Borders).

    Inherits from Model class. Implements the detection pipeline with a
    predict method utilizing YOLO-specific preprocessing (letterbox resize),
    inference and postprocessing (NMS + box decoding).

    Attributes:
        model_type (str): Type of the model. Inherited from Model.
        preprocessing (YoloPreprocessing): YOLO specific preprocessing.
        inference_model (ModelInference): Model to make predictions. Inherited from Model.
        postprocessing (YOLODetectorPostprocessing): YOLO detector specific postprocessing.

    Methods:
        predict: Runs the detection pipeline with preprocessing, model
            inference and postprocessing.

    """

    def __init__(self, model_type, preprocessings: List[YoloPreprocessing], model_inference: ModelInference,
                 postprocessings: List[YOLODetectorPostprocessing]):
        """Initialize YOLODetectionModel by inheriting from Model

        Args:
            preprocessings (YoloPreprocessing): YOLO specific preprocessing.
            model_inference (ModelInference): Model to make predictions.
            postprocessing (YOLODetectorPostprocessing): YOLO detector specific postprocessing.

        """
        super().__init__(model_type, preprocessings, model_inference, postprocessings)

    def predict(self, img: Union[Path, np.ndarray]):
        """Runs the detection pipeline (preprocessing + inference + postprocessing).

        Args:
            img (Union[Path, np.ndarray]): Image to detect objects in.

        Returns:
            Detected bounding boxes (see YOLODetectorPostprocessing).
        """
        tensor, pad_ratio, pad_extra, pad_to_size, _  = self.preprocessings[0](img)

        inf_result = self.inference_model.predict([tensor,])


        padding_meta = {
            'pad_to_size': pad_to_size,
            'pad_extra': pad_extra,
            'ratio': pad_ratio,
        }

        bboxes = np.squeeze(inf_result)

        result = self.postprocessings[0](bboxes, padding_meta=padding_meta, resize=True)
        return result

    def predict_fv(self, img: Union[Path, np.ndarray]):
        """Detection pipeline without postprocessing (raw boxes, no NMS/decoding).

        Args:
            img (Union[Path, np.ndarray]): Image to predict on

        Returns:
            Result of inference without postprocessing
        """

        tensor, pad_ratio, pad_add_extra, pad_add_to_size, _ = self.preprocessings[0](img)
        inf_result = self.inference_model.predict([tensor,])
        bboxes = np.squeeze(inf_result)
        return bboxes


class YOLOOBBDetectionModel(Model):
    """YOLO oriented-bbox (OBB) detection model.

    Runs the standard ultralytics OBB ONNX export (NCHW /255 input,
    output ``[1, 4+nc+1, n_anchors]``) and returns oriented detections
    in original-image coordinates via ``YOLOOBBDetectorPostprocessing``.
    """

    def __init__(self, model_type, preprocessings: List[OBBPreprocessing], model_inference: ModelInference,
                 postprocessings: List[YOLOOBBDetectorPostprocessing]):
        super().__init__(model_type, preprocessings, model_inference, postprocessings)

    def predict(self, img: Union[Path, np.ndarray]):
        tensor, pad_ratio, pad_extra, pad_to_size, _ = self.preprocessings[0](img)
        inf_result = self.inference_model.predict([tensor, ])

        padding_meta = {
            'pad_to_size': pad_to_size,
            'pad_extra': pad_extra,
            'ratio': pad_ratio,
        }
        vector = np.squeeze(inf_result[0] if isinstance(inf_result, list) else inf_result)
        result = self.postprocessings[0](vector, padding_meta=padding_meta, resize=True)
        return result

    def predict_fv(self, img: Union[Path, np.ndarray]):
        tensor, *_ = self.preprocessings[0](img)
        inf_result = self.inference_model.predict([tensor, ])
        return np.squeeze(inf_result[0] if isinstance(inf_result, list) else inf_result)


class YOLOSegmentionModel(Model):
    """YOLOSegmentorModel class for segmentation using YOLO detection.

    Attributes:
        preprocessing (YoloPreprocessing): Image preprocessing object
        model_inference (ModelInference): Model inference object
        postprocessing (List[Union[YoloDetectorPostprocessing,
                                   YoloSegmentorPostprocessing]]): List of postprocessing objects

    """
    def __init__(self, model_type, preprocessings: List[YoloPreprocessing], model_inference: ModelInference,
                 postprocessings: List[Union[YOLODetectorPostprocessing, YOLOSegmentorPostprocessing]]):
        """Initializes YoloSegmentorModel by inheriting from Model



        """

        super().__init__(model_type, preprocessings, model_inference, postprocessings)

    def predict(self, img: Union[Path, np.ndarray]):
        """Runs segmentation pipeline with postprocessing

        Args:
            img: Input image

        Returns:
            nms_prediction: Processed bboxes
            masks: Predicted masks
            segments: Output image segments
        """

        tensor, pad_ratio, pad_extra, pad_to_size, img_shape  = self.preprocessings[0](img)
        inf_result = self.inference_model.predict([tensor,])


        padding_meta = {
            'pad_to_size': pad_to_size,
            'pad_extra': pad_extra,
            'ratio': pad_ratio,
        }
        bboxes, masks = np.squeeze(inf_result[0]), np.squeeze(inf_result[1])


        nms_prediction = self.postprocessings[0](bboxes, padding_meta=padding_meta, resize=True, numpy=True)
        if len(nms_prediction) == 0:
            return None, None, None

        masks, segments = self.postprocessings[1](masks, nms_prediction[:, 6:], nms_prediction[:, :4], pad_extra, img_shape, upsample=True)
        return nms_prediction[:, :6], masks, segments

    def predict_fv(self, img: Union[Path, np.ndarray]):
        """Returns raw network output without postprocessing

        Args:
            img: Input image

        Returns:
            bboxes: Raw bboxes from network
            masks: Raw masks from network
        """
        tensor, pad_ratio, pad_add_extra, pad_add_to_size, _ = self.preprocessings[0](img)
        inf_result = self.inference_model.predict([tensor,])

        bboxes, masks = (np.squeeze(inf_result[0]), np.squeeze(inf_result[1])) \
            if isinstance(inf_result, list) else (np.squeeze(inf_result), None)

        # print(bboxes)

        return bboxes, masks