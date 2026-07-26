import numpy as np
import threading
from pathlib import Path
import platform
from importlib import import_module
from typing import List


# ONNX tensor element type -> numpy dtype. Anything not listed falls back to
# float32 (the historical default), so existing float models are unaffected while
# uint8-input models (the v2 OCR nets) get the correct dtype instead of a cast
# that onnxruntime would reject.
_ORT_TYPE_TO_NP = {
    'tensor(float)': np.float32,
    'tensor(float16)': np.float16,
    'tensor(double)': np.float64,
    'tensor(uint8)': np.uint8,
    'tensor(int8)': np.int8,
    'tensor(int32)': np.int32,
    'tensor(int64)': np.int64,
    'tensor(bool)': np.bool_,
}


def _np_dtype_for(ort_input) -> np.dtype:
    return np.dtype(_ORT_TYPE_TO_NP.get(ort_input.type, np.float32))


class ModelInference:
    """Class for making inferences from different model types.

    This class handles loading models in various formats such as TensorFlow, ONNX,
    OpenVINO etc. and making predictions from them.

    Attributes:
        model: The loaded model object.
        device (str): Device to use for inference - 'cpu' or 'gpu'.
    """
    def __init__(self,
                 model_path: Path,
                 device: str = 'gpu',
                 verbose=False,
                 ):
        """
        Initializes the ModelInference class.

        Args:
          model_path (Path): Path to model file or directory.
          device (str): Device to use for inference - 'cpu' or 'gpu'.
          verbose (bool): Whether to print debug statements.
        """

        self.device = device
        self.verbose = verbose

        if model_path.suffix == '.h5':
            self.tf = import_module('tensorflow')
            self.__load_h5(model_path)
            self.predict = self.__predict_saved_model
            if verbose:
                print("[+] H5 inference loaded")

        elif model_path.is_dir():
            self.tf = import_module('tensorflow')
            self.__load_saved_model(model_path)
            self.predict = self.__predict_saved_model
            if verbose:
                print("[+] Saved_model inference loaded")

        elif model_path.suffix == '.pb':
            self.tf = import_module('tensorflow')
            self.__load_pb(model_path)
            self.predict = self.__predict_frozen
            if verbose:
                print("[+] PB inference loaded")

        elif model_path.suffix == '.tflite':
            self.tf = import_module('tensorflow')
            self.__load_tflite(model_path)
            self.predict = self.__predict_tflite
            if verbose:
                print("[+] TFlite inference loaded")

        elif model_path.suffix == '.onnx':
            self.ort = import_module('onnxruntime')
            self.__load_onnx(model_path)
            self.predict = self.__predict_onnx
            if verbose:
                print("[+] ONNX inference loaded")

        elif model_path.suffix == '.ir':
            self.openvino = import_module('openvino')
            self.__load_openvino(model_path)
            self.predict = self.__predict_openvino
            if verbose:
                print("[+] OpenVINO model loaded")


        elif model_path.suffix == '.mlmodel':
            self.ct = import_module('coremltools')
            if platform.system() != 'Darwin':
                raise Exception("MLModel Not supported on Windows and Linux")
            self.__load_coreml(model_path)
            self.predict = self.__predict_coreml
            if verbose:
                print("[+] CoreML inference loaded")

        else:
            raise Exception("Unsupported model type TODO")


    def predict(self, tensors: List[np.ndarray]):
        """Makes a prediction on the input tensor.

        Runs inference on the loaded model.

        Args:
           tensor (numpy.ndarray): Input tensor for model.

        Returns:
           numpy.ndarray: Output prediction

        """
        raise NotImplemented("Need to implement this method")

    def __load_h5(self, model_path:Path):
        if self.device == 'gpu':
            self.model = self.tf.keras.models.load_model(model_path)
        else:
            with self.tf.device('/cpu:0'):
                self.model = self.tf.keras.models.load_model(model_path)
        print("[+] H5 model loaded")

    def __load_saved_model(self, model_path:Path):
        if self.device == 'gpu':
            self.model = self.tf.saved_model.load(model_path)
        else:
            with self.tf.device('/cpu:0'):
                self.model = self.tf.saved_model.load(model_path)


    def __load_pb(self, model_path:Path):
        graph_path = model_path.as_posix()
        with self.tf.compat.v1.gfile.GFile(graph_path, "rb") as f:
            graph_def = self.tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.tf.Graph().as_default() as graph:
            self.tf.import_graph_def(
                graph_def,
            )
        self.model = graph

        inp_name, inp_shape = self.model.get_operations()[0]._outputs[0].name, \
            self.model.get_operations()[0]._outputs[0].shape[1:]


        self.outp_name = []

        for i, op in enumerate(graph.get_operations()):
            if op.type == 'Identity':
                self.outp_name.append(op._outputs[0].name)



        with self.tf.compat.v1.Session(graph=self.model) as sess:
            sess.run(self.outp_name, feed_dict={inp_name: np.zeros(np.append(1, np.array(inp_shape)))})

    def __load_tflite(self, model_path: Path):
        self.model = self.tf.lite.Interpreter(model_path.as_posix())
        self.outputs = sorted(self.model.get_output_details(), key=lambda d: d['name'])

    def __load_onnx(self, model_path: Path):
        onnx_model_path = model_path.as_posix()
        if self.device == 'gpu':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider', ]
        self.model = self.ort.InferenceSession(onnx_model_path, providers=providers)
        inputs = self.model.get_inputs()
        ort_inputs = {}
        for inp in inputs:
            # batch=1, and replace dynamic axes (None / str names) with a safe size.
            # 1 is too small for strided OCR backbones (a 1px width collapses to an
            # empty feature map), so dynamic spatial dims use 320.
            shape = [1] + [d if isinstance(d, int) else 320 for d in inp.shape[1:]]
            dtype = _np_dtype_for(inp)
            if np.issubdtype(dtype, np.integer):
                ort_inputs[inp.name] = np.random.randint(0, 256, size=shape).astype(dtype)
            else:
                ort_inputs[inp.name] = np.random.rand(*shape).astype(dtype)
        # Warmup is best-effort priming (primes the CUDA graph/kernel-selection
        # so the first REAL call isn't the one paying that cost) - never let
        # it block model loading. Always surface a failure, not just under
        # verbose: a silently-skipped warmup would otherwise look identical to
        # a slow first real call, which is exactly the kind of thing this is
        # meant to prevent from going unnoticed.
        try:
            self.model.run(None, ort_inputs)
        except Exception as e:
            print(f"[!] ONNX warmup skipped for {onnx_model_path}: {e}")

    def __load_openvino(self, model_path: Path):
        core = self.openvino.Core()
        ov_model = self.openvino.convert_model(model_path)
        self.model = core.compile_model(ov_model, device_name=self.device.upper())
        # Calling a CompiledModel directly reuses one shared infer request, which
        # raises "Infer Request is busy" when the pipeline runs stages
        # concurrently. Give each thread its own request off the same compiled
        # model (compilation and weights stay shared).
        self._ov_local = threading.local()

    def __load_coreml(self, model_path: Path):
        self.model = self.ct.models.MLModel(model_path.as_posix())


    def __predict_saved_model(self, tensor: np.ndarray):
        if self.device == 'cpu':
            with self.tf.device('/cpu:0'):
                result = self.model(tensor).numpy()
        else:
            result = self.model(tensor).numpy()
        pred = result[0] if len(result) == 1 else result
        return pred


    def __predict_frozen(self, tensor: np.ndarray):

        inp_name = self.model.get_operations()[0]._outputs[0].name
        with self.tf.compat.v1.Session(graph=self.model) as sess:
            result = sess.run(self.outp_name, feed_dict={inp_name: tensor})
        pred = result[0] if len(result)==1 else result
        return pred
    def __predict_tflite(self, tensor: np.ndarray):
        tensor = tensor.astype(self.model.get_input_details()[0]['dtype'])
        self.model.allocate_tensors()
        self.model.set_tensor(self.model.get_input_details()[0]['index'], tensor)
        self.model.invoke()

        result = []
        for output in self.outputs:
            result.append(self.model.get_tensor(output['index']))
        pred = result[0] if len(result) == 1 else result
        return pred

    def __predict_onnx(self, tensors:  List[np.ndarray]):
        inputs = self.model.get_inputs()
        # cast each tensor to the dtype the model actually declares (float32 for
        # the legacy nets, uint8 for the v2 OCR nets) rather than always float32.
        ort_inputs = {inp.name: tensors[i].astype(_np_dtype_for(inp)) for i, inp in enumerate(inputs)}

        result = self.model.run(None, ort_inputs)
        # pred = result[0] if len(result) == 1 else result
        return result

    def __predict_openvino(self, tensor: np.ndarray):
        request = getattr(self._ov_local, 'request', None)
        if request is None:
            request = self.model.create_infer_request()
            self._ov_local.request = request
        request.infer(tensor)
        outputs = [request.get_output_tensor(i).data
                   for i in range(len(self.model.outputs))]
        return outputs[0] if len(outputs) == 1 else outputs

    def __predict_coreml(self, tensor: np.ndarray):
        pred = self.model.predict({'image': tensor})
        return pred