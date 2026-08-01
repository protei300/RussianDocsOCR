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

    This class handles loading models in various formats such as ONNX,
    OpenVINO etc. and making predictions from them.

    Attributes:
        model: The loaded model object.
        device (str): Device to use for inference - 'cpu' or 'gpu'.
    """
    def __init__(self,
                 model_path: Path,
                 device: str = 'gpu',
                 verbose=False,
                 runtime: str = None,
                 ):
        """
        Initializes the ModelInference class.

        Args:
          model_path (Path): Path to model file or directory.
          device (str): Device to use for inference - 'cpu' or 'gpu'.
          verbose (bool): Whether to print debug statements.
          runtime (str): Which runtime to load the artifact with ('ONNX',
            'OpenVINO', 'CoreML'). Defaults to picking it from the file
            extension. OpenVINO reads .onnx directly, so its model.json points
            at the ONNX artifact and names the runtime explicitly instead of
            shipping a second copy of the weights under a .ir extension.
        """

        self.device = device
        self.verbose = verbose
        runtime = (runtime or '').lower()

        # Concurrent Run() on ONE CUDA session wedges the GPU: reproduced by
        # hammering the words detector from 8 threads, which hangs at ~200 calls
        # with the GPU pinned at 100%, or dies with
        # "CUDA error cudaErrorIllegalAddress" mid-graph. Serialised through this
        # lock the same 2400 calls pass in 22s. The lock is PER SESSION, so
        # stages that run different models concurrently
        # (Pipeline._quality_and_borders_parallel) keep their speedup; only
        # repeated calls into the same model (Pipeline._split_words) queue up.
        # CPU sessions are left unlocked - the failure is CUDA-specific and a
        # lock there would only cost throughput.
        self._run_lock = threading.Lock() if device == 'gpu' else None

        if runtime == 'openvino' or (not runtime and model_path.suffix == '.ir'):
            self.openvino = import_module('openvino')
            self.__load_openvino(model_path)
            self.predict = self.__predict_openvino
            if verbose:
                print("[+] OpenVINO model loaded")

        elif runtime == 'coreml' or (not runtime and model_path.suffix == '.mlmodel'):
            self.ct = import_module('coremltools')
            if platform.system() != 'Darwin':
                raise Exception("MLModel Not supported on Windows and Linux")
            self.__load_coreml(model_path)
            self.predict = self.__predict_coreml
            if verbose:
                print("[+] CoreML inference loaded")

        elif runtime in ('onnx', '') and model_path.suffix == '.onnx':
            self.ort = import_module('onnxruntime')
            self.__load_onnx(model_path)
            self.predict = self.__predict_onnx
            if verbose:
                print("[+] ONNX inference loaded")

        else:
            raise Exception(f"Unsupported model '{model_path}' "
                            f"(runtime={runtime or 'auto'}, suffix='{model_path.suffix}'). "
                            f"Supported runtimes: ONNX, OpenVINO, CoreML; "
                            f"supported files: .onnx, .ir, .mlmodel")


    def predict(self, tensors: List[np.ndarray]):
        """Makes a prediction on the input tensor.

        Runs inference on the loaded model.

        Args:
           tensor (numpy.ndarray): Input tensor for model.

        Returns:
           numpy.ndarray: Output prediction

        """
        raise NotImplemented("Need to implement this method")

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


    def __predict_onnx(self, tensors:  List[np.ndarray]):
        inputs = self.model.get_inputs()
        # cast each tensor to the dtype the model actually declares (float32 for
        # the detectors, uint8 for the OCR nets) rather than always float32.
        ort_inputs = {inp.name: tensors[i].astype(_np_dtype_for(inp)) for i, inp in enumerate(inputs)}

        if self._run_lock is None:
            return self.model.run(None, ort_inputs)
        with self._run_lock:                      # CUDA only - see __init__
            return self.model.run(None, ort_inputs)

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