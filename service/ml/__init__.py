"""The library-integration layer — the only place that imports ``document_processing``.

Everything else in the service talks to the recognition library through this
package's narrow surface:

    from service.ml import runtime
    with runtime.lease_pipeline() as pipeline:
        payload = runtime.recognise(pipeline, image_path)

Keeping the import surface in one package is what makes the rest of the service
testable without 215 MB of ONNX models, and what makes a future port to another
language a bounded job: only this package has to be rewritten.

Read ``runtime.py`` first — its module docstring lists the non-obvious rules of
using ``Pipeline`` correctly. They are not guesses; each one is a real trap that
costs a debugging session if you get it wrong.
"""

from service.ml.errors import PipelineBusy, RecognitionError

__all__ = ["PipelineBusy", "RecognitionError"]
