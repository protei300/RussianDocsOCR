"""Exceptions raised by the integration layer.

Deliberately few and deliberately specific: the worker needs to tell apart
"try again later" from "this document will never work", because retrying a
corrupt JPEG forever is just as wrong as giving up on a transient CUDA hiccup.
"""


class RecognitionError(Exception):
    """Base class for anything the integration layer raises."""

    #: Whether retrying the same input could plausibly succeed.
    transient = False


class PipelineBusy(RecognitionError):
    """No pipeline instance became free within the lease timeout.

    Transient by definition — the job goes back on the queue rather than being
    marked failed. Seeing this repeatedly means a previous job wedged and never
    released its lease (see ``runtime.lease_pipeline``).
    """

    transient = True


class ImageUnreadable(RecognitionError):
    """The uploaded bytes could not be decoded as an image.

    Deterministic: the same bytes will fail the same way forever, so the worker
    must not retry.
    """

    transient = False


class RuntimeNotReady(RecognitionError):
    """A recognition was requested before the models finished loading.

    Transient: model loading takes ~15 s at startup and the service accepts
    uploads immediately, so this is an expected race right after boot.
    """

    transient = True
