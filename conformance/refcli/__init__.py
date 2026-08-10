"""The Python REFERENCE implementation of the conformance CLI.

This is the only package in `conformance/` allowed to import
`document_processing` (and `service.ml.transform`, which owns the view-model
shape). The checker must stay ignorant of both — see `conformance/__init__.py`.

Run it as a module:

    python -m conformance.refcli info
    python -m conformance.refcli recognize --image samples/DL_2011/1_CR_DL_2010.jpg
    python -m conformance.refcli probe --image ... --dump-dir out/ --upto rotate
    python -m conformance.refcli regen
"""
