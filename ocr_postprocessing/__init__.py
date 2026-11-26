"""Post-processing of the registration-stamp OCR: CTC matrices of a stamp's fields in, the same fields
improved by the dictionaries (KLADR, FMS department codes) out. See README.md."""
from .ctc import CTCAlphabet, greedy, join
from .postprocess import LABELS, NET_CLASSES, OCRPostprocessor, References

__all__ = ['CTCAlphabet', 'greedy', 'join', 'LABELS', 'NET_CLASSES', 'References', 'OCRPostprocessor']
