from pathlib import Path
from typing import Union

import numpy as np

from ..base_module import BaseModule


class RegistrationStamps(BaseModule):
    """Finds the registration stamps on the residence registration page of the
    Russian internal passport (INTPASSPORTADDR), issue #27.

    A YOLO-OBB detector (the same family and I/O as AddressLinesDetector) run on
    the same canvas as the address lines. Each stamp is one oriented box around the
    whole impression, signature and unit code included, classed as

    * ``registered``   - «Зарегистрирован»: carries the address lines;
    * ``deregistered`` - «Снят с регистрационного учёта»: carries no address;
    * ``unclear``      - a stamp the detector could not tell apart.

    The stamp is located, not read: which address lines belong to it is decided
    by the pipeline (lines lie inside their stamp), and dates inside a stamp are
    not read yet.
    """

    #: Model label -> kind reported in the results.
    KINDS = {'Stamp_registered': 'registered',
             'Stamp_deregistered': 'deregistered',
             'Stamp_unclear': 'unclear'}

    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        self.model_name = 'RegistrationStamps'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects the stamps.

        Returns:
            dict with oriented boxes as [cx, cy, w, h, angle, conf, cls, label].
        """
        img = self.load_img(img)
        obboxes = self.model.predict(img)
        return {self.model_name: {'obbox': obboxes}}
