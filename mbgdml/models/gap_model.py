# MIT License
#
# Copyright (c) 2022-2023, Alex M. Maldonado
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import hashlib
import numpy as np
from .base import Model
from ..logger import GDMLLogger

try:
    import quippy

    _HAS_QUIPPY = True
except ImportError:
    _HAS_QUIPPY = False

log = GDMLLogger(__name__)

# pylint: disable-next=invalid-name
class gapModel(Model):
    def __init__(
        self,
        model_path,
        comp_ids,
        criteria=None,
    ):
        """
        Parameters
        ----------
        model_path : :obj:`str`
            Path to GAP xml file.
        comp_ids : ``iterable``
            Model component IDs that relate entity IDs of a structure to a
            fragment label.
        criteria : :obj:`mbgdml.descriptors.Criteria`, default: :obj:`None`
            Initialized descriptor criteria for accepting a structure based on
            a descriptor and cutoff.
        """
        assert _HAS_QUIPPY

        super().__init__(criteria)
        self.type = "gap"
        self.gap = quippy.potential.Potential(param_filename=model_path)
        if isinstance(comp_ids, (list, tuple)):
            comp_ids = np.array(comp_ids)
        self.comp_ids = comp_ids
        self.nbody_order = len(comp_ids)

        # GAP MD5
        with open(model_path, "r", encoding="utf-8") as f:
            gap_lines = f.readlines()

        md5_hash = hashlib.md5()
        for gap_line in gap_lines:
            md5_hash.update(hashlib.md5(repr(gap_line).encode()).digest())
        self.md5 = md5_hash.hexdigest()
