# MIT License
# 
# Copyright (c) 2022, Alex M. Maldonado
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

import logging
import numpy as np
from .base import model

log = logging.getLogger(__name__)
     
class schnetModel(model):

    def __init__(
        self, model_path, comp_ids, device, criteria_desc_func=None,
        criteria_cutoff=None
    ):
        """
        Parameters
        ----------
        model_path : :obj:`str`
            Path to SchNet PyTorch model.
        comp_ids : ``iterable``
            Model component IDs that relate entity IDs of a structure to a
            fragment label.
        device : :obj:`str`
            The device where the model and tensors will be stored. For example,
            ``'cpu'`` and ``'cuda'``.
        criteria_desc_func : ``callable``, default: ``None``
            A descriptor used to filter :math:`n`-body structures from being
            predicted.
        criteria_cutoff : :obj:`float`, default: ``None``
            Value of ``criteria_desc_func`` where the mlModel will not predict
            the :math:`n`-body contribution of. If ``None``, no cutoff will be
            enforced.
        """
        global schnetpack, torch, ase
        import schnetpack, torch, ase

        super().__init__(criteria_desc_func, criteria_cutoff)
        self.type = 'schnet'
        self.spk_model = torch.load(
            model_path, map_location=torch.device(device)
        )
        self.device = device
        if isinstance(comp_ids, list) or isinstance(comp_ids, tuple):
            comp_ids = np.array(comp_ids)
        self.comp_ids = comp_ids
        self.nbody_order = len(comp_ids)

        # SchNet MD5
        import hashlib
        md5_hash = hashlib.md5()
        for param in self.spk_model.parameters():
            md5_hash.update(
                hashlib.md5(
                    param.cpu().detach().numpy().flatten()
                ).digest()
            )
        self.md5 = md5_hash.hexdigest()

