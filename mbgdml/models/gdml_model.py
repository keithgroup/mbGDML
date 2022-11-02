# MIT License
# 
# Copyright (c) 2018-2022 Stefan Chmiela, Gregory Fonseca
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
from ..utils import md5_data
from .._gdml.desc import _from_r

log = logging.getLogger(__name__)

class gdmlModel(model):

    def __init__(
        self, model, criteria_desc_func=None, criteria_cutoff=None
    ):
        """
        Parameters
        ----------
        model : :obj:`str` or :obj:`dict`
            Path to GDML npz model or :obj:`dict`.
        criteria_desc_func : ``callable``, default: ``None``
            A descriptor used to filter :math:`n`-body structures from being
            predicted.
        criteria_cutoff : :obj:`float`, default: ``None``
            Value of ``criteria_desc_func`` where the mlModel will not predict
            the :math:`n`-body contribution of. If ``None``, no cutoff will be
            enforced.
        """
        super().__init__(criteria_desc_func, criteria_cutoff)

        self.type = 'gdml'

        if isinstance(model, str):
            model = dict(np.load(model, allow_pickle=True))
        elif not isinstance(model, dict):
            raise TypeError(f'{type(model)} is not string or dict')
        self._model_dict = model

        self.z = model['z']
        self.comp_ids = model['comp_ids']
        self.nbody_order = len(self.comp_ids)

        self.sig = model['sig']
        self.n_atoms = self.z.shape[0]
        self.desc_func = _from_r

        self.lat_and_inv = (
            (model['lattice'], np.linalg.inv(model['lattice']))
            if 'lattice' in model
            else None
        )
        self.n_train = model['R_desc'].shape[1]
        self.stdev = model['std'] if 'std' in model else 1.0
        self.integ_c = model['c']
        self.n_perms = model['perms'].shape[0]
        self.tril_perms_lin = model['tril_perms_lin']

        self.use_torch = False
        self.R_desc_perms = (
            np.tile(model['R_desc'].T, self.n_perms)[:, self.tril_perms_lin]
            .reshape(self.n_train, self.n_perms, -1, order='F')
            .reshape(self.n_train * self.n_perms, -1)
        )
        self.R_d_desc_alpha_perms = (
            np.tile(model['R_d_desc_alpha'], self.n_perms)[:, self.tril_perms_lin]
            .reshape(self.n_train, self.n_perms, -1, order='F')
            .reshape(self.n_train * self.n_perms, -1)
        )

        if 'alphas_E' in model.keys():
            self.alphas_E_lin = np.tile(
                model['alphas_E'][:, None], (1, n_perms)
            ).ravel()
        else:
            self.alphas_E_lin = None
    
    @property
    def md5(self):
        """Unique MD5 hash of model.

        :type: :obj:`bytes`
        """
        md5_properties_all = [
            'md5_train', 'z', 'R_desc', 'R_d_desc_alpha', 'sig', 'alphas_F'
        ]
        md5_properties = []
        for key in md5_properties_all:
            if key in self._model_dict.keys():
                md5_properties.append(key)
        md5_string = md5_data(self._model_dict, md5_properties)
        return md5_string

