# MIT License
# 
# Copyright (c) 2020-2022, Alex M. Maldonado
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

# pylint: disable=E1101

import os
import numpy as np
from .basedata import mbGDMLData
from ..utils import md5_data
from .. import __version__ as mbgdml_version


class mbModel(mbGDMLData):
    """A class to load, inspect, and modify GDML models.

    Attributes
    ----------
    model_path : :obj:`str`
        Path to the npz file.
    model : `npz`
        GDML model for predicting energies and forces.
    """
    
    def __init__(self, *args):
        if len(args) == 1:
            self.load(args[0])

    @property
    def code_version(self):
        """mbGDML version used to train the model.

        Raises
        ------
        AttributeError
            If accessing information when no model is loaded.
        """

        if hasattr(self, '_model_data'):
            return self._model_data['code_version'][()]
        else:
            raise AttributeError('No model is loaded.')
    
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
            if key in self.model.keys():
                md5_properties.append(key)
        md5_string = md5_data(self.model, md5_properties)
        return md5_string

    def load(self, model_path):
        """Loads GDML model.
        
        Parameters
        ----------
        model_path : :obj:`str`
            Path to mbGDML model.

        Raises
        ------
        AttributeError
            If npz file is not a GDML model.
        """
        self.path = model_path
        self.name = os.path.splitext(os.path.basename(model_path))[0]
        model = dict(np.load(model_path, allow_pickle=True))

        if model['type'][()] != 'm':
            raise AttributeError('This npz is not a mbGDML model.')

        self.z = model['z']
        self.r_unit = str(model['r_unit'][()])
        self.e_unit = str(model['e_unit'][()])
        
        for key in ['criteria', 'z_slice', 'cutoff', 'entity_ids', 'comp_ids']:
            if key in model.keys():
                data = model[key]
                if key in ['criteria', 'z_slice', 'cutoff']:
                    data = data[()]
                if key == 'criteria':
                    data = str(data)
                setattr(self, key, data)

        self.model = model
    
    def add_modifications(self, dset):
        """mbGDML-specific modifications of models.

        Transfers information from data set to model.

        Parameters
        ----------
        dset : :obj:`dict`
            Dictionary of a mbGDML data set.
        """
        dset_keys = ['entity_ids', 'comp_ids', 'criteria', 'z_slice', 'cutoff']
        for key in dset_keys:
            self.model[key] = dset[key]
        self.model['mbgdml_version'] = np.array(mbgdml_version)
        self.model['md5'] = np.array(self.md5, dtype='S32')
