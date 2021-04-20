# MIT License
# 
# Copyright (c) 2020-2021, Alex M. Maldonado
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
from mbgdml.data import mbGDMLData
from mbgdml.utils import md5_data
from mbgdml import __version__


class mbModel(mbGDMLData):
    """A class to load, inspect, and modify GDML models.

    Attributes
    ----------
    model_path : :obj:`str`
        Path to the npz file.
    model : :obj:`numpy.npzfile`
        GDML model for predicting energies and forces.
    """
    
    def __init__(self):
        pass

    @property
    def code_version(self):
        """sGDML code version used to train the model.

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
            'md5_train', 'z', 'R_desc', 'R_d_desc_alpha', 'interact_cut_off',
            'sig', 'alphas_F'
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
        self.model = dict(np.load(model_path, allow_pickle=True))

        if self.model['type'][()] != 'm':
            raise AttributeError('This npz is not a mbGDML model.')
    
    def add_modifications(self):
        """mbGDML-specific modifications of models.

        - Adds system information if possible which could include ``'system'``,
          ``'solvent'``, and ``'cluster_size'``.
        - Adds mbGDML version with ``'mbgdml_version'``.
        """
        self.add_system_info(self.model)
        self.model['mbgdml_version'] = np.array(__version__)
        self.model['md5'] = np.array(self.md5, dtype='S32')
    
    def add_manybody_info(self, mb_order, mb_models_md5):
        """Adds many-body (mb) information to GDML model.
        
        Parameters
        ----------
        mb_order : :obj:`int`
            The max order of many-body predictions removed from the data set.
        mb_models_md5 : :obj:`list` [:obj:`str`]
            The MD5 hashes of the models used to remove n-body contributions
            from data set.

        Raises
        ------
        AttributeError
            If accessing information when no model is loaded.
        """

        if not hasattr(self, 'model'):
            raise AttributeError('There is no mbGDML model loaded.')

        self.model['mb'] = mb_order
        self.model['mb_models_md5'] = mb_models_md5
        
