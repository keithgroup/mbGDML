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


import numpy as np
from mbgdml.data import mbGDMLData


class mbGDMLModel(mbGDMLData):
    """A class to load, inspect, and modify GDML models.

    Attributes
    ----------
    model_path : str
        Path to the npz file.
    model : numpy.npzfile
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


    def load(self, model_path):
        """Loads GDML model.
        
        Parameters
        ----------
        model_path : str
            Path to GDML model.

        Raises
        ------
        AttributeError
            If npz file is not a GDML model.
        """
        self.model_path = model_path
        self.model = np.load(model_path, allow_pickle=True)
        self._model_data = dict(self.model_npz)

        if self._model_data['type'][()] != 'm':
            raise AttributeError('This npz is not a GDML model.')
    

    def get_model_name(self, log_path):
        """Retrives GDML model's name from log file.
        
        Parameters
        ----------
        log_path : str
            Path to the log file.
        """

        for line in reversed(list(open(log_path))):
            if 'This is your model file' in line:
                self.name = line.split(':')[-1][2:-6]
                break
    

    def add_manybody_info(self, mb_order):
        """Adds many-body (mb) information to GDML model.
        
        Parameters
        ----------
        mb_order : int
            The max order of many-body predictions removed from the dataset.

        Raises
        ------
        AttributeError
            If accessing information when no model is loaded.
        """

        if not hasattr(self, 'model_data'):
            raise AttributeError('There is no model loaded.')

        self.model_data['mb'] = mb_order
        
