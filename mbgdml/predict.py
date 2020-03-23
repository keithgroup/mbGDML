# MIT License
# 
# Copyright (c) 2020, Alex M. Maldonado
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

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

import numpy as np

from sgdml import predict

class MBGDMLPredict():

    def __init__(self, model_path, dataset_path):
        self.model_path = model_path
        self.dataset_path = dataset_path


    def remove_nbody_contributions(
        self, raw_dataset, nbody_model
    ):
        """Creates new GDML dataset with GDML predicted n-body predictions
        removed.

        To employ the many body expansion, we need GDML models that predict
        n-body corrections/contributions. This provides the appropriate dataset
        for training an (n+1)-body mbGDML model.

        Args:
            raw_dataset (dict): GDML dataset containing n-body
                contributions to be removed.
            nbody_model (dict): GDML model that predicts n-body contributions.
        """