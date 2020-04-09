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

"""Analyses for mbGDML models."""
import numpy as np
from mbgdml.data import mbGDMLPredictset

class NBodyContributions:

    def __init__(self):
        pass
    
    
    def load_predictset(self, predictset_path):
        self.predictset = mbGDMLPredictset()
        self.predictset.read(predictset_path)

    def force_similarity(self, predict_force, true_force):
        """Compute modified cosine similarity of two force vectors.

        Computes 1 - cosine_similarity. Two exact vectors will thus have a
        similarity of 0, orthogal vectors will be 1, and equal but opposite
        will be 2.
        
        Args:
            predict_force (np.ndarray): Array of the predicted force vector by
                GDML.
            true_force (np.ndarray): Array of the true force vector of the same
                shape as predict_force
        
        Returns:
            float: [description]
        """

        similarity = np.dot(predict_force, true_force) / \
                     (np.linalg.norm(predict_force) * \
                      np.linalg.norm(true_force))

        similarity = float(1 - similarity)
        return similarity

    # TODO heat map that computes cosine similarity for 1-body up to n-body contributions of force vectors