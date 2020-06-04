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

# pylint: disable=E1101


import os
import re
import numpy as np
from cclib.io import ccread
from cclib.parser.utils import convertor
from sgdml import __version__
from sgdml.utils import io as sgdml_io
from mbgdml.data import mbGDMLData
from mbgdml import utils
import mbgdml.solvents as solvents
from mbgdml.predict import mbGDMLPredict


class mbGDMLPredictset(mbGDMLData):
    """
    A predict set is a data set with mbGDML predicted energy and forces instead
    of training data.

    When analyzing many structures using mbGDML it is easier (and faster) to
    predict all many-body contributions once and then analyze the stored data.
    The predict set accomplishes just this.
    """

    def __init__(self):
        pass
        
    
    def load(self, predictset_path):
        """Reads predict data set and loads data.
        
        Args:
            predictset_path (str): Path to predict data set.
        """
        predictset = np.load(predictset_path, allow_pickle=True)
        predictset = dict(predictset)
        for file in predictset:
            if type(predictset[file][()]) == np.str_:
                setattr(self, file, str(predictset[file][()]))
            else:
                setattr(self, file, predictset[file][()])

    
    def sum_contributions(self, struct_num, nbody_order):
        """
        Returns the energy and force of a structure at a
        specific many-body order.

        Predict sets have data that is broken down into many-body and 'total'
        contributions. Many-body contributions provide the total for that order;
        for example, 'E_3' gives you the total contribution (or correction) of
        all three bodies evaluated in the structure. This is not the total
        energy with one-body, two-body, and three-body corrections.

        This function returns the 'total energy' that includes the specified
        nbody_order and lower corrections.

        Parameters
        ----------
        struct_num : int
            Specifies the index of the structure in the self.R array and the
            energy and force arrays.
        nbody_order : int
            Highest many-body order corrections to include.
        
        Returns
        -------
        tuple
            Energy and force of the structure with all many-body corrections
            up to nbody_order.
        """

        if not hasattr(self, 'type'):
            raise AttributeError('Please read a predict set first.')
        
        nbody_index = 1
        while hasattr(self, f'E_{nbody_index}') and \
              hasattr(self, f'F_{nbody_index}') and \
              nbody_index <= nbody_order:

            E_cont = getattr(self, f'E_{nbody_index}')
            F_cont = getattr(self, f'F_{nbody_index}')

            if nbody_index == 1:
                E = E_cont['T'][struct_num]
                F = F_cont['T'][struct_num]
            else:
                E += E_cont['T'][struct_num]
                F += F_cont['T'][struct_num]

            nbody_index += 1
        
        return (E, F)
    

    def nbody_predictions(self, nbody_order):

        if not hasattr(self, 'R'):
            raise AttributeError('No coordinates;'
                                 'please read a predict set first.')
        else:
            num_structures = self.R.shape[0]

            for structure in range(0, num_structures):
                e, f = self.sum_contributions(structure, nbody_order)

                e = np.array([e])
                f = np.array([f])

                if structure == 0:
                    E = e
                    F = f
                else:
                    E = np.concatenate((E, e))
                    F = np.concatenate((F, f))

        return (E, F)

    def load_dataset(self, dataset_path):
        """
        Loads data set in preparation to create a predict set.
        """
        self.dataset_path = dataset_path
        self.dataset = dict(np.load(dataset_path))
    

    def load_models(self, model_paths):
        """
        Loads model(s) in preparation to create a predict set.
        """
        self.model_paths = model_paths
        self.mbgdml = mbGDMLPredict(model_paths)


    def create_predictset(self):
        """
        Creates a predict set from loaded data set and models.
        """

        if not hasattr(self, 'dataset') or not hasattr(self, 'mbgdml'):
            raise AttributeError('Please load a data set and mbGDML models.')

        num_config = self.dataset['R'].shape[0]
        name = str(self.dataset['name'][()]).replace(
            'dataset', 'predictset'
        )
        
        self.predictset = {
            'type': 'p',  # Designates predictions.
            'code_version': __version__,  # sGDML version.
            'name': name,
            'theory': self.dataset['theory'],
            'z': self.dataset['z'],
            'R': self.dataset['R'],
            'r_unit': self.dataset['r_unit'],
            'E_true': self.dataset['E'],
            'e_unit': self.dataset['e_unit'],
            'F_true': self.dataset['F'],
        }

        # Predicts and stores energy and forces.
        all_E = {}
        all_F = {}
        for i in range(num_config):
            print(f'Predicting structure {i} out of {num_config - 1} ...')
            e, f = self.mbgdml.decomposed_predict(
                self.dataset['z'].tolist(), self.dataset['R'][i]
            )

            for order in e:
                if i == 0:
                    all_E[order] = e[order]
                    all_F[order] = f[order]
                    
                    if order == 'T':
                        all_E[order] = all_E[order]
                        all_F[order] = np.array([all_F[order]])
                    else:
                        for combo in e[order]:
                            all_E[order][combo] = all_E[order][combo]
                            all_F[order][combo] = np.array(
                                [all_F[order][combo]]
                            )
                else:
                    if order == 'T':
                        all_E[order] = np.concatenate(
                            (all_E[order], e[order]),
                            axis=0
                        )
                        all_F[order] = np.concatenate(
                            (all_F[order], np.array([f[order]])),
                            axis=0
                        )
                    else:
                        for combo in e[order]:
                            all_E[order][combo] = np.concatenate(
                                (all_E[order][combo], e[order][combo]),
                                axis=0
                            )
                            all_F[order][combo] = np.concatenate(
                                (all_F[order][combo],
                                 np.array([f[order][combo]])),
                                 axis=0
                            )


        # Loop through all_E and all_F and add their keys to dataset
        for order in all_E:
            E_name = f'E_{order}'
            F_name = f'F_{order}'
            self.predictset[E_name] = all_E[order]
            self.predictset[F_name] = all_F[order]
        
        for data in self.predictset:
            setattr(self, data, self.predictset[data])


