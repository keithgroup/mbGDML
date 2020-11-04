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
from sgdml import __version__
from mbgdml.data import mbGDMLData
from mbgdml.predict import mbGDMLPredict

# TODO finish documenting
class mbGDMLPredictset(mbGDMLData):
    """A predict set is a data set with mbGDML predicted energy and forces
    instead of training data.

    When analyzing many structures using mbGDML it is easier (and faster) to
    predict all many-body contributions once and then analyze the stored data.
    The predict set accomplishes just this.

    Attributes
    ----------
    predictset : `dict`
        Dictionary of loaded npz predict set file.
    sgdml_version : `str`
        The sGDML Python package version used for predictions.
    name : `str`
        File name of the predict set.
    theory : `str`
        Specifies the level of theory used for GDML training.
    n_z : `int`
        Number of atoms in each structure.
    z : `numpy.ndarray`
        Atomic numbers of all atoms in every structure (same in each one) with
        shape ``n_z``.
    n_R : `int`
        Number of structures in the predict set.
    R : `numpy.ndarray`
        Atomic coordinates of every structure with shape ``(n_R, n_z, 3)``.
    r_unit : `str`
        Units of space for the structures' coordinates.
    e_unit : `str`
        Units of energy.
    E_true : `numpy.ndarray`
        Reference energies with shape ``n_R``.
    F_true : `numpy.ndarray`
        Reference atomic forces with shape ``(n_R, n_z, 3)``.
    """

    def __init__(self, *args):
        for arg in args:
            self.load(arg)
    
    def load(self, predictset_path):
        """Reads predict data set and loads data.
        
        Parameters
        ----------
        predictset_path : str
            Path to predict set npz file.
        """
        predictset = np.load(predictset_path, allow_pickle=True)
        self.predictset = dict(predictset)
        self.sgdml_version = str(predictset['code_version'][()])
        self.name = str(predictset['name'][()])
        self.theory = str(predictset['theory'][()])
        self.z = predictset['z']
        self.n_z = self.z.shape[0]
        self.R = predictset['R']
        self.n_R = self.R.shape[0]
        self.r_unit = str(predictset['r_unit'][()])
        self.e_unit = str(predictset['e_unit'][()])
        self.E_true = predictset['E_true']
        self.F_true = predictset['F_true']
    
    def _get_total_contributions(self, nbody_order):
        """N-body energy and atomic forces contributions of all structures.

        Parameters
        ----------
        structure : `int`
            The index of the desired structure.
        nbody_order : `int`
            Desired n-body order contributions.
        
        Returns
        -------
        tuple : (`numpy.ndarray`)
            Energies, shape of ``n_R``, and atomic forces, shape of
            (``n_R``, ``n_z``, 3), contributions of all structures. 
        
        Raises
        ------
        AttributeError
            If there is no predict set.
        """
        if not hasattr(self, 'R'):
            raise AttributeError('Please load or create a predict set first.'
            )
        else:
            E_cont = self.predictset[f'E_{nbody_order}'][()]['T']
            F_cont = self.predictset[f'F_{nbody_order}'][()]['T']
            return E_cont, F_cont

    def nbody_predictions(self, max_nbody_order):
        """Energies and forces of all structures up to and including a specific
        n-body order.

        Predict sets have data that is broken down into many-body contributions.
        This function sums the many-body contributions up to the specified
        level; for example, `3` returns the energy and force predictions when
        including one, two, and three body contributions/corrections.

        Parameters
        ----------
        max_nbody_order : `int`
            Highest many-body order corrections to include.
        
        Returns
        -------
        tuple (`numpy.ndarray`)
            Energies and forces of all structures with all many-body corrections
            up to nbody_order.
        """
        nbody_index = 1
        while nbody_index <= max_nbody_order:
            E_cont, F_cont = self._get_total_contributions(nbody_index)
            if nbody_index == 1:
                E = E_cont
                F = F_cont
            else:
                E = np.add(E, E_cont)
                F = np.add(F, F_cont)
            nbody_index += 1  
        return E, F

    def load_dataset(self, dataset_path):
        """Loads data set in preparation to create a predict set.

        Parameters
        ----------
        dataset_path : `str`
            Path to data set.
        """
        self.dataset_path = dataset_path
        self.dataset = dict(np.load(dataset_path))
    
    def load_models(self, model_paths):
        """Loads model(s) in preparation to create a predict set.

        Parameters
        ----------
        model_paths : `list` [`str`]
            Paths to GDML models in assending order of n-body corrections, e.g.,
            ['/path/to/1body-model.npz', '/path/to/2body-model.npz'].
        """
        self.model_paths = model_paths
        self.mbgdml = mbGDMLPredict(model_paths)

    def create_predictset(self, name=''):
        """Creates a predict set from loaded data set and models.

        Parameters
        ----------
        name : `str`, optional
            The desired file name for the predict set. Defaults to the name of
            the data set with any occurrence of 'data' changed to 'predict'.
        
        Raises
        ------
        AttributeError
            If no data set or GDML models were loaded beforehand.
        """
        if not hasattr(self, 'dataset'):
            raise AttributeError('Please load a data set.')
        if not hasattr(self, 'mbgdml'):
            raise AttributeError('Please load GDML models.')
        num_config = self.dataset['R'].shape[0]
        if name == '':
            name = str(self.dataset['name'][()]).replace(
                'data', 'predict'
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

