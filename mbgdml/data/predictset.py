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
from sgdml import __version__ as sgdml_version
from mbgdml.data import mbGDMLData
from mbgdml.predict import mbGDMLPredict

# TODO finish documenting
class predictSet(mbGDMLData):
    """A predict set is a data set with mbGDML predicted energy and forces
    instead of training data.

    When analyzing many structures using mbGDML it is easier (and faster) to
    predict all many-body contributions once and then analyze the stored data.
    The predict set accomplishes just this.

    Attributes
    ----------
    sgdml_version : :obj:`str`
        The sGDML Python package version used for predictions.
    name : :obj:`str`
        File name of the predict set.
    theory : :obj:`str`
        Specifies the level of theory used for GDML training.
    """

    def __init__(self, *args):
        self.name = 'predictset'
        self.type = 'p'
        self._loaded = False
        self._predicted = False
        for arg in args:
            self.load(arg)
    

    def _calc_contributions(self):
        """
        """
        all_E = {}
        all_F = {}
        for i in range(self.n_R):
            print(f'Predicting structure {i} out of {self.n_R} ...')
            e, f = self.mbgdml.decomposed_predict(
                self.z, self.R[i]
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
        return all_E, all_F

    @property
    def predictset(self):
        """Contains all data as :obj:`numpy.ndarray` objects.

        :type: :obj:`dict`
        """        
        predictset = {
            'type': np.array('p'),
            #'code_version': self.sgdml_version,
            'name': np.array(self.name),
            'theory': np.array(self.theory),
            'z': self.z,
            'R': self.R,
            'r_unit': np.array(self.r_unit),
            'E_true': self.E_true,
            'e_unit': np.array(self.e_unit),
            'F_true': self.F_true,
        }

        if self._loaded == False and self._predicted == False:
            if not hasattr(self, 'dataset') or not hasattr(self, 'mbgdml'):
                raise AttributeError(
                    'No data can be predicted or is not loaded.'
                )
            else: 
                all_E, all_F = self._calc_contributions()
        
                for order in all_E:
                    setattr(self, f'_E_{order}', all_E[order])
                    setattr(self, f'_F_{order}', all_F[order])
                self.sgdml_version = sgdml_version
                self._predicted = True
        
        # predictset['sgdml_version'] = np.array(self.sgdml_version)

        n_index = 1
        while hasattr(self, f'_E_{n_index}'):
            predictset[f'E_{n_index}'] = getattr(self, f'_E_{n_index}')
            predictset[f'F_{n_index}'] = getattr(self, f'_F_{n_index}')
            n_index += 1
        
        return predictset
    
    @property
    def e_unit(self):
        """Units of energy. Options are ``'eV'``, ``'hartree'``,
        ``'kcal/mol'``, and ``'kJ/mol'``.

        :type: :obj:`str`
        """
        return self._e_unit

    @e_unit.setter
    def e_unit(self, var):
        self._e_unit = var

    @property
    def E_true(self):
        """True energies from data set.

        :type: :obj:`numpy.ndarray`
        """
        return self._E_true


    @property
    def F_true(self):
        """True forces from data set.

        :type: :obj:`numpy.ndarray`
        """
        return self._F_true

    
    def load(self, predictset_path):
        """Reads predict data set and loads data.
        
        Parameters
        ----------
        predictset_path : :obj:`str`
            Path to predict set ``.npz`` file.
        """
        predictset = dict(np.load(predictset_path, allow_pickle=True))
        #self.sgdml_version = str(predictset['code_version'][()])
        self.name = str(predictset['name'][()])
        self.theory = str(predictset['theory'][()])
        self._z = predictset['z']
        self._R = predictset['R']
        self._r_unit = str(predictset['r_unit'][()])
        self._e_unit = str(predictset['e_unit'][()])
        self._E_true = predictset['E_true']
        self._F_true = predictset['F_true']
        n_index = 1
        while f'E_{n_index}' in predictset.keys():
            setattr(self, f'_E_{n_index}', predictset[f'E_{n_index}'])
            setattr(self, f'_F_{n_index}', predictset[f'F_{n_index}'])
            n_index += 1
        self._loaded = True
    
    
    def _get_total_contributions(self, nbody_order):
        """N-body energy and atomic forces contributions of all structures.

        Parameters
        ----------
        structure : :obj:`int`
            The index of the desired structure.
        nbody_order : :obj:`int`
            Desired n-body order contributions.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            Energy n-order corrections of structures.
        :obj:`numpy.ndarray`
            Forces n-order corrections of structures.
        
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

    def nbody_predictions(self, nbody_order):
        """Energies and forces of all structures up to and including a specific
        n-body order.

        Predict sets have data that is broken down into many-body contributions.
        This function sums the many-body contributions up to the specified
        level; for example, ``3`` returns the energy and force predictions when
        including one, two, and three body contributions/corrections.

        Parameters
        ----------
        nbody_order : :obj:`int`
            Highest many-body order corrections to include.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            Energy of structures up to and including n-order corrections.
        :obj:`numpy.ndarray`
            Forces of structures up to an including n-order corrections.
        """
        for nbody_index in range(1, nbody_order + 1):
            E_cont, F_cont = self._get_total_contributions(nbody_index)
            if nbody_index == 1:
                E = E_cont
                F = F_cont
            else:
                E = np.add(E, E_cont)
                F = np.add(F, F_cont)
        return E, F

    def load_dataset(self, dataset_path):
        """Loads data set in preparation to create a predict set.

        Parameters
        ----------
        dataset_path : :obj:`str`
            Path to data set.
        """
        self.dataset_path = dataset_path
        self.dataset = dict(np.load(dataset_path))
        self.theory = str(self.dataset['theory'][()])
        self._z = self.dataset['z']
        self._R = self.dataset['R']
        self._r_unit = str(self.dataset['r_unit'][()])
        self._e_unit = str(self.dataset['e_unit'][()])
        self._E_true = self.dataset['E']
        self._F_true = self.dataset['F']
    
    def load_models(self, model_paths):
        """Loads model(s) in preparation to create a predict set.

        Parameters
        ----------
        model_paths : :obj:`list` [:obj:`str`]
            Paths to GDML models in assending order of n-body corrections, e.g.,
            ['/path/to/1body-model.npz', '/path/to/2body-model.npz'].
        """
        self.model_paths = model_paths
        self.mbgdml = mbGDMLPredict(model_paths)

