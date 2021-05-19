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
from mbgdml.predict import mbPredict


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
    entity_ids : :obj:`numpy.ndarray`
        An array specifying which atoms belong to what entities
        (e.g., molecules). Similar to PDBx/mmCIF ``_atom_site.label_entity_ids``
        data item.
    comp_ids : :obj:`numpy.ndarray`
        A 2D array relating ``entity_ids`` to a chemical component/species
        id or label (``comp_id``). The first column is the unique ``entity_id``
        and the second is a unique ``comp_id`` for that chemical species.
        Each ``comp_id`` is reused for the same chemical species.
    """

    def __init__(self, *args):
        self.name = 'predictset'
        self.type = 'p'
        self._loaded = False
        self._predicted = False
        for arg in args:
            self.load(arg)
    

    def _calc_contributions(self, ignore_criteria=True):
        """

        As currently written this only works if you ignore all criteria.
        """
        all_E = {}
        all_F = {}
        for i in range(self.n_R):
            print(f'Predicting structure {i+1} out of {self.n_R}')
            e, f = self.mbgdml.decomposed_predict(
                self.z, self.R[i], self.entity_ids, self.comp_ids,
                ignore_criteria=ignore_criteria, store_each=True
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
            'entity_ids': self.entity_ids,
            'comp_ids': self.comp_ids
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

        
        for E_key in [
            i for i in self.__dict__.keys() \
            if '_E_' in i and 'true' not in i and '_T' not in i
        ]:
            predictset[f'{E_key[1:]}'] = getattr(self, f'{E_key}')
        for F_key in [
            i for i in self.__dict__.keys() \
            if '_F_' in i and 'true' not in i and '_T' not in i
        ]:
            predictset[f'{F_key[1:]}'] = getattr(self, f'{F_key}')
        
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
        self.name = str(predictset['name'][()])
        self.theory = str(predictset['theory'][()])
        self._z = predictset['z']
        self._R = predictset['R']
        self._r_unit = str(predictset['r_unit'][()])
        self._e_unit = str(predictset['e_unit'][()])
        self._E_true = predictset['E_true']
        self._F_true = predictset['F_true']
        self.entity_ids = predictset['entity_ids']
        self.comp_ids = predictset['comp_ids']

        for E_key in [i for i in predictset.keys() if 'E_' in i and 'true' not in i]:
            setattr(self, f'_{E_key}', predictset[f'{E_key}'])
        for F_key in [i for i in predictset.keys() if 'F_' in i and 'true' not in i]:
            setattr(self, f'_{F_key}', predictset[f'{F_key}'])

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

    def nbody_predictions(self, nbody_orders):
        """Energies and forces of all structures up to and including a specific
        n-body order.

        Predict sets have data that is broken down into many-body contributions.
        This function sums the many-body contributions up to the specified
        level; for example, ``3`` returns the energy and force predictions when
        including one, two, and three body contributions/corrections.

        Parameters
        ----------
        nbody_orders : :obj:`list` [:obj:`int`]
            N-body orders to include.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            Energy of structures up to and including n-order corrections.
        :obj:`numpy.ndarray`
            Forces of structures up to an including n-order corrections.
        """
        E = np.zeros(self.E_true.shape)
        F = np.zeros(self.F_true.shape)
        for nbody_index in nbody_orders:
            E_cont, F_cont = self._get_total_contributions(nbody_index)
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
        self.dataset = dict(np.load(dataset_path, allow_pickle=True))
        self.theory = str(self.dataset['theory'][()])
        self._z = self.dataset['z']
        self._R = self.dataset['R']
        self._r_unit = str(self.dataset['r_unit'][()])
        self._e_unit = str(self.dataset['e_unit'][()])
        self._E_true = self.dataset['E']
        self._F_true = self.dataset['F']
        self.entity_ids = self.dataset['entity_ids']
        self.comp_ids = self.dataset['comp_ids']
    
    def load_models(self, model_paths):
        """Loads model(s) in preparation to create a predict set.

        Parameters
        ----------
        model_paths : :obj:`list` [:obj:`str`]
            Paths to GDML models in assending order of n-body corrections, e.g.,
            ['/path/to/1body-model.npz', '/path/to/2body-model.npz'].
        """
        self.model_paths = model_paths
        self.mbgdml = mbPredict(model_paths)

