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


import numpy as np
from .. import __version__ as mbgdml_version
from .basedata import mbGDMLData
from ..predict import mbPredict


class predictSet(mbGDMLData):
    """A predict set is a data set with mbGDML predicted energy and forces
    instead of training data.

    When analyzing many structures using mbGDML it is easier (and faster) to
    predict all many-body contributions once and then analyze the stored data.
    The predict set accomplishes just this.

    Attributes
    ----------
    name : :obj:`str`
        File name of the predict set.
    theory : :obj:`str`
        Specifies the level of theory used for GDML training.
    entity_ids : :obj:`numpy.ndarray`
        A uniquely identifying integer specifying what atoms belong to
        which entities. Entities can be a related set of atoms, molecules,
        or functional group. For example, a water and methanol molecule
        could be ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.
    comp_ids : :obj:`numpy.ndarray`
        Relates ``entity_id`` to a fragment label for chemical components
        or species. Labels could be ``WAT`` or ``h2o`` for water, ``MeOH``
        for methanol, ``bz`` for benzene, etc. There are no standardized
        labels for species. The index of the label is the respective
        ``entity_id``. For example, a water and methanol molecule could
        be ``['h2o', 'meoh']``.
    """

    def __init__(self, *args):
        self.name = 'predictset'
        self.type = 'p'
        self._loaded = False
        self._predicted = False
        for arg in args:
            self.load(arg)  
    
    def prepare(
        self, z, R, entity_ids, comp_ids, ignore_criteria=False
    ):
        """Prepares a predict set by calculated the decomposed energy and
        force contributions.
        """

        self.E_decomp, self.F_decomp = self.mbgdml.predict_decomposed(
            z, R, entity_ids, comp_ids,
            ignore_criteria=ignore_criteria, store_each=True
        )
        
        self.mbgdml_version = mbgdml_version
        self._predicted = True

    def asdict(self):
        """Converts object into a custom :obj:`dict`.

        Returns
        -------
        :obj:`dict`
        """     
        predictset = {
            'type': np.array('p'),
            'sgdml_version': np.array(self.sgdml_version),
            'mbgdml_version': np.array(self.mbgdml_version),
            'name': np.array(self.name),
            'theory': np.array(self.theory),
            'z': self.z,
            'R': self.R,
            'r_unit': np.array(self.r_unit),
            'E_true': self.E_true,
            'e_unit': np.array(self.e_unit),
            'F_true': self.F_true,
            'E_decomp': self.E_decomp,
            'F_decomp': self.F_decomp,
            'entity_ids': self.entity_ids,
            'comp_ids': self.comp_ids,
            'models_order': self.models_order,
            'models_md5': self.models_md5
        }

        if self._loaded == False and self._predicted == False:
            if not hasattr(self, 'dataset') or not hasattr(self, 'mbgdml'):
                raise AttributeError(
                    'No data can be predicted or is not loaded.'
                )
            else: 
                self.prepare(self.z, self.R, self.entity_ids, self.comp_ids)
        
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
        self.sgdml_version = str(predictset['sgdml_version'][()])
        self.mbgdml_version = str(predictset['mbgdml_version'][()])
        self._z = predictset['z']
        self._R = predictset['R']
        self.E_decomp = predictset['E_decomp']
        self.F_decomp = predictset['F_decomp']
        self._r_unit = str(predictset['r_unit'][()])
        self._e_unit = str(predictset['e_unit'][()])
        self._E_true = predictset['E_true']
        self._F_true = predictset['F_true']
        self.entity_ids = predictset['entity_ids']
        self.comp_ids = predictset['comp_ids']
        self.models_order = predictset['models_order']
        self.models_md5 = predictset['models_md5']

        self._loaded = True

    def nbody_predictions(self, nbody_orders):
        """Energies and forces of all structures .

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
        for nbody_order in nbody_orders:
            E_nbody = [i[nbody_order]['T'] for i in self.E_decomp]
            F_nbody = [i[nbody_order]['T'] for i in self.F_decomp]
            E = np.add(E, E_nbody)
            F = np.add(F, F_nbody)
        return E, F

    def load_dataset(self, dataset_path):
        """Loads data set in preparation to create a predict set.

        Parameters
        ----------
        dataset_path : :obj:`str`
            Path to data set.
        """
        self.dataset_path = dataset_path
        dataset = dict(np.load(dataset_path, allow_pickle=True))
        self.dataset = dataset
        self.theory = str(dataset['theory'][()])
        self._z = dataset['z']
        self._R = dataset['R']
        self._r_unit = str(dataset['r_unit'][()])
        self._e_unit = str(dataset['e_unit'][()])
        self._E_true = dataset['E']
        self._F_true = dataset['F']
        self.entity_ids = dataset['entity_ids']
        self.comp_ids = dataset['comp_ids']
    
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

        from mbgdml.data import mbModel
        models_md5 = []
        models_order = []
        for model_path in model_paths:
            model = mbModel(model_path)
            models_order.append(len(set(model.entity_ids)))
            try:
                models_md5.append(model.md5)
            except AttributeError:
                pass
        self.models_order = np.array(models_order)
        self.models_md5 = np.array(models_md5)

