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

    When analyzing many structures using mbGDML it is easier to
    predict all many-body contributions once and then analyze the stored data.

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

    def __init__(self, pset=None):
        """
        Parameters
        ----------
        pset : :obj:`str` or :obj:`dict`, optional
            Predict set path or dictionary to initialize with.
        """
        self.name = 'predictset'
        self.type = 'p'
        self.mbgdml_version = mbgdml_version
        self._loaded = False
        self._predicted = False
        if pset is not None:
            self.load(pset)  
    
    def prepare(
        self, z, R, entity_ids, comp_ids, ignore_criteria=False
    ):
        """Prepares a predict set by calculated the decomposed energy and
        force contributions.
        """

        self.E_decomp, self.F_decomp = self.predict.predict_decomposed(
            z, R, entity_ids, comp_ids,
            ignore_criteria=ignore_criteria, store_each=True
        )
        self._predicted = True

    def asdict(self):
        """Converts object into a custom :obj:`dict`.

        Returns
        -------
        :obj:`dict`
        """     
        pset = {
            'type': np.array('p'),
            'version': np.array(self.mbgdml_version),
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
        
        return pset
    
    @property
    def e_unit(self):
        """Units of energy. Options are ``eV``, ``hartree``, ``kcal/mol``, and
        ``kJ/mol``.

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
    
    @E_true.setter
    def E_true(self, var):
        self._E_true = var

    @property
    def F_true(self):
        """True forces from data set.

        :type: :obj:`numpy.ndarray`
        """
        return self._F_true
    
    @F_true.setter
    def F_true(self, var):
        self._F_true = var
 
    def load(self, pset):
        """Reads predict data set and loads data.
        
        Parameters
        ----------
        pset : :obj:`str` or :obj:`dict`
            Path to predict set ``.npz`` file or a dictionary.
        """
        if isinstance(pset, str):
            pset = dict(np.load(pset, allow_pickle=True))
        elif not isinstance(pset, dict):
            raise TypeError(f'{type(pset)} is not supported.')

        self.name = str(pset['name'][()])
        self.theory = str(pset['theory'][()])
        self.version = str(pset['version'][()])
        self.z = pset['z']
        self.R = pset['R']
        self.E_decomp = pset['E_decomp']
        self.F_decomp = pset['F_decomp']
        self.r_unit = str(pset['r_unit'][()])
        self.e_unit = str(pset['e_unit'][()])
        self.E_true = pset['E_true']
        self.F_true = pset['F_true']
        self.entity_ids = pset['entity_ids']
        self.comp_ids = pset['comp_ids']
        self.models_order = pset['models_order']
        self.models_md5 = pset['models_md5']

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

    def load_dataset(self, dset):
        """Loads data set in preparation to create a predict set.

        Parameters
        ----------
        dset : :obj:`str` or :obj:`dict`
            Path to data set or :obj:`dict` with at least the following data:

            ``z`` (:obj:`numpy.ndarray`, ndim: ``1``) - 
                Atomic numbers.

            ``R`` (:obj:`numpy.ndarray`, ndim: ``3``) - 
                Cartesian coordinates.

            ``E`` (:obj:`numpy.ndarray`, ndim: ``1``) - 
                Reference, or true, energies of the structures we will predict.

            ``F`` (:obj:`numpy.ndarray`, ndim: ``3``) - 
                Reference, or true, forces of the structures we will predict.

            ``entity_ids`` (:obj:`numpy.ndarray`, ndim: ``1``) - 
                An array specifying which atoms belong to which entities.

            ``comp_ids`` (:obj:`numpy.ndarray`, ndim: ``1``) - 
                An array relating ``entity_id`` to a fragment label for chemical
                components or species.

            ``theory`` (:obj:`str`) - 
                The level of theory used to compute energy and forces.

            ``r_unit`` (:obj:`str`) - 
                Units of distance.
                
            ``e_unit`` (:obj:`str`) - 
                Units of energy.
        """
        if isinstance(dset, str):
            dset = dict(np.load(dset, allow_pickle=True))
        elif not isinstance(dset, dict):
            raise TypeError(f'{type(dset)} is not supported')

        self.z = dset['z']
        self.R = dset['R']
        self.E_true = dset['E']
        self.F_true = dset['F']
        self.entity_ids = dset['entity_ids']
        self.comp_ids = dset['comp_ids']

        if isinstance(dset['theory'], np.ndarray):
            self.theory = str(dset['theory'].item())
        else:
            self.theory = dset['theory']
        if isinstance(dset['r_unit'], np.ndarray):
            self.r_unit = str(dset['r_unit'].item())
        else:
            self.r_unit = dset['r_unit']
        if isinstance(dset['e_unit'], np.ndarray):
            self.e_unit = str(dset['e_unit'].item())
        else:
            self.e_unit = dset['e_unit']
    
    def load_models(self, models, use_torch=False):
        """Loads model(s) in preparation to create a predict set.

        Parameters
        ----------
        models : :obj:`list` of :obj:`str` or :obj:`dict`
            GDML models in ascending order of n-body corrections (e.g., 1-, 2-
            and 3-body models).
        use_torch : :obj:`bool`, default: ``False``
            Use PyTorch to make predictions.
        """
        self.predict = mbPredict(models, use_torch=use_torch)

        from mbgdml.data import mbModel
        models_md5, models_order = [], []
        for model in self.predict.models:
            models_order.append(len(set(model['entity_ids'])))
            try:
                models_md5.append(model['md5'])
            except AttributeError:
                pass
        self.models_order = np.array(models_order)
        self.models_md5 = np.array(models_md5)

