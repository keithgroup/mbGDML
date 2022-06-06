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

import itertools

import numpy as np
from ._gdml.predict import GDMLPredict
from . import criteria

try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True

class mbPredict():
    """Predict energies and forces of structures using many-body GDML models.
    """

    def __init__(self, models, use_torch=False):
        """ 
        Parameters
        ----------
        models : :obj:`list` of :obj:`str` or :obj:`dict`
            Contains paths or dictionaries of many-body GDML models.
        use_torch : :obj:`bool`, default: ``False``
            Use PyTorch to make predictions.
        """
        self._load_models(models, use_torch)
    

    def _load_models(self, models, use_torch):
        """Loads models and prepares GDMLPredict.
        
        Parameters
        ----------
        models : :obj:`list` [:obj:`str`]
            Contains paths to either standard or many-body GDML models.
        """
        self.gdmls, self.models, self.entity_ids, self.comp_ids = [], [], [], []
        self.criteria, self.z_slice, self.cutoff = [], [], []
        for model in models:
            if isinstance(model, str):
                loaded = np.load(model, allow_pickle=True)
                model = dict(loaded)
            self.models.append(model)
            gdml = GDMLPredict(model, use_torch=use_torch)
            self.gdmls.append(gdml)

            if model['criteria'] == '':
                self.criteria.append(None)
            else:
                # Stores that actual criteria function from the criteria module.
                self.criteria.append(getattr(criteria, str(model['criteria'])))
            self.z_slice.append(model['z_slice'])
            self.cutoff.append(model['cutoff'])
            self.entity_ids.append(model['entity_ids'])
            self.comp_ids.append(model['comp_ids'])
    
    def _generate_entity_combinations(self, r_entity_ids_per_model_entity):
        """Generator for entity combinations where each entity comes from a
        specified list.

        Parameters
        ----------
        r_entity_ids_per_model_entity : :obj:`list` [:obj:`numpy.ndarray`]
            A list of ``entity_ids`` that match the ``comp_id`` of each
            model ``entity_id``. Note that the index of the
            :obj:`numpy.ndarray` is equal to the model ``entity_id`` and the
            values are ``r`` ``entity_ids`` that match the ``comp_id``.
        """
        nbody_combinations = itertools.product(*r_entity_ids_per_model_entity)
        # Excludes combinations that have repeats (e.g., (0, 0) and (1, 1. 2)).
        nbody_combinations = itertools.filterfalse(
            lambda x: len(set(x)) <  len(x), nbody_combinations
        )
        # At this point, there are still duplicates in this iterator.
        # For example, (0, 1) and (1, 0) are still included.
        for combination in nbody_combinations:
            if sorted(combination) == list(combination):
                yield combination


    def _calculate(
        self, r, entity_ids, comp_ids, model, gdml, ignore_criteria=False, 
        store_each=True
    ):
        """The actual calculate/predict function for a single GDML model.

        Predicts the energy and force contribution from a single many-body
        GDML model.
        
        Parameters
        ----------
        r : :obj:`numpy.ndarray`
            Cartesian coordinates of all atoms of the structure specified in 
            the same order as z. The array should have shape (n, 3) where n is 
            the number of atoms. This is a single structure.
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
        model : :obj:`dict`
            The dictionary of the loaded npz file. Stored in ``self.models``.
        gdml : :obj:`mbgdml._gdml.predict.GDMLPredict`
            Object used to predict energies and forces of the structure defined 
            in ``r``.
        ignore_criteria : :obj:`bool`, optional
            Whether to take into account structure criteria and their cutoffs
            for each model. Defaults to ``False``.
        store_each : :obj:`bool`, optional
            Store each n-body combination's contribution in the :obj:`dict`.
            Defaults to ``True``. Changing to ``False`` often speeds up
            predictions.
        
        Returns
        -------
        :obj:`dict`
            Energy contributions of the structure. Dictionary keys specify the
            molecule combination. For example, ``'0,2'`` contains the
            predictions of two-body energy of the dimer containing molecules
            ``0`` and ``2``. Also contains a ``'T'`` key representing the total
            n-body contribution for the whole system.
        :obj:`dict`
            Force contributions of the structure. Dictionary keys specify the
            molecule combination. For example, ``'1,2,5'`` contains the
            predictions of three-body energy of the trimer containing molecules
            ``1``, ``2``, and ``5``. Also contains a ``'T'`` key representing
            the total n-body contribution for the whole system.
        """
        r_dim = r.ndim

        # 'T' is for total
        E_contributions = {'T': 0.0}
        F_contributions = {'T': np.zeros(r.shape)}

        # Models have a specific entity order that needs to be conserved in the
        # predictions. Here, we create a ``entity_idxs`` list where each item
        # is a list of all entities in ``r`` that match the model entity.
        r_entity_ids_per_model_entity = []
        for model_comp_id in model['comp_ids']:
            matching_entity_ids = np.where(model_comp_id == comp_ids)[0]
            r_entity_ids_per_model_entity.append(matching_entity_ids)

        nbody_combinations = self._generate_entity_combinations(
            r_entity_ids_per_model_entity
        )

        # Getting all contributions for each molecule combination (comb).
        for comb_entity_ids in nbody_combinations:

            # Gets indices of all atoms in the combination of molecules.
            # r_idx is a list of the atoms for the entity_id combination.
            r_idx = []
            for entity_id in comb_entity_ids:
                r_idx.extend(np.where(entity_ids == entity_id)[0])
            
            # Checks criteria if present and desired.
            if not ignore_criteria:
                # If there is a cutoff specified.
                if model['cutoff'].shape != (0,):
                    r_criteria = getattr(criteria, str(model['criteria']))
                    valid_r, _ = r_criteria(
                        model['z'], r[r_idx], model['z_slice'],
                        entity_ids[r_idx], cutoff=model['cutoff']
                    )
                    if not valid_r:
                        # Do not include this contribution.
                        continue
            
            # Predicts energies and forces.
            if r_dim == 2:  # A single structure.
                r_comp = r[r_idx]
                e, f = gdml.predict(r_comp.flatten())
                e = e[0]
            elif r_dim == 3:  # Multiple structures.
                raise ValueError('Dimensions of R should be 2')

            # Adds contributions prediced from model.
            if store_each:
                entity_label = ','.join(
                    str(entity_id) for entity_id in comb_entity_ids
                )
                E_contributions[entity_label] = e
                F_contributions[entity_label] = f.reshape(len(r_idx), 3)

            # Adds contributions to total energy and forces.
            E_contributions['T'] += e
            F_contributions['T'][r_idx] += f.reshape(len(r_idx), 3)
        
        return E_contributions, F_contributions


    def predict_decomposed(
        self, z, R, entity_ids, comp_ids, ignore_criteria=False,
        store_each=True
    ):
        """Computes predicted total energy and atomic forces decomposed by
        many-body order.
        
        Parameters
        ----------
        z : :obj:`numpy.ndarray`
            A ``(n,)`` shape array of type :obj:`numpy.int32` containing atomic
            numbers of atoms in the structures in order as they appear.
        R : :obj:`numpy.ndarray`
            Cartesian coordinates of all atoms of the structure specified in 
            the same order as ``z``. The array can be two or three dimensional.
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
        ignore_criteria : :obj:`bool`, optional
            Whether to take into account structure criteria and their cutoffs
            for each model. Defaults to ``False``.
        store_each : :obj:`bool`, optional
            Store each n-body combination's contribution in the :obj:`dict`.
            Defaults to ``True``. Changing to ``False`` often speeds up
            predictions.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            A 1D array where each element is a :obj:`dict` for each structure.
            Each :obj:`dict` contains the total energy and its breakdown by
            total n-body order, and by entity combination. For example,
            if interested in the contribution of entities ``0`` and ``1``
            (``2``-body contribution) of the 5th structure, we would access
            this information with ``[4][2]['0,1']``. Getting the
            total energy of all contributions for that structure would be
            ``[4]['T']`` and for all 1-body contributions would be
            ``[4][1]['T']``. Each element's dictionary could have the following
            keys.

            ``'T'``
                Total energy of all n-body orders and contributions.
            ``1``
                All 1-body energy contributions. Usually there is no criteria
                for 1-body models so all entities are typically included.

                ``'T'``
                    Total 1-body contributions for the structure.
                
                ``'0'``
                    The 1-body contribution for entity ``'0'``.

            ``2``
                All 2-body energy contributions. Not all combination are
                included.

                ``'T'``
                    Total 2-body contributions for the structure.

                ``'0,1'``
                    The 2-body contribution for the dimer containing the ``0``
                    and ``1`` entities.
        :obj:`numpy.ndarray`
            Same as the energies array above but for atomic forces.
        """
        # Ensures R has three dimensions.
        if R.ndim == 2:
            R = np.array([R])

        E = np.empty(R.shape[0], dtype='object')
        F = np.empty(R.shape[0], dtype='object')

        # Adds contributions from all models.
        for i_r in range(len(R)):
            e = {'T': 0.0}
            f = {'T': np.zeros(R[i_r].shape)}
            for j in range(len(self.gdmls)):
                gdml = self.gdmls[j]
                model = self.models[j]
                nbody_order = int(len(set(self.entity_ids[j])))

                e[nbody_order], f[nbody_order] = \
                    self._calculate(
                        R[i_r], entity_ids, comp_ids, model, gdml,
                        ignore_criteria=ignore_criteria, store_each=store_each
                    )

                # Adds contributions to total energy and forces.
                if e[nbody_order]['T'] == 0.0:
                    e[nbody_order]['T'] = np.nan
                    f[nbody_order]['T'][:] = np.nan
                else:
                    e['T'] += e[nbody_order]['T']
                    f['T'] += f[nbody_order]['T']
            
            # If the total energy after all model predictions is zero we assume
            # that the structure falls outside all model criteria. This usually
            # only occurs with only one model. We replace the zero with NaN to 
            if e['T'] == 0.0:
                e['T'] = np.nan
                f['T'][:] = np.nan
            
            E[i_r] = e
            F[i_r] = f
        
        return E, F

    def predict(self, z, R, entity_ids, comp_ids, ignore_criteria=False):
        """Predicts total energy and atomic forces using many-body GDML models.
        
        Parameters
        ----------
        z : :obj:`numpy.ndarray`
            Atomic numbers of all atoms in the system.
        R : :obj:`numpy.ndarray`
            Cartesian coordinates of all atoms of the structure specified in 
            the same order as ``z``. The array can be two or three dimensional.
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
        ignore_criteria : :obj:`bool`, optional
            Whether to take into account structure criteria and their cutoffs
            for each model. Defaults to ``False``.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            Total energy of the system.
        :obj:`numpy.ndarray`
            Atomic forces of the system in the same shape as ``R``.
        """
        E_decomp, F_decomp = self.predict_decomposed(
            z, R, entity_ids, comp_ids, ignore_criteria=ignore_criteria,
            store_each=False
        )

        E = np.array([e['T'] for e in E_decomp])
        F = np.array([f['T'] for f in F_decomp])

        return E, F

    def remove_nbody(self, ref_dataset, ignore_criteria=False, store_each=False):
        """Removes mbGDML prediced energies and forces from a reference data
        set.

        Parameters
        ----------
        ref_dataset : :obj:`dict`
            Contains all data as :obj:`numpy.ndarray` objects.
        ignore_criteria : :obj:`bool`, optional
            Whether to take into account structure criteria and their cutoffs
            for each model. Defaults to ``False``.
        store_each : :obj:`bool`, optional
            Store each n-body combination's contribution in the :obj:`dict`.
            Defaults to ``False``.
        """
        nbody_dataset = ref_dataset
        z = nbody_dataset['z']
        R = nbody_dataset['R']
        E = nbody_dataset['E']
        F = nbody_dataset['F']
        num_config = R.shape[0]
        entity_num = len(set(nbody_dataset['entity_ids']))
        
        # Removing all n-body contributions for every configuration.
        for config in range(num_config):
            if (config+1)%500 == 0:
                print(f'Predicted {config+1} out of {num_config}')
            if z.ndim == 1:
                z_predict = z
            else:
                z_predict = z[config]
            e, f = self.predict_decomposed(
                z_predict, R[config], nbody_dataset['entity_ids'],
                nbody_dataset['comp_ids'], ignore_criteria=ignore_criteria,
                store_each=store_each
            )
            F[config] -= f
            E[config] -= e

        # Updates dataset.
        nbody_dataset['E'] = np.array(E)
        nbody_dataset['E_min'] = np.array(np.min(E.ravel()))
        nbody_dataset['E_max'] = np.array(np.max(E.ravel()))
        nbody_dataset['E_mean'] = np.array(np.mean(E.ravel()))
        nbody_dataset['E_var'] = np.array(np.var(E.ravel()))
        nbody_dataset['F'] = np.array(F)
        nbody_dataset['F_min'] = np.array(np.min(F.ravel()))
        nbody_dataset['F_max'] = np.array(np.max(F.ravel()))
        nbody_dataset['F_mean'] = np.array(np.mean(F.ravel()))
        nbody_dataset['F_var'] = np.array(np.var(F.ravel()))
        nbody_dataset['mb'] = np.array(entity_num)

        # Tries to add model md5 hashes to data set
        mb_models_md5 = []
        for model in self.models:
            if 'md5' in model.keys():
                mb_models_md5.append(model['md5'][()])
        nbody_dataset['mb_models_md5'] = mb_models_md5
        
        # Generating new data set name
        name_old = str(nbody_dataset['name'][()])
        nbody_label = str(entity_num) + 'body'
        name = '-'.join([name_old, nbody_label])
        nbody_dataset['name'] = np.array(name)

        return nbody_dataset
