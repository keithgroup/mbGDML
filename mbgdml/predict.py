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

import itertools

import numpy as np
from sgdml.predict import GDMLPredict
from torch.functional import cartesian_prod
import mbgdml.solvents as solvents
from mbgdml import criteria

class mbPredict():
    """Predict energies and forces of structures using GDML models.
    """

    def __init__(self, models):
        """Sets GDML models to be used for many-body predictions.
        
        Parameters
        ----------
        models : :obj:`list` [:obj:`str`]
            Contains paths to either standard or many-body GDML models.
        """
        self._load_models(models)
    

    def _load_models(self, models):
        """Loads models and preprares GDMLPredict.
        
        Parameters
        ----------
        models : :obj:`list` [:obj:`str`]
            Contains paths to either standard or many-body GDML models.
        """
        self.gdmls = []
        self.models = []
        self.entity_ids = []
        self.comp_ids = []
        self.criteria = []
        self.z_slice = []
        self.cutoff = []
        for model in models:
            loaded = np.load(model, allow_pickle=True)
            model = dict(loaded)
            self.models.append(model)
            gdml = GDMLPredict(loaded)
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
            :obj:`numpy.ndarray``is equal to the model ``entity_id`` and the
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
            An array specifying which atoms belong to what entities
            (e.g., molecules) for ``r``.
        comp_ids : :obj:`numpy.ndarray`
            A 2D array relating ``entity_ids`` to a chemical component/species
            id or label (``comp_id``) for ``r``. The first column is the unique
            ``entity_id`` and the second is a unique ``comp_id`` for that
            chemical species. Each ``comp_id`` is reused for the same chemical
            species.
        model : :obj:`dict`
            The dictionary of the loaded npz file. Stored in ``self.models``.
        gdml : :obj:`sgdml.predict.GDMLPredict`
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

        # 'T' is for total
        E_contributions = {'T': 0.0}
        F_contributions = {'T': np.zeros(r.shape)}

        # Models have a specific entity order that needs to be conserved in the
        # predictions. Here, we create a ``entity_idxs`` list where each item
        # is a list of all entities in ``r`` that match the model entity.
        r_entity_ids_per_model_entity = []
        for model_comp_id in model['comp_ids']:
            matching_entity_ids = np.where(comp_ids[:,1] == model_comp_id[1])[0]
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
            
            # Predicts energies
            e, f = gdml.predict(r[r_idx].flatten())

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


    def decomposed_predict(
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
            A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is
            the number of structures and ``n`` is the number of atoms with three 
            Cartesian components.
        entity_ids : :obj:`numpy.ndarray`
            An array specifying which atoms belong to what entities
            (e.g., molecules).
        comp_ids : :obj:`numpy.ndarray`
            A 2D array relating ``entity_ids`` to a chemical component/species
            id or label (``comp_id``). The first column is the unique
            ``entity_id`` and the second is a unique ``comp_id`` for that
            chemical species. Each ``comp_id`` is reused for the same chemical
            species.
        ignore_criteria : :obj:`bool`, optional
            Whether to take into account structure criteria and their cutoffs
            for each model. Defaults to ``False``.
        store_each : :obj:`bool`, optional
            Store each n-body combination's contribution in the :obj:`dict`.
            Defaults to ``True``. Changing to ``False`` often speeds up
            predictions.
        
        Returns
        -------
        tuple
            Dictionaries of many-body contributions of total energies
            (float) and forces (np.ndarray) of the structure. Note that
            forces has the same shape as R. Each key of the dictionary is
            the order of the many-body model. Within this dictionary is a
            total, 'T', and molecule combinations identified in the system
            and their contributions.
        """
        # Gets system information from dataset.
        # This assumes the system is only solvent.
        if type(z) != list:
            z = z.tolist()

        # 'T' is for total
        E_contributions = {'T': 0.0}
        F_contributions = {'T': np.zeros(R.shape)}

        # Adds contributions from all models.
        for i in range(len(self.gdmls)):
            gdml = self.gdmls[i]
            model = self.models[i]
            nbody_order = int(len(set(self.entity_ids[i])))

            E_contributions[nbody_order], F_contributions[nbody_order] = \
                self._calculate(
                    R, entity_ids, comp_ids, model, gdml,
                    ignore_criteria=ignore_criteria, store_each=store_each
                )

            # Adds contributions to total energy and forces.
            E_contributions['T'] += E_contributions[nbody_order]['T']
            F_contributions['T'] += F_contributions[nbody_order]['T']
        
        return E_contributions, F_contributions


    def predict(self, z, R, entity_ids, comp_ids, ignore_criteria=False):
        """Predicts total energy and atomic forces using many-body GDML models.
        
        Parameters
        ----------
        z : :obj:`numpy.ndarray`
            Atomic numbers of all atoms in the system.
        R : :obj:`numpy.ndarray`
            Cartesian coordinates of all atoms of the structure specified in 
            the same order as ``z``. The array should have shape ``(n, 3)``
            where ``n`` is the number of atoms.
        entity_ids : :obj:`numpy.ndarray`
            An array specifying which atoms belong to what entities
            (e.g., molecules).
        comp_ids : :obj:`numpy.ndarray`
            A 2D array relating ``entity_ids`` to a chemical component/species
            id or label (``comp_id``). The first column is the unique
            ``entity_id`` and the second is a unique ``comp_id`` for that
            chemical species. Each ``comp_id`` is reused for the same chemical
            species.
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
        e, f = self.decomposed_predict(
            z, R, entity_ids, comp_ids, ignore_criteria=ignore_criteria,
            store_each=False
        )
        
        return e['T'], f['T']


    def remove_nbody(self, ref_dataset):
        """Removes mbGDML prediced energies and forces from a reference data
        set.

        Parameters
        ----------
        ref_dataset : :obj:`dict`
            Contains all data as :obj:`numpy.ndarray` objects.
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
            if z.ndim == 1:
                z_predict = z
            else:
                z_predict = z[config]
            e, f = self.predict(
                z_predict, R[config], nbody_dataset['entity_ids'],
                nbody_dataset['comp_ids'], ignore_criteria=False
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
