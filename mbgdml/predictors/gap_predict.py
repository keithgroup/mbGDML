# MIT License
# 
# Copyright (c) 2022, Alex M. Maldonado
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

import logging
import numpy as np

log = logging.getLogger(__name__)

# Possible ray task.
def predict_gap(
    z, r, entity_ids, entity_combs, model, periodic_cell, ignore_criteria=False,
    **kwargs
):
    """Predict total :math:`n`-body energy and forces of a single structure.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`, ndim: ``1``
        Atomic numbers of all atoms in ``r`` (in the same order).
    r : :obj:`numpy.ndarray`, ndim: ``2``
        Cartesian coordinates of a single structure to predict.
    entity_ids : :obj:`numpy.ndarray`, ndim: ``1``
        1D array specifying which atoms belong to which entities.
    entity_combs : ``iterable``
        Entity ID combinations (e.g., ``(53,)``, ``(0, 2)``,
        ``(32, 55, 293)``, etc.) to predict using this model. These are used
        to slice ``r`` with ``entity_ids``.
    model : :obj:`mbgdml.models.gapModel`
        GAP model containing all information need to make predictions.
    periodic_cell : :obj:`mbgdml.periodic.Cell`, default: ``None``
        Use periodic boundary conditions defined by this object.
    ignore_criteria : :obj:`bool`, default: ``False``
        Ignore any criteria for predictions; i.e., all :math:`n`-body
        structures will be predicted.
    
    Returns
    -------
    :obj:`float`
        Predicted :math:`n`-body energy.
    :obj:`numpy.ndarray`
        Predicted :math:`n`-body forces.
    """
    assert r.ndim == 2
    E = 0.
    F = np.zeros(r.shape)

    if periodic_cell is not None:
        periodic = True
    else:
        periodic = False

    # Getting all contributions for each molecule combination (comb).
    first_r = True
    for entity_id_comb in entity_combs:
        log.debug(f'Entity combination: {entity_id_comb}')

        # Gets indices of all atoms in the combination of molecules.
        # r_slice is a list of the atoms for the entity_id combination.
        r_slice = []
        for entity_id in entity_id_comb:
            r_slice.extend(np.where(entity_ids == entity_id)[0])
        
        z_comp = z[r_slice]
        r_comp = r[r_slice]

        if first_r:
            atoms = ase.Atoms(z_comp)
            atoms.set_calculator(model.gap)
            first_r = False
        
        # If we are using a periodic cell we convert r_comp into coordinates
        # we can use in many-body expansions.
        if periodic:
            r_comp = periodic_cell.r_mic(r_comp)
            if r_comp is None:
                # Any atomic pairwise distance was larger than cutoff.
                continue
        
        # Checks criteria cutoff if present and desired.
        if model.criteria_cutoff is not None and not ignore_criteria:
            _, crit_val = model.criteria_desc_func(
                z_comp, r_comp, None, entity_ids[r_slice]
            )
            if crit_val >= model.criteria_cutoff:
                # Do not include this contribution.
                continue
        
        # Predicts energies and forces.
        atoms.set_positions(r_comp)
        e = atoms.get_potential_energy()
        f = atoms.get_forces()

        # Adds contributions to total energy and forces.
        E += e
        F[r_slice] += f
    
    return E, F

def predict_gap_decomp(
    z, r, entity_ids, entity_combs, model, ignore_criteria=False, **kwargs
):
    """Predict all :math:`n`-body energies and forces of a single structure.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`, ndim: ``1``
        Atomic numbers of all atoms in ``r`` (in the same order).
    r : :obj:`numpy.ndarray`, ndim: ``2``
        Cartesian coordinates of a single structure to predict.
    entity_ids : :obj:`numpy.ndarray`, ndim: ``1``
        1D array specifying which atoms belong to which entities.
    entity_combs : ``iterable``
        Entity ID combinations (e.g., ``(53,)``, ``(0, 2)``,
        ``(32, 55, 293)``, etc.) to predict using this model. These are used
        to slice ``r`` with ``entity_ids``.
    model : :obj:`mbgdml.models.gdmlModel`
        GDML model containing all information need to make predictions.
    
    Returns
    -------
    :obj:`numpy.ndarray`, ndim: ``1``
        Energies of all possible :math:`n`-body structures. Some elements
        can be :obj:`numpy.nan` if they are beyond the criteria cutoff.
    :obj:`numpy.ndarray`, ndim: ``3``
        Atomic forces of all possible :math:`n`-body structure. Some
        elements can be :obj:`numpy.nan` if they are beyond the criteria
        cutoff.
    :obj:`numpy.ndarray`, ndim: ``2``
        All possible :math:`n`-body combinations of ``r`` (i.e., entity ID
        combinations).
    """
    assert r.ndim == 2
    
    if entity_combs.ndim == 1:
        n_atoms = np.count_nonzero(entity_ids == entity_combs[0])
    else:
        n_atoms = 0
        for i in entity_combs[0]:
            n_atoms += np.count_nonzero(entity_ids == i)
    
    E = np.empty(len(entity_combs), dtype=np.float64)
    F = np.empty((len(entity_combs), n_atoms, 3), dtype=np.float64)
    E[:] = np.nan
    F[:] = np.nan

    # Getting all contributions for each molecule combination (comb).
    first_r = True
    for i in range(len(entity_combs)):
        entity_id_comb = entity_combs[i]
        # Gets indices of all atoms in the combination of molecules.
        # r_slice is a list of the atoms for the entity_id combination.
        r_slice = []
        for entity_id in entity_id_comb:
            r_slice.extend(np.where(entity_ids == entity_id)[0])
        
        z_comp = z[r_slice]
        r_comp = r[r_slice]
        if first_r:
            atoms = ase.Atoms(z_comp)
            atoms.set_calculator(model.gap)
            first_r = False
        
        # Checks criteria cutoff if present and desired.
        if model.criteria_cutoff is not None and not ignore_criteria:
            _, crit_val = model.criteria_desc_func(
                z_comp, r_comp, None, entity_ids[r_slice]
            )
            if crit_val >= model.criteria_cutoff:
                # Do not include this contribution.
                continue
        
        # Predicts energies and forces.
        atoms.set_positions(r_comp)
        E[i] = atoms.get_potential_energy()
        F[i] = atoms.get_forces()
    
    return E, F, entity_combs