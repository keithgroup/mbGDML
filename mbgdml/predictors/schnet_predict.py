# MIT License
#
# Copyright (c) 2022-2023, Alex M. Maldonado
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

import numpy as np
import ase
from ..logger import GDMLLogger

try:
    import schnetpack

    _HAS_SCHNETPACK = True
except ImportError:
    _HAS_SCHNETPACK = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

log = GDMLLogger(__name__)


# pylint: disable-next=unused-argument
def predict_schnet(Z, R, entity_ids, entity_combs, model, periodic_cell, **kwargs):
    r"""Predict total :math:`n`-body energy and forces of a single structure.

    Parameters
    ----------
    Z : :obj:`numpy.ndarray`, ndim: ``1``
        Atomic numbers of all atoms in ``r`` (in the same order).
    R : :obj:`numpy.ndarray`, ndim: ``2``
        Cartesian coordinates of a single structure to predict.
    entity_ids : :obj:`numpy.ndarray`, ndim: ``1``
        1D array specifying which atoms belong to which entities.
    entity_combs : ``iterable``
        Entity ID combinations (e.g., ``(53,)``, ``(0, 2)``,
        ``(32, 55, 293)``, etc.) to predict using this model. These are used
        to slice ``r`` with ``entity_ids``.
    model : :obj:`mbgdml.models.schnetModel`
        GAP model containing all information need to make predictions.
    periodic_cell : :obj:`mbgdml.periodic.Cell`, default: :obj:`None`
        Use periodic boundary conditions defined by this object.

    Returns
    -------
    :obj:`float`
        Predicted :math:`n`-body energy.
    :obj:`numpy.ndarray`
        Predicted :math:`n`-body forces.
    """
    assert _HAS_SCHNETPACK and _HAS_TORCH

    assert R.ndim == 2
    E = 0.0
    F = np.zeros(R.shape)

    # pylint: disable-next=no-member
    atom_conv = schnetpack.data.atoms.AtomsConverter(device=torch.device(model.device))

    periodic = bool(periodic_cell)

    for entity_id_comb in entity_combs:
        log.debug("Entity combination: %r", entity_id_comb)

        # Gets indices of all atoms in the combination of molecules.
        # r_slice is a list of the atoms for the entity_id combination.
        r_slice = []
        for entity_id in entity_id_comb:
            r_slice.extend(np.where(entity_ids == entity_id)[0])

        z_comp = Z[r_slice]
        r_comp = R[r_slice]

        # If we are using a periodic cell we convert r_comp into coordinates
        # we can use in many-body expansions.
        if periodic:
            r_comp = periodic_cell.r_mic(r_comp)
            if r_comp is None:
                # Any atomic pairwise distance was larger than cutoff.
                continue

        # Checks criteria cutoff if present and desired.
        if model.criteria is not None:
            accept_r, _ = model.criteria.accept(z_comp, r_comp)
            if not accept_r:
                # Do not include this contribution.
                continue

        # Making predictions
        pred = model.spk_model(atom_conv(ase.Atoms(z_comp, r_comp)))
        E += pred["energy"].cpu().detach().numpy()[0][0]
        F[r_slice] += pred["forces"].cpu().detach().numpy()[0]

    return E, F


# pylint: disable-next=unused-argument
def predict_schnet_decomp(Z, R, entity_ids, entity_combs, model, **kwargs):
    r"""Predict total :math:`n`-body energy and forces of a single structure.

    Parameters
    ----------
    Z : :obj:`numpy.ndarray`, ndim: ``1``
        Atomic numbers of all atoms in ``r`` (in the same order).
    R : :obj:`numpy.ndarray`, ndim: ``2``
        Cartesian coordinates of a single structure to predict.
    entity_ids : :obj:`numpy.ndarray`, ndim: ``1``
        1D array specifying which atoms belong to which entities.
    entity_combs : ``iterable``
        Entity ID combinations (e.g., ``(53,)``, ``(0, 2)``,
        ``(32, 55, 293)``, etc.) to predict using this model. These are used
        to slice ``r`` with ``entity_ids``.
    model : :obj:`mbgdml.models.schnetModel`
        GAP model containing all information need to make predictions.

    Returns
    -------
    :obj:`float`
        Predicted :math:`n`-body energy.
    :obj:`numpy.ndarray`
        Predicted :math:`n`-body forces.
    :obj:`numpy.ndarray`, ndim: ``2``
        All possible :math:`n`-body combinations of ``r`` (i.e., entity ID
        combinations).
    """
    assert R.ndim == 2

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

    # pylint: disable-next=no-member
    atom_conv = schnetpack.data.atoms.AtomsConverter(device=torch.device(model.device))

    for i, entity_id_comb in enumerate(entity_combs):
        log.debug("Entity combination: %r", entity_id_comb)

        # Gets indices of all atoms in the combination of molecules.
        # r_slice is a list of the atoms for the entity_id combination.
        r_slice = []
        for entity_id in entity_id_comb:
            r_slice.extend(np.where(entity_ids == entity_id)[0])

        z_comp = Z[r_slice]
        r_comp = R[r_slice]

        # Checks criteria cutoff if present and desired.
        if model.criteria is not None:
            accept_r, _ = model.criteria.accept(z_comp, r_comp)
            if not accept_r:
                # Do not include this contribution.
                continue

        # Making predictions
        pred = model.spk_model(atom_conv(ase.Atoms(z_comp, r_comp)))
        E[i] = pred["energy"].cpu().detach().numpy()[0][0]
        F[i] = pred["forces"].cpu().detach().numpy()[0]

    return E, F, entity_combs
