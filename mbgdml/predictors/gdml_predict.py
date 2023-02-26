# MIT License
#
# Copyright (c) 2018-2022 Stefan Chmiela, Gregory Fonseca
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
from ..logger import GDMLLogger
from ..stress import virial_atom_loop

log = GDMLLogger(__name__)

# This calculation is too fast to be a ray task.
# pylint: disable-next=too-many-statements
def _predict_gdml_wkr(
    r_desc,
    r_d_desc,
    n_atoms,
    sig,
    n_perms,
    R_desc_perms,
    R_d_desc_alpha_perms,
    alphas_E_lin,
    wkr_start_stop=None,
    chunk_size=None,
):
    r"""Compute (part) of a GDML prediction.

    Every prediction is a linear combination involving the training points used for
    this model. This function evaluates that combination for the range specified by
    `wkr_start_stop`. This workload can optionally be processed in chunks,
    which can be faster as it requires less memory to be allocated.

    Note
    ----
    It is sufficient to provide either the parameter ``r`` or ``r_desc_d_desc``.
    The other one can be set to :obj:`None`.

    Returns
    -------
    :obj:`numpy.ndarray`
        Partial prediction of all force components and energy (appended to
        array as last element).
    """
    # pylint: disable=invalid-name
    ###   mbGDML CHANGE   ###
    n_train = int(R_desc_perms.shape[0] / n_perms)
    ###   mbGDML CHANGE END   ###

    wkr_start, wkr_stop = (0, n_train) if wkr_start_stop is None else wkr_start_stop
    if chunk_size is None:
        chunk_size = n_train

    ###   mbGDML CHANGE   ###
    dim_d = (n_atoms * (n_atoms - 1)) // 2
    dim_i = 3 * n_atoms
    dim_c = chunk_size * n_perms
    ###   mbGDML CHANGE END   ###

    # Pre-allocate memory.
    diff_ab_perms = np.empty((dim_c, dim_d), dtype=np.double)
    a_x2 = np.empty((dim_c,), dtype=np.double)
    mat52_base = np.empty((dim_c,), dtype=np.double)

    # avoid divisions (slower)
    sig_inv = 1.0 / sig
    mat52_base_fact = 5.0 / (3 * sig**3)
    diag_scale_fact = 5.0 / sig
    sqrt5 = np.sqrt(5.0)

    E_F = np.zeros((dim_d + 1,), dtype=np.double)
    F = E_F[1:]

    wkr_start *= n_perms
    wkr_stop *= n_perms

    b_start = wkr_start
    for b_stop in list(range(wkr_start + dim_c, wkr_stop, dim_c)) + [wkr_stop]:

        rj_desc_perms = R_desc_perms[b_start:b_stop, :]
        rj_d_desc_alpha_perms = R_d_desc_alpha_perms[b_start:b_stop, :]

        # Resize pre-allocated memory for last iteration,
        # if chunk_size is not a divisor of the training set size.
        # Note: It's faster to process equally sized chunks.
        c_size = b_stop - b_start
        if c_size < dim_c:
            diff_ab_perms = diff_ab_perms[:c_size, :]
            a_x2 = a_x2[:c_size]
            mat52_base = mat52_base[:c_size]

        np.subtract(
            np.broadcast_to(r_desc, rj_desc_perms.shape),
            rj_desc_perms,
            out=diff_ab_perms,
        )
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

        np.exp(-norm_ab_perms * sig_inv, out=mat52_base)
        mat52_base *= mat52_base_fact
        np.einsum(
            "ji,ji->j", diff_ab_perms, rj_d_desc_alpha_perms, out=a_x2
        )  # colum wise dot product

        F += (a_x2 * mat52_base).dot(diff_ab_perms) * diag_scale_fact
        mat52_base *= norm_ab_perms + sig
        F -= mat52_base.dot(rj_d_desc_alpha_perms)

        # Energies are automatically predicted with a flipped sign here
        # (because -E are trained, instead of E)
        E_F[0] += a_x2.dot(mat52_base)

        # Energies are automatically predicted with a flipped sign here
        # (because -E are trained, instead of E)
        if alphas_E_lin is not None:

            K_fe = diff_ab_perms * mat52_base[:, None]
            F += alphas_E_lin[b_start:b_stop].dot(K_fe)

            K_ee = (
                1 + (norm_ab_perms * sig_inv) * (1 + norm_ab_perms / (3 * sig))
            ) * np.exp(-norm_ab_perms * sig_inv)
            E_F[0] += K_ee.dot(alphas_E_lin[b_start:b_stop])

        b_start = b_stop

    out = E_F[: dim_i + 1]

    # Descriptor has less entries than 3N, need to extend size of the 'E_F' array.
    if dim_d < dim_i:
        out = np.empty((dim_i + 1,))
        out[0] = E_F[0]

    # 'r_d_desc.T.dot(F)' for our special representation of 'r_d_desc'
    r_d_desc = r_d_desc[None, ...]
    F = F[None, ...]

    ###   mbGDML CHANGE   ###
    n = np.max((r_d_desc.shape[0], F.shape[0]))
    i, j = np.tril_indices(n_atoms, k=-1)

    out_F = np.zeros((n, n_atoms, n_atoms, 3))
    out_F[:, i, j, :] = r_d_desc * F[..., None]
    out_F[:, j, i, :] = -out_F[:, i, j, :]
    out[1:] = out_F.sum(axis=1).reshape(n, -1)
    ###   mbGDML CHANGE END   ###

    return out


# Possible ray task.
# pylint: disable-next=too-many-branches
def predict_gdml(Z, R, entity_ids, entity_combs, model, periodic_cell, **kwargs):
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
    model : :obj:`mbgdml.models.gdmlModel`
        GDML model containing all information need to make predictions.
    periodic_cell : :obj:`mbgdml.periodic.Cell`, default: :obj:`None`
        Use periodic boundary conditions defined by this object.

    Returns
    -------
    :obj:`float`
        Predicted :math:`n`-body energy.
    :obj:`numpy.ndarray`
        Predicted :math:`n`-body forces.
    """
    assert R.ndim == 2
    E = 0.0
    F = np.zeros(R.shape)

    alchemy_scalers = kwargs.get("alchemy_scalers", None)
    compute_virial = kwargs.get("compute_virial", False)

    periodic = bool(periodic_cell)
    compute_virial = bool(periodic and compute_virial)
    if compute_virial:
        virial = np.zeros((3, 3), dtype=np.float64)

    # Getting all contributions for each molecule combination (comb).
    for entity_id_comb in entity_combs:

        # Gets indices of all atoms in the combination of molecules.
        # r_slice is a list of the atoms for the entity_id combination.
        r_slice = []
        for entity_id in entity_id_comb:
            r_slice.extend(np.where(entity_ids == entity_id)[0])

        z = Z[r_slice]
        r = R[r_slice]
        # Note: We store the original coordinates in case the virial is requested

        # If we are using a periodic cell we convert r_comp into coordinates
        # we can use in many-body expansions.
        if periodic:
            r = periodic_cell.r_mic(r)
            if r is None:
                # Any atomic pairwise distance was larger than cutoff.
                continue

        # Checks criteria cutoff if present and desired.
        if model.criteria is not None:
            accept_r, _ = model.criteria.accept(z, r)
            if not accept_r:
                # Do not include this contribution.
                continue

        # Predicts energies and forces.
        r_desc, r_d_desc = model.desc_func(r.flatten(), model.lat_and_inv)
        wkr_args = (
            r_desc,
            r_d_desc,
            len(z),
            model.sig,
            model.n_perms,
            model.R_desc_perms,
            model.R_d_desc_alpha_perms,
            model.alphas_E_lin,
        )
        out = _predict_gdml_wkr(*wkr_args)

        out *= model.stdev
        e = out[0] + model.integ_c
        f = out[1:].reshape((len(z), 3))

        # Scale contribution if entity is included.
        if alchemy_scalers not in (None, []):
            for alchemy_scaler in alchemy_scalers:
                if alchemy_scaler.entity_id in entity_id_comb:
                    e = alchemy_scaler.scale(e)
                    f = alchemy_scaler.scale(f)

        # Adds contributions to total energy and forces.
        E += e
        F[r_slice] += f

        if compute_virial:
            virial += virial_atom_loop(r, f)

    if compute_virial:
        return E, F, virial

    return E, F


def predict_gdml_decomp(Z, R, entity_ids, entity_combs, model, **kwargs):
    r"""Predict all :math:`n`-body energies and forces of a single structure.

    Parameters
    ----------
    Z : :obj:`numpy.ndarray`, ndim: ``1``
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

    alchemy_scalers = kwargs.get("alchemy_scalers", None)

    # Getting all contributions for each molecule combination (comb).
    for i, entity_id_comb in enumerate(entity_combs):
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

        # Predicts energies and forces.
        r_desc, r_d_desc = model.desc_func(r_comp.flatten(), model.lat_and_inv)
        out = _predict_gdml_wkr(
            r_desc,
            r_d_desc,
            n_atoms,
            model.sig,
            model.n_perms,
            model.R_desc_perms,
            model.R_d_desc_alpha_perms,
            model.alphas_E_lin,
        )

        out *= model.stdev
        E[i] = out[0] + model.integ_c
        F[i] = out[1:].reshape((n_atoms, 3))

        # Scale contribution if entity is included.
        if alchemy_scalers not in (None, []):
            for alchemy_scaler in alchemy_scalers:
                if alchemy_scaler.entity_id in entity_id_comb:
                    E[i] = alchemy_scaler.scale(E[i])
                    F[i] = alchemy_scaler.scale(F[i])

    return E, F, entity_combs
