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

import logging
import numpy as np
import scipy as sp

log = logging.getLogger(__name__)

"""
Models
------

Classes that are used to stores information from ML models. These objects are
passed into prediction functions in this module.
"""

class mlModel(object):
    """"""

    def __init__(self, criteria_desc_func=None, criteria_cutoff=None):
        """
        Parameters
        ----------
        criteria_desc : ``callable``, default: ``None``
            A descriptor used to filter :math:`n`-body structures from being
            predicted.
        criteria_cutoff : :obj:`float`, default: ``None``
            Value of ``criteria_desc`` where the mlWorker will not predict the
            :math:`n`-body contribution of. If ``None``, no cutoff will be
            enforced.
        """
        self.criteria_desc_func = criteria_desc_func
        self.criteria_cutoff = criteria_cutoff


class gdmlModel(mlModel):

    def __init__(
        self, model, criteria_desc_func=None, criteria_cutoff=None,
        use_ray=False
    ):
        """
        Parameters
        ----------
        model : :obj:`str` or :obj:`dict`
            Path to GDML npz model or :obj:`dict`.
        criteria_desc_func : ``callable``, default: ``None``
            A descriptor used to filter :math:`n`-body structures from being
            predicted.
        criteria_cutoff : :obj:`float`, default: ``None``
            Value of ``criteria_desc_func`` where the mlWorker will not predict
            the :math:`n`-body contribution of. If ``None``, no cutoff will be
            enforced.
        use_ray : :obj:`bool`, default: ``False``
        """
        super().__init__(criteria_desc_func, criteria_cutoff)

        if isinstance(model, str):
            model = dict(np.load(model, allow_pickle=True))
        elif not isinstance(model, dict):
            raise TypeError(f'{type(model)} is not string or dict')
        
        self._pred_data_keys = [
            'sig', 'n_perms', 'desc_func', 'R_desc_perms',
            'R_d_desc_alpha_perms', 'alphas_E_lin'
        ]

        self.z = model['z']
        self.comp_ids = model['comp_ids']

        self.sig = model['sig']
        self.n_atoms = self.z.shape[0]
        interact_cut_off = (
            model['interact_cut_off'] if 'interact_cut_off' in model else None
        )
        self.desc_func = _from_r

        self.lat_and_inv = (
            (model['lattice'], np.linalg.inv(model['lattice']))
            if 'lattice' in model
            else None
        )
        self.n_train = model['R_desc'].shape[1]
        self.stdev = model['std'] if 'std' in model else 1.0
        self.integ_c = model['c']
        self.n_perms = model['perms'].shape[0]
        self.tril_perms_lin = model['tril_perms_lin']

        self.use_torch = False
        self.R_desc_perms = (
            np.tile(model['R_desc'].T, self.n_perms)[:, self.tril_perms_lin]
            .reshape(self.n_train, self.n_perms, -1, order='F')
            .reshape(self.n_train * self.n_perms, -1)
        )
        self.R_d_desc_alpha_perms = (
            np.tile(model['R_d_desc_alpha'], self.n_perms)[:, self.tril_perms_lin]
            .reshape(self.n_train, self.n_perms, -1, order='F')
            .reshape(self.n_train * self.n_perms, -1)
        )

        if 'alphas_E' in model.keys():
            self.alphas_E_lin = np.tile(
                model['alphas_E'][:, None], (1, n_perms)
            ).ravel()
        else:
            self.alphas_E_lin = None

        self.use_ray = use_ray





"""
Prediction workers
------------------

Functions that make or support ML predictions. Used in
:obj:`mbgdml.mbe.mbePredict``.
"""

from ._gdml.desc import _pdist, _r_to_desc, _r_to_d_desc, _from_r

# This calculation is too fast to be a ray task.
def _predict_gdml_wkr(
    r_desc, r_d_desc, n_atoms, sig, n_perms, R_desc_perms,
    R_d_desc_alpha_perms, alphas_E_lin, stdev, integ_c, wkr_start_stop=None
):
    """Compute (part) of a GDML prediction.

    Every prediction is a linear combination involving the training points used for
    this model. This function evaluates that combination for the range specified by
    `wkr_start_stop`. This workload can optionally be processed in chunks,
    which can be faster as it requires less memory to be allocated.

    Note
    ----
    It is sufficient to provide either the parameter ``r`` or ``r_desc_d_desc``.
    The other one can be set to ``None``.

    Parameters
    ----------

    Returns
    -------
    :obj:`numpy.ndarray`
        Partial prediction of all force components and energy (appended to
        array as last element).
    """
    n_train = int(R_desc_perms.shape[0] / n_perms)
    wkr_start, wkr_stop = (0, n_train)

    dim_d = (n_atoms * (n_atoms - 1)) // 2
    dim_i = 3 * n_atoms
    dim_c = n_train * n_perms

    # Pre-allocate memory.
    diff_ab_perms = np.empty((dim_c, dim_d))
    a_x2 = np.empty((dim_c,))
    mat52_base = np.empty((dim_c,))

    # avoid divisions (slower)
    sig_inv = 1.0 / sig
    mat52_base_fact = 5.0 / (3 * sig ** 3)
    diag_scale_fact = 5.0 / sig
    sqrt5 = np.sqrt(5.0)

    E_F = np.zeros((dim_d + 1,))
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
            'ji,ji->j', diff_ab_perms, rj_d_desc_alpha_perms, out=a_x2
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

    n = np.max((r_d_desc.shape[0], F.shape[0]))
    i, j = np.tril_indices(n_atoms, k=-1)

    out_F = np.zeros((n, n_atoms, n_atoms, 3))
    out_F[:, i, j, :] = r_d_desc * F[..., None]
    out_F[:, j, i, :] = -out_F[:, i, j, :]
    out[1:] = out_F.sum(axis=1).reshape(n, -1)
    
    out *= stdev
    E = out[0] + integ_c
    F = out[1:].reshape((n_atoms), 3)

    return E, F

# Possible ray task.
def predict_gdml(z, r, entity_ids, nbody_gen, model):
    """Predict energies and forces of a single structures.

    Parameters
    ----------
    model : :obj:`mbgdml.predict.gdmlModel``
        GDML model.
    z : :obj:`numpy.ndarray`, ndim: ``1``
        Atomic numbers of all atoms in ``r`` (in the same order).
    r : :obj:`numpy.ndarray`, ndim: ``2``
        Cartesian coordinates of a single structure to predict.
    entity_ids : :obj:`numpy.ndarray`, ndim: ``1``
        1D array specifying which atoms belong to which entities.
    nbody_gen : ``iterable``
        Entity ID combinations (e.g., ``(53,)``, ``(0, 2)``,
        ``(32, 55, 293)``, etc.) to predict using this model. These are used
        to slice ``R``.
    
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

    # Getting all contributions for each molecule combination (comb).
    for comb_entity_ids in nbody_gen:

        # Gets indices of all atoms in the combination of molecules.
        # r_slice is a list of the atoms for the entity_id combination.
        r_slice = []
        for entity_id in comb_entity_ids:
            r_slice.extend(np.where(entity_ids == entity_id)[0])
        
        # Checks criteria cutoff if present and desired.
        if model.criteria_cutoff is not None:
            _, crit_val = model.criteria_desc_func(
                model.z, r[r_slice], None, entity_ids[r_slice]
            )
            if crit_val >= model.criteria_cutoff:
                # Do not include this contribution.
                continue
        
        # Predicts energies and forces.
        z_comp = z[r_slice]
        r_comp = r[r_slice]
        r_desc, r_d_desc = model.desc_func(
            r_comp.flatten(), model.lat_and_inv
        )
        wkr_args = (
            r_desc, r_d_desc, len(z_comp), model.sig, model.n_perms,
            model.R_desc_perms, model.R_d_desc_alpha_perms,
            model.alphas_E_lin, model.stdev, model.integ_c
        ) 
        e, f = _predict_gdml_wkr(*wkr_args)

        # Adds contributions to total energy and forces.
        E += e
        F[r_slice] += f
    
    return E, F
