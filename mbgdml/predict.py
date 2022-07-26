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
import logging
import numpy as np
import scipy as sp
from .utils import md5_data

log = logging.getLogger(__name__)

"""
Models
------

Classes that are used to stores information from ML models. These objects are
passed into prediction functions in this module.
"""

class model(object):
    """A parent class for machine learning model objects.
    
    Attributes
    ----------
    md5 : :obj:`str`
        A property that creates a unique MD5 hash for the model. This is
        primarily only used in the creation of predict sets.
    nbody_order : :obj:`int`
        What order of :math:`n`-body contributions does this model predict?
        This is easily determined by taking the length of component IDs for the
        model.
    """

    def __init__(self, criteria_desc_func=None, criteria_cutoff=None):
        """
        Parameters
        ----------
        criteria_desc : ``callable``, default: ``None``
            A descriptor used to filter :math:`n`-body structures from being
            predicted.
        criteria_cutoff : :obj:`float`, default: ``None``
            Value of ``criteria_desc`` where the mlModel will not predict the
            :math:`n`-body contribution of. If ``None``, no cutoff will be
            enforced.
        """
        self.criteria_desc_func = criteria_desc_func
        # Make sure cutoff is a single value (weird extraction from npz).
        if isinstance(criteria_cutoff, np.ndarray):
            if len(criteria_cutoff) == 0:
                criteria_cutoff = None
            elif len(criteria_cutoff) == 1:
                criteria_cutoff = criteria_cutoff[0]
        self.criteria_cutoff = criteria_cutoff


class gdmlModel(model):

    def __init__(
        self, model, criteria_desc_func=None, criteria_cutoff=None
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
            Value of ``criteria_desc_func`` where the mlModel will not predict
            the :math:`n`-body contribution of. If ``None``, no cutoff will be
            enforced.
        """
        super().__init__(criteria_desc_func, criteria_cutoff)

        self.type = 'gdml'

        if isinstance(model, str):
            model = dict(np.load(model, allow_pickle=True))
        elif not isinstance(model, dict):
            raise TypeError(f'{type(model)} is not string or dict')
        self._model_dict = model

        self.z = model['z']
        self.comp_ids = model['comp_ids']
        self.nbody_order = len(self.comp_ids)

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
    
    @property
    def md5(self):
        """Unique MD5 hash of model.

        :type: :obj:`bytes`
        """
        md5_properties_all = [
            'md5_train', 'z', 'R_desc', 'R_d_desc_alpha', 'sig', 'alphas_F'
        ]
        md5_properties = []
        for key in md5_properties_all:
            if key in self._model_dict.keys():
                md5_properties.append(key)
        md5_string = md5_data(self._model_dict, md5_properties)
        return md5_string

class gapModel(model):

    def __init__(
        self, model_path, comp_ids, criteria_desc_func=None,
        criteria_cutoff=None
    ):
        """
        Parameters
        ----------
        model_path : :obj:`str`
            Path to GAP xml file.
        comp_ids : ``iterable``
            Model component IDs that relate entity IDs of a structure to a
            fragment label.
        criteria_desc_func : ``callable``, default: ``None``
            A descriptor used to filter :math:`n`-body structures from being
            predicted.
        criteria_cutoff : :obj:`float`, default: ``None``
            Value of ``criteria_desc_func`` where the mlModel will not predict
            the :math:`n`-body contribution of. If ``None``, no cutoff will be
            enforced.
        """
        global quippy
        import quippy

        super().__init__(criteria_desc_func, criteria_cutoff)
        self.type = 'gap'
        self.gap = quippy.potential.Potential(
            param_filename=model_path
        )
        if isinstance(comp_ids, list) or isinstance(comp_ids, tuple):
            comp_ids = np.array(comp_ids)
        self.comp_ids = comp_ids
        self.nbody_order = len(comp_ids)

        # GAP MD5
        with open(model_path, 'r') as f:
            gap_lines = f.readlines()
        import hashlib
        md5_hash = hashlib.md5()
        for i in range(len(gap_lines)):
            md5_hash.update(
                hashlib.md5(repr(gap_lines[i]).encode()).digest()
            )
        self.md5 = md5_hash.hexdigest()
        
class schnetModel(model):

    def __init__(
        self, model_path, comp_ids, device, criteria_desc_func=None,
        criteria_cutoff=None
    ):
        """
        Parameters
        ----------
        model_path : :obj:`str`
            Path to GAP xml file.
        comp_ids : ``iterable``
            Model component IDs that relate entity IDs of a structure to a
            fragment label.
        device : :obj:`str`
            The device where the model and tensors will be stored. For example,
            ``'cpu'`` and ``'cuda'``.
        criteria_desc_func : ``callable``, default: ``None``
            A descriptor used to filter :math:`n`-body structures from being
            predicted.
        criteria_cutoff : :obj:`float`, default: ``None``
            Value of ``criteria_desc_func`` where the mlModel will not predict
            the :math:`n`-body contribution of. If ``None``, no cutoff will be
            enforced.
        """
        global schnetpack, torch, ase
        import schnetpack, torch, ase

        super().__init__(criteria_desc_func, criteria_cutoff)
        self.type = 'schnet'
        self.spk_model = torch.load(
            model_path, map_location=torch.device(device)
        )
        self.device = device
        if isinstance(comp_ids, list) or isinstance(comp_ids, tuple):
            comp_ids = np.array(comp_ids)
        self.comp_ids = comp_ids
        self.nbody_order = len(comp_ids)

        # SchNet MD5
        import hashlib
        md5_hash = hashlib.md5()
        for param in self.spk_model.parameters():
            md5_hash.update(
                hashlib.md5(
                    param.cpu().detach().numpy().flatten()
                ).digest()
            )
        self.md5 = md5_hash.hexdigest()


"""
Prediction workers
------------------

Functions that make or support ML predictions. Used in
:obj:`mbgdml.mbe.mbePredict``.
"""

################
###   GDML   ###
################

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
    diff_ab_perms = np.empty((dim_c, dim_d), dtype=np.double)
    a_x2 = np.empty((dim_c,), dtype=np.double)
    mat52_base = np.empty((dim_c,), dtype=np.double)

    # avoid divisions (slower)
    sig_inv = 1.0 / sig
    mat52_base_fact = 5.0 / (3 * sig ** 3)
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

    return out

# Possible ray task.
def predict_gdml(z, r, entity_ids, entity_combs, model, ignore_criteria=False):
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
    model : :obj:`mbgdml.predict.gdmlModel`
        GDML model containing all information need to make predictions.
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

    # Getting all contributions for each molecule combination (comb).
    for entity_id_comb in entity_combs:

        # Gets indices of all atoms in the combination of molecules.
        # r_slice is a list of the atoms for the entity_id combination.
        r_slice = []
        for entity_id in entity_id_comb:
            r_slice.extend(np.where(entity_ids == entity_id)[0])
        
        # Checks criteria cutoff if present and desired.
        if model.criteria_cutoff is not None and not ignore_criteria:
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
        out = _predict_gdml_wkr(*wkr_args)

        out *= model.stdev
        e = out[0] + model.integ_c
        f = out[1:].reshape((len(z_comp), 3))

        # Adds contributions to total energy and forces.
        E += e
        F[r_slice] += f
    
    return E, F

def predict_gdml_decomp(
    z, r, entity_ids, entity_combs, model, ignore_criteria=False
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
    model : :obj:`mbgdml.predict.gdmlModel`
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
    for i in range(len(entity_combs)):
        entity_id_comb = entity_combs[i]
        # Gets indices of all atoms in the combination of molecules.
        # r_slice is a list of the atoms for the entity_id combination.
        r_slice = []
        for entity_id in entity_id_comb:
            r_slice.extend(np.where(entity_ids == entity_id)[0])
        
        # Checks criteria cutoff if present and desired.
        if model.criteria_cutoff is not None and not ignore_criteria:
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
        out = _predict_gdml_wkr(
            r_desc, r_d_desc, n_atoms, model.sig, model.n_perms,
            model.R_desc_perms, model.R_d_desc_alpha_perms,
            model.alphas_E_lin, model.stdev, model.integ_c
        )

        out *= model.stdev
        E[i] = out[0] + model.integ_c
        F[i] = out[1:].reshape((n_atoms, 3))
    
    return E, F, entity_combs

################
###    GAP   ###
################

import ase

# Possible ray task.
def predict_gap(z, r, entity_ids, entity_combs, model, ignore_criteria=False):
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
    model : :obj:`mbgdml.predict.gapModel`
        GAP model containing all information need to make predictions.
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
    z, r, entity_ids, entity_combs, model, ignore_criteria=False
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
    model : :obj:`mbgdml.predict.gdmlModel`
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

######################
###   SchNetPack   ###
######################

def predict_schnet(z, r, entity_ids, entity_combs, model, ignore_criteria=False):
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
    model : :obj:`mbgdml.predict.gapModel`
        GAP model containing all information need to make predictions.
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

    atom_conv = schnetpack.data.atoms.AtomsConverter(
        device=torch.device(model.device)
    )

    for entity_id_comb in entity_combs:
        log.debug(f'Entity combination: {entity_id_comb}')

        # Gets indices of all atoms in the combination of molecules.
        # r_slice is a list of the atoms for the entity_id combination.
        r_slice = []
        for entity_id in entity_id_comb:
            r_slice.extend(np.where(entity_ids == entity_id)[0])
        
        z_comb = z[r_slice]
        r_comb = r[r_slice]

        # Checks criteria cutoff if present and desired.
        if model.criteria_cutoff is not None and not ignore_criteria:
            _, crit_val = model.criteria_desc_func(
                z_comb, r_comb, None, entity_ids[r_slice]
            )
            if crit_val >= model.criteria_cutoff:
                continue
        
        # Making predictions
        pred = model.spk_model(atom_conv(ase.Atoms(z_comb, r_comb)))
        E += pred['energy'].cpu().detach().numpy()[0][0]
        F[r_slice] += pred['forces'].cpu().detach().numpy()[0]
    
    return E, F

def predict_schnet_decomp(
    z, r, entity_ids, entity_combs, model, ignore_criteria=False
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
    model : :obj:`mbgdml.predict.gapModel`
        GAP model containing all information need to make predictions.
    ignore_criteria : :obj:`bool`, default: ``False``
        Ignore any criteria for predictions; i.e., all :math:`n`-body
        structures will be predicted.
    
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

    atom_conv = schnetpack.data.atoms.AtomsConverter(
        device=torch.device(model.device)
    )

    for i in range(len(entity_combs)):
        entity_id_comb = entity_combs[i]

        log.debug(f'Entity combination: {entity_id_comb}')

        # Gets indices of all atoms in the combination of molecules.
        # r_slice is a list of the atoms for the entity_id combination.
        r_slice = []
        for entity_id in entity_id_comb:
            r_slice.extend(np.where(entity_ids == entity_id)[0])
        
        z_comb = z[r_slice]
        r_comb = r[r_slice]

        # Checks criteria cutoff if present and desired.
        if model.criteria_cutoff is not None and not ignore_criteria:
            _, crit_val = model.criteria_desc_func(
                z_comb, r_comb, None, entity_ids[r_slice]
            )
            if crit_val >= model.criteria_cutoff:
                continue
        
        # Making predictions
        pred = model.spk_model(atom_conv(ase.Atoms(z_comb, r_comb)))
        E[i] = pred['energy'].cpu().detach().numpy()[0][0]
        F[i] = pred['forces'].cpu().detach().numpy()[0]
    
    return E, F, entity_combs
