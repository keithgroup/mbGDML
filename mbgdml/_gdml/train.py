# MIT License
# 
# Copyright (c) 2018-2020, Stefan Chmiela
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

import numpy as np
import scipy as sp
from .sample import draw_strat_sample
from .perm import find_perms
from .desc import Desc
from .predict import GDMLPredict
from ..utils import md5_data
from .. import __version__

import multiprocessing as mp
Pool = mp.get_context('fork').Pool
from functools import partial
import warnings

try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True

import logging
log = logging.getLogger(__name__)

def _share_array(arr_np, typecode_or_type):
    """Return a ctypes array allocated from shared memory with data from a
    NumPy array.

    Parameters
    ----------
    arr_np : :obj:`numpy.ndarray`
        NumPy array.
    typecode_or_type : char or ``ctype``
        Either a ctypes type or a one character typecode of the
        kind used by the Python array module.

    Returns
    -------
        Array of ``ctype``.
    """
    arr = mp.RawArray(typecode_or_type, arr_np.ravel())
    return arr, arr_np.shape

def _assemble_kernel_mat_wkr(
    j, tril_perms_lin, sig, use_E_cstr=False, exploit_sym=False, cols_m_limit=None
):
    """Compute one row and column of the force field kernel matrix.

    The Hessian of the Matern kernel is used with n = 2 (twice
    differentiable). Each row and column consists of matrix-valued
    blocks, which encode the interaction of one training point with all
    others. The result is stored in shared memory (a global variable).

    Parameters
    ----------
    j : :obj:`int`
        Index of training point.
    tril_perms_lin : :obj:`numpy.ndarray` of :obj:`int`
        1D array containing all recovered permutations expanded as one large
        permutation to be applied to a tiled copy of the object to be permuted.
    sig : :obj:`int` or :obj:`float`
        Hyperparameter sigma (kernel length scale).
    use_E_cstr : :obj:`bool`, optional
        True: include energy constraints in the kernel,
        False: default (s)GDML kernel.
    exploit_sym : :obj:`bool`, optional
        Do not create symmetric entries of the kernel matrix twice
        (this only works for specific inputs for ``cols_m_limit``)
    cols_m_limit : :obj:`int`, optional
        Limit the number of columns (include training points 1-``M``).
        Note that each training points consists of multiple columns.

    Returns
    -------
    :obj:`int`
        Number of kernel matrix blocks created, divided by 2
        (symmetric blocks are always created at together).
    """

    global glob

    R_desc = np.frombuffer(glob['R_desc']).reshape(glob['R_desc_shape'])
    R_d_desc = np.frombuffer(glob['R_d_desc']).reshape(glob['R_d_desc_shape'])
    K = np.frombuffer(glob['K']).reshape(glob['K_shape'])

    desc_func = glob['desc_func']

    n_train, dim_d = R_d_desc.shape[:2]
    n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    dim_i = 3 * n_atoms
    n_perms = int(len(tril_perms_lin) / dim_d)

    if type(j) is tuple:  # selective/"fancy" indexing
        # (block index in final K, block index global, indices of partials within block)
        (K_j, j,keep_idxs_3n,) = j
        blk_j = slice(K_j, K_j + len(keep_idxs_3n))

    else:  # sequential indexing
        blk_j = slice(j * dim_i, (j + 1) * dim_i)
        keep_idxs_3n = slice(None)  # same as [:]

    # TODO: document this exception
    if use_E_cstr and not (cols_m_limit is None or cols_m_limit == n_train):
        raise ValueError(
            '\'use_E_cstr\'- and \'cols_m_limit\'-parameters are mutually exclusive!'
        )

    # Create permutated variants of 'rj_desc' and 'rj_d_desc'.
    rj_desc_perms = np.reshape(
        np.tile(R_desc[j, :], n_perms)[tril_perms_lin], (n_perms, -1), order='F'
    )

    # convert descriptor back to full representation
    rj_d_desc = desc_func.d_desc_from_comp(R_d_desc[j,:,:])[0][:,keep_idxs_3n]
    rj_d_desc_perms = np.reshape(
        np.tile(rj_d_desc.T, n_perms)[:, tril_perms_lin], (-1, dim_d, n_perms)
    )

    mat52_base_div = 3 * sig ** 4
    sqrt5 = np.sqrt(5.0)
    sig_pow2 = sig ** 2

    dim_i_keep = rj_d_desc.shape[1]
    diff_ab_outer_perms = np.empty((dim_d, dim_i_keep))
    diff_ab_perms = np.empty((n_perms, dim_d))
    ri_d_desc = np.zeros((1, dim_d, dim_i)) # must be zeros!
    k = np.empty((dim_i, dim_i_keep))

    for i in range(j if exploit_sym else 0, n_train):

        blk_i = slice(i * dim_i, (i + 1) * dim_i)

        np.subtract(R_desc[i, :], rj_desc_perms, out=diff_ab_perms)

        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
        mat52_base_perms = np.exp(-norm_ab_perms / sig) / mat52_base_div * 5

        np.einsum(
            'ki,kj->ij',
            diff_ab_perms * mat52_base_perms[:, None] * 5,
            np.einsum('ki,jik -> kj', diff_ab_perms, rj_d_desc_perms),
            out=diff_ab_outer_perms
        )

        diff_ab_outer_perms -= np.einsum(
            'ikj,j->ki',
            rj_d_desc_perms,
            (sig_pow2 + sig * norm_ab_perms) * mat52_base_perms,
        )

        #ri_d_desc = desc_func.d_desc_from_comp(R_d_desc[i, :, :])[0]
        desc_func.d_desc_from_comp(R_d_desc[i, :, :], out=ri_d_desc)

        #K[blk_i, blk_j] = ri_d_desc[0].T.dot(diff_ab_outer_perms)
        np.dot(ri_d_desc[0].T, diff_ab_outer_perms, out=k)
        K[blk_i, blk_j] = k

        # this will never be called with 'keep_idxs_3n' set to anything else than [:]
        if exploit_sym and (cols_m_limit is None or i < cols_m_limit):
            K[blk_j, blk_i] = K[blk_i, blk_j].T

    if use_E_cstr:

        E_off = K.shape[0] - n_train, K.shape[1] - n_train
        blk_j_full = slice(j * dim_i, (j + 1) * dim_i)
        for i in range(n_train):

            diff_ab_perms = R_desc[i, :] - rj_desc_perms
            norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

            K_fe = (
                5
                * diff_ab_perms
                / (3 * sig ** 3)
                * (norm_ab_perms[:, None] + sig)
                * np.exp(-norm_ab_perms / sig)[:, None]
            )
            K_fe = -np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)
            K[blk_j_full, E_off[1] + i] = K_fe  # vertical
            K[E_off[0] + i, blk_j] = K_fe[keep_idxs_3n]  # lower horizontal

            K[E_off[0] + i, E_off[1] + j] = K[E_off[0] + j, E_off[1] + i] = -(
                1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            ).dot(np.exp(-norm_ab_perms / sig))

    return blk_j.stop - blk_j.start


class GDMLTrain(object):
    """Train GDML force fields.

    This class is used to train models using different closed-form
    and numerical solvers. GPU support is provided
    through PyTorch (requires optional `torch` dependency to be
    installed) for some solvers.
    """
    def __init__(self, max_processes=None, use_torch=False):
        """
        Parameters
        ----------
        max_processes : :obj:`int`, default: ``None``
            Limit the max. number of processes. Otherwise all CPU cores are
            used. This parameters has no effect if `use_torch=True`
        use_torch : boolean, default: ``False``
            Use PyTorch to calculate predictions (if supported by solver).

        Raises
        ------
        Exception
            If multiple instances of this class are created.
        ImportError
            If the optional PyTorch dependency is missing, but PyTorch features are used.
        """

        global glob
        if 'glob' not in globals():  # Don't allow more than one instance of this class.
            glob = {}
        else:
            raise Exception(
                'You can not create multiple instances of this class. '
                'Please reuse your first one.'
            )
        
        self._max_processes = max_processes
        self._use_torch = use_torch

        if use_torch and not _has_torch:
            raise ImportError(
                'Optional PyTorch dependency not found!'
            )

    def __del__(self):

        global glob

        if 'glob' in globals():
            del glob

    def create_task(
        self, train_dataset, n_train, valid_dataset, n_valid, sig, lam=1e-15,
        use_sym=True, use_E=True, use_E_cstr=False, use_cprsn=False,
        solver='analytic', solver_tol=1e-4, interact_cut_off=None,
        idxs_train=None, idxs_valid=None,
    ):
        """Create a data structure of custom type ``task``.

        These data structures serve as recipes for model creation,
        summarizing the configuration of one particular training run.
        Training and test points are sampled from the provided dataset,
        without replacement. If the same dataset if given for training
        and testing, the subsets are drawn without overlap.

        Each task also contains a choice for the hyper-parameters of the
        training process and the MD5 fingerprints of the used datasets.

        Parameters
        ----------
        train_dataset : :obj:`dict`
            Data structure of custom type ``dataset`` containing
            train dataset.
        n_train : :obj:`int`
            Number of training points to sample.
        valid_dataset : :obj:`dict`
            Data structure of custom type ``dataset`` containing
            validation dataset.
        n_valid : :obj:`int`
            Number of validation points to sample.
        sig : :obj:`int`
            Hyperparameter sigma (kernel length scale).
        lam : :obj:`float`, default: ``1e-15``
            Hyper-parameter lambda (regularization strength).
        use_sym : bool, default: ``True``
            True: include symmetries (sGDML), False: GDML.
        use_E : :obj:`bool`, optional
            ``True``: reconstruct force field with corresponding potential energy surface,

            ``False``: ignore energy during training, even if energy labels are available
            in the dataset. The trained model will still be able to predict
            energies up to an unknown integration constant. Note, that the
            energy predictions accuracy will be untested.
        use_E_cstr : bool, default: ``False``
            True: include energy constraints in the kernel,
            False: default (s)GDML.
        use_cprsn : bool, default: ``False``
            True: compress kernel matrix along symmetric degrees of
            freedom,
            False: train using full kernel matrix

        Returns
        -------
        :obj:`dict`
            Data structure of custom type ``task``.

        Raises
        ------
        ValueError
            If a reconstruction of the potential energy surface is requested,
            but the energy labels are missing in the dataset.
        """
        t_create_task = log.t_start()
        log.info(
            '-----------------------------------\n'
            '|   Creating GDML training task   |\n'
            '-----------------------------------\n'
        )

        log.log_model(
            {'z': train_dataset['z'], 'n_train': n_train, 'n_valid': n_valid,
            'sig': sig, 'lam': lam, 'use_sym': use_sym, 'use_E': use_E,
            'use_E_cstr': use_E_cstr, 'use_cprsn': use_cprsn, 'type': 't'}
        )

        if use_E and 'E' not in train_dataset:
            raise ValueError(
                'No energy labels found in dataset!\n'
                + 'By default, force fields are always reconstructed including the\n'
                + 'corresponding potential energy surface (this can be turned off).\n'
                + 'However, the energy labels are missing in the provided dataset.\n'
            )

        use_E_cstr = use_E and use_E_cstr
        
        log.info(
            '\nDataset splitting\n'
            '-----------------'
        )
        md5_train_keys = ['z', 'R', 'F']
        md5_valid_keys = ['z', 'R', 'F']
        if 'E' in train_dataset.keys():
            md5_train_keys.insert(2, 'E')
        if 'E' in valid_dataset.keys():
            md5_valid_keys.insert(2, 'E')
        md5_train = md5_data(train_dataset, md5_train_keys)
        md5_valid = md5_data(valid_dataset, md5_valid_keys)
        
        log.info(
            '\n#   Training   #'
        )
        log.info(f'MD5: {md5_train}')
        log.info(f'Size : {n_train}')
        if idxs_train is None:
            log.info('Drawing structures from the dataset')
            if 'E' in train_dataset:
                log.info(
                    'Energies are included in the dataset\n'
                    'Using the Freedman-Diaconis rule'
                )
                idxs_train = draw_strat_sample(
                    train_dataset['E'], n_train
                )
            else:
                log.info(
                    'Energies are not included in the dataset\n'
                    'Randomly selecting structures'
                )
                idxs_train = np.random.choice(
                    np.arange(train_dataset['F'].shape[0]),
                    size=n_train,
                    replace=False,
                )
        else:
            log.info('Training indices were manually specified')
            idxs_train = np.array(idxs_train)
            log.log_array(idxs_train, level=10)

        # Handles validation indices.
        log.info(
            '\n#   Validation   #'
        )
        log.info(f'MD5: {md5_valid}')
        log.info(f'Size : {n_valid}')
        if idxs_valid is not None:
            log.info('Validation indices were manually specified')
            idxs_valid = np.array(idxs_valid)
            log.log_array(idxs_valid, level=10)
        else:
            log.info('Drawing structures from the dataset')
            excl_idxs = (
                idxs_train if md5_train == md5_valid else np.array([], dtype=np.uint)
            )
            log.debug(f'Excluded {len(excl_idxs)} structures')
            log.log_array(excl_idxs, level=10)

            if 'E' in valid_dataset:
                idxs_valid = draw_strat_sample(
                    valid_dataset['E'], n_valid, excl_idxs=excl_idxs,
                )
            else:
                idxs_valid_cands = np.setdiff1d(
                    np.arange(valid_dataset['F'].shape[0]), excl_idxs,
                    assume_unique=True
                )
                idxs_valid = np.random.choice(
                    idxs_valid_cands, size=n_valid, replace=False
                )

        R_train = train_dataset['R'][idxs_train, :, :]
        task = {
            'type': 't',
            'code_version': __version__,
            'dataset_name': train_dataset['name'].astype(str),
            'dataset_theory': train_dataset['theory'].astype(str),
            'z': train_dataset['z'],
            'R_train': R_train,
            'F_train': train_dataset['F'][idxs_train, :, :],
            'idxs_train': idxs_train,
            'md5_train': md5_train,
            'idxs_valid': idxs_valid,
            'md5_valid': md5_valid,
            'sig': sig,
            'lam': lam,
            'use_E': use_E,
            'use_E_cstr': use_E_cstr,
            'use_sym': use_sym,
            'use_cprsn': use_cprsn,
            'solver_name': solver,
            'solver_tol': solver_tol,
            'interact_cut_off': interact_cut_off,
        }

        if use_E:
            task['E_train'] = train_dataset['E'][idxs_train]

        lat_and_inv = None
        if 'lattice' in train_dataset:
            log.info('\nLattice was found in the dataset')
            log.debug(train_dataset['lattice'])
            task['lattice'] = train_dataset['lattice']

            try:
                lat_and_inv = (task['lattice'], np.linalg.inv(task['lattice']))
            except np.linalg.LinAlgError:
                raise ValueError(
                    'Provided dataset contains invalid lattice vectors (not invertible). Note: Only rank 3 lattice vector matrices are supported.'
                )

        if 'r_unit' in train_dataset and 'e_unit' in train_dataset:
            log.info(f'\nCoordinate unit : {train_dataset["r_unit"]}')
            log.info(f'Energy unit : {train_dataset["e_unit"]}')
            task['r_unit'] = train_dataset['r_unit']
            task['e_unit'] = train_dataset['e_unit']

        if use_sym:
            n_train = R_train.shape[0]
            R_train_sync_mat = R_train
            if n_train > 1000:
                log.info(
                    'Symmetry search has been restricted to a random subset\n'
                    'of 1000 training points for faster convergence.'
                )
                R_train_sync_mat = R_train[
                    np.random.choice(n_train, 1000, replace=False), :, :
                ]

            task['perms'] = find_perms(
                R_train_sync_mat,
                train_dataset['z'],
                lat_and_inv=None,
                max_processes=self._max_processes,
            )

        else:
            task['perms'] = np.arange(train_dataset['R'].shape[1])[
                None, :
            ]  # no symmetries

        # Which atoms can we keep, if we exclude all symmetric ones?
        n_perms = task['perms'].shape[0]
        if use_cprsn and n_perms > 1:

            _, cprsn_keep_idxs = np.unique(
                np.sort(task['perms'], axis=0), axis=1, return_index=True
            )

            task['cprsn_keep_atoms_idxs'] = cprsn_keep_idxs
        
        log.t_stop(
            t_create_task, message='\nCreating task took {time} s'
        )
        return task

    def create_model(
        self, task, solver, R_desc, R_d_desc, tril_perms_lin, std, alphas_F,
        alphas_E=None, solver_resid=None, solver_iters=None, norm_y_train=None,
        inducing_pts_idxs=None,
    ):
        """Create a data structure of custom type ``model``.
        """
        n_train, dim_d = R_d_desc.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)

        if 'cprsn_keep_atoms_idxs' in task:
            cprsn_keep_idxs = task['cprsn_keep_atoms_idxs']

            desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=self._max_processes,
            )

            R_d_desc_full = desc.d_desc_from_comp(R_d_desc).reshape(
                n_train, dim_d, n_atoms, 3
            )
            R_d_desc_full = R_d_desc_full[:, :, cprsn_keep_idxs, :].reshape(
                n_train, dim_d, -1
            )

            r_d_desc_alpha = np.einsum(
                'kji,ki->kj', R_d_desc_full, alphas_F.reshape(n_train, -1)
            )

        else:
            i, j = np.tril_indices(n_atoms, k=-1)
            alphas_F_exp = alphas_F.reshape(-1, n_atoms, 3)

            r_d_desc_alpha = np.einsum(
                'kji,kji->kj', R_d_desc, alphas_F_exp[:, j, :] - alphas_F_exp[:, i, :]
            )

        model = {
            'type': 'm',
            'code_version': __version__,
            'dataset_name': task['dataset_name'],
            'dataset_theory': task['dataset_theory'],
            'solver_name': solver,
            'solver_tol': task['solver_tol'],
            'norm_y_train': norm_y_train,
            'z': task['z'],
            'idxs_train': task['idxs_train'],
            'md5_train': task['md5_train'],
            'idxs_valid': task['idxs_valid'],
            'md5_valid': task['md5_valid'],
            'n_test': 0,
            'md5_test': None,
            'f_err': {'mae': np.nan, 'rmse': np.nan},
            'R_desc': R_desc.T,
            'R_d_desc_alpha': r_d_desc_alpha,
            'interact_cut_off': task['interact_cut_off'],
            'c': 0.0,
            'std': std,
            'sig': task['sig'],
            'lam': task['lam'],
            'alphas_F': alphas_F,
            'perms': task['perms'],
            'tril_perms_lin': tril_perms_lin,
            'use_E': task['use_E'],
            'use_cprsn': task['use_cprsn'],
        }

        if solver_resid is not None:
            model['solver_resid'] = solver_resid  # residual of solution (cg solver)

        if solver_iters is not None:
            model[
                'solver_iters'
            ] = solver_iters  # number of iterations performed to obtain solution (cg solver)

        if inducing_pts_idxs is not None:
            model['inducing_pts_idxs'] = inducing_pts_idxs

        if task['use_E']:
            model['e_err'] = {'mae': np.nan, 'rmse': np.nan}

            if task['use_E_cstr']:
                model['alphas_E'] = alphas_E

        if 'lattice' in task:
            model['lattice'] = task['lattice']

        if 'r_unit' in task and 'e_unit' in task:
            model['r_unit'] = task['r_unit']
            model['e_unit'] = task['e_unit']

        return model

    def train(self, task):
        """Train a model based on a task.

        Parameters
        ----------
        task : :obj:`dict`
            Data structure of custom type ``task``.

        Returns
        -------
        :obj:`dict`
            Data structure of custom type ``model``.

        Raises
        ------
        ValueError
            If the provided dataset contains invalid lattice vectors.
        """

        task = dict(task)  # make mutable

        solver = task['solver_name']
        assert solver == 'analytic' or solver == 'cg'  # or solver == 'fk'

        n_train, n_atoms = task['R_train'].shape[:2]

        desc = Desc(
            n_atoms,
            interact_cut_off=task['interact_cut_off'],
            max_processes=self._max_processes,
        )

        n_perms = task['perms'].shape[0]
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        dim_i = 3 * n_atoms
        dim_d = desc.dim

        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')

        lat_and_inv = None
        if 'lattice' in task:
            try:
                lat_and_inv = (task['lattice'], np.linalg.inv(task['lattice']))
            except np.linalg.LinAlgError:
                raise ValueError(
                    'Provided dataset contains invalid lattice vectors (not invertible).'
                    'Note: Only rank 3 lattice vector matrices are supported.'
                )

        R = task['R_train'].reshape(n_train, -1)
        R_desc, R_d_desc = desc.from_R(
            R,
            lat_and_inv=lat_and_inv
        )

        # Generate label vector.
        E_train_mean = None
        y = task['F_train'].ravel().copy()
        if task['use_E'] and task['use_E_cstr']:
            E_train = task['E_train'].ravel().copy()
            E_train_mean = np.mean(E_train)

            y = np.hstack((y, -E_train + E_train_mean))
            # y = np.hstack((n*Ft, (1-n)*Et))
        y_std = np.std(y)
        y /= y_std

        num_iters = None  # number of iterations performed (cg solver)
        resid = None  # residual of solution
        if solver == 'analytic':

            alphas = self.solve_analytic(
                task, desc, R_desc, R_d_desc, tril_perms_lin, y,
            )

        else:
            raise ValueError(f'{solver} solver is not supported')

        alphas_E = None
        alphas_F = alphas
        if task['use_E_cstr']:
            alphas_E = alphas[-n_train:]
            alphas_F = alphas[:-n_train]

        model = self.create_model(
            task,
            solver,
            R_desc,
            R_d_desc,
            tril_perms_lin,
            y_std,
            alphas_F,
            alphas_E=alphas_E,
            solver_resid=resid,
            solver_iters=num_iters,
            norm_y_train=np.linalg.norm(y),
            inducing_pts_idxs=inducing_pts_idxs if solver == 'cg' else None,
        )

        # Recover integration constant.
        # Note: if energy constraints are included in the kernel (via 'use_E_cstr'), do not
        # compute the integration constant, but simply set it to the mean of the training energies
        # (which was subtracted from the labels before training).
        if model['use_E']:
            c = (
                self._recov_int_const(model, task, R_desc=R_desc, R_d_desc=R_d_desc)
                if E_train_mean is None
                else E_train_mean
            )
            if c is None:
                # Something does not seem right. Turn off energy predictions for this model, only output force predictions.
                model['use_E'] = False
            else:
                model['c'] = c

        return model
    
    def solve_analytic(self, task, desc, R_desc, R_d_desc, tril_perms_lin, y):
        log.info(
            '\n-------------------------\n'
            '|   Analytical solver   |\n'
            '-------------------------\n'
        )

        sig = task['sig']
        lam = task['lam']
        use_E_cstr = task['use_E_cstr']
        log.log_model(task)

        n_train, dim_d = R_d_desc.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        dim_i = 3 * n_atoms

        # Compress kernel based on symmetries
        col_idxs = np.s_[:]
        if 'cprsn_keep_atoms_idxs' in task:

            cprsn_keep_idxs = task['cprsn_keep_atoms_idxs']
            cprsn_keep_idxs_lin = (
                np.arange(dim_i).reshape(n_atoms, -1)[cprsn_keep_idxs, :].ravel()
            )

            col_idxs = (
                cprsn_keep_idxs_lin[:, None] + np.arange(n_train) * dim_i
            ).T.ravel()

        log.info('\nAssembling kernel matrix')
        t_assemble = log.t_start()
        K = self._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            desc,
            use_E_cstr=use_E_cstr,
            col_idxs=col_idxs,
        )
        log.t_stop(t_assemble)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            if K.shape[0] == K.shape[1]:

                K[np.diag_indices_from(K)] -= lam  # regularize

                try:
                    t_cholesky = log.t_start()
                    log.info('Solving linear system (Cholesky factorization)')
                    # Cholesky
                    L, lower = sp.linalg.cho_factor(
                        -K, overwrite_a=True, check_finite=False
                    )
                    alphas = -sp.linalg.cho_solve(
                        (L, lower), y, overwrite_b=True, check_finite=False
                    )
                    
                    log.t_stop(
                        t_cholesky, message='Done in {time} s'
                    )
                except np.linalg.LinAlgError:  # try a solver that makes less assumptions
                    log.t_stop(
                        t_cholesky, message='Cholesky factorization failed in {time} s'
                    )
                    log.info('Solving linear system (LU factorization)')

                    try:
                        # LU
                        t_lu = log.t_start()
                        alphas = sp.linalg.solve(
                            K, y, overwrite_a=True, overwrite_b=True,
                            check_finite=False
                        )
                        log.t_stop(
                            t_lu, message='Done in {time} s'
                        )
                    except MemoryError:
                        log.t_stop(
                            t_lu, message='LU factorization failed in {time} s', level=50
                        )
                        log.critical(
                            'Not enough memory to train this system using a closed form solver.\n'
                            + 'Please reduce the size of the training set or consider one of the approximate solver options.'
                        )
                        sys.exit()

                except MemoryError:
                    log.critical(
                        'Not enough memory to train this system using a closed form solver.\n'
                        + 'Please reduce the size of the training set or consider one of the approximate solver options.'
                    )
                    sys.exit()
            else:
                log.info('Solving overdetermined linear system (least squares approximation)')
                t_least_squares = log.t_start()
                # least squares for non-square K
                alphas = np.linalg.lstsq(K, y, rcond=-1)[0]
                log.t_stop(t_least_squares)
        
        return alphas

    def _recov_int_const(self, model, task, R_desc=None, R_d_desc=None):
        """
        Estimate the integration constant for a force field model.

        The offset between the energies predicted for the original training
        data and the true energy labels is computed in the least square sense.
        Furthermore, common issues with the user-provided datasets are self
        diagnosed here.

        Parameters
        ----------
        model : :obj:`dict`
            Data structure of custom type ``model``.
        task : :obj:`dict`
            Data structure of custom type ``task``.
        R_desc : :obj:`numpy.ndarray`, optional
            An 2D array of size M x D containing the descriptors of dimension
            D for M molecules.
        R_d_desc : :obj:`numpy.ndarray`, optional
            A 2D array of size M x D x 3N containing of the
            descriptor Jacobians for M molecules. The descriptor
            has dimension D with 3N partial derivatives with
            respect to the 3N Cartesian coordinates of each atom.

        Returns
        -------
        float
            Estimate for the integration constant.

        Raises
        ------
        ValueError
            If the sign of the force labels in the dataset from
            which the model emerged is switched (e.g. gradients
            instead of forces).
        ValueError
            If inconsistent/corrupted energy labels are detected
            in the provided dataset.
        ValueError
            If different scales in energy vs. force labels are
            detected in the provided dataset.
        """
        gdml = GDMLPredict(
            model, max_processes=self._max_processes
        )

        n_train = task['E_train'].shape[0]
        R = task['R_train'].reshape(n_train, -1)

        E_pred, _ = gdml.predict(R, R_desc=R_desc, R_d_desc=R_d_desc)
        E_ref = np.squeeze(task['E_train'])

        e_fact = np.linalg.lstsq(
            np.column_stack((E_pred, np.ones(E_ref.shape))), E_ref, rcond=-1
        )[0][0]
        corrcoef = np.corrcoef(E_ref, E_pred)[0, 1]

        if np.sign(e_fact) == -1:
            log.warning(
                'The provided dataset contains gradients instead of force labels (flipped sign). Please correct!\n'
                + 'Note: The energy prediction accuracy of the model will thus neither be validated nor tested in the following steps!'
            )
            return None

        if corrcoef < 0.95:
            log.warning(
                'Inconsistent energy labels detected!'
            )
            log.warning(
                'The predicted energies for the training data are only weakly\n'
                f'correlated with the reference labels (correlation coefficient {corrcoef:.2f})\n'
                'which indicates that the issue is most likely NOT just a unit conversion error.\n'
            )
            return None

        if np.abs(e_fact - 1) > 1e-1:
            log.warning(
                'Different scales in energy vs. force labels detected!\n'
            )
            return None

        # Least squares estimate for integration constant.
        return np.sum(E_ref - E_pred) / E_ref.shape[0]

    def _assemble_kernel_mat(
        self, R_desc, R_d_desc, tril_perms_lin, sig, desc, use_E_cstr=False,
        col_idxs=np.s_[:]
    ):
        """
        Compute force field kernel matrix.

        The Hessian of the Matern kernel is used with n = 2 (twice
        differentiable). Each row and column consists of matrix-valued blocks,
        which encode the interaction of one training point with all others. The
        result is stored in shared memory (a global variable).

        Parameters
        ----------
        R_desc : :obj:`numpy.ndarray`
            Array containing the descriptor for each training point.
        R_d_desc : :obj:`numpy.ndarray`
            Array containing the gradient of the descriptor for
            each training point.
        tril_perms_lin : :obj:`numpy.ndarray`
            1D array containing all recovered permutations
            expanded as one large permutation to be applied to a
            tiled copy of the object to be permuted.
        sig : int
            Hyperparameter sigma (kernel length scale).
        use_E_cstr : :obj:`bool`, optional
            True: include energy constraints in the kernel,
            False: default (s)GDML kernel.
        cols_m_limit : :obj:`int`, optional
            Only generate the columns up to index 'cols_m_limit'. This creates
            a M*3N x cols_m_limit*3N kernel matrix, instead of M*3N x M*3N.
        cols_3n_keep_idxs : :obj:`numpy.ndarray`, optional
            Only generate columns with the given indices in the 3N x 3N
            kernel function. The resulting kernel matrix will have dimension
            M*3N x M*len(cols_3n_keep_idxs).

        Returns
        -------
        :obj:`numpy.ndarray`
            Force field kernel matrix.
        """
        global glob

        n_train, dim_d = R_d_desc.shape[:2]
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)

        # Determine size of kernel matrix.
        K_n_rows = n_train * dim_i
        if isinstance(col_idxs, slice):  # indexed by slice
            K_n_cols = len(range(*col_idxs.indices(K_n_rows)))
        else:  # indexed by list

            # TODO: throw exception with description
            assert len(col_idxs) == len(set(col_idxs))  # assume no duplicate indices

            # TODO: throw exception with description
            # Note: This function does not support unsorted (ascending) index arrays.
            assert np.array_equal(col_idxs, np.sort(col_idxs))

            K_n_cols = len(col_idxs)

        # Account for additional rows and columns due to energy constraints in the kernel matrix.
        if use_E_cstr:
            K_n_rows += n_train
            K_n_cols += n_train

        # Make sure no indices are outside of the valid range.
        if K_n_cols > K_n_rows:
            raise ValueError('Columns indexed beyond range.')

        exploit_sym = False
        cols_m_limit = None

        # Check if range is a subset of training points (as opposed to a subset of partials of multiple points).
        is_M_subset = (
            isinstance(col_idxs, slice)
            and (col_idxs.start is None or col_idxs.start % dim_i == 0)
            and (col_idxs.stop is None or col_idxs.stop % dim_i == 0)
            and col_idxs.step is None
        )
        if is_M_subset:
            M_slice_start = (
                None if col_idxs.start is None else int(col_idxs.start / dim_i)
            )
            M_slice_stop = None if col_idxs.stop is None else int(col_idxs.stop / dim_i)
            M_slice = slice(M_slice_start, M_slice_stop)

            J = range(*M_slice.indices(n_train))

            if M_slice_start is None:
                exploit_sym = True
                cols_m_limit = M_slice_stop

        else:

            if isinstance(col_idxs, slice):
                random = list(range(*col_idxs.indices(n_train * dim_i)))
            else:
                random = col_idxs

            # M - number training
            # N - number atoms

            n_idxs = np.mod(random, dim_i)
            m_idxs = (np.array(random) / dim_i).astype(int)
            m_idxs_uniq = np.unique(m_idxs)  # which points to include?

            m_n_idxs = [
                list(n_idxs[np.where(m_idxs == m_idx)]) for m_idx in m_idxs_uniq
            ]

            m_n_idxs_lens = [len(m_n_idx) for m_n_idx in m_n_idxs]

            m_n_idxs_lens.insert(0, 0)
            blk_start_idxs = list(
                np.cumsum(m_n_idxs_lens[:-1])
            )  # index within K at which each block starts

            # tuples: (block index in final K, block index global, indices of partials within block)
            J = list(zip(blk_start_idxs, m_idxs_uniq, m_n_idxs))

        K = mp.RawArray('d', K_n_rows * K_n_cols)
        glob['K'], glob['K_shape'] = K, (K_n_rows, K_n_cols)
        glob['R_desc'], glob['R_desc_shape'] = _share_array(R_desc, 'd')
        glob['R_d_desc'], glob['R_d_desc_shape'] = _share_array(R_d_desc, 'd')

        glob['desc_func'] = desc

        pool = Pool(self._max_processes)

        todo, done = K_n_cols, 0
        for done_wkr in pool.imap_unordered(
            partial(
                _assemble_kernel_mat_wkr,
                tril_perms_lin=tril_perms_lin,
                sig=sig,
                use_E_cstr=use_E_cstr,
                exploit_sym=exploit_sym,
                cols_m_limit=cols_m_limit,
            ),
            J,
        ):
            done += done_wkr

        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).

        # Release some memory.
        glob.pop('K', None)
        glob.pop('R_desc', None)
        glob.pop('R_d_desc', None)

        return np.frombuffer(K).reshape(glob['K_shape'])


def get_test_idxs(model, dataset, n_test=None):
    """Gets dataset indices for testing a model.

    Parameters
    ----------
    model : :obj:`dict`
        Model to test.
    dataset : :obj:`dict`
        Dataset to be used for testing.
    n_test : :obj:`int`, default: ``None``
        Number of points to include in test indices. Defaults to all available
        indices.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        Structure indices for model testing.
    """
    # exclude training and/or test sets from validation set if necessary
    excl_idxs = np.empty((0,), dtype=np.uint)
    if dataset['md5'] == model['md5_train']:
        excl_idxs = np.concatenate([excl_idxs, model['idxs_train']]).astype(
            np.uint
        )
    if dataset['md5'] == model['md5_valid']:
        excl_idxs = np.concatenate([excl_idxs, model['idxs_valid']]).astype(
            np.uint
        )

    n_data = dataset['F'].shape[0]
    n_data_eff = n_data - len(excl_idxs)
    if n_test is None:
        n_test = n_data_eff

    if n_data_eff == 0:
        log.warning('No unused points for testing in provided dataset.')
        return
    
    log.info(
        f'Test set size was automatically set to {n_data_eff} points.'
    )

    if 'E' in dataset:
        test_idxs = draw_strat_sample(
            dataset['E'], n_test, excl_idxs=excl_idxs
        )
    else:
        test_idxs = np.delete(np.arange(n_data), excl_idxs)

        log.warning(
            'Test dataset will be sampled with no guidance from energy labels (randomly)!\n'
            + 'Note: Larger test datasets are recommended due to slower convergence of the error.'
        )
    
    return test_idxs


def add_valid_errors(
    model, dataset, overwrite=False, max_processes=None, use_torch=False
):
    """Calculate and add energy and force validation errors to a model.
    
    Parameters
    ----------
    model : :obj:`dict`
        Trained GDML model.
    dataset : :obj:`dict`
        Validation dataset.
    overwrite : :obj:`bool`, default: ``False``
        Will overwrite validation errors in model if they already exist.
    
    Return
    ------
    :obj:`dict`
        Validation errors with the following keys.

            ``'force'`` which contains ``'mae'`` and ``'rmse'``

            ``'energy'`` which contains ``'mae'`` and ``'rmse'``
    :obj:`dict`
        Model with validation errors.
    """
    log.info(
        '\n------------------------\n'
        '|   Model Validation   |\n'
        '------------------------'
    )
    if model['use_E']:
        e_err = np.array(model['e_err']).item()
    f_err = np.array(model['f_err']).item()
    is_model_validated = not (np.isnan(f_err['mae']) or np.isnan(f_err['rmse']))
    if is_model_validated and not overwrite:
        log.warning('Model is already validated and overwrite is False.')
        return

    n_valid, e_mae, e_rmse, f_mae, f_rmse = model_errors(
        model, dataset, is_valid=True, max_processes=max_processes,
        use_torch=use_torch
    )

    model['n_test'] = 0  # flag the model as not tested

    results = {
        'force': {'mae': f_mae, 'rmse': f_rmse}
    }
    model['f_err'] = results['force']

    if model['use_E']:
        results['energy'] = {
            'mae': e_mae,
            'rmse': e_rmse,
        }
        model['e_err'] = results['energy']
    else:
        results['energy'] = {
            'mae': None,
            'rmse': None,
        }
    
    return results, model

def save_model(model, model_path):
    np.savez_compressed(model_path, **model)

def model_errors(
    model, dataset, is_valid=False, n_test=None, max_processes=None,
    use_torch=False
):
    """Computes model errors for either validation or testing purposes.

    Parameters
    ----------
    model : :obj:`dict`
        Trained GDML model.
    dataset : :obj:`dict`
        Dataset to test the model against.
    is_valid : :obj:`bool`, default: ``False``
        Is this for model validation? Determines how we get the structure
        indices.
    n_test : :obj:`int`, default: ``None``
        Number of desired testing indices. ``None`` will test against all
        available structures.
    
    Returns
    -------
    :obj:`int`
        Number of structures predicted.
    :obj:`float`
        Energy MAE
    :obj:`float`
        Energy RMSE
    :obj:`float`
        Force MAE
    :obj:`float`
        Force RMSE
    """
    num_workers, batch_size = 0, 0

    if not np.array_equal(model['z'], dataset['z']):
        raise AssistantError(
            'Atom composition or order in dataset does not match the one in model'
        )

    if ('lattice' in model) is not ('lattice' in dataset):
        if 'lattice' in model:
            raise AssistantError(
                'Model contains lattice vectors, but dataset does not.'
            )
        elif 'lattice' in dataset:
            raise AssistantError(
                'Dataset contains lattice vectors, but model does not.'
            )

    if is_valid:
        test_idxs = model['idxs_valid']
    else:
        log.info('\n---------------------\n'
            '|   Model Testing   |\n'
            '---------------------\n'
        )
        log.log_model(model)
        test_idxs = get_test_idxs(model, dataset, n_test=n_test)

    z = dataset['z']
    R = dataset['R'][test_idxs, :, :]
    F = dataset['F'][test_idxs, :, :]

    if model['use_E']:
        E = dataset['E'][test_idxs]

    try:
        gdml_predict = GDMLPredict(
            model, max_processes=max_processes, use_torch=use_torch
        )
    except:
        raise
    
    log.info(f'\nPredicting {len(test_idxs)} structures')
    b_size = min(1000, len(test_idxs))
    log.info(f'Batch size : {b_size} structures\n')

    if not use_torch:
        if num_workers == 0 or batch_size == 0:
            log.info('Optimizing parallelism')

            gps, is_from_cache = gdml_predict.prepare_parallel(
                n_bulk=b_size, return_is_from_cache=True
            )
            num_workers, batch_size, bulk_mp = (
                gdml_predict.num_workers,
                gdml_predict.chunk_size,
                gdml_predict.bulk_mp,
            )

            if is_from_cache:
                log.info('Taken from cache')
            log.info(f'Using {num_workers} workers with chunks of {batch_size}')
        else:
            gdml_predict._set_num_workers(num_workers)
            gdml_predict._set_batch_size(batch_size)
            gdml_predict._set_bulk_mp(bulk_mp)

    n_atoms = z.shape[0]

    if model['use_E']:
        e_mae_sum, e_rmse_sum = 0, 0
    f_mae_sum, f_rmse_sum = 0, 0
    cos_mae_sum, cos_rmse_sum = 0, 0
    mag_mae_sum, mag_rmse_sum = 0, 0
    
    t_pred = log.t_start()
    n_done = 0
    for b_range in _batch(list(range(len(test_idxs))), b_size):

        n_done_step = len(b_range)
        n_done += n_done_step

        r = R[b_range].reshape(n_done_step, -1)
        e_pred, f_pred = gdml_predict.predict(r)

        # energy error
        if model['use_E']:
            e = E[b_range]
            e_mae, e_mae_sum, e_rmse, e_rmse_sum = _online_err(
                np.squeeze(e) - e_pred, 1, n_done, e_mae_sum, e_rmse_sum
            )

        # force component error
        f = F[b_range].reshape(n_done_step, -1)
        f_mae, f_mae_sum, f_rmse, f_rmse_sum = _online_err(
            f - f_pred, 3 * n_atoms, n_done, f_mae_sum, f_rmse_sum
        )
    
    log.info(f"\nForce MAE  : {f_mae:.5f}")
    log.info(f"Force RMSE : {f_rmse:.5f}")
    if model['use_E']:
        log.info(f"Energy MAE  : {e_mae:.5f}")
        log.info(f"Energy RMSE : {e_rmse:.5f}")
    t_elapsed = log.t_stop(
        t_pred, message='\nTook {time} s'
    )
    log.info(f'Prediction rate : {n_done/t_elapsed:.2f} structures per second')
    if model['use_E']:
        return len(test_idxs), e_mae, e_rmse, f_mae, f_rmse
    else:
        return len(test_idxs), None, None, f_mae, f_rmse

def _online_err(err, size, n, mae_n_sum, rmse_n_sum):
    err = np.abs(err)

    mae_n_sum += np.sum(err) / size
    mae = mae_n_sum / n

    rmse_n_sum += np.sum(err ** 2) / size
    rmse = np.sqrt(rmse_n_sum / n)

    return mae, mae_n_sum, rmse, rmse_n_sum

def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]