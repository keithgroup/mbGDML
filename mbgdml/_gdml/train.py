# MIT License
#
# Copyright (c) 2018-2022, Stefan Chmiela
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

# pylint: disable=too-many-branches, too-many-statements

from functools import partial
import multiprocessing as mp
import numpy as np
import psutil

from .solvers.analytic import Analytic
from .predict import GDMLPredict
from .desc import Desc
from .sample import draw_strat_sample
from .perm import find_perms
from ..utils import md5_data, chunk_array
from ..losses import mae, rmse
from ..logger import GDMLLogger
from .. import _version

mbgdml_version = _version.get_versions()["version"]

Pool = mp.get_context("fork").Pool

try:
    import torch
except ImportError:
    _HAS_TORCH = False
else:
    _HAS_TORCH = True

try:
    _TORCH_MPS_IS_AVAILABLE = torch.backends.mps.is_available()
except (NameError, AttributeError):
    _TORCH_MPS_IS_AVAILABLE = False
_TORCH_MPS_IS_AVAILABLE = False

try:
    _TORCH_CUDA_IS_AVAILABLE = torch.cuda.is_available()
except (NameError, AttributeError):
    _TORCH_CUDA_IS_AVAILABLE = False

# TODO: remove exception handling once iterative solver ships
try:
    from .solvers.iterative import Iterative
except ImportError:
    pass

log = GDMLLogger(__name__)


def _share_array(arr_np, typecode_or_type):
    r"""Return a ctypes array allocated from shared memory with data from a
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
    r"""Compute one row and column of the force field kernel matrix.

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
        Include energy constraints in the kernel. This can
        sometimes be helpful in tricky cases.
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
    # pylint: disable=invalid-name, undefined-variable
    global glob  # pylint: disable=global-variable-not-assigned

    R_desc = np.frombuffer(glob["R_desc"]).reshape(glob["R_desc_shape"])
    R_d_desc = np.frombuffer(glob["R_d_desc"]).reshape(glob["R_d_desc_shape"])
    K = np.frombuffer(glob["K"]).reshape(glob["K_shape"])

    desc_func = glob["desc_func"]

    n_train, dim_d = R_d_desc.shape[:2]
    n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    dim_i = 3 * n_atoms
    n_perms = int(len(tril_perms_lin) / dim_d)

    # (block index in final K, block index global, indices of partials within block)
    if isinstance(j, tuple):  # Selective/"fancy" indexing
        (
            K_j,
            j,
            keep_idxs_3n,
        ) = j
        blk_j = slice(K_j, K_j + len(keep_idxs_3n))

    else:  # Sequential indexing
        K_j = j * dim_i if j < n_train else n_train * dim_i + (j % n_train)
        blk_j = slice(K_j, K_j + dim_i) if j < n_train else slice(K_j, K_j + 1)
        keep_idxs_3n = slice(None)  # same as [:]

    # Note: The modulo-operator wraps around the index pointer on the training points
    # when energy constraints are used in the kernel. In that case each point is
    # accessed twice.

    # Create permutated variants of 'rj_desc' and 'rj_d_desc'.
    rj_desc_perms = np.reshape(
        np.tile(R_desc[j % n_train, :], n_perms)[tril_perms_lin],
        (n_perms, -1),
        order="F",
    )

    rj_d_desc = desc_func.d_desc_from_comp(R_d_desc[j % n_train, :, :])[0][
        :, keep_idxs_3n
    ]  # convert descriptor back to full representation

    rj_d_desc_perms = np.reshape(
        np.tile(rj_d_desc.T, n_perms)[:, tril_perms_lin], (-1, dim_d, n_perms)
    )

    mat52_base_div = 3 * sig**4
    sqrt5 = np.sqrt(5.0)
    sig_pow2 = sig**2

    dim_i_keep = rj_d_desc.shape[1]
    diff_ab_outer_perms = np.empty((dim_d, dim_i_keep))
    diff_ab_perms = np.empty((n_perms, dim_d))
    ri_d_desc = np.zeros((1, dim_d, dim_i))  # must be zeros!
    k = np.empty((dim_i, dim_i_keep))

    if (
        j < n_train
    ):  # This column only contains second and first derivative constraints.

        # for i in range(j if exploit_sym else 0, n_train):
        for i in range(0, n_train):

            blk_i = slice(i * dim_i, (i + 1) * dim_i)

            # diff_ab_perms = R_desc[i, :] - rj_desc_perms
            np.subtract(R_desc[i, :], rj_desc_perms, out=diff_ab_perms)

            norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
            mat52_base_perms = np.exp(-norm_ab_perms / sig) / mat52_base_div * 5

            np.einsum(
                "ki,kj->ij",
                diff_ab_perms * mat52_base_perms[:, None] * 5,
                np.einsum("ki,jik -> kj", diff_ab_perms, rj_d_desc_perms),
                out=diff_ab_outer_perms,
            )

            diff_ab_outer_perms -= np.einsum(
                "ikj,j->ki",
                rj_d_desc_perms,
                (sig_pow2 + sig * norm_ab_perms) * mat52_base_perms,
            )

            # ri_d_desc = desc_func.d_desc_from_comp(R_d_desc[i, :, :])[0]
            desc_func.d_desc_from_comp(R_d_desc[i, :, :], out=ri_d_desc)

            # K[blk_i, blk_j] = ri_d_desc[0].T.dot(diff_ab_outer_perms)
            np.dot(ri_d_desc[0].T, diff_ab_outer_perms, out=k)
            K[blk_i, blk_j] = k

            # this will never be called with 'keep_idxs_3n' set to anything else
            # than [:]
            if exploit_sym and (cols_m_limit is None or i < cols_m_limit):
                K[blk_j, blk_i] = K[blk_i, blk_j].T

            # First derivative constraints
            if use_E_cstr:

                K_fe = (
                    5
                    * diff_ab_perms
                    / (3 * sig**3)
                    * (norm_ab_perms[:, None] + sig)
                    * np.exp(-norm_ab_perms / sig)[:, None]
                )

                K_fe = -np.einsum("ik,jki -> j", K_fe, rj_d_desc_perms)

                E_off_i = n_train * dim_i  # , K.shape[1] - n_train
                K[E_off_i + i, blk_j] = K_fe

    else:

        if use_E_cstr:

            # rj_d_desc = desc_func.d_desc_from_comp(R_d_desc[j % n_train, :, :])[0][
            #    :, :
            # ]  # convert descriptor back to full representation

            # rj_d_desc_perms = np.reshape(
            #    np.tile(rj_d_desc.T, n_perms)[:, tril_perms_lin], (-1, dim_d, n_perms)
            # )

            E_off_i = n_train * dim_i  # Account for 'alloc_extra_rows'!.
            # blk_j_full = slice((j % n_train) * dim_i, ((j % n_train) + 1) * dim_i)
            # for i in range((j % n_train) if exploit_sym else 0, n_train):
            for i in range(0, n_train):

                ri_desc_perms = np.reshape(
                    np.tile(R_desc[i, :], n_perms)[tril_perms_lin],
                    (n_perms, -1),
                    order="F",
                )

                ri_d_desc = desc_func.d_desc_from_comp(R_d_desc[i, :, :])[
                    0
                ]  # convert descriptor back to full representation
                ri_d_desc_perms = np.reshape(
                    np.tile(ri_d_desc.T, n_perms)[:, tril_perms_lin],
                    (-1, dim_d, n_perms),
                )

                diff_ab_perms = R_desc[j % n_train, :] - ri_desc_perms

                norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

                K_fe = (
                    5
                    * diff_ab_perms
                    / (3 * sig**3)
                    * (norm_ab_perms[:, None] + sig)
                    * np.exp(-norm_ab_perms / sig)[:, None]
                )

                K_fe = -np.einsum("ik,jki -> j", K_fe, ri_d_desc_perms)

                blk_i_full = slice(i * dim_i, (i + 1) * dim_i)
                K[blk_i_full, K_j] = K_fe  # vertical

                K[E_off_i + i, K_j] = -(
                    1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
                ).dot(np.exp(-norm_ab_perms / sig))

    return blk_j.stop - blk_j.start


class GDMLTrain:
    r"""Train GDML force fields.

    This class is used to train models using different closed-form
    and numerical solvers. GPU support is provided
    through PyTorch (requires optional ``torch`` dependency to be
    installed) for some solvers.

    """

    def __init__(self, max_memory=None, max_processes=None, use_torch=False):
        """
        Parameters
        ----------
        max_memory : :obj:`int`, default: :obj:`None`
            Limit the maximum memory usage. This is a soft limit that cannot
            always be enforced.
        max_processes : :obj:`int`, default: :obj:`None`
            Limit the max. number of processes. Otherwise all CPU cores are
            used. This parameters has no effect if ``use_torch=True``
        use_torch : :obj:`bool`, default: ``False``
            Use PyTorch to calculate predictions (if supported by solver).

        Raises
        ------
        Exception
            If multiple instances of this class are created.
        ImportError
            If the optional PyTorch dependency is missing, but PyTorch features are
            used.
        """

        global glob  # pylint: disable=global-variable-undefined
        if "glob" not in globals():  # Don't allow more than one instance of this class.
            glob = {}
        else:
            raise Exception(
                "You can not create multiple instances of this class. "
                "Please reuse your first one."
            )

        total_memory = psutil.virtual_memory().total // 2**30  # bytes to GB)
        self._max_memory = (
            min(max_memory, total_memory) if max_memory is not None else total_memory
        )

        total_cpus = mp.cpu_count()
        self._max_processes = (
            min(max_processes, total_cpus) if max_processes is not None else total_cpus
        )

        self._use_torch = use_torch

        if use_torch and not _HAS_TORCH:
            raise ImportError(
                "Optional PyTorch dependency not found! Please install PyTorch."
            )

    def __del__(self):
        global glob  # pylint: disable=global-variable-undefined
        if "glob" in globals():
            del glob

    def create_task(
        self,
        train_dataset,
        n_train,
        valid_dataset,
        n_valid,
        sig,
        lam=1e-10,
        perms=None,
        use_sym=True,
        use_E=True,
        use_E_cstr=False,
        use_cprsn=False,
        solver=None,
        solver_tol=1e-4,
        idxs_train=None,
        idxs_valid=None,
    ):
        r"""Create a data structure, :obj:`dict`, of custom type ``task``.

        These data structures serve as recipes for model creation,
        summarizing the configuration of one particular training run.
        Training and test points are sampled from the provided dataset,
        without replacement. If the same dataset if given for training
        and testing, the subsets are drawn without overlap.

        Each task also contains a choice for the hyperparameters of the
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
        lam : :obj:`float`, default: ``1e-10``
            Hyperparameter lambda (regularization strength).

            .. note::

                Early sGDML models used ``1e-15``.
        perms : :obj:`numpy.ndarray`, optional
            An 2D array of size P x N containing P possible permutations
            of the N atoms in the system. This argument takes priority over the ones
            provided in the training dataset. No automatic discovery is run
            when this argument is provided.
        use_sym : :obj:`bool`, default: ``True``
            True: include symmetries (sGDML), False: GDML.
        use_E : :obj:`bool`, optional
            ``True``: reconstruct force field with corresponding potential energy
            surface,

            ``False``: ignore energy during training, even if energy labels are
            available in the dataset. The trained model will still be able to predict
            energies up to an unknown integration constant. Note, that the
            energy predictions accuracy will be untested.
        use_E_cstr : :obj:`bool`, default: ``False``
            Include energy constraints in the kernel. This can
            sometimes be helpful in tricky cases.
        use_cprsn : :obj:`bool`, default: ``False``
            Compress the kernel matrix along symmetric degrees of freedom.
            If ``False``, training is done on the full kernel matrix.
        solver : :obj:`str`, default: :obj:`None`
            Type of solver to use for training. ``'analytic'`` is currently the
            only option and defaults to this.

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
        # pylint: disable=invalid-name
        ###   mbGDML ADD   ###
        log.info(
            "-----------------------------------\n"
            "|   Creating GDML training task   |\n"
            "-----------------------------------\n"
        )

        log.log_model(
            {
                "Z": train_dataset["Z"],
                "n_train": n_train,
                "n_valid": n_valid,
                "sig": sig,
                "lam": lam,
                "use_sym": use_sym,
                "use_E": use_E,
                "use_E_cstr": use_E_cstr,
                "use_cprsn": use_cprsn,
                "type": "t",
            }
        )
        ###   mbGDML ADD END   ###

        if use_E and "E" not in train_dataset:
            raise ValueError(
                "No energy labels found in dataset!\n"
                + "By default, force fields are always reconstructed including the\n"
                + "corresponding potential energy surface (this can be turned off).\n"
                + "However, the energy labels are missing in the provided dataset.\n"
            )

        use_E_cstr = use_E and use_E_cstr

        # Is not needed in this function (mbGDML CHANGED)
        # n_atoms = train_dataset['R'].shape[1]

        ###   mbGDML CHANGE   ###
        log.info("\nDataset splitting\n-----------------")
        md5_train_keys = ["Z", "R", "F"]
        md5_valid_keys = ["Z", "R", "F"]
        if "E" in train_dataset.keys():
            md5_train_keys.append("E")
        if "E" in valid_dataset.keys():
            md5_valid_keys.append("E")
        md5_train = md5_data(train_dataset, md5_train_keys)
        md5_valid = md5_data(valid_dataset, md5_valid_keys)

        log.info("\n#   Training   #")
        log.info("MD5 : %s", md5_train)
        log.info("Size : %d", n_train)
        if idxs_train is None:
            log.info("Drawing structures from the dataset")
            if "E" in train_dataset:
                log.info(
                    "Energies are included in the dataset\n"
                    "Using the Freedman-Diaconis rule"
                )
                idxs_train = draw_strat_sample(train_dataset["E"], n_train)
            else:
                log.info(
                    "Energies are not included in the dataset\n"
                    "Randomly selecting structures"
                )
                idxs_train = np.random.choice(
                    np.arange(train_dataset["F"].shape[0]),
                    n_train,
                    replace=False,
                )
        else:
            log.info("Training indices were manually specified")
            idxs_train = np.array(idxs_train)
            log.log_array(idxs_train, level=10)

        # Handles validation indices.
        log.info("\n#   Validation   #")
        log.info("MD5 : %s", md5_valid)
        log.info("Size : %d", n_valid)
        if idxs_valid is not None:
            log.info("Validation indices were manually specified")
            idxs_valid = np.array(idxs_valid)
            log.log_array(idxs_valid, level=10)
        else:
            log.info("Drawing structures from the dataset")
            excl_idxs = (
                idxs_train if md5_train == md5_valid else np.array([], dtype=np.uint)
            )
            log.debug("Excluded %r structures", len(excl_idxs))
            log.log_array(excl_idxs, level=10)

            if "E" in valid_dataset:
                idxs_valid = draw_strat_sample(
                    valid_dataset["E"],
                    n_valid,
                    excl_idxs=excl_idxs,
                )
            else:
                idxs_valid_cands = np.setdiff1d(
                    np.arange(valid_dataset["F"].shape[0]),
                    excl_idxs,
                    assume_unique=True,
                )
                idxs_valid = np.random.choice(idxs_valid_cands, n_valid, replace=False)
        ###   mbGDML CHANGE END   ###

        R_train = train_dataset["R"][idxs_train, :, :]
        task = {
            "type": "t",
            "code_version": mbgdml_version,
            "dataset_name": train_dataset["name"].astype(str),
            "dataset_theory": train_dataset["theory"].astype(str),
            "Z": train_dataset["Z"],
            "R_train": R_train,
            "F_train": train_dataset["F"][idxs_train, :, :],
            "idxs_train": idxs_train,
            "md5_train": md5_train,
            "idxs_valid": idxs_valid,
            "md5_valid": md5_valid,
            "sig": sig,
            "lam": lam,
            "use_E": use_E,
            "use_E_cstr": use_E_cstr,
            "use_sym": use_sym,
            "use_cprsn": use_cprsn,
            "solver_name": solver,
            "solver_tol": solver_tol,
        }

        if use_E:
            task["E_train"] = train_dataset["E"][idxs_train]

        lat_and_inv = None
        if "lattice" in train_dataset:
            log.info("\nLattice was found in the dataset")
            log.debug(train_dataset["lattice"])
            task["lattice"] = train_dataset["lattice"]

            try:
                lat_and_inv = (task["lattice"], np.linalg.inv(task["lattice"]))
            except np.linalg.LinAlgError as e:
                raise ValueError(
                    "Provided dataset contains invalid lattice vectors "
                    "(not invertible). Note: Only rank 3 lattice vector matrices "
                    "are supported."
                ) from e

        if "r_unit" in train_dataset and "e_unit" in train_dataset:
            log.info("\nCoordinate unit : %s", train_dataset["r_unit"])
            log.info("Energy unit : %s", train_dataset["e_unit"])
            task["r_unit"] = train_dataset["r_unit"]
            task["e_unit"] = train_dataset["e_unit"]

        if use_sym:

            # No permutations provided externally.
            if perms is None:

                if (
                    "perms" in train_dataset
                ):  # take perms from training dataset, if available

                    n_perms = train_dataset["perms"].shape[0]
                    log.info("Using %d permutations included in dataset", n_perms)

                    task["perms"] = train_dataset["perms"]

                else:  # find perms from scratch

                    n_train = R_train.shape[0]
                    R_train_sync_mat = R_train
                    if n_train > 1000:
                        R_train_sync_mat = R_train[
                            np.random.choice(n_train, 1000, replace=False), :, :
                        ]
                        log.info(
                            "Symmetry search has been restricted to a random subset of "
                            "1000/%d training points for faster convergence.",
                            n_train,
                        )

                    # TODO: PBCs disabled when matching (for now).
                    task["perms"] = find_perms(
                        R_train_sync_mat,
                        train_dataset["Z"],
                        # lat_and_inv=None,
                        lat_and_inv=lat_and_inv,
                        max_processes=self._max_processes,
                    )

            else:  # use provided perms

                n_atoms = len(task["Z"])
                n_perms, perms_len = perms.shape

                if perms_len != n_atoms:
                    raise ValueError(  # TODO: Document me
                        "Provided permutations do not match the number of atoms in "
                        "dataset."
                    )

                log.info("Using %d externally provided permutations", n_perms)
                task["perms"] = perms

        else:
            task["perms"] = np.arange(train_dataset["R"].shape[1])[
                None, :
            ]  # no symmetries

        return task

    def create_model(
        self,
        task,
        solver,
        R_desc,
        R_d_desc,
        tril_perms_lin,
        std,
        alphas_F,
        alphas_E=None,
    ):
        r"""Create a data structure, :obj:`dict`, of custom type ``model``.

        These data structures contain the trained model are everything
        that is needed to generate predictions for new inputs.
        Each task also contains the MD5 fingerprints of the used datasets.

        Parameters
        ----------
        task : :obj:`dict`
            Data structure of custom type ``task`` from which the model emerged.
        solver : :obj:`str`
            Identifier string for the solver that has been used to
            train this model.
        R_desc : :obj:`numpy.ndarray`, ndim: ``2``
            An array of size :math:`M \\times D` containing the descriptors of
            dimension :math:`D` for :math:`M` molecules.
        R_d_desc : :obj:`numpy.ndarray`, ndim: ``2``
            An array of size :math:`M \\times D \\times 3N` containing of the
            descriptor Jacobians for :math:`M` molecules. The descriptor has
            dimension :math:`D` with :math:`3N` partial derivatives with respect
            to the :math:`3N` Cartesian coordinates of each atom.
        tril_perms_lin : :obj:`numpy.ndarray`, ndim: ``1``
            An array containing all recovered permutations expanded as one large
            permutation to be applied to a tiled copy of the object to be
            permuted.
        std : :obj:`float`
            Standard deviation of the training labels.
        alphas_F : :obj:`numpy.ndarray`, ndim: ``1``
            An array of size :math:`3NM` containing of the linear coefficients
            that correspond to the force constraints.
        alphas_E : :obj:`numpy.ndarray`, ndim: ``1``, optional
            An array of size N containing of the linear coefficients that
            correspond to the energy constraints. Only used if ``use_E_cstr``
            is ``True``.

        Returns
        -------
        :obj:`dict`
            Data structure of custom type ``model``.
        """
        # pylint: disable=invalid-name
        _, dim_d = R_d_desc.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)

        desc = Desc(
            n_atoms,
            max_processes=self._max_processes,
        )

        dim_i = desc.dim_i
        R_d_desc_alpha = desc.d_desc_dot_vec(R_d_desc, alphas_F.reshape(-1, dim_i))

        model = {
            "type": "m",
            "code_version": mbgdml_version,
            "dataset_name": task["dataset_name"],
            "dataset_theory": task["dataset_theory"],
            "solver_name": solver,
            "Z": task["Z"],
            "idxs_train": task["idxs_train"],
            "md5_train": task["md5_train"],
            "idxs_valid": task["idxs_valid"],
            "md5_valid": task["md5_valid"],
            "n_test": 0,
            "md5_test": None,
            "f_err": {"mae": np.nan, "rmse": np.nan},
            "R_desc": R_desc.T,
            "R_d_desc_alpha": R_d_desc_alpha,
            "c": 0.0,
            "std": std,
            "sig": task["sig"],
            "lam": task["lam"],
            "alphas_F": alphas_F,
            "perms": task["perms"],
            "tril_perms_lin": tril_perms_lin,
            "use_E": task["use_E"],
            "use_cprsn": task["use_cprsn"],
        }

        if task["use_E"]:
            model["e_err"] = {"mae": np.nan, "rmse": np.nan}

            if task["use_E_cstr"]:
                model["alphas_E"] = alphas_E

        if "lattice" in task:
            model["lattice"] = task["lattice"]

        if "r_unit" in task and "e_unit" in task:
            model["r_unit"] = task["r_unit"]
            model["e_unit"] = task["e_unit"]

        return model

    def train_labels(self, F, use_E, use_E_cstr, E=None):
        r"""Compute custom train labels.

        By default, they are the raveled forces scaled by the standard
        deviation. Energy labels are included if ``use_E`` and ``use_E_cstr``
        are ``True``.

        .. note::

            We use negative energies with the mean removed for training labels
            (if requested).

        Parameters
        ----------
        F : :obj:`numpy.ndarray`, ndim: ``3``
            Train set forces.
        use_E : :obj:`bool`
            If energies are used during training.
        use_E_cstr : :obj:`bool`
            If energy constraints are being added to the kernel.
        E : :obj:`numpy.ndarray`, default: :obj:`None`
            Train set energies.

        Returns
        -------
        :obj:`numpy.ndarray`
            Train labels including forces and possibly energies.
        :obj:`float`
            Standard deviation of training labels.
        :obj:`float`
            Mean energy. If energy labels are not included then this is
            :obj:`None`.
        """
        # pylint: disable=invalid-name
        y = F.ravel().copy()

        if use_E and use_E_cstr:
            E_train = E.ravel().copy()
            E_train_mean = np.mean(E_train)

            y = np.hstack((y, -E_train + E_train_mean))
        else:
            E_train_mean = None

        y_std = np.std(y)
        y /= y_std

        return y, y_std, E_train_mean

    # pylint: disable-next=invalid-name
    def train(self, task, require_E_eval=False):
        r"""Train a model based on a task.

        Parameters
        ----------
        task : :obj:`dict`
            Data structure of custom type ``task`` from
            :meth:`~mbgdml._gdml.train.GDMLTrain.create_task`.
        require_E_eval : :obj:`bool`, default: ``False``
            Require energy evaluation.

        Returns
        -------
        :obj:`dict`
            Data structure of custom type ``model`` from
            :meth:`~mbgdml._gdml.train.GDMLTrain.create_model`.

        Raises
        ------
        ValueError
            If the provided dataset contains invalid lattice vectors.
        """
        # pylint: disable=invalid-name
        task = dict(task)  # make mutable

        n_train, n_atoms = task["R_train"].shape[:2]

        desc = Desc(
            n_atoms,
            max_processes=self._max_processes,
        )

        n_perms = task["perms"].shape[0]
        tril_perms = np.array([desc.perm(p) for p in task["perms"]])

        # dim_i = 3 * n_atoms  # Unused variable
        dim_d = desc.dim

        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        tril_perms_lin = (tril_perms + perm_offsets).flatten("F")

        lat_and_inv = None
        if "lattice" in task:
            try:
                lat_and_inv = (task["lattice"], np.linalg.inv(task["lattice"]))
            except np.linalg.LinAlgError as e:
                raise ValueError(
                    "Provided dataset contains invalid lattice vectors (not "
                    "invertible). Note: Only rank 3 lattice vector matrices are "
                    "supported."
                ) from e

        R = task["R_train"].reshape(n_train, -1)
        R_desc, R_d_desc = desc.from_R(R, lat_and_inv=lat_and_inv)

        # Generate label vector.
        if task["use_E"]:
            E_train = task["use_E"]
        else:
            E_train = None
        y, y_std, E_train_mean = self.train_labels(
            task["F_train"], task["use_E"], task["use_E_cstr"], E=E_train
        )

        max_memory_bytes = self._max_memory * 1024**3

        # Memory cost of analytic solver
        est_bytes_analytic = Analytic.est_memory_requirement(n_train, n_atoms)

        # Memory overhead (solver independent)
        est_bytes_overhead = y.nbytes
        est_bytes_overhead += R.nbytes
        est_bytes_overhead += R_desc.nbytes
        est_bytes_overhead += R_d_desc.nbytes

        solver_keys = {}

        use_analytic_solver = (
            est_bytes_analytic + est_bytes_overhead
        ) < max_memory_bytes

        ###   mbGDML CHANGE START   ###
        if task["solver_name"] is None:
            # Fall back to analytic solver
            use_analytic_solver = True
        else:
            solver = task["solver_name"]
            assert solver in ("analytic", "iterative")
            use_analytic_solver = bool(solver == "analytic")
        ###   mbGDML CHANGE END   ###

        if use_analytic_solver:
            mem_req_mb = (est_bytes_analytic + est_bytes_overhead) * 1e-6  # MB
            log.info(
                "Using analytic solver (expected memory requirement: ~%.2f MB)",
                mem_req_mb,
            )
            ###   mbGDML CHANGE START   ###
            analytic = Analytic(self, desc)
            alphas = analytic.solve(task, R_desc, R_d_desc, tril_perms_lin, y)
            solver_keys["norm_y_train"] = np.linalg.norm(y)
            ###   mbGDML CHANGE END   ###

        else:
            # Iterative solver
            max_n_inducing_pts = Iterative.max_n_inducing_pts(
                n_train, n_atoms, max_memory_bytes
            )
            est_bytes_iterative = Iterative.est_memory_requirement(
                n_train, max_n_inducing_pts, n_atoms
            )
            mem_req_mb = (est_bytes_iterative + est_bytes_overhead) * 1e-6  # MB
            log.info(
                "Using iterative solver (expected memory requirement: ~%.2f MB)",
                mem_req_mb,
            )

            alphas_F = task["alphas0_F"] if "alphas0_F" in task else None
            alphas_E = task["alphas0_E"] if "alphas0_E" in task else None

            iterative = Iterative(
                self,
                desc,
                self._max_memory,
                self._max_processes,
                self._use_torch,
            )
            (
                alphas,
                solver_keys["solver_tol"],
                solver_keys[
                    "solver_iters"
                ],  # number of iterations performed (cg solver)
                solver_keys["solver_resid"],  # residual of solution
                _,  # train_rmse
                solver_keys["inducing_pts_idxs"],
                is_conv,
            ) = iterative.solve(
                task,
                R_desc,
                R_d_desc,
                tril_perms_lin,
                y,
                y_std,
            )

            solver_keys["norm_y_train"] = np.linalg.norm(y)

            if not is_conv:
                log.warning("Iterative solver did not converge!")
                log.info("Troubleshooting tips:")
                log.info(
                    "(1) Are the provided geometries highly correlated (i.e. very "
                    "similar to each other)?"
                )
                log.info("(2) Try a larger length scale (sigma) parameter.")
                log.warning(
                    "We will continue with this unconverged model, but its accuracy "
                    "will likely be very bad."
                )

        alphas_E = None
        alphas_F = alphas
        if task["use_E_cstr"]:
            alphas_E = alphas[-n_train:]
            alphas_F = alphas[:-n_train]

        model = self.create_model(
            task,
            "analytic" if use_analytic_solver else "cg",
            R_desc,
            R_d_desc,
            tril_perms_lin,
            y_std,
            alphas_F,
            alphas_E=alphas_E,
        )
        model.update(solver_keys)

        # Recover integration constant.
        # Note: if energy constraints are included in the kernel (via 'use_E_cstr'),
        # do not compute the integration constant, but simply set it to the mean of
        # the training energies (which was subtracted from the labels before training).
        if model["use_E"]:
            c = (
                self._recov_int_const(
                    model,
                    task,
                    R_desc=R_desc,
                    R_d_desc=R_d_desc,
                    require_E_eval=require_E_eval,
                )
                if E_train_mean is None
                else E_train_mean
            )
            model["c"] = c

        return model

    def _recov_int_const(
        self, model, task, R_desc=None, R_d_desc=None, require_E_eval=False
    ):
        r"""Estimate the integration constant for a force field model.

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
        R_desc : :obj:`numpy.ndarray`, ndim: ``2``, optional
            An array of size :math:`M \\times D` containing the descriptors of
            dimension :math:`D` for :math:`M` molecules.
        R_d_desc : :obj:`numpy.ndarray`, ndim: ``2``, optional
            An array of size :math:`M \\times D \\times 3N` containing of the
            descriptor Jacobians for :math:`M` molecules. The descriptor
            has dimension :math:`D` with :math:`3N` partial derivatives with
            respect to the :math:`3N` Cartesian coordinates of each atom.
        require_E_eval : :obj:`bool`, default: ``False``
            Force the computation and return of the integration constant
            regardless if there are significant errors.

        Returns
        -------
        :obj:`float`
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
        # pylint: disable=invalid-name
        gdml_predict = GDMLPredict(
            model,
            max_memory=self._max_memory,
            max_processes=self._max_processes,
            use_torch=self._use_torch,
        )

        gdml_predict.set_R_desc(R_desc)
        gdml_predict.set_R_d_desc(R_d_desc)

        n_train = task["E_train"].shape[0]
        R = task["R_train"].reshape(n_train, -1)

        E_pred, _ = gdml_predict.predict(R)
        E_ref = np.squeeze(task["E_train"])

        e_fact = np.linalg.lstsq(
            np.column_stack((E_pred, np.ones(E_ref.shape))), E_ref, rcond=-1
        )[0][0]
        corrcoef = np.corrcoef(E_ref, E_pred)[0, 1]

        if not require_E_eval:
            if np.sign(e_fact) == -1:
                log.warning(
                    "The provided dataset contains gradients instead of force labels "
                    "(flipped sign). Please correct!\nNote: The energy prediction "
                    "accuracy of the model will thus neither be validated nor tested "
                    "in the following steps!"
                )
                return None

            if corrcoef < 0.95:
                log.warning("Inconsistent energy labels detected!")
                log.warning(
                    "The predicted energies for the training data are only weakly\n"
                    "correlated with the reference labels (correlation "
                    "coefficient %.2f)\nwhich indicates that the issue is most "
                    "likely NOT just a unit conversion error.\n",
                    corrcoef,
                )
                return None

            if np.abs(e_fact - 1) > 1e-1:
                log.warning("Different scales in energy vs. force labels detected!\n")
                return None

        del gdml_predict

        # Least squares estimate for integration constant.
        return np.sum(E_ref - E_pred) / E_ref.shape[0]

    def _assemble_kernel_mat(
        self,
        R_desc,
        R_d_desc,
        tril_perms_lin,
        sig,
        desc,
        use_E_cstr=False,
        col_idxs=np.s_[:],
        alloc_extra_rows=0,
    ):
        r"""Compute force field kernel matrix.

        The Hessian of the Matern kernel is used with n = 2 (twice
        differentiable). Each row and column consists of matrix-valued blocks,
        which encode the interaction of one training point with all others. The
        result is stored in shared memory (a global variable).

        Parameters
        ----------
        R_desc : :obj:`numpy.ndarray`, ndim: ``2``, optional
            An array of size :math:`M \\times D` containing the descriptors of
            dimension :math:`D` for :math:`M` molecules.
        R_d_desc : :obj:`numpy.ndarray`, ndim: ``2``, optional
            An array of size :math:`M \\times D \\times 3N` containing of the
            descriptor Jacobians for :math:`M` molecules. The descriptor
            has dimension :math:`D` with :math:`3N` partial derivatives with
            respect to the :math:`3N` Cartesian coordinates of each atom.
        tril_perms_lin : :obj:`numpy.ndarray`, ndim: ``1``
            An array containing all recovered permutations
            expanded as one large permutation to be applied to a
            tiled copy of the object to be permuted.
        sig : :obj:`int`
            Hyperparameter sigma (kernel length scale).
        use_E_cstr : :obj:`bool`, optional
            Include energy constraints in the kernel. This can
            sometimes be helpful in tricky cases.
        cols_m_limit : :obj:`int`, optional
            Only generate the columns up to index ``cols_m_limit``. This creates
            a :math:`M3N \\times` ``cols_m_limit``:math:`3N` kernel matrix,
            instead of :math:`M3N \\times M3N`.
        cols_3n_keep_idxs : :obj:`numpy.ndarray`, optional
            Only generate columns with the given indices in the
            :math:`3N \\times 3N` kernel function. The resulting kernel matrix
            will have dimension :math:`M3N \\times M`
            ``len(cols_3n_keep_idxs)``.

        Returns
        -------
        :obj:`numpy.ndarray`
            Force field kernel matrix.
        """
        # pylint: disable=invalid-name
        global glob  # pylint: disable=global-variable-not-assigned

        n_train, dim_d = R_d_desc.shape[:2]
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)

        # Determine size of kernel matrix.
        K_n_rows = n_train * dim_i

        # Account for additional rows (and columns) due to energy constraints in the
        # kernel matrix.
        if use_E_cstr:
            K_n_rows += n_train

        if isinstance(col_idxs, slice):  # indexed by slice
            K_n_cols = len(range(*col_idxs.indices(K_n_rows)))
        else:  # indexed by list

            assert len(col_idxs) == len(set(col_idxs))  # assume no duplicate indices

            # Note: This function does not support unsorted (ascending) index arrays.
            assert np.array_equal(col_idxs, np.sort(col_idxs))

            K_n_cols = len(col_idxs)

        # Make sure no indices are outside of the valid range.
        if K_n_cols > K_n_rows:
            raise ValueError("Columns indexed beyond range.")

        exploit_sym = False
        cols_m_limit = None

        # Check if range is a subset of training points (as opposed to a subset of
        # partials of multiple points).
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

            J = range(*M_slice.indices(n_train + (n_train if use_E_cstr else 0)))

            if M_slice_start is None:
                exploit_sym = True
                cols_m_limit = M_slice_stop

        else:

            if isinstance(col_idxs, slice):
                # random = list(range(*col_idxs.indices(n_train * dim_i)))
                col_idxs = list(range(*col_idxs.indices(K_n_rows)))

            # Separate column indices of force-force and force-energy constraints.
            cond = col_idxs >= (n_train * dim_i)
            ff_col_idxs, fe_col_idxs = col_idxs[~cond], col_idxs[cond]

            # M - number training
            # N - number atoms

            # Column indices that go beyond force-force correlations need a different
            # treatment.
            n_idxs = np.concatenate(
                [np.mod(ff_col_idxs, dim_i), np.zeros(fe_col_idxs.shape, dtype=int)]
            )

            m_idxs = np.concatenate([np.array(ff_col_idxs) // dim_i, fe_col_idxs])
            m_idxs_uniq = np.unique(m_idxs)  # which points to include?

            m_n_idxs = [
                list(n_idxs[np.where(m_idxs == m_idx)]) for m_idx in m_idxs_uniq
            ]
            m_n_idxs_lens = [len(m_n_idx) for m_n_idx in m_n_idxs]

            m_n_idxs_lens.insert(0, 0)
            blk_start_idxs = list(
                np.cumsum(m_n_idxs_lens[:-1])
            )  # index within K at which each block starts

            # tuples: (block index in final K, block index global, indices of partials
            # within block)
            J = list(zip(blk_start_idxs, m_idxs_uniq, m_n_idxs))

        if self._use_torch:
            # pylint: disable=no-member, invalid-name, global-variable-undefined
            if not _HAS_TORCH:
                raise ImportError("PyTorch not found! Please install PyTorch.")

            K = np.empty((K_n_rows + alloc_extra_rows, K_n_cols))

            if J is not list:
                J = list(J)

            global torch_assemble_done
            _, torch_assemble_done = K_n_cols, 0

            if _TORCH_CUDA_IS_AVAILABLE:
                torch_device = "cuda"
            elif _TORCH_MPS_IS_AVAILABLE:
                torch_device = "mps"
            else:
                torch_device = "cpu"

            R_desc_torch = torch.from_numpy(R_desc).to(torch_device)  # N, d
            R_d_desc_torch = torch.from_numpy(R_d_desc).to(torch_device)

            # pylint: disable-next=import-outside-toplevel
            from .torchtools import GDMLTorchAssemble

            torch_assemble = GDMLTorchAssemble(
                J,
                tril_perms_lin,
                sig,
                use_E_cstr,
                R_desc_torch,
                R_d_desc_torch,
                out=K[:K_n_rows, :],
            )

            # Enable data parallelism
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                torch_assemble = torch.nn.DataParallel(torch_assemble)
            torch_assemble.to(torch_device)

            torch_assemble.forward(torch.arange(len(J)))
            del torch_assemble

            del R_desc_torch
            del R_d_desc_torch

            return K

        K = mp.RawArray("d", (K_n_rows + alloc_extra_rows) * K_n_cols)
        glob["K"], glob["K_shape"] = K, (K_n_rows + alloc_extra_rows, K_n_cols)
        glob["R_desc"], glob["R_desc_shape"] = _share_array(R_desc, "d")
        glob["R_d_desc"], glob["R_d_desc_shape"] = _share_array(R_d_desc, "d")

        glob["desc_func"] = desc

        pool = None
        map_func = map
        if self._max_processes != 1 and mp.cpu_count() > 1:
            pool = Pool(
                (self._max_processes or mp.cpu_count()) - 1
            )  # exclude main process
            map_func = pool.imap_unordered

        _, done = K_n_cols, 0
        for done_wkr in map_func(
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

        if pool is not None:
            pool.close()
            # Wait for the worker processes to terminate (to measure total runtime
            # correctly).
            pool.join()
            pool = None

        # Release some memory.
        glob.pop("K", None)
        glob.pop("R_desc", None)
        glob.pop("R_d_desc", None)

        return np.frombuffer(K).reshape((K_n_rows + alloc_extra_rows), K_n_cols)


def get_test_idxs(model, dataset, n_test=None):
    r"""Gets dataset indices for testing a model.

    Parameters
    ----------
    model : :obj:`dict`
        Model to test.
    dataset : :obj:`dict`
        Dataset to be used for testing.
    n_test : :obj:`int`, default: :obj:`None`
        Number of points to include in test indices. Defaults to all available
        indices.

    Returns
    -------
    :obj:`numpy.ndarray`
        Structure indices for model testing.
    """
    # exclude training and/or test sets from validation set if necessary
    excl_idxs = np.empty((0,), dtype=np.uint)

    def convert_md5(md5_value):
        if isinstance(md5_value, np.ndarray):
            return str(md5_value.item())

        return str(md5_value)

    md5_dset = convert_md5(dataset["md5"])
    md5_train = convert_md5(model["md5_train"])
    md5_valid = convert_md5(model["md5_valid"])

    if md5_dset == md5_train:
        excl_idxs = np.concatenate([excl_idxs, model["idxs_train"]]).astype(np.uint)
    if md5_dset == md5_valid:
        excl_idxs = np.concatenate([excl_idxs, model["idxs_valid"]]).astype(np.uint)

    n_data = dataset["F"].shape[0]
    n_data_eff = n_data - len(excl_idxs)
    if n_test is None:
        n_test = n_data_eff

    if n_data_eff == 0:
        log.warning("No unused points for testing in provided dataset.")
        return None

    log.info("Test set size was automatically set to %d points.", n_data_eff)

    if "E" in dataset:
        test_idxs = draw_strat_sample(dataset["E"], n_test, excl_idxs=excl_idxs)
    else:
        test_idxs = np.delete(np.arange(n_data), excl_idxs)

        log.warning(
            "Test dataset will be sampled with no guidance from energy labels "
            "(randomly)!\nNote: Larger test datasets are recommended due to slower "
            "convergence of the error."
        )

    return test_idxs


def add_valid_errors(
    model, dataset, overwrite=False, max_processes=None, use_torch=False
):
    r"""Calculate and add energy and force validation errors to a model.

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
        Force and energy validation errors as arrays with keys ``'force'`` and
        ``'energy'``.
    :obj:`dict`
        Model with validation errors.
    """
    log.info(
        "\n------------------------\n"
        "|   Model Validation   |\n"
        "------------------------"
    )
    f_err = np.array(model["f_err"]).item()
    is_model_validated = not (np.isnan(f_err["mae"]) or np.isnan(f_err["rmse"]))
    if is_model_validated and not overwrite:
        log.warning("Model is already validated and overwrite is False.")
        return None

    _, E_errors, F_errors = model_errors(
        model, dataset, is_valid=True, max_processes=max_processes, use_torch=use_torch
    )

    model["n_test"] = 0  # flag the model as not tested

    results = {"force": F_errors}
    model["f_err"] = {"mae": mae(F_errors), "rmse": rmse(F_errors)}

    if model["use_E"]:
        results["energy"] = E_errors
        model["e_err"] = {"mae": mae(E_errors), "rmse": rmse(E_errors)}
    else:
        results["energy"] = None
    return results, model


def save_model(model, model_path):
    np.savez_compressed(model_path, **model)


def model_errors(
    model, dataset, is_valid=False, n_test=None, max_processes=None, use_torch=False
):
    r"""Computes model errors for either validation or testing purposes.

    Parameters
    ----------
    model : :obj:`dict`
        Trained GDML model.
    dataset : :obj:`dict`
        Dataset to test the model against.
    is_valid : :obj:`bool`, default: ``False``
        Is this for model validation? Determines how we get the structure
        indices.
    n_test : :obj:`int`, default: :obj:`None`
        Number of desired testing indices. :obj:`None` will test against all
        available structures.

    Returns
    -------
    :obj:`int`
        Number of structures predicted.
    :obj:`numpy.ndarray`
        (ndim: ``1``) Energy errors or :obj:`None`.
    :obj:`numpy.ndarray`
        (ndim: ``3``) Force errors.
    """
    # pylint: disable=protected-access
    num_workers, batch_size = 0, 0

    if not np.array_equal(model["Z"], dataset["Z"]):
        raise ValueError(
            "Atom composition or order in dataset does not match the one in model"
        )

    if ("lattice" in model) is not ("lattice" in dataset):
        if "lattice" in model:
            raise ValueError("Model contains lattice vectors, but dataset does not.")

        if "lattice" in dataset:
            raise ValueError("Dataset contains lattice vectors, but model does not.")

    if is_valid:
        test_idxs = model["idxs_valid"]
    else:
        log.info(
            "\n---------------------\n"
            "|   Model Testing   |\n"
            "---------------------\n"
        )
        log.log_model(model)
        test_idxs = get_test_idxs(model, dataset, n_test=n_test)

    Z = dataset["Z"]
    R = dataset["R"][test_idxs, :, :]
    F = dataset["F"][test_idxs, :, :]
    n_R = R.shape[0]

    if model["use_E"]:
        E = dataset["E"][test_idxs]

    gdml_predict = GDMLPredict(model, max_processes=max_processes, use_torch=use_torch)

    log.info("\nPredicting %r structures", len(test_idxs))
    b_size = min(1000, len(test_idxs))
    log.info("Batch size : %d structures\n", b_size)

    if not use_torch:
        log.debug("Using CPU (use_torch = False)")
        if num_workers == 0 or batch_size == 0:
            gdml_predict.prepare_parallel(n_bulk=b_size, return_is_from_cache=True)
            num_workers, batch_size, bulk_mp = (
                gdml_predict.num_workers,
                gdml_predict.chunk_size,
                gdml_predict.bulk_mp,
            )
        else:
            gdml_predict._set_num_workers(num_workers)
            gdml_predict._set_chunk_size(batch_size)
            gdml_predict._set_bulk_mp(bulk_mp)

    n_atoms = Z.shape[0]

    E_pred, F_pred = np.empty(n_R), np.empty(R.shape)
    t_pred = log.t_start()
    n_done = 0
    for b_range in chunk_array(list(range(len(test_idxs))), b_size):
        log.info("%d done", n_done)
        n_done_step = len(b_range)
        n_done += n_done_step

        r = R[b_range].reshape(n_done_step, -1)
        e_pred, f_pred = gdml_predict.predict(r)

        F_pred[b_range] = f_pred.reshape((len(b_range), n_atoms, 3))
        if model["use_E"]:
            E_pred[b_range] = e_pred
    log.info("%d done", n_done)
    del gdml_predict

    t_elapsed = log.t_stop(t_pred, message="\nTook {time} s")
    pred_rate = n_R / t_elapsed
    log.info("Prediction rate : %.2f structures per second", pred_rate)

    # Force errors
    log.info("Computing force (and energy) errors")
    F_errors = F_pred - F
    F_mae = mae(F_errors)
    F_rmse = rmse(F_errors)
    log.info("\nForce MAE  : %.5f", F_mae)
    log.info("Force RMSE : %.5f", F_rmse)

    # Energy errors
    if model["use_E"]:
        E_errors = E_pred - E
        E_mae = mae(E_errors)
        E_rmse = rmse(E_errors)
        log.info("Energy MAE  : %.5f", E_mae)
        log.info("Energy RMSE : %.5f", E_rmse)

    if model["use_E"]:
        return len(test_idxs), E_errors, F_errors

    return len(test_idxs), None, F_errors
