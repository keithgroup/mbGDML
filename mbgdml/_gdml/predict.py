# MIT License
#
# Copyright (c) 2018-2022, Stefan Chmiela, Gregory Fonseca
# Copyright (c) 2020-2023, Alex M. Maldonado
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

import sys
import os
import multiprocessing as mp
import timeit
from functools import partial
import numpy as np
import psutil

from .. import __version__
from ..logger import GDMLLogger
from .desc import Desc

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

Pool = mp.get_context("fork").Pool

log = GDMLLogger(__name__)


def share_array(arr_np):
    """
    Return a ctypes array allocated from shared memory with data from a
    NumPy array of type `float`.

    Parameters
    ----------
    arr_np : :obj:`numpy.ndarray`
        NumPy array.

    Returns
    -------
    array of ``ctype``
    """

    arr = mp.RawArray("d", arr_np.ravel())
    return arr, arr_np.shape


# pylint: disable-next=too-many-statements
def _predict_wkr(
    r, r_desc_d_desc, lat_and_inv, glob_id, wkr_start_stop=None, chunk_size=None
):
    """
    Compute (part) of a prediction.

    Every prediction is a linear combination involving the training points used for
    this model. This function evaluates that combination for the range specified by
    `wkr_start_stop`. This workload can optionally be processed in chunks,
    which can be faster as it requires less memory to be allocated.

    Note
    ----
    It is sufficient to provide either the parameter ``r`` or ``r_desc_d_desc``.
    The other one can be set to :obj:`None`.

    Parameters
    ----------
    r : :obj:`numpy.ndarray`
        An array of size 3N containing the Cartesian
        coordinates of each atom in the molecule.
    r_desc_d_desc : :obj:`tuple` of :obj:`numpy.ndarray`
        A tuple made up of:
            (1) An array of size D containing the descriptors
            of dimension D for the molecule.
            (2) An array of size D x 3N containing the
            descriptor Jacobian for the molecules. It has dimension
            D with 3N partial derivatives with respect to the 3N
            Cartesian coordinates of each atom.
    lat_and_inv : :obj:`tuple` of :obj:`numpy.ndarray`
        Tuple of 3 x 3 matrix containing lattice vectors as columns and
        its inverse.
    glob_id : :obj:`int`
        Identifier of the global namespace that this
        function is supposed to be using (zero if only one
        instance of this class exists at the same time).
    wkr_start_stop : :obj:`tuple` of :obj:`int`, optional
        Range defined by the indices of first and last (exclusive)
        sum element. The full prediction is generated if this parameter
        is not specified.
    chunk_size : :obj:`int`, optional
        Chunk size. The whole linear combination is evaluated in a large
        vector operation instead of looping over smaller chunks if this
        parameter is left unspecified.

    Returns
    -------
    :obj:`numpy.ndarray`
        Partial prediction of all force components and energy (appended to
        array as last element).
    """
    global globs  # pylint: disable=global-variable-not-assigned
    glob = globs[glob_id]
    sig, n_perms = glob["sig"], glob["n_perms"]

    desc_func = glob["desc_func"]

    R_desc_perms = np.frombuffer(glob["R_desc_perms"]).reshape(
        glob["R_desc_perms_shape"]
    )
    R_d_desc_alpha_perms = np.frombuffer(glob["R_d_desc_alpha_perms"]).reshape(
        glob["R_d_desc_alpha_perms_shape"]
    )

    if "alphas_E_lin" in glob:
        alphas_E_lin = np.frombuffer(glob["alphas_E_lin"]).reshape(
            glob["alphas_E_lin_shape"]
        )

    r_desc, r_d_desc = r_desc_d_desc or desc_func.from_R(
        r, lat_and_inv, max_processes=1
    )  # no additional forking during parallelization

    n_train = int(R_desc_perms.shape[0] / n_perms)

    wkr_start, wkr_stop = (0, n_train) if wkr_start_stop is None else wkr_start_stop
    if chunk_size is None:
        chunk_size = n_train

    dim_d = desc_func.dim
    dim_i = desc_func.dim_i
    dim_c = chunk_size * n_perms

    # Pre-allocate memory.
    diff_ab_perms = np.empty((dim_c, dim_d))
    a_x2 = np.empty((dim_c,))
    mat52_base = np.empty((dim_c,))

    # avoid divisions (slower)
    sig_inv = 1.0 / sig
    mat52_base_fact = 5.0 / (3 * sig**3)
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

        # Resize pre-allocated memory for last iteration, if chunk_size is not a
        # divisor of the training set size.
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

        # Note: Energies are automatically predicted with a flipped sign here
        # (because -E are trained, instead of E)
        E_F[0] += a_x2.dot(mat52_base)

        # Note: Energies are automatically predicted with a flipped sign here
        # (because -E are trained, instead of E)
        if "alphas_E_lin" in glob:

            K_fe = diff_ab_perms * mat52_base[:, None]  # pylint: disable=invalid-name
            F += alphas_E_lin[b_start:b_stop].dot(K_fe)

            # pylint: disable-next=invalid-name
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

    out[1:] = desc_func.vec_dot_d_desc(
        r_d_desc,
        F,
    )  # 'r_d_desc.T.dot(F)' for our special representation of 'r_d_desc'

    return out


class GDMLPredict:

    # pylint: disable=too-many-branches, too-many-statements

    def __init__(
        self,
        model,
        batch_size=None,
        num_workers=None,
        max_memory=None,
        max_processes=None,
        use_torch=False,
    ):
        r"""Query trained sGDML force fields.

        This class is used to load a trained model and make energy and
        force predictions for new geometries. GPU support is provided
        through PyTorch (requires optional ``torch`` dependency to be
        installed).

        .. important::

            This is only used in
            :meth:`mbgdml._gdml.train.GDMLTrain._recov_int_const`
            and :func:`mbgdml._gdml.train.model_errors`.

        .. note::

            The parameters ``batch_size`` and ``num_workers`` are only
            relevant if this code runs on a CPU. Both can be set
            automatically via the function ``prepare_parallel``.
            Note: Running calculations via PyTorch is only
            recommended with available GPU hardware. CPU calculations
            are faster with our NumPy implementation.

        Parameters
        ----------
        model : :obj:`dict`
            Data structure that holds all parameters of the trained model.
            This object is the output of `GDMLTrain.train`
        batch_size : :obj:`int`, optional
            Chunk size for processing parallel tasks.
        num_workers : :obj:`int`, optional
            Number of parallel workers.
        max_memory : :obj:`int`, optional
            Limit the max. memory usage [GB]. This is only a
            soft limit that can not always be enforced.
        max_processes : :obj:`int`, optional
            Limit the max. number of processes. Otherwise all CPU cores are
            used. This parameters has no effect if `use_torch=True`
        use_torch : :obj:`bool`, optional
            Use PyTorch to calculate predictions
        """
        log.debug("Initializing GDMLPredict object")
        global globs  # pylint: disable=global-variable-undefined
        if "globs" not in globals():
            log.debug("globs not found in globals; creating it now.")
            globs = []

        # Create a personal global space for this model at a new index
        # Note: do not call delete entries in this list, since 'self.glob_id' is
        # static. Instead, setting them to None conserves positions while still
        # freeing up memory.
        globs.append({})
        self.glob_id = len(globs) - 1
        glob = globs[self.glob_id]

        total_memory = psutil.virtual_memory().total // 2**30  # bytes to GB)
        self.max_memory = (
            min(max_memory, total_memory) if max_memory is not None else total_memory
        )
        log.debug("max_memory : %r MB", self.max_memory)

        total_cpus = mp.cpu_count()
        self.max_processes = (
            min(max_processes, total_cpus) if max_processes is not None else total_cpus
        )
        log.debug("max_processes : %r", self.max_processes)

        if "type" not in model or not (model["type"] == "m" or model["type"] == b"m"):
            log.critical("The provided data structure is not a valid model.")
            sys.exit()

        self.n_atoms = model["Z"].shape[0]
        log.debug("n_atoms in model : %r", self.n_atoms)

        log.debug("Creating Desc object")
        self.desc = Desc(self.n_atoms, max_processes=max_processes)
        glob["desc_func"] = self.desc

        # Cache for iterative training mode.
        self.R_desc = None
        self.R_d_desc = None

        self.lat_and_inv = (
            (model["lattice"], np.linalg.inv(model["lattice"]))
            if "lattice" in model
            else None
        )

        log.debug("Unpacking model")
        self.n_train = model["R_desc"].shape[1]
        glob["sig"] = model["sig"]

        self.std = model["std"] if "std" in model else 1.0
        self.c = model["c"]

        n_perms = model["perms"].shape[0]
        self.n_perms = n_perms
        glob["n_perms"] = n_perms

        self.tril_perms_lin = model["tril_perms_lin"]

        self.torch_predict = None
        self.use_torch = use_torch
        if use_torch:
            log.debug("use_torch is True; setting up now")
            if not _HAS_TORCH:
                raise ImportError("Optional PyTorch dependency not found!")

            # pylint: disable-next=import-outside-toplevel
            from .torchtools import GDMLTorchPredict

            self.torch_predict = GDMLTorchPredict(
                model,
                self.lat_and_inv,
                max_memory=max_memory,
                max_processes=max_processes,
            )

            # Enable data parallelism
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                self.torch_predict = torch.nn.DataParallel(self.torch_predict)

            # Send model to device
            # self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if _TORCH_CUDA_IS_AVAILABLE:
                self.torch_device = "cuda"
            elif _TORCH_MPS_IS_AVAILABLE:
                self.torch_device = "mps"
            else:
                self.torch_device = "cpu"

            while True:
                try:
                    self.torch_predict.to(self.torch_device)
                except RuntimeError as e:
                    if "out of memory" in str(e):

                        if _TORCH_CUDA_IS_AVAILABLE:
                            torch.cuda.empty_cache()

                        model = self.torch_predict
                        if isinstance(self.torch_predict, torch.nn.DataParallel):
                            model = model.module

                        # model caches the permutations, this could be why it is too
                        # large
                        if model.get_n_perm_batches() == 1:
                            model.set_n_perm_batches(
                                model.get_n_perm_batches() + 1
                            )  # uncache
                        else:
                            self.log.critical(
                                "Not enough memory on device (RAM or GPU memory). "
                                "There is no hope!"
                            )
                            print()
                            os._exit(1)
                    else:
                        raise e
                else:
                    break
        else:

            # Precompute permuted training descriptors and its first derivatives
            # multiplied with the coefficients.

            R_desc_perms = (
                np.tile(model["R_desc"].T, n_perms)[:, self.tril_perms_lin]
                .reshape(self.n_train, n_perms, -1, order="F")
                .reshape(self.n_train * n_perms, -1)
            )
            glob["R_desc_perms"], glob["R_desc_perms_shape"] = share_array(R_desc_perms)

            R_d_desc_alpha_perms = (
                np.tile(model["R_d_desc_alpha"], n_perms)[:, self.tril_perms_lin]
                .reshape(self.n_train, n_perms, -1, order="F")
                .reshape(self.n_train * n_perms, -1)
            )
            (
                glob["R_d_desc_alpha_perms"],
                glob["R_d_desc_alpha_perms_shape"],
            ) = share_array(R_d_desc_alpha_perms)

            if "alphas_E" in model:
                alphas_E_lin = np.tile(model["alphas_E"][:, None], (1, n_perms)).ravel()
                glob["alphas_E_lin"], glob["alphas_E_lin_shape"] = share_array(
                    alphas_E_lin
                )

        # Parallel processing configuration
        log.debug("Setting up parallel processing")

        self.bulk_mp = False  # Bulk predictions with multiple processes?
        self.pool = None

        # How many workers in addition to main process?
        num_workers = num_workers or (self.max_processes - 1)  # exclude main process
        self.num_workers = num_workers
        self._set_num_workers(num_workers, force_reset=True)

        # Size of chunks in which each parallel task will be processed
        # (unit: number of training samples)
        # This parameter should be as large as possible, but it depends on the size of
        # available memory.
        self._set_chunk_size(batch_size)

        log.debug("Done initializing GDMLPredict object")

    def __del__(self):

        global globs  # pylint: disable=global-variable-not-assigned, invalid-name

        try:
            # Changed from terminate to avoid throwing sigterm during tests.
            self.pool.close()
            self.pool.join()
            self.pool = None
        except Exception as e:  # pylint: disable=unused-variable
            pass

        if "globs" in globals() and globs is not None and self.glob_id < len(globs):
            globs[self.glob_id] = None

    ## Public ##

    # pylint: disable-next=invalid-name
    def set_R_desc(self, R_desc):
        r"""Store a reference to the training geometry descriptors.

        This can accelerate iterative model training.

        Parameters
        ----------
        R_desc : :obj:`numpy.ndarray`, optional
            An 2D array of size M x D containing the descriptors of dimension
            D for M molecules.
        """

        self.R_desc = R_desc

    # pylint: disable-next=invalid-name
    def set_R_d_desc(self, R_d_desc):
        r"""Store a reference to the training geometry descriptor Jacobians.

        This function must be called before ``set_alphas()`` can be used.

        This routine is used during iterative model training.

        Parameters
        ----------
        R_d_desc : :obj:`numpy.ndarray`, optional
            A 2D array of size M x D x 3N containing of the descriptor Jacobians
            for M molecules. The descriptor has dimension D with 3N partial
            derivatives with respect to the 3N Cartesian coordinates of each
            atom.
        """

        self.R_d_desc = R_d_desc

        if self.use_torch:
            model = self.torch_predict
            if isinstance(self.torch_predict, torch.nn.DataParallel):
                model = model.module

            model.set_R_d_desc(R_d_desc)

    def set_alphas(self, alphas_F, alphas_E=None):
        r"""Reconfigure the current model with a new set of regression parameters.
        ``R_d_desc`` needs to be set for this function to work.

        This routine is used during iterative model training.

        Parameters
        ----------
        alphas_F : :obj:`numpy.ndarray`
            1D array containing the new model parameters.
        alphas_E : :obj:`numpy.ndarray`, optional
            1D array containing the additional new model parameters, if
            energy constraints are used in the kernel (`use_E_cstr=True`)
        """

        if self.use_torch:

            model = self.torch_predict
            if isinstance(self.torch_predict, torch.nn.DataParallel):
                model = model.module

            model.set_alphas(alphas_F, alphas_E=alphas_E)

        else:

            assert self.R_d_desc is not None

            global globs  # pylint: disable=global-variable-not-assigned
            glob = globs[self.glob_id]

            dim_i = self.desc.dim_i
            # pylint: disable-next=invalid-name
            R_d_desc_alpha = self.desc.d_desc_dot_vec(
                self.R_d_desc, alphas_F.reshape(-1, dim_i)
            )

            # pylint: disable-next=invalid-name
            R_d_desc_alpha_perms_new = np.tile(R_d_desc_alpha, self.n_perms)[
                :, self.tril_perms_lin
            ].reshape(self.n_train, self.n_perms, -1, order="F")

            R_d_desc_alpha_perms = np.frombuffer(glob["R_d_desc_alpha_perms"])
            np.copyto(R_d_desc_alpha_perms, R_d_desc_alpha_perms_new.ravel())

            if alphas_E is not None:

                # pylint: disable-next=invalid-name
                alphas_E_lin_new = np.tile(alphas_E[:, None], (1, self.n_perms)).ravel()

                alphas_E_lin = np.frombuffer(glob["alphas_E_lin"])
                np.copyto(alphas_E_lin, alphas_E_lin_new)

    def _set_num_workers(self, num_workers=None, force_reset=False):
        r"""Set number of processes to use during prediction.

        If ``bulk_mp is True``, each worker handles the whole generation of single
        prediction (this if for querying multiple geometries at once)

        If ``bulk_mp is False``, each worker may handle only a part of a prediction
        (chunks are defined in ``'wkr_starts_stops'``). In that scenario multiple
        processes are used to distribute the work of generating a single
        prediction.

        This number should not exceed the number of available CPU cores.

        Note
        ----
        This parameter can be optimally determined using
        ``prepare_parallel``.

        Parameters
        ----------
        num_workers : :obj:`int`, optional
            Number of processes (maximum value is set if :obj:`None`).
        force_reset : :obj:`bool`, optional
            Force applying the new setting.
        """
        # pylint: disable=access-member-before-definition

        log.debug("Setting the number of workers in GDMLPredict")
        log.debug("force_reset : %r", force_reset)
        log.debug("Local num_workers : %r", num_workers)
        log.debug("self.num_workers : %r", self.num_workers)
        log.debug("self.pool : %r", self.pool)

        if force_reset or self.num_workers is not num_workers:

            if self.pool is not None:
                log.debug("Resetting pool")
                log.debug("Running self.pool.close()")
                # Changed from terminate to avoid throwing sigterm during tests.
                self.pool.close()
                log.debug("Running self.pool.join()")
                self.pool.join()
                self.pool = None
                log.debug("self.pool is now None")

            self.num_workers = 0
            if num_workers is None or num_workers > 0:
                log.debug("Running Pool(%r)", num_workers)
                self.pool = Pool(num_workers)
                # pylint: disable-next=protected-access
                self.num_workers = self.pool._processes

        log.info("Set number of workers : %r", self.num_workers)

        # Data ranges for processes
        if self.bulk_mp or self.num_workers < 2:
            # wkr_starts = [self.n_train]
            wkr_starts = [0]
        else:
            wkr_starts = list(
                range(
                    0,
                    self.n_train,
                    int(np.ceil(float(self.n_train) / self.num_workers)),
                )
            )
        wkr_stops = wkr_starts[1:] + [self.n_train]

        self.wkr_starts_stops = list(zip(wkr_starts, wkr_stops))

    def _set_chunk_size(self, chunk_size=None):
        r"""Set chunk size for each worker process.

        Every prediction is generated as a linear combination of the training
        points that the model is comprised of. If multiple workers are available
        (and bulk mode is disabled), each one processes an (approximately equal)
        part of those training points. Then, the chunk size determines how much of
        a processes workload is passed to NumPy's underlying low-level routines at
        once. If the chunk size is smaller than the number of points the worker is
        supposed to process, it processes them in multiple steps using a loop. This
        can sometimes be faster, depending on the available hardware.

        Note
        ----
        This parameter can be optimally determined using
        ``prepare_parallel``.

        Parameters
        ----------
        chunk_size : :obj:`int`, default: :obj:`None`
            Chunk size (maximum value is set if `None`).
        """

        if chunk_size is None:
            chunk_size = self.n_train

        log.info("Set chunk size : %r", chunk_size)
        self.chunk_size = chunk_size

    def _set_bulk_mp(self, bulk_mp=False):
        """
        Toggles bulk prediction mode.

        If bulk prediction is enabled, the prediction is parallelized across
        input geometries, i.e. each worker generates the complete prediction for
        one query. Otherwise (depending on the number of available CPU cores) the
        input geometries are process sequentially, but every one of them may be
        processed by multiple workers at once (in chunks).

        Note
        ----
        This parameter can be optimally determined using `prepare_parallel`.

        Parameters
        ----------
        bulk_mp : :obj:`bool`, optional
            Enable or disable bulk prediction mode.
        """

        bulk_mp = bool(bulk_mp)
        if bulk_mp:
            log.info("Using one worker per structure (i.e., bulk)")
        else:
            log.info("Using multiple workers per structure")
        if self.bulk_mp is not bulk_mp:
            self.bulk_mp = bulk_mp

            # Reset data ranges for processes stored in 'wkr_starts_stops'
            self._set_num_workers(self.num_workers)

    def prepare_parallel(
        self, n_bulk=1, n_reps=1, return_is_from_cache=False
    ):  # noqa: C901
        """
        Find and set the optimal parallelization parameters for the
        currently loaded model, running on a particular system. The result
        also depends on the number of geometries ``n_bulk`` that will be
        passed at once when calling the `predict` function.

        This function runs a benchmark in which the prediction routine is
        repeatedly called ``n_reps``-times (default: 1) with varying parameter
        configurations, while the runtime is measured for each one. The
        optimal parameters are then cached for fast retrieval in future
        calls of this function.

        We recommend calling this function after initialization of this
        class, as it will drastically increase the performance of the
        ``predict`` function.

        Note
        ----
        Depending on the parameter ``n_reps``, this routine may take
        some seconds/minutes to complete. However, once a
        statistically significant number of benchmark results has
        been gathered for a particular configuration, it starts
        returning almost instantly.

        Parameters
        ----------
        n_bulk : :obj:`int`, optional
            Number of geometries that will be passed to the
            `predict` function in each call (performance
            will be optimized for that exact use case).
        n_reps : :obj:`int`, optional
            Number of repetitions (bigger value: more
            accurate, but also slower).
        return_is_from_cache : :obj:`bool`, optional
            If enabled, this function returns a second value
            indicating if the returned results were obtained
            from cache.

        Returns
        -------
        :obj:`int`
            Force and energy prediction speed in geometries per second.
        :obj:`bool`, optional
            Return, whether this function obtained the results from cache.
        """
        t_parallel = log.t_start()
        log.info("#   Optimizing parallelism   #")

        # No benchmarking necessary if prediction is running on GPUs.
        if self.use_torch:
            log.info("Skipping multi-CPU benchmark, since torch is enabled.")
            log.t_stop(t_parallel)
            return None

        # Retrieve cached benchmark results, if available.
        bmark_result = self._load_cached_bmark_result(n_bulk)
        if bmark_result is not None:

            num_workers, chunk_size, bulk_mp, gps = bmark_result

            self._set_chunk_size(chunk_size)
            self._set_num_workers(num_workers)
            self._set_bulk_mp(bulk_mp)

            if return_is_from_cache:
                is_from_cache = True
                log.t_stop(t_parallel, message="Parallel optimization took {time} s\n")
                return gps, is_from_cache

            log.t_stop(t_parallel, message="Parallel optimization took {time} s\n")
            return gps

        warm_up_done = False
        log.info("Preparing benchmark runs")
        best_results = []
        last_i = None

        best_gps = 0
        gps_min = 0.0

        best_params = None

        r_dummy = np.random.rand(n_bulk, self.n_atoms * 3)

        def _dummy_predict():
            self.predict(r_dummy)

        bulk_mp_rng = [True, False] if n_bulk > 1 else [False]
        for bulk_mp in bulk_mp_rng:
            self._set_bulk_mp(bulk_mp)

            if bulk_mp is False:
                last_i = 0

            num_workers_rng = list(range(0, self.max_processes))
            if bulk_mp:
                num_workers_rng.reverse()  # benchmark converges faster this way

            for num_workers in num_workers_rng:
                if not bulk_mp and num_workers != 0 and self.n_train % num_workers != 0:
                    continue

                self._set_num_workers(num_workers)

                best_gps = 0
                gps_rng = (np.inf, 0.0)  # min and max per num_workers

                min_chunk_size = (
                    min(self.n_train, n_bulk)
                    if bulk_mp or num_workers < 2
                    else int(np.ceil(self.n_train / num_workers))
                )
                chunk_size_rng = list(range(min_chunk_size, 0, -1))

                chunk_size_rng_sizes = [
                    chunk_size
                    for chunk_size in chunk_size_rng
                    if min_chunk_size % chunk_size == 0
                ]

                i_done = 0
                i_dir = 1
                i = 0 if last_i is None else last_i

                while 0 <= i < len(chunk_size_rng_sizes):

                    chunk_size = chunk_size_rng_sizes[i]
                    self._set_chunk_size(chunk_size)

                    i_done += 1

                    log.info("Timing predictions")
                    if warm_up_done is False:
                        timeit.timeit(_dummy_predict, number=10)
                        warm_up_done = True

                    gps = n_bulk * n_reps / timeit.timeit(_dummy_predict, number=n_reps)
                    log.info("Geometries per second : %r", gps)

                    gps_rng = (
                        min(gps_rng[0], gps),
                        max(gps_rng[1], gps),
                    )  # min and max per num_workers

                    # gps still going up?
                    # AND: gps not lower than the lowest overall?
                    if gps < best_gps:
                        if (
                            # there is no point in turning if this is the second
                            # batch size in the range
                            i_dir > 0
                            and i_done == 2
                            and chunk_size != chunk_size_rng_sizes[1]
                        ):  # do we turn?
                            i -= 2 * i_dir
                            i_dir = -1
                            continue

                        if chunk_size == chunk_size_rng_sizes[1]:
                            i -= 1 * i_dir
                        break

                    best_gps = gps
                    best_params = num_workers, chunk_size, bulk_mp

                    if (
                        not bulk_mp and n_bulk > 1
                    ):  # stop search early when multiple cpus are available and the
                        # 1 cpu case is tested
                        if (
                            gps < gps_min
                        ):  # if the batch size run is lower than the lowest
                            # overall, stop right here
                            # print('breaking here')
                            break

                    i += 1 * i_dir

                last_i = i - 1 * i_dir
                i_dir = 1

                if len(best_results) > 0:
                    overall_best_gps = max(best_results, key=lambda x: x[1])[1]
                    if best_gps < overall_best_gps:
                        # print('breaking, because best of last test was worse than
                        # overall best so far')
                        break

                gps_min = gps_rng[0]  # FIX me: is this the overall min?

                best_results.append(
                    (best_params, best_gps)
                )  # best results per num_workers

        (num_workers, chunk_size, bulk_mp), gps = max(best_results, key=lambda x: x[1])

        # Cache benchmark results.
        self._save_cached_bmark_result(n_bulk, num_workers, chunk_size, bulk_mp, gps)
        log.info("Best config : %r worker(s), %r chunk size", num_workers, chunk_size)

        self._set_chunk_size(chunk_size)
        self._set_num_workers(num_workers)
        self._set_bulk_mp(bulk_mp)

        if return_is_from_cache:
            is_from_cache = False
            log.t_stop(t_parallel, message="Parallel optimization took {time} s\n")
            return gps, is_from_cache

        log.t_stop(t_parallel, message="Parallel optimization took {time} s\n")
        return gps

    def _save_cached_bmark_result(self, n_bulk, num_workers, chunk_size, bulk_mp, gps):

        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        bmark_file = "_bmark_cache.npz"
        bmark_path = os.path.join(pkg_dir, bmark_file)

        bkey = f"{self.n_atoms}-{self.n_train}-{n_bulk}-{self.max_processes}"

        if os.path.exists(bmark_path):
            log.info("Appending benchmark results to existing cache")

            with np.load(bmark_path, allow_pickle=True) as bmark:
                bmark = dict(bmark)

                bmark["runs"] = np.append(bmark["runs"], bkey)
                bmark["num_workers"] = np.append(bmark["num_workers"], num_workers)
                bmark["batch_size"] = np.append(bmark["batch_size"], chunk_size)
                bmark["bulk_mp"] = np.append(bmark["bulk_mp"], bulk_mp)
                bmark["gps"] = np.append(bmark["gps"], gps)
        else:
            log.info("Caching new benchmark results")
            bmark = {
                "code_version": __version__,
                "runs": [bkey],
                "gps": [gps],
                "num_workers": [num_workers],
                "batch_size": [chunk_size],
                "bulk_mp": [bulk_mp],
            }

        np.savez_compressed(bmark_path, **bmark)

    def _load_cached_bmark_result(self, n_bulk):

        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        bmark_file = "_bmark_cache.npz"
        bmark_path = os.path.join(pkg_dir, bmark_file)
        log.debug("Looking for benchmark file at %s", bmark_path)

        bkey = f"{self.n_atoms}-{self.n_train}-{n_bulk}-{self.max_processes}"

        if not os.path.exists(bmark_path):
            log.info("No benchmark cache found")
            return None

        with np.load(bmark_path, allow_pickle=True) as bmark:
            log.info("Benchmark cache found")
            # Keep collecting benchmark runs, until we have at least three.
            run_idxs = np.where(bmark["runs"] == bkey)[0]
            log.info("Found %d benchmark runs", len(run_idxs))
            if len(run_idxs) >= 3:

                config_keys = []
                for run_idx in run_idxs:
                    config_keys.append(
                        f"{bmark['num_workers'][run_idx]}-"
                        f"{bmark['batch_size'][run_idx]}-"
                        f"{bmark['bulk_mp'][run_idx]}"
                    )

                values, uinverse = np.unique(config_keys, return_index=True)

                best_mean = -1
                best_gps = 0
                for i, config_key in enumerate(zip(values, uinverse)):
                    mean_gps = np.mean(
                        bmark["gps"][
                            np.where(np.array(config_keys) == config_key[0])[0]
                        ]
                    )

                    if best_gps == 0 or best_gps < mean_gps:
                        best_mean = i
                        best_gps = mean_gps

                best_idx = run_idxs[uinverse[best_mean]]
                num_workers = bmark["num_workers"][best_idx]
                chunk_size = bmark["batch_size"][best_idx]
                bulk_mp = bmark["bulk_mp"][best_idx]

                log.info(
                    "Best config : %r worker(s), %r chunk size", num_workers, chunk_size
                )
                return num_workers, chunk_size, bulk_mp, best_gps

            log.info("Not enough runs (need at 3)")

        return None

    # pylint: disable-next=invalid-name
    def get_GPU_batch(self):
        """
        Get batch size used by the GPU implementation to process bulk
        predictions (predictions for multiple input geometries at once).

        This value is determined on-the-fly depending on the available GPU
        memory.
        """
        if self.use_torch:

            model = self.torch_predict
            if isinstance(model, torch.nn.DataParallel):
                model = model.module

            return model._batch_size()  # pylint: disable=protected-access

        return None

    # pylint: disable-next=invalid-name
    def predict(self, R=None, return_E=True):
        r"""Predict energy and forces for multiple geometries.

        This function can run on the GPU, if the optional PyTorch dependency is
        installed and ``use_torch=True`` was specified during
        initialization of this class.

        Optionally, the descriptors and descriptor Jacobians for the
        same geometries can be provided, if already available from some
        previous calculations.

        Note
        ----
        The order of the atoms in ``R`` is not arbitrary and must
        be the same as used for training the model.

        Parameters
        ----------
        R : :obj:`numpy.ndarray`, optional
            An 2D array of size M x 3N containing the Cartesian coordinates of
            each atom of M molecules. If this parameter is omitted, the training
            error is returned. Note that the training geometries need to be set
            right after initialization using ``set_R()`` for this to work.
        return_E : :obj:`bool`, default: ``True``
            If ``False``, only the forces are returned.

        Returns
        -------
        :obj:`numpy.ndarray`
            Energies stored in an 1D array of size M.
            Unless ``return_E is False``.
        :obj:`numpy.ndarray`
            Forces stored in an 2D array of size M x 3N.
        """

        # Add singleton dimension if input is (,3N).
        if R is not None and R.ndim == 1:
            R = R[None, :]

        if self.use_torch:  # multi-GPU (or CPU if no GPUs are available)

            R_torch = torch.arange(self.n_train)  # pylint: disable=no-member
            if R is None:
                if self.R_d_desc is None:
                    log.critical(
                        "A reference to the training geometry descriptors needs to be "
                        "set (using 'set_R_d_desc()') for this function to work "
                        "without arguments (using PyTorch)."
                    )
                    print()
                    sys.exit(1)
            else:
                R_torch = (
                    # pylint: disable=no-member
                    torch.from_numpy(R.reshape(-1, self.n_atoms, 3))
                    .type(torch.float32)
                    .to(self.torch_device)
                )

            model = self.torch_predict
            if R_torch.shape[0] < torch.cuda.device_count() and isinstance(
                model, torch.nn.DataParallel
            ):
                model = self.torch_predict.module
            E_torch_F_torch = model.forward(R_torch, return_E=return_E)

            if return_E:
                E_torch, F_torch = E_torch_F_torch
                E = E_torch.cpu().numpy()
            else:
                (F_torch,) = E_torch_F_torch

            F = F_torch.cpu().numpy().reshape(-1, 3 * self.n_atoms)

        else:  # multi-CPU

            # Use precomputed descriptors in training mode.
            is_desc_in_cache = self.R_desc is not None and self.R_d_desc is not None

            if R is None and not is_desc_in_cache:
                log.critical(
                    "A reference to the training geometry descriptors and Jacobians "
                    "needs to be set for this function to work without arguments."
                )
                print()
                sys.exit(1)

            assert is_desc_in_cache or R is not None

            dim_i = 3 * self.n_atoms
            n_pred = self.R_desc.shape[0] if R is None else R.shape[0]

            E_F = np.empty((n_pred, dim_i + 1))

            if (
                self.bulk_mp and self.num_workers > 0
            ):  # One whole prediction per worker (and multiple workers).

                _predict_wo_r_or_desc = partial(
                    _predict_wkr,
                    lat_and_inv=self.lat_and_inv,
                    glob_id=self.glob_id,
                    wkr_start_stop=None,
                    chunk_size=self.chunk_size,
                )

                for i, e_f in enumerate(
                    self.pool.imap(
                        partial(_predict_wo_r_or_desc, None)
                        if is_desc_in_cache
                        else partial(_predict_wo_r_or_desc, r_desc_d_desc=None),
                        zip(self.R_desc, self.R_d_desc) if is_desc_in_cache else R,
                    )
                ):
                    E_F[i, :] = e_f

            else:  # Multiple workers per prediction (or just one worker).

                for i in range(n_pred):

                    if is_desc_in_cache:
                        r_desc, r_d_desc = self.R_desc[i], self.R_d_desc[i]
                    else:
                        r_desc, r_d_desc = self.desc.from_R(R[i], self.lat_and_inv)

                    _predict_wo_wkr_starts_stops = partial(
                        _predict_wkr,
                        None,
                        (r_desc, r_d_desc),
                        self.lat_and_inv,
                        self.glob_id,
                        chunk_size=self.chunk_size,
                    )

                    if self.num_workers == 0:
                        E_F[i, :] = _predict_wo_wkr_starts_stops()
                    else:
                        E_F[i, :] = sum(
                            self.pool.imap_unordered(
                                _predict_wo_wkr_starts_stops, self.wkr_starts_stops
                            )
                        )

            E_F *= self.std
            F = E_F[:, 1:]
            E = E_F[:, 0] + self.c

        ret = (F,)
        if return_E:
            ret = (E,) + ret

        return ret
