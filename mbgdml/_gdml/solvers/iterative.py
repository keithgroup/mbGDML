# MIT License
#
# Copyright (c) 2018-2022, Stefan Chmiela
# Copyright (c) 2023, Alex M. Maldonado
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

# pylint: disable=invalid-name,consider-using-f-string,global-variable-undefined
# pylint: disable=global-variable-not-assigned, no-member

import os
import inspect
import collections
import timeit
import numpy as np
import scipy as sp

from ..predict import GDMLPredict
from ...logger import GDMLLogger

log = GDMLLogger(__name__)

try:
    import torch
except ImportError:
    _HAS_TORCH = False
else:
    _HAS_TORCH = True

CG_STEPS_HIST_LEN = (
    100  # number of past steps to consider when calculating solver effectiveness
)
# if solver effectiveness is less than that percentage after 'CG_STEPS_HIST_LEN'-steps,
# a solver restart is triggered (with stronger preconditioner)
EFF_RESTART_THRESH = 0

MAX_NUM_RESTARTS = 6


class CGRestartException(Exception):
    pass


class Iterative:
    def __init__(
        self,
        gdml_train,
        desc,
        max_memory,
        max_processes,
        use_torch,
    ):
        r"""The sGDML :class:`sgdml.solvers.iterative.Iterative` class."""
        log.info("Initializing iterative solver")
        self.gdml_train = gdml_train
        self.gdml_predict = None
        self.desc = desc

        self._max_memory = max_memory
        self._max_processes = max_processes
        self._use_torch = use_torch

    def _init_precon_operator(
        self, task, R_desc, R_d_desc, tril_perms_lin, inducing_pts_idxs
    ):
        r"""Solve for :math:`\alpha`.

        Parameters
        ----------
        task : :obj:`dict`
            Properties of the training task.
        R_desc : :obj:`numpy.ndarray`
            Array containing the descriptor for each training point.
            Computed from :func:`~mbgdml._gdml.desc._r_to_desc`.
        R_d_desc : :obj:`numpy.ndarray`
            Array containing the gradient of the descriptor for
            each training point. Computed from
            :func:`~mbgdml._gdml.desc._r_to_d_desc`.
        tril_perms_lin : :obj:`numpy.ndarray`, ndim: ``1``
            An array containing all recovered permutations expanded as one large
            permutation to be applied to a tiled copy of the object to be
            permuted.
        inducing_pts_idxs : :obj:`numpy.ndarray`, ndim: ``1``
            Indices of inducing points.
        """

        lam = task["lam"]
        lam_inv = 1.0 / lam

        sig = task["sig"]

        use_E_cstr = task["use_E_cstr"]

        L_inv_K_mn = self._nystroem_cholesky_factor(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            lam,
            use_E_cstr=use_E_cstr,
            col_idxs=inducing_pts_idxs,
        )

        L_inv_K_mn = np.ascontiguousarray(L_inv_K_mn)

        lev_scores = np.einsum(
            "i...,i...->...", L_inv_K_mn, L_inv_K_mn
        )  # compute leverage scores because it is basically free once we got the factor

        _, n = L_inv_K_mn.shape

        # pylint: disable-next=condition-evals-to-constant
        if self._use_torch and False:  # TURNED OFF!
            _torch_device = "cuda" if torch.cuda.is_available() else "cpu"
            L_inv_K_mn_torch = torch.from_numpy(L_inv_K_mn).to(_torch_device)

        global is_primed
        is_primed = False

        def _P_vec(v):

            global is_primed
            if not is_primed:
                is_primed = True
                return v

            # pylint: disable-next=condition-evals-to-constant
            if self._use_torch and False:  # TURNED OFF!

                v_torch = torch.from_numpy(v).to(_torch_device)[:, None]
                return (
                    L_inv_K_mn_torch.t().mm(L_inv_K_mn_torch.mm(v_torch)) - v_torch
                ).cpu().numpy() * lam_inv

            ret = L_inv_K_mn.T.dot(L_inv_K_mn.dot(v))
            ret -= v
            ret *= lam_inv

            return ret

        return sp.sparse.linalg.LinearOperator((n, n), matvec=_P_vec), lev_scores

    def _init_kernel_operator(self, task, R_desc, R_d_desc, tril_perms_lin, lam, n):

        n_train = R_desc.shape[0]

        # dummy alphas
        v_F = np.zeros((n - n_train, 1)) if task["use_E_cstr"] else np.zeros((n, 1))
        v_E = np.zeros((n_train, 1)) if task["use_E_cstr"] else None

        # Note: The standard deviation is set to 1.0, because we are predicting
        # normalized labels here.
        model = self.gdml_train.create_model(
            task, "cg", R_desc, R_d_desc, tril_perms_lin, 1.0, v_F, alphas_E=v_E
        )

        self.gdml_predict = GDMLPredict(
            model,
            max_memory=self._max_memory,
            max_processes=self._max_processes,
            use_torch=self._use_torch,
        )

        self.gdml_predict.set_R_desc(R_desc)  # only needed on CPU
        self.gdml_predict.set_R_d_desc(R_d_desc)

        if not self._use_torch:

            self.gdml_predict.prepare_parallel(n_bulk=n_train)

        global is_primed
        is_primed = False

        def _K_vec(v):

            global is_primed
            if not is_primed:
                is_primed = True
                return v

            v_F, v_E = v, None
            if task["use_E_cstr"]:
                v_F, v_E = v[:-n_train], v[-n_train:]

            self.gdml_predict.set_alphas(v_F, alphas_E=v_E)

            pred = self.gdml_predict.predict(return_E=task["use_E_cstr"])
            if task["use_E_cstr"]:
                e_pred, f_pred = pred
                pred = np.hstack((f_pred.ravel(), -e_pred))
            else:
                pred = pred[0].ravel()

            pred -= lam * v
            return pred

        return sp.sparse.linalg.LinearOperator((n, n), matvec=_K_vec)

    def _nystroem_cholesky_factor(
        self,
        R_desc,
        R_d_desc,
        tril_perms_lin,
        sig,
        lam,
        use_E_cstr,
        col_idxs,
        task_name="",
    ):
        log.info("Assembling kernel [m x k] (%s)", task_name)

        dim_d = R_desc.shape[1]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        n = R_desc.shape[0] * n_atoms * 3 + (R_desc.shape[0] if use_E_cstr else 0)
        m = len(
            range(*col_idxs.indices(n)) if isinstance(col_idxs, slice) else col_idxs
        )

        # pylint: disable-next=protected-access
        K_nmm = self.gdml_train._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            self.desc,
            use_E_cstr=use_E_cstr,
            col_idxs=col_idxs,
            alloc_extra_rows=m,
        )

        # Store (psd) copy of K_mm in lower part of this oversized K_(n+m)m matrix.
        K_nmm[-m:, :] = -K_nmm[col_idxs, :]

        K_nm = K_nmm[:-m, :]
        K_mm = K_nmm[-m:, :]

        log.debug("Cholesky fact. (1/2) [k x k] (%s)", task_name)

        # Additional regularization is almost always necessary here
        # (hence pre_reg=True).
        K_mm, lower = self._cho_factor_stable(K_mm, pre_reg=True)  # overwrites input!
        L_mm = K_mm
        # del K_mm

        log.debug("m tri. solves (1/2) [k x k] (%s)", task_name)

        b_start, b_size = 0, int(n / 4)  # update in percentage steps of 25
        for b_stop in list(range(b_size, n, b_size)) + [n]:

            K_nm[b_start:b_stop, :] = sp.linalg.solve_triangular(
                L_mm,
                K_nm[b_start:b_stop, :].T,
                lower=lower,
                trans="T",
                overwrite_b=True,
                check_finite=False,
            ).T
            b_start = b_stop

        del L_mm

        K_nmm[-m:, :] = K_nm.T.dot(K_nm)
        K_nmm[-m:, :][np.diag_indices_from(K_nmm[-m:, :])] += lam
        inner = K_nmm[-m:, :]

        log.debug("Cholesky fact. (2/2) [k x k] (%s)", task_name)

        L_lower = self._cho_factor_stable(
            inner, eps_mag_max=-14
        )  # Do not regularize more than 1e-14.
        if L_lower is not None:
            K_nmm[-m:, :], lower = L_lower
            L = K_nmm[-m:, :]
            del inner
        else:

            log.debug("QR fact. (alt.) [k x k] (%s)", task_name)

            K_nmm[-m:, :] = 0
            K_nmm[-m:, :][np.diag_indices(m)] = np.sqrt(lam)

            K_nmm[-m:, :] = np.linalg.qr(K_nmm, mode="r")
            L = K_nmm[-m:, :]
            lower = False

        log.debug("m tri. solves (2/2) [k x k] (%s)", task_name)

        b_start, b_size = 0, int(n / 4)  # update in percentage steps of 25
        for b_stop in list(range(b_size, n, b_size)) + [n]:

            K_nm[b_start:b_stop, :] = sp.linalg.solve_triangular(
                L,
                K_nm[b_start:b_stop, :].T,
                lower=lower,
                trans="T",
                overwrite_b=True,
                check_finite=False,
            ).T  # Note: Overwrites K_nm to save memory
            b_start = b_stop

        del L

        return K_nm.T

    def _lev_scores(
        self,
        R_desc,
        R_d_desc,
        tril_perms_lin,
        sig,
        lam,
        use_E_cstr,
        n_inducing_pts,
    ):

        n_train, dim_d = R_d_desc.shape[:2]
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)

        # Convert from training points to actual columns.
        # dim_m = (
        #    np.maximum(1, n_inducing_pts // 4) * dim_i
        # )  # only use 1/4 of inducing points for leverage score estimate
        dim_m = dim_i * min(n_inducing_pts, 10)

        # Which columns to use for leverage score approximation?
        lev_approx_idxs = np.sort(
            np.random.choice(
                n_train * dim_i + (n_train if use_E_cstr else 0), dim_m, replace=False
            )
        )  # random subset of columns

        L_inv_K_mn = self._nystroem_cholesky_factor(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            lam,
            use_E_cstr=use_E_cstr,
            col_idxs=lev_approx_idxs,
            task_name="lev. scores",
        )

        lev_scores = np.einsum("i...,i...->...", L_inv_K_mn, L_inv_K_mn)
        return lev_scores

    def inducing_pts_from_lev_scores(self, lev_scores, N):

        # Sample 'N' columns with probabilities proportional to the leverage scores.
        inducing_pts_idxs = np.random.choice(
            np.arange(lev_scores.size),
            N,
            replace=False,
            p=lev_scores / lev_scores.sum(),
        )

        return np.sort(inducing_pts_idxs)

    # performs a cholesky decomposition of a matrix, but regularizes the matrix
    # (if needed) until its positive definite
    # pylint: disable-next=inconsistent-return-statements
    def _cho_factor_stable(self, M, pre_reg=False, eps_mag_max=1):
        """Performs a Cholesky decomposition of a matrix, but regularizes as needed
        until its positive definite.

        Parameters
        ----------
        M : :obj:`numpy.ndarray`
            Matrix to factorize.
        pre_reg : boolean, optional
            Regularize M right away (machine precision), before
            trying to factorize it (default: False).

        Returns
        -------
        :obj:`numpy.ndarray`
            Matrix whose upper or lower triangle contains the Cholesky factor of a.
            Other parts of the matrix contain random data.
        :obj:`bool`
            Flag indicating whether the factor is in the lower or upper triangle
        """

        eps = np.finfo(float).eps
        eps_mag = int(np.floor(np.log10(eps)))

        if pre_reg:
            M[np.diag_indices_from(M)] += eps
            # if additional regularization is necessary, start from the next order of
            # magnitude
            eps_mag += 1

        for reg in 10.0 ** np.arange(
            eps_mag, eps_mag_max + 1
        ):  # regularize more and more aggressively (strongest regularization: 1)
            try:

                L, lower = sp.linalg.cho_factor(
                    M, overwrite_a=False, check_finite=False
                )

            except np.linalg.LinAlgError as e:

                if "not positive definite" in str(e):
                    log.debug(
                        "Cholesky solver needs more aggressive regularization "
                        "(adding %s to diagonal)",
                        reg,
                    )
                    M[np.diag_indices_from(M)] += reg
                else:
                    raise e
            else:
                return L, lower

        log.critical(
            "Failed to factorize despite strong regularization (max: {%r})!",
            10.0**eps_mag_max,
        )
        log.critical("You could try a larger sigma.")
        os._exit(1)  # pylint: disable=protected-access

    # pylint: disable=too-many-statements
    def solve(
        self,
        task,
        R_desc,
        R_d_desc,
        tril_perms_lin,
        y,
        y_std,
        tol=1e-4,
        save_progr_callback=None,
    ):
        r"""Iteratively solve for :math:`\alpha`."""

        global num_iters, start, resid, avg_tt, m  # , P_t

        n_train, n_atoms = task["R_train"].shape[:2]
        dim_i = 3 * n_atoms

        sig = task["sig"]
        lam = task["lam"]

        # these keys are only present if the task was created from an existing model
        alphas0_F = task["alphas0_F"] if "alphas0_F" in task else None
        alphas0_E = task["alphas0_E"] if "alphas0_E" in task else None
        num_iters0 = task["solver_iters"] if "solver_iters" in task else 0

        # Number of inducing points to use for Nystrom approximation.
        max_memory_bytes = self._max_memory * 1024**3
        max_n_inducing_pts = Iterative.max_n_inducing_pts(
            n_train, n_atoms, max_memory_bytes
        )
        n_inducing_pts = min(n_train, max_n_inducing_pts)
        n_inducing_pts_init = (
            len(task["inducing_pts_idxs"]) // (3 * n_atoms)
            if "inducing_pts_idxs" in task
            else None
        )

        log.info(
            "Building preconditioner (k=%d ind. point%s)",
            n_inducing_pts,
            "s" if n_inducing_pts > 1 else "",
        )

        lev_scores = None
        if n_inducing_pts_init is not None and n_inducing_pts_init == n_inducing_pts:
            inducing_pts_idxs = task["inducing_pts_idxs"]  # reuse old inducing points
        else:
            # Determine good inducing points.
            lev_scores = self._lev_scores(
                R_desc,
                R_d_desc,
                tril_perms_lin,
                sig,
                lam,
                task["use_E_cstr"],
                n_inducing_pts,
            )

            dim_m = n_inducing_pts * dim_i
            inducing_pts_idxs = self.inducing_pts_from_lev_scores(lev_scores, dim_m)

        start = timeit.default_timer()
        P_op, lev_scores = self._init_precon_operator(
            task,
            R_desc,
            R_d_desc,
            tril_perms_lin,
            inducing_pts_idxs,
        )
        stop = timeit.default_timer()  # pylint: disable=unused-variable

        n = P_op.shape[0]
        K_op = self._init_kernel_operator(
            task, R_desc, R_d_desc, tril_perms_lin, lam, n
        )

        num_iters = int(num_iters0)

        start = 0
        resid = 0
        avg_tt = 0

        global alpha_t, eff, steps_hist, callback_disp_str

        alpha_t = None
        if alphas0_F is not None:  # TODO: improve me: this will not work with E_cstr
            alpha_t = -alphas0_F  # pylint: disable=invalid-unary-operand-type

        if alphas0_E is not None:
            # pylint: disable-next=invalid-unary-operand-type
            alpha_t = np.hstack((alpha_t, -alphas0_E))

        steps_hist = collections.deque(
            maxlen=CG_STEPS_HIST_LEN
        )  # moving average window for step history

        def _cg_status(xk):

            global num_iters, start, resid, alpha_t, avg_tt, eff, steps_hist
            global callback_disp_str, P_t

            stop = timeit.default_timer()
            tt = 0.0 if start == 0 else (stop - start)
            avg_tt += tt
            start = timeit.default_timer()

            old_resid = resid
            resid = inspect.currentframe().f_back.f_locals["resid"]

            step = 0 if num_iters == num_iters0 else resid - old_resid
            steps_hist.append(step)

            steps_hist_arr = np.array(steps_hist)
            steps_hist_all = np.abs(steps_hist_arr).sum()
            steps_hist_ratio = (
                (-steps_hist_arr.clip(max=0).sum() / steps_hist_all)
                if steps_hist_all > 0
                else 1
            )
            eff = (
                0 if num_iters == num_iters0 else (int(100 * steps_hist_ratio) - 50) * 2
            )

            if tt > 0.0 and num_iters % int(np.ceil(1.0 / tt)) == 0:  # once per second

                train_rmse = resid / np.sqrt(len(y))
                log.debug("Training error (RMSE): forces %r", train_rmse)
                log.debug(
                    "{:d} iter @ {} iter/s [eff: {:d}%], k={:d}",
                    num_iters,
                    "{:.2f}".format(1.0 / tt),
                    eff,
                    n_inducing_pts,
                )

            # Write out current solution as a model file once every 2 minutes
            # (give or take).
            if (
                tt > 0.0
                and num_iters % int(np.ceil(2 * 60.0 / tt)) == 0
                and num_iters % 10 == 0
            ):

                log.debug("Saving model checkpoint")

                # TODO: support for +E constraints (done?)
                alphas_F, alphas_E = -xk, None
                if task["use_E_cstr"]:
                    n_train = task["R_train"].shape[0]
                    alphas_F, alphas_E = -xk[:-n_train], -xk[-n_train:]

                unconv_model = self.gdml_train.create_model(
                    task,
                    "cg",
                    R_desc,
                    R_d_desc,
                    tril_perms_lin,
                    y_std,
                    alphas_F,
                    alphas_E=alphas_E,
                )

                solver_keys = {
                    "solver_tol": tol,
                    "solver_iters": num_iters
                    + 1,  # number of iterations performed (cg solver)
                    "solver_resid": resid,  # residual of solution
                    "norm_y_train": np.linalg.norm(y),
                    "inducing_pts_idxs": inducing_pts_idxs,
                }

                unconv_model.update(solver_keys)

                # recover integration constant
                self.gdml_predict.set_alphas(alphas_F, alphas_E=alphas_E)
                E_pred, _ = self.gdml_predict.predict()

                E_pred *= y_std

                unconv_model["c"] = 0
                if "E_train" in task:
                    E_ref = np.squeeze(task["E_train"])
                    unconv_model["c"] = np.mean(E_ref - E_pred)

                if save_progr_callback is not None:
                    save_progr_callback(unconv_model)

            num_iters += 1

            n_train = task["idxs_train"].shape[0]
            if (
                len(steps_hist) == CG_STEPS_HIST_LEN
                and eff <= EFF_RESTART_THRESH
                and n_inducing_pts < n_train
            ):
                alpha_t = xk
                raise CGRestartException

        num_restarts = 0
        while True:
            try:
                # allow 10x as many iterations as theoretically needed
                # (at perfect precision)
                alphas, info = sp.sparse.linalg.cg(
                    -K_op,
                    y,
                    x0=alpha_t,
                    M=P_op,
                    tol=tol,  # norm(residual) <= max(tol*norm(b), atol)
                    atol=None,
                    maxiter=3 * n_atoms * n_train * 10,
                    callback=_cg_status,
                )
                alphas = -alphas

            except CGRestartException:

                num_restarts += 1
                steps_hist.clear()

                if num_restarts == MAX_NUM_RESTARTS:
                    info = 1  # convergence to tolerance not achieved
                    alphas = alpha_t
                    break

                num_restarts_left = MAX_NUM_RESTARTS - num_restarts - 1
                log.debug(
                    "Restarts left before giving up: {}{}.".format(
                        num_restarts_left,
                        " (final trial)" if num_restarts_left == 0 else "",
                    )
                )

                # TODO: keep using same number of points

                n_inducing_pts = min(
                    int(np.ceil(1.2 * n_inducing_pts)), n_train
                )  # increase in increments (ignoring memory limits...)

                dim_m = n_inducing_pts * dim_i
                inducing_pts_idxs = self.inducing_pts_from_lev_scores(lev_scores, dim_m)

                del P_op
                P_op, lev_scores = self._init_precon_operator(
                    task,
                    R_desc,
                    R_d_desc,
                    tril_perms_lin,
                    inducing_pts_idxs,
                )

            else:
                break

        is_conv = info == 0

        is_conv_warn_str = "" if is_conv else " (NOT CONVERGED)"
        log.debug("Training on {:,} points{}".format(n_train, is_conv_warn_str))

        train_rmse = resid / np.sqrt(len(y))

        return alphas, tol, num_iters, resid, train_rmse, inducing_pts_idxs, is_conv

    @staticmethod
    def max_n_inducing_pts(n_train, n_atoms, max_memory_bytes):

        SQUARE_FACT = 5
        LINEAR_FACT = 4

        to_bytes = 8
        to_dof = (3 * n_atoms) ** 2 * to_bytes

        sq_factor = LINEAR_FACT * n_train * to_dof
        ny_factor = SQUARE_FACT * to_dof

        n_inducing_pts = (
            np.sqrt(sq_factor**2 + 4.0 * ny_factor * max_memory_bytes) - sq_factor
        ) / (2 * ny_factor)
        n_inducing_pts = int(n_inducing_pts)

        return min(n_inducing_pts, n_train)

    @staticmethod
    def est_memory_requirement(n_train, n_inducing_pts, n_atoms):

        SQUARE_FACT = 5
        LINEAR_FACT = 4

        est_bytes = LINEAR_FACT * n_train * n_inducing_pts * (3 * n_atoms) ** 2 * 8

        est_bytes += (
            SQUARE_FACT * n_inducing_pts * n_inducing_pts * (3 * n_atoms) ** 2 * 8
        )

        return est_bytes
