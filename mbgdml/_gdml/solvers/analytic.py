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

import sys
import warnings
import numpy as np
import scipy as sp

from ...logger import GDMLLogger

log = GDMLLogger(__name__)


class Analytic:
    def __init__(self, gdml_train, desc):
        r"""The sGDML :class:`sgdml.solvers.analytic.Analytic` class."""
        self.gdml_train = gdml_train
        self.desc = desc

    def solve(self, task, R_desc, R_d_desc, tril_perms_lin, y):
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
        y : :obj:`numpy.ndarray`, ndim: ``1``
            The train labels computed in
            :meth:`~mbgdml._gdml.train.GDMLTrain.train_labels`.
        """
        # pylint: disable=invalid-name
        log.info(
            "\n-------------------------\n"
            "|   Analytical solver   |\n"
            "-------------------------\n"
        )

        sig = task["sig"]
        lam = task["lam"]
        use_E_cstr = task["use_E_cstr"]
        log.log_model(task)

        n_train, dim_d = R_d_desc.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        dim_i = 3 * n_atoms

        # Compress kernel based on symmetries
        col_idxs = np.s_[:]
        if "cprsn_keep_atoms_idxs" in task:

            cprsn_keep_idxs = task["cprsn_keep_atoms_idxs"]
            cprsn_keep_idxs_lin = (
                np.arange(dim_i).reshape(n_atoms, -1)[cprsn_keep_idxs, :].ravel()
            )

            col_idxs = (
                cprsn_keep_idxs_lin[:, None] + np.arange(n_train) * dim_i
            ).T.ravel()

        log.info("\nAssembling kernel matrix")
        t_assemble = log.t_start()
        K = self.gdml_train._assemble_kernel_mat(  # pylint: disable=protected-access
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            self.desc,
            use_E_cstr=use_E_cstr,
            col_idxs=col_idxs,
        )
        log.t_stop(t_assemble)

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"Ill-conditioned matrix")

            if K.shape[0] == K.shape[1]:

                K[np.diag_indices_from(K)] -= lam  # regularize

                try:
                    t_cholesky = log.t_start()
                    log.info("Solving linear system (Cholesky factorization)")
                    # Cholesky
                    L, lower = sp.linalg.cho_factor(
                        -K, overwrite_a=True, check_finite=False
                    )
                    alphas = -sp.linalg.cho_solve(
                        (L, lower), y, overwrite_b=True, check_finite=False
                    )

                    log.t_stop(t_cholesky, message="Done in {time} s")
                except np.linalg.LinAlgError:
                    # try a solver that makes less assumptions
                    log.t_stop(
                        t_cholesky, message="Cholesky factorization failed in {time} s"
                    )
                    log.info("Solving linear system (LU factorization)")

                    try:
                        # LU
                        t_lu = log.t_start()
                        alphas = sp.linalg.solve(
                            K, y, overwrite_a=True, overwrite_b=True, check_finite=False
                        )
                        log.t_stop(t_lu, message="Done in {time} s")
                    except MemoryError:
                        log.t_stop(
                            t_lu,
                            message="LU factorization failed in {time} s",
                            level=50,
                        )
                        log.critical(
                            "Not enough memory to train this system using a closed "
                            "form solver.\nPlease reduce the size of the training set "
                            "or consider one of the approximate solver options."
                        )
                        sys.exit()

                except MemoryError:
                    log.critical(
                        "Not enough memory to train this system using a closed "
                        "form solver.\nPlease reduce the size of the training set "
                        "or consider one of the approximate solver options."
                    )
                    sys.exit()
            else:
                log.info(
                    "Solving overdetermined linear system (least squares approximation)"
                )
                t_least_squares = log.t_start()
                # least squares for non-square K
                alphas = np.linalg.lstsq(K, y, rcond=-1)[0]
                log.t_stop(t_least_squares)

        return alphas

    @staticmethod
    def est_memory_requirement(n_train, n_atoms):
        est_bytes = 3 * (n_train * 3 * n_atoms) ** 2 * 8  # K + factor(s) of K
        est_bytes += (n_train * 3 * n_atoms) * 8  # alpha
        return est_bytes
