# MIT License
#
# Copyright (c) 2018-2020, Stefan Chmiela
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

import os
import shutil
import numpy as np
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer

from .models import gdmlModel
from .analysis.problematic import ProblematicStructures
from .predictors import predict_gdml
from ._gdml.train import GDMLTrain, model_errors, add_valid_errors
from ._gdml.train import save_model, get_test_idxs
from .utils import save_json
from .losses import loss_f_rmse, mae, rmse
from .logger import GDMLLogger

log = GDMLLogger(__name__)


class mbGDMLTrain:
    r"""Train many-body GDML models."""

    def __init__(
        self,
        entity_ids,
        comp_ids,
        use_sym=True,
        use_E=True,
        use_E_cstr=False,
        use_cprsn=False,
        solver="analytic",
        lam=1e-10,
        solver_tol=1e-4,
        use_torch=False,
        max_processes=None,
    ):
        """
        Parameters
        ----------
        entity_ids : :obj:`numpy.ndarray`
            Model ``entity_ids``.
        comp_ids : :obj:`numpy.ndarray`
            Model ``comp_ids``.
        use_sym : :obj:`bool`, default: ``True``
            If to identify and include symmetries when training GDML models.
            This usually increases training and prediction times, but comes
            with accuracy improvements.
        use_E : :obj:`bool`, default: ``True``
            Whether or not to reconstruct the potential energy surface
            (``True``) with or (``False``) without energy labels. It is
            highly recommended to train with energies.
        use_E_cstr : :obj:`bool`, default: ``False``
            Whether or not to include energies as a part of the model training.
            Meaning ``True`` will add another column of alphas that will be
            trained to the energies. This is rarely
            useful for higher order *n*-body models.
        use_cprsn : :obj:`bool`, default: ``False``
            Compresses the kernel matrix along symmetric degrees of freedom to
            try to reduce training time. Usually does not provide significant
            benefits.
        solver : :obj:`str`, default: ``'analytic'``
            The GDML solver to use, either ``analytic`` or ``iterative``.
        lam : :obj:`float`, default: ``1e-10``
            Hyper-parameter lambda (regularization strength). This generally
            does not need to change.
        solver_tol : :obj:`float`, default: ``1e-4``
           Solver tolerance.
        use_torch : :obj:`bool`, default: ``False``
            Use PyTorch to enable GPU acceleration.
        max_processes : :obj:`int`, default: :obj:`None`
            The maximum number of cores to use for the training process. Will
            automatically calculate if not specified.
        """
        self.entity_ids = entity_ids
        self.comp_ids = comp_ids

        self.use_sym = use_sym
        self.use_E = use_E
        self.use_E_cstr = use_E_cstr
        self.use_cprsn = use_cprsn
        self.solver = solver
        self.lam = lam
        self.solver_tol = solver_tol
        self.interact_cut_off = None
        self.use_torch = use_torch
        self.max_processes = max_processes

        self.GDMLTrain = GDMLTrain(max_processes=max_processes, use_torch=use_torch)
        self.sigma_grid = [
            2,
            25,
            50,
            100,
            200,
            300,
            400,
            500,
            700,
            900,
            1100,
            1500,
            2000,
            2500,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
            20000,
            50000,
            100000,
        ]
        r"""Determining reasonable ``sigma_bounds`` is difficult without some
        prior experience with the system. Even then, the optimal ``sigma``
        can drastically change depending on the training set size.

        ``sigma_grid`` will assist with determining optimal
        ``sigma_bounds`` by first performing a course grid search. The
        Bayesian optimization will start with the bounds of the grid-search
        minimum. It is recommended to choose a large ``sigma_bounds`` as
        large as your ``sigma_grid``; it will be updated internally.

        The number of probes done during the initial grid search will be
        subtracted from the Bayesian optimization ``init_points``.

        We recommend that the grid includes several lower sigmas (< 50), a
        few medium sigmas (< 500), and several large sigmas that span up to
        at least 1000 for higher-order models.

        **Default**

        .. code-block:: python

            [
                2, 25, 50, 100, 200, 300, 400, 500, 700, 900, 1100, 1500, 2000,
                2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000,
                50000, 100000
            ]

        :type: :obj:`list`
        """
        self.bayes_opt_params = {
            "init_points": 10,
            "n_iter": 10,
            "alpha": 1e-7,
            "acq": "ucb",
            "kappa": 1.5,
        }
        r"""Bayesian optimization parameters.

        **Default**

        .. code-block:: python

            {
                'init_points': 10, 'n_iter': 10, 'alpha': 1e-7, 'acq': 'ucb',
                'kappa': 1.5
            }


        :type: :obj:`dict`
        """
        self.bayes_opt_params_final = None
        r"""Bayesian optimization parameters for the final model.

        If :obj:`None`, then ``bayes_opt_params`` are used.

        Default: :obj:`None`

        :type: :obj:`dict`
        """
        self.sigma_bounds = (2, 400)
        r"""Kernel length scale bounds for the Bayesian optimization.

        This is only used if ``sigma_grid`` if :obj:`None`.

        Default: ``(2, 400)``

        :type: :obj:`tuple`
        """
        self.check_energy_pred = True
        r"""Will return the model with the lowest loss that predicts reasonable
        energies. If ``False``, the model with the lowest loss is not
        checked for reasonable energy predictions.

        Sometimes, GDML kernels are unable to accurately reconstruct potential
        energies even if the force predictions are accurate. This is
        sometimes prevalent in many-body models with low and high sigmas
        (i.e., sigmas less than 5 or greater than 500).

        Default: ``True``

        :type: :obj:`bool`
        """
        self.loss_func = loss_f_rmse
        r"""Loss function for validation. The input of this function is the
        dictionary of :obj:`mbgdml._gdml.train.add_valid_errors` which
        contains ``force`` and ``energy`` errors.

        Default: :obj:`mbgdml.losses.loss_f_rmse`

        :type: ``callable``
        """
        self.loss_kwargs = {}
        r"""Loss function keyword arguments with the exception of the validation
        ``results`` dictionary.

        :type: :obj:`dict`
        """
        self.require_E_eval = True  # pylint: disable=invalid-name
        r"""Require energy evaluation regardless even if they are terrible.

        If ``False``, it defaults to sGDML behavior (this does not work well
        with n-body training).

        Default: ``True``

        :type: :obj:`bool`
        """
        self.keep_tasks = False
        r"""Keep all models trained during the train task if ``True``. They are
        removed by default.

        :type: :obj:`bool`
        """
        self.bayes_opt_n_check_rising = 4
        r"""Number of additional ``sigma_grid`` probes to check if loss
        continues to rise after finding a minima.

        We often perform a grid search prior to Bayesian optimization.
        Sometimes, with :math:`n`-body training, the loss will start rising
        but then fall again to a lower value. Thus, we do some extra (larger)
        sigmas to check if the loss will fall again. If it does, then we
        restart the grid search.

        :type: :obj:`int`
        """

    def min_memory_analytic(self, n_train, n_atoms):
        r"""Minimum memory recommendation for training analytically.

        GDML currently only supports closed form solutions (i.e., analytically).
        Thus, the entire kernel matrix must be in memory which requires
        :math:`(M * 3N)^2` double precision (8 byte) entries. This provides
        a rough estimate for memory requirements.

        Parameters
        ----------
        n_train : :obj:`int`
            Number of training structures.
        n_atoms : :obj:`int`
            Number of atoms in a single structure.

        Returns
        -------
        :obj:`float`
            Minimum memory requirements in MB.
        """
        return (8 * ((n_train * 3 * n_atoms) ** 2)) * 1e-6

    def __del__(self):
        # pylint: disable=undefined-variable, global-variable-undefined
        global glob

        if "glob" in globals():
            del glob

    def create_task(
        self,
        train_dataset,
        n_train,
        valid_dataset,
        n_valid,
        sigma,
        train_idxs=None,
        valid_idxs=None,
    ):
        r"""Create a single training task that can be used as a template for
        hyperparameter searches.

        Parameters
        ----------
        train_dataset : :obj:`mbgdml.data.DataSet`
            Dataset for training a model on.
        n_train : :obj:`int`
            The number of training points to sample.
        valid_dataset : :obj:`mbgdml.data.DataSet`
            Dataset for validating a model on.
        n_valid : :obj:`int`
            The number of validation points to sample, without replacement.
        sigma : :obj:`float` or :obj:`int`
            Kernel length scale of the desired model.
        train_idxs : :obj:`numpy.ndarray`, default: :obj:`None`
            The specific indices of structures to train the model on. If
            :obj:`None` will automatically sample the training data set.
        valid_idxs : :obj:`numpy.ndarray`, default: :obj:`None`
            The specific indices of structures to validate models on.
            If :obj:`None`, structures will be automatically determined.
        """
        self.task = self.GDMLTrain.create_task(
            train_dataset,
            n_train,
            valid_dataset,
            n_valid,
            sigma,
            lam=self.lam,
            use_sym=self.use_sym,
            use_E=self.use_E,
            use_E_cstr=self.use_E_cstr,
            use_cprsn=self.use_cprsn,
            solver=self.solver,
            solver_tol=self.solver_tol,
            idxs_train=train_idxs,
            idxs_valid=valid_idxs,
        )
        return self.task

    def train_model(self, task):
        r"""Trains a GDML model from a task.

        Parameters
        ----------
        task : :obj:`dict`
            Training task.
        n_train : :obj:`int`
            The number of training points to sample.

        Returns
        -------
        :obj:`dict`
            Trained (not validated or tested) model.
        """
        model = self.GDMLTrain.train(task, require_E_eval=self.require_E_eval)
        return model

    def test_model(self, model, dataset, n_test=None):
        r"""Test model and add mbGDML modifications.

        Parameters
        ----------
        model : :obj:`mbgdml.models.gdmlModel`
            Model to test.
        dataset : :obj:`dict`
            Test dataset.

        Returns
        -------
        :obj:`mbgdml.models.gdmlModel`
            Tested and finalized many-body GDML model.
        """
        n_test, E_errors, F_errors = model_errors(
            model.model_dict,
            dataset,
            is_valid=False,
            n_test=n_test,
            max_processes=self.max_processes,
            use_torch=self.use_torch,
        )
        model.model_dict["n_test"] = np.array(n_test)

        results_test = {"force": {"mae": mae(F_errors), "rmse": rmse(F_errors)}}
        if E_errors is not None:
            results_test["energy"] = {"mae": mae(E_errors), "rmse": rmse(E_errors)}

        # Adding mbGDML-specific modifications to model.
        model.add_modifications(dataset)
        if "mb" in dataset.keys():
            model.model_dict["mb"] = int(dataset["mb"][()])
        if "mb_models_md5" in dataset.keys():
            model.model_dict["mb_models_md5"] = dataset["mb_models_md5"]

        return model, results_test

    def save_valid_csv(self, save_dir, valid_json):
        r"""Writes a CSV summary file for validation statistics.

        This is just easier to see the trend of sigma and validation error.

        Parameters
        ----------
        save_dir : :obj:`str`
            Where to save the CSV file.
        valid_json : :obj:`dict`
            The validation json curated during the training routine.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel

        sigmas = np.array(valid_json["sigmas"])
        sigma_argsort = np.argsort(sigmas)

        sigmas = sigmas[sigma_argsort]
        losses = np.array(valid_json["losses"])[sigma_argsort]
        e_mae = np.array(valid_json["energy"]["mae"])[sigma_argsort]
        e_rmse = np.array(valid_json["energy"]["rmse"])[sigma_argsort]
        f_mae = np.array(valid_json["force"]["mae"])[sigma_argsort]
        f_rmse = np.array(valid_json["force"]["rmse"])[sigma_argsort]

        df = pd.DataFrame(
            {
                "sigma": sigmas,
                "losses": losses,
                "e_mae": e_mae,
                "e_rmse": e_rmse,
                "f_mae": f_mae,
                "f_rmse": f_rmse,
            }
        )
        df.to_csv(os.path.join(save_dir, "valid.csv"), index=False)

    def save_idxs(self, model, dataset, save_dir, n_test):
        r"""Saves npy files of the dataset splits (training, validation, and
        test).

        Parameters
        ----------
        model : :obj:`mbgdml.models.gdmlModel`
            Many-body GDML model.
        dataset : :obj:`dict`
            Dataset used for training, validation, and testing.
        """
        log.info("\n#   Writing train, valid, and test indices   #")
        train_idxs = model.model_dict["idxs_train"]
        valid_idxs = model.model_dict["idxs_valid"]

        np.save(os.path.join(save_dir, "train_idxs"), train_idxs)
        np.save(os.path.join(save_dir, "valid_idxs"), valid_idxs)

        test_idxs = get_test_idxs(model.model_dict, dataset, n_test=n_test)
        np.save(os.path.join(save_dir, "test_idxs"), test_idxs)

    def plot_bayes_opt_gp(self, optimizer):
        r"""Prepare a plot of the Bayesian optimization Gaussian process.

        Parameters
        ----------
        optimizer : ``bayes_opt.BayesianOptimization``

        Returns
        -------
        ``object``
            A matplotlib figure object.
        """
        # pylint: disable=protected-access, import-outside-toplevel
        import matplotlib.pyplot as plt

        params = optimizer.space.params.flatten()
        losses = -optimizer.space.target
        sigma_bounds = optimizer._space.bounds[0]
        lower_bound = min(min(params), sigma_bounds[0])
        upper_bound = max(max(params), sigma_bounds[1])

        x = np.linspace(lower_bound, upper_bound, 10000)
        # pylint: disable-next=invalid-name
        mu, sigma = optimizer._gp.predict(x.reshape(-1, 1), return_std=True)

        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(5.5, 4), constrained_layout=True
        )

        ax.plot(x, -mu, label="Prediction")
        ax.fill_between(
            x,
            -mu - 1.9600 * sigma,
            -mu + 1.9600 * sigma,
            alpha=0.1,
            label=r"95% confidence",
        )
        ax.scatter(params, losses, c="red", s=8, zorder=10, label="Observations")

        ax.set_xlabel("Sigma")
        ax.set_ylabel(self.loss_func.__name__)

        ax.set_xlim(left=lower_bound, right=upper_bound)

        plt.legend()

        return fig

    # pylint: disable-next=too-many-branches, too-many-statements
    def bayes_opt(
        self,
        dataset,
        model_name,
        n_train,
        n_valid,
        n_test=None,
        save_dir=".",
        is_final=False,
        use_domain_opt=False,
        plot_bo=True,
        train_idxs=None,
        valid_idxs=None,
        overwrite=False,
        write_json=True,
        write_idxs=True,
    ):
        r"""Train a GDML model using Bayesian optimization for sigma.

        Uses the `Bayesian optimization
        <https://github.com/fmfn/BayesianOptimization>`__
        package to automatically find the optimal sigma. This will maximize
        the negative validation loss.

        ``gp_params`` can be used to specify options to
        ``BayesianOptimization.maximize()`` method.

        A sequential domain reduction optimizer is used to accelerate the
        convergence to an optimal sigma (when requested).

        Parameters
        ----------
        dataset : :obj:`mbgdml.data.DataSet`
            Dataset to train, validate, and test a model on.
        model_name : :obj:`str`
            User-defined model name without the ``'.npz'`` file extension.
        n_train : :obj:`int`
            The number of training points to use.
        n_valid : :obj:`int`
            The number of validation points to use.
        n_test : :obj:`int`, default: :obj:`None`
            The number of test points to test the validated GDML model.
            Defaults to testing all available structures.
        save_dir : :obj:`str`, default: ``'.'``
            Path to train and save the mbGDML model. Defaults to current
            directory.
        is_final : :obj:`bool`
            If we use ``bayes_opt_params_final`` or not.
        use_domain_opt : :obj:`bool`, default: ``False``
            Whether to use a sequential reduction optimizer or not. This
            sometimes crashes.
        plot_bo : :obj:`bool`, default: ``True``
            Plot the Bayesian optimization Gaussian process.
        train_idxs : :obj:`numpy.ndarray`, default: :obj:`None`
            The specific indices of structures to train the model on. If
            :obj:`None` will automatically sample the training data set.
        valid_idxs : :obj:`numpy.ndarray`, default: :obj:`None`
            The specific indices of structures to validate models on.
            If :obj:`None`, structures will be automatically determined.
        overwrite : :obj:`bool`, default: ``False``
            Overwrite existing files.
        write_json : :obj:`bool`, default: ``True``
            Write a JSON file containing information about the training job.
        write_idxs : :obj:`bool`, default: ``True``
            Write npy files for training, validation, and test indices.

        Returns
        -------
        :obj:`dict`
            Optimal many-body GDML model.
        ``bayes_opt.BayesianOptimization``
            The Bayesian optimizer object.
        """
        t_job = log.t_start()

        log.info(
            "#############################\n"
            "#         Training          #\n"
            "#   Bayesian Optimization   #\n"
            "#############################\n"
        )

        if write_json:
            job_json = {
                "model": {},
                "testing": {},
                "validation": {},
                "training": {"idxs": []},
            }

        if train_idxs is not None:
            assert n_train == len(train_idxs)
            assert len(set(train_idxs)) == len(train_idxs)
        if valid_idxs is not None:
            assert n_valid == len(valid_idxs)
            assert len(set(valid_idxs)) == len(valid_idxs)

        dset_dict = dataset.asdict(gdml_keys=True)

        task_dir = os.path.join(save_dir, "tasks")
        if os.path.exists(task_dir):
            if overwrite:
                shutil.rmtree(task_dir)
            else:
                raise FileExistsError("Task directory already exists")
        os.makedirs(task_dir, exist_ok=overwrite)

        task = self.create_task(
            dset_dict,
            n_train,
            dset_dict,
            n_valid,
            None,  # Do not specify sigma; should not have any effect.
            train_idxs=train_idxs,
            valid_idxs=valid_idxs,
        )

        valid_json = {
            "sigmas": [],
            "losses": [],
            "force": {"mae": [], "rmse": []},
            "energy": {"mae": [], "rmse": []},
            "idxs": [],
        }

        def opt_func(sigma):
            r"""Computes the (negative) validation loss that Bayesian
            optimization tries to optimize.

            Also updates ``valid_json``.

            Parameters
            ----------
            sigma : :obj:`float`
                Kernel length scale for a validation run.

            Returns
            -------
            :obj:`float`
                Negative loss.
            """
            task["sig"] = sigma
            model_trial = self.train_model(task)

            if len(valid_json["idxs"]) == 0:
                valid_json["idxs"] = model_trial["idxs_valid"].tolist()

            valid_results, model_trial = add_valid_errors(
                model_trial,
                dset_dict,
                overwrite=True,
                max_processes=self.max_processes,
                use_torch=self.use_torch,
            )

            # pylint: disable-next=invalid-name
            l = self.loss_func(valid_results, **self.loss_kwargs)
            valid_json["losses"].append(l)

            valid_json["sigmas"].append(float(model_trial["sig"]))
            valid_json["energy"]["mae"].append(mae(valid_results["energy"]))
            valid_json["energy"]["rmse"].append(rmse(valid_results["energy"]))
            valid_json["force"]["mae"].append(mae(valid_results["force"]))
            valid_json["force"]["rmse"].append(rmse(valid_results["force"]))

            model_trail_path = os.path.join(task_dir, f"model-{float(sigma)}.npz")

            save_model(model_trial, model_trail_path)

            return -l

        sigma_bounds = self.sigma_bounds
        opt_kwargs = {"f": opt_func, "pbounds": {"sigma": sigma_bounds}, "verbose": 0}
        if use_domain_opt:
            bounds_transformer = SequentialDomainReductionTransformer()
            opt_kwargs["bounds_transformer"] = bounds_transformer
        optimizer = BayesianOptimization(**opt_kwargs)

        # Use optimizer.probe to run grid search and to update sigma_bounds.
        # If no minimum is found, the provided sigma_bounds are used.
        # For every probe we will reduce the number of initial points in the
        # Bayesian optimization.
        if is_final:
            if self.bayes_opt_params_final is not None:
                gp_params = self.bayes_opt_params_final
            else:
                gp_params = self.bayes_opt_params
        else:
            gp_params = self.bayes_opt_params
        if gp_params is not None and "init_points" in gp_params:
            init_points = gp_params["init_points"]
        else:
            init_points = 5  # BayesianOptimization default.
        sigma_grid = self.sigma_grid
        if sigma_grid is not None:
            log.info("#   Initial grid search   #")
            sigma_grid.sort()

            def probe_sigma(sigma):
                optimizer.probe({"sigma": sigma}, lazy=False)

            def constant_loss_rising(valid_json, min_idxs_orig):
                r"""Checks for sustained rising loss.

                Parameters
                ----------
                valid_json : :obj:`dict`
                    Data from validation calculations.
                min_idxs_orig : :obj:`int`
                    Index of the minimum.

                Returns
                -------
                :obj:`bool`
                    If the loss has continued to rise after identifying a
                    minimum.
                """
                if self.check_energy_pred:
                    if valid_json["energy"]["rmse"][min_idxs_orig] is None:
                        # Restart to find a better minimum that predicts energies.
                        return False

                losses = valid_json["losses"]
                # Check if losses start falling after minimum.
                for i in range(min_idxs_orig + 1, len(losses)):
                    if losses[i] < losses[i - 1]:
                        # We will restart the grid search to find another minimum.
                        return False

                return True

            loss_rising = False
            do_extra = (
                self.bayes_opt_n_check_rising
            )  # Extra sigmas to check after losses rise.
            for i, sigma in enumerate(sigma_grid):
                probe_sigma(sigma)
                init_points -= 1

                # Check to see if losses have started to rise.
                # Only check if loss_rising is False to avoid changing back
                # to False if the losses start falling (for do_extra).
                if len(valid_json["sigmas"]) >= 2 and not loss_rising:
                    if valid_json["losses"][-1] > valid_json["losses"][-2]:
                        loss_rising = True
                        min_idxs_orig = len(valid_json["sigmas"]) - 2

                if loss_rising:
                    # We do extra sigmas to ensure we did not have premature
                    # rising loss.
                    if i != len(sigma_grid) - 1 and do_extra > 0:
                        do_extra -= 1
                        continue

                    # Once do_extra is complete we check losses.
                    restart_grid_search = constant_loss_rising(
                        valid_json, min_idxs_orig
                    )

                    if not restart_grid_search:
                        # Losses started lowering again.
                        # Or we checked for energy predictions and it failed.
                        # Restart the grid search.
                        loss_rising = False
                        do_extra = self.bayes_opt_n_check_rising
                    else:
                        # Loss has continued to rise; good chance we found the minimum.

                        # Update the sigma upper bound to the largest sigma
                        # we have already checked.
                        sigma_bounds = (sigma_bounds[0], np.max(valid_json["sigmas"]))
                        optimizer.set_bounds({"sigma": sigma_bounds})

                        # We stop grid search and start Bayesian optimization.
                        break

        if init_points < 0:
            init_points = 0
            gp_params["init_points"] = init_points

        log.info("#   Bayesian optimization   #")
        optimizer.maximize(**gp_params)

        results = optimizer.res
        losses = np.array([-res["target"] for res in results])
        min_idxs = np.argsort(losses)

        for idx in min_idxs:
            sigma_best = results[idx]["params"]["sigma"]
            if self.check_energy_pred:
                e_rmse = valid_json["energy"]["rmse"][idx]
                if e_rmse is None:
                    # Go to the next lowest loss.
                    continue
                # Found the lowest loss.
                break
            break

        model_best_path = os.path.join(task_dir, f"model-{sigma_best}.npz")
        model_best = gdmlModel(model_best_path, comp_ids=self.comp_ids)

        # Testing model
        model_best, results_test = self.test_model(model_best, dset_dict, n_test=n_test)

        # Including final JSON stuff and writing.
        if write_json:
            job_json["training"]["idxs"] = model_best.model_dict["idxs_train"].tolist()
            job_json["testing"]["n_test"] = int(model_best.model_dict["n_test"][()])
            job_json["testing"]["energy"] = results_test["energy"]
            job_json["testing"]["force"] = results_test["force"]

            job_json["model"]["sigma"] = float(model_best.model_dict["sig"][()])
            job_json["model"]["n_symm"] = len(model_best.model_dict["perms"])
            job_json["validation"] = valid_json

            save_json(os.path.join(save_dir, "training.json"), job_json)

            self.save_valid_csv(save_dir, valid_json)

        if write_idxs:
            self.save_idxs(model_best, dset_dict, save_dir, n_test)

        if not self.keep_tasks:
            shutil.rmtree(task_dir)

        # Saving model.
        save_path = os.path.join(save_dir, model_name)
        model_best.save(save_path)

        # Bayesian optimization plot
        if plot_bo:
            fig = self.plot_bayes_opt_gp(optimizer)
            fig.savefig(os.path.join(save_dir, "bayes_opt.png"), dpi=600)

        log.t_stop(t_job, message="\nJob duration : {time} s", precision=2)
        return model_best.model_dict, optimizer

    # pylint: disable-next=too-many-branches, too-many-statements
    def grid_search(
        self,
        dataset,
        model_name,
        n_train,
        n_valid,
        n_test=None,
        save_dir=".",
        train_idxs=None,
        valid_idxs=None,
        overwrite=False,
        write_json=True,
        write_idxs=True,
    ):
        r"""Train a GDML model using a grid search for sigma.

        Usually, the validation errors will decrease until an optimal sigma is
        found then start to increase (overfitting). We sort ``sigmas`` from
        lowest to highest and stop the search once the loss function starts
        increasing.

        Parameters
        ----------
        dataset : :obj:`mbgdml.data.DataSet`
            Dataset to train, validate, and test a model on.
        model_name : :obj:`str`
            User-defined model name without the ``'.npz'`` file extension.
        n_train : :obj:`int`
            The number of training points to use.
        n_valid : :obj:`int`
            The number of validation points to use.
        n_test : :obj:`int`, default: :obj:`None`
            The number of test points to test the validated GDML model.
            Defaults to testing all available structures.
        save_dir : :obj:`str`, default: ``'.'``
            Path to train and save the mbGDML model. Defaults to current
            directory.
        train_idxs : :obj:`numpy.ndarray`, default: :obj:`None`
            The specific indices of structures to train the model on. If
            :obj:`None` will automatically sample the training data set.
        valid_idxs : :obj:`numpy.ndarray`, default: :obj:`None`
            The specific indices of structures to validate models on.
            If :obj:`None`, structures will be automatically determined.
        overwrite : :obj:`bool`, default: ``False``
            Overwrite existing files.
        write_json : :obj:`bool`, default: ``True``
            Write a JSON file containing information about the training job.
        write_idxs : :obj:`bool`, default: ``True``
            Write npy files for training, validation, and test indices.

        Returns
        -------
        :obj:`dict`
            GDML model with an optimal hyperparameter found via grid search.
        """
        t_job = log.t_start()
        log.info(
            "###################\n"
            "#    Training     #\n"
            "#   Grid Search   #\n"
            "###################\n"
        )
        if write_json:
            write_json = True
            job_json = {
                "model": {},
                "testing": {},
                "validation": {},
                "training": {"idxs": []},
            }
        else:
            write_json = False

        if train_idxs is not None:
            assert n_train == len(train_idxs)
            assert len(set(train_idxs)) == len(train_idxs)
        if valid_idxs is not None:
            assert n_valid == len(valid_idxs)
            assert len(set(valid_idxs)) == len(valid_idxs)

        dset_dict = dataset.asdict(gdml_keys=True)
        sigmas = self.sigma_grid
        sigmas.sort()

        task_dir = os.path.join(save_dir, "tasks")
        if os.path.exists(task_dir):
            if overwrite:
                shutil.rmtree(task_dir)
            else:
                raise FileExistsError("Task directory already exists")
        os.makedirs(task_dir, exist_ok=overwrite)

        task = self.create_task(
            dset_dict,
            n_train,
            dset_dict,
            n_valid,
            sigmas[0],
            train_idxs=train_idxs,
            valid_idxs=valid_idxs,
        )

        # Starting grid search
        trial_model_paths = []
        valid_json = {
            "sigmas": [],
            "losses": [],
            "force": {"mae": [], "rmse": []},
            "energy": {"mae": [], "rmse": []},
            "idxs": [],
        }
        model_best_path = None
        for next_sigma in sigmas[1:]:
            model_trial = self.train_model(task)

            if len(valid_json["idxs"]) == 0:
                valid_json["idxs"] = model_trial["idxs_valid"].tolist()

            valid_results, model_trial = add_valid_errors(
                model_trial,
                dset_dict,
                overwrite=True,
                max_processes=self.max_processes,
                use_torch=self.use_torch,
            )

            valid_json["losses"].append(
                self.loss_func(valid_results, **self.loss_kwargs)
            )
            valid_json["sigmas"].append(model_trial["sig"])
            valid_json["energy"]["mae"].append(mae(valid_results["energy"]))
            valid_json["energy"]["rmse"].append(rmse(valid_results["energy"]))
            valid_json["force"]["mae"].append(mae(valid_results["force"]))
            valid_json["force"]["rmse"].append(rmse(valid_results["force"]))

            model_trail_path = os.path.join(
                task_dir, f'model-trial-sig{model_trial["sig"]}.npz'
            )
            save_model(model_trial, model_trail_path)
            trial_model_paths.append(model_trail_path)

            if len(valid_json["losses"]) > 1:
                if valid_json["losses"][-1] > valid_json["losses"][-2]:
                    log.info("\nValidation errors are rising")
                    log.info("Terminating grid search")
                    model_best_path = trial_model_paths[-2]
                    on_grid_bounds = False
                    break

            task["sig"] = next_sigma

        # Determine best model and checking optimal sigma.
        if model_best_path is None:
            model_best_path = trial_model_paths[-1]
            on_grid_bounds = True
            next_search_sign = ">"

        model_best = gdmlModel(model_best_path, comp_ids=self.comp_ids)
        sigma_best = model_best.model_dict["sig"].item()
        if sigma_best == sigmas[0]:
            on_grid_bounds = True
            next_search_sign = "<"

        if on_grid_bounds:
            log.warning("Optimal sigma is on the bounds of grid search")
            log.warning("This model is not optimal")
            log.warning(
                "Extend your grid search to be %r %r", next_search_sign, sigma_best
            )

        # Testing model
        model_best, results_test = self.test_model(model_best, dset_dict, n_test=n_test)

        # Including final JSON stuff and writing.
        if write_json:
            job_json["training"]["idxs"] = model_best.model_dict["idxs_train"].tolist()
            job_json["testing"]["n_test"] = int(model_best.model_dict["n_test"][()])
            job_json["testing"]["energy"] = results_test["energy"]
            job_json["testing"]["force"] = results_test["force"]

            job_json["model"]["sigma"] = model_best.model_dict["sig"][()]
            job_json["model"]["n_symm"] = len(model_best.model_dict["perms"])
            # TODO: Can we sort validation data to make it easier to follow?
            job_json["validation"] = valid_json

            save_json(os.path.join(save_dir, "training.json"), job_json)

            self.save_valid_csv(save_dir, valid_json)

        if write_idxs:
            self.save_idxs(model_best, dset_dict, save_dir, n_test)

        if not self.keep_tasks:
            shutil.rmtree(task_dir)

        # Saving model.
        save_path = os.path.join(save_dir, model_name)
        model_best.save(save_path)

        log.t_stop(t_job, message="\nJob duration : {time} s", precision=2)
        return model_best.model_dict

    def active_train(
        self,
        dataset,
        model_name,
        n_train_init,
        n_train_final,
        n_valid,
        model0=None,
        n_train_step=100,
        n_test=None,
        save_dir=".",
        overwrite=False,
        write_json=True,
        write_idxs=True,
    ):
        r"""Trains a GDML model by using Bayesian optimization and
        adding problematic (high error) structures to the training set.

        Trains a GDML model with :meth:`mbgdml.train.mbGDMLTrain.bayes_opt`.

        Parameters
        ----------
        dataset : :obj:`mbgdml.data.DataSet`
            Data set to split into training, validation, and test sets are
            derived from.
        model_name : :obj:`str`
            User-defined model name without the ``'.npz'`` file extension.
        n_train_init : :obj:`int`
            Initial size of the training set. If ``model0`` is provided, this
            is the size of that model.
        n_train_final : :obj:`int`
            Training set size of the final model.
        n_valid : :obj:`int`
            Size of the validation set to be used for each training task.
            Different structures are sampled for each training task.
        model0 : :obj:`dict`, default: :obj:`None`
            Initial model to start training with. Training indices will be taken from
            here.
        n_train_step : :obj:`int`, default: ``100``
            Number of problematic structures to add to the training set for
            each iteration.
        n_test : :obj:`int`, default: :obj:`None`
            The number of test points to test the validated GDML model.
            Defaults to testing all available structures.
        save_dir : :obj:`str`, default: ``'.'``
            Path to train and save the mbGDML model.
        overwrite : :obj:`bool`, default: ``False``
            Overwrite existing files.
        write_json : :obj:`bool`, default: ``True``
            Write a JSON file containing information about the training job.
        write_idxs : :obj:`bool`, default: ``True``
            Write npy files for training, validation, and test indices.
        """
        log.log_package()

        t_job = log.t_start()
        log.info(
            "###################\n"
            "#    Active       #\n"
            "#    Training     #\n"
            "###################\n"
        )
        log.info("Initial training set size : %r", n_train_init)
        if model0 is not None:
            log.info("Initial model was provided")
            log.info("Taking training indices from model")
            train_idxs = model0["idxs_train"]
            n_train = len(train_idxs)
            log.log_array(train_idxs, level=10)

            prob_s = ProblematicStructures(
                [gdmlModel(model0, comp_ids=self.comp_ids)], predict_gdml
            )
            save_dir_i = os.path.join(save_dir, f"train{n_train}")
            os.makedirs(save_dir_i, exist_ok=overwrite)
            prob_idxs = prob_s.find(dataset, n_train_step, save_dir=save_dir_i)
            train_idxs = np.concatenate((train_idxs, prob_idxs))
            log.info("Extended the training set to %d", len(train_idxs))
            n_train = len(train_idxs)
        else:
            log.info("An initial model will be trained")
            n_train = n_train_init
            train_idxs = None

        sigma_bounds = self.sigma_bounds
        while n_train <= n_train_final:

            save_dir_i = os.path.join(save_dir, f"train{n_train}")
            model, _ = self.bayes_opt(
                dataset,
                model_name + f"-train{n_train}",
                n_train,
                n_valid,
                n_test=n_test,
                save_dir=save_dir_i,
                train_idxs=train_idxs,
                valid_idxs=None,
                overwrite=overwrite,
                write_json=write_json,
                write_idxs=write_idxs,
            )

            # Check sigma bounds
            buffer = 5
            lower_buffer = min(sigma_bounds) + buffer
            upper_buffer = max(sigma_bounds) - buffer
            if model["sig"] <= lower_buffer:
                log.warning(
                    "WARNING: Optimal sigma is within %r from the lower sigma bound",
                    buffer,
                )
            elif model["sig"] >= upper_buffer:
                log.warning(
                    "WARNING: Optimal sigma is within %r from the upper sigma bound",
                    buffer,
                )

            train_idxs = model["idxs_train"]
            prob_s = ProblematicStructures(
                [gdmlModel(model, comp_ids=self.comp_ids)], predict_gdml
            )
            prob_idxs = prob_s.find(dataset, n_train_step, save_dir=save_dir_i)

            train_idxs = np.concatenate((train_idxs, prob_idxs))
            n_train = len(train_idxs)

        log.t_stop(t_job, message="\nJob duration : {time} s", precision=2)

        return model
