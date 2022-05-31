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

import os
import shutil
from mbgdml.data import mbModel
from mbgdml.data import dataSet

from ._gdml.train import *

def loss_f_rmse(results):
    """Returns the force RMSE.

    Parameters
    ----------
    results : :obj:`dict`
        Validation results.
    """
    return results['force']['rmse']

class mbGDMLTrain:
    """Train many-body GDML models.
    """

    def __init__(
        self, use_sym=True, use_E=True, use_E_cstr=False, use_cprsn=False,
        solver='analytic', lam=1e-15, solver_tol=1e-4, interact_cut_off=None,
        use_torch=False, max_processes=None
    ):
        """

        Parameters
        ----------
        use_sym : :obj:`bool`, optional
            Whether or not to use (``True``) symmetric or (``False``)
            nonsymmetric GDML. Usually ``True`` is recommended.
        use_E : :obj:`bool`, optional
            Whether or not to reconstruct the potential energy surface
            (``True``) with or (``False``) without energy labels. Almost always
            should be ``True``.
        use_E_cstr : :obj:`bool`, optional
            Whether or not to include energies as a part of the model training.
            Meaning ``True`` will add another column of alphas that will be
            trained to the energies. This is not necessary, but is sometimes
            useful for higher order *n*-body models. Defaults to ``True``.
        use_cprsn : :obj:`bool`, optional
            Compresses the kernel matrix along symmetric degrees of freedom to
            try to reduce training time. Usually does not provide significant
            benefits. Defaults to ``False``.
        solver : :obj:`str`, optional
            The sGDML solver to use. Currently the only option is
            ``'analytic'``.
        lam : float, optional
            Hyper-parameter lambda (regularization strength).
        solver_tol : float, optional
           Solver tolerance.
        interact_cut_off : :obj:`float`, optional
            Untested option. Not recommended and turned off.
        use_torch : :obj:`bool`, optional
            Use PyTorch to enable GPU acceleration.
        max_processes : :obj:`int`, default: ``None``
            The maximum number of cores to use for the training process. Will
            automatically calculate if not specified.
        """
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
    
    def train_model(
        self, train_dataset, n_train, valid_dataset, n_valid, sigma,
        train_idxs=None, valid_idxs=None
    ):
        """Creates a task and trains a GDML model.

        Parameters
        ----------
        train_dataset : :obj:`mbgdml.data.dataSet`
            Dataset for training a model on.
        n_train : :obj:`int`
            The number of training points to sample.
        valid_dataset : :obj:`mbgdml.data.dataSet`
            Dataset for validating a model on.
        n_valid : :obj:`int`
            The number of validation points to sample, without replacement.
        sigma : :obj:`float` or :obj:`int`
            Kernel length scale of the desired model.
        train_idxs : :obj:`numpy.ndarray`, default: ``None``
            The specific indices of structures to train the model on. If
            ``None`` will automatically sample the training data set.
        valid_idxs : :obj:`numpy.ndarray`, default: ``None``
            The specific indices of structures to validate models on.
            If ``None``, structures will be automatically determined.
        
        Returns
        -------
        :obj:`dict`
            Trained (not validated or tested) model.
        """
        train = GDMLTrain()
        task = train.create_task(
            train_dataset, n_train, valid_dataset, n_valid, sigma, lam=self.lam,
            use_sym=self.use_sym, use_E=self.use_E, use_E_cstr=self.use_E_cstr,
            use_cprsn=self.use_cprsn, solver=self.solver,
            solver_tol=self.solver_tol, interact_cut_off=self.interact_cut_off,
            idxs_train=train_idxs, idxs_valid=valid_idxs
        )
        model = train.train(task)
        return model
    
    def test_model(self, model, dataset, n_test=None):
        """Test model and add mbGDML modifications.
        
        Parameters
        ----------
        model : :obj:`mbgdml.data.mbModel`
            Model to test.
        dataset : :obj:`dict`
            Training dataset
        
        Returns
        -------
        :obj:`mbgdml.data.mbModel`
            Tested and finalized many-body GDML model.
        """
        n_test, e_mae, e_rmse, f_mae, f_rmse = model_errors(
            model.model, dataset, is_valid=False, n_test=n_test,
            max_processes=self.max_processes, use_torch=self.use_torch
        )
        model.model['n_test'] = np.array(n_test)

        results_test = {
            'energy': {'mae': e_mae, 'rmse': e_rmse},
            'force': {'mae': f_mae, 'rmse': f_rmse},
        }
        
        # Adding mbGDML-specific modifications to model.
        model.add_modifications(dataset)
        if 'mb' in dataset.keys():
            model.model['mb'] = int(dataset['mb'][()])
        if 'mb_models_md5' in dataset.keys():
            model_best.model['mb_models_md5'] = dataset['mb_models_md5']
        
        return model, results_test
    
    def save_idxs(self, model, dataset, save_dir, n_test):
        """Saves npy files of the dataset splits (training, validation, and
        test).

        Parameters
        ----------
        model : :obj:`mbgdml.data.mbModel`
            Many-body GDML model.
        dataset : :obj:`dict`
            Dataset used for training, validation, and testing.
        """
        train_idxs = model.model['idxs_train']
        valid_idxs = model.model['idxs_valid']

        np.save(os.path.join(save_dir, 'train-idxs'), train_idxs)
        np.save(os.path.join(save_dir, 'valid-idxs'), valid_idxs)

        test_idxs = get_test_idxs(model.model, dataset, n_test=n_test)
        np.save(os.path.join(save_dir, 'test-idxs'), test_idxs)
    
    def save_json(self, json_dict, save_dir, json_name='log'):
        """Save JSON file.

        Parameters
        ----------
        json_dict : :obj:`dict`
            JSON dictionary to be saved.
        save_dir : :obj:`str`
            Where to save the JSON file.
        json_name : :obj:`str`
            File name.
        """
        import json
        from cclib.io.cjsonwriter import JSONIndentEncoder
            
        json_string = json.dumps(
            json_dict, cls=JSONIndentEncoder, indent=4
        )

        json_path = os.path.join(save_dir, f'{json_name}.json')
        with open(json_path, 'w') as f:
            f.write(json_string)
    
    def bayes_opt(
        self, dataset, model_name, n_train, n_valid,
        sigma_bounds=(2, 300), n_test=None, save_dir='.',
        gp_params={'init_points': 10, 'n_iter': 10, 'alpha': 0.001},
        loss=loss_f_rmse, train_idxs=None, valid_idxs=None,
        overwrite=False, write_json=False, write_idxs=True, bo_verbose=2
    ):
        """Train a GDML model using Bayesian optimization for sigma.

        Notes
        -----
        Uses the `Bayesian optimization <https://github.com/fmfn/BayesianOptimization>`_
        package to automatically find the optimal sigma. This will maximize
        the negative validation loss.

        ``gp_params`` can be used to specify options to
        ``BayesianOptimization.maximize()`` method.


        Parameters
        ----------
        dataset : :obj:`mbgdml.data.dataSet`
            Dataset to train, validate, and test a model on.
        model_name : :obj:`str`
            User-defined model name without the ``'.npz'`` file extension.
        n_train : :obj:`int`
            The number of training points to use.
        n_valid : :obj:`int`
            The number of validation points to use.
        sigma_bounds : :obj:`tuple`, default: ``(2, 300)``
            Kernel length scale bounds for the Bayesian optimization.
        n_test : :obj:`int`, default: ``None``
            The number of test points to test the validated GDML model.
            Defaults to testing all available structures.
        save_dir : :obj:`str`, default: ``'.'``
            Path to train and save the mbGDML model. Defaults to current
            directory.
        gp_params : :obj:`dict`
            Gaussian process parameters. Others can be included.

            ``init_points``
                How many steps of random exploration you want to perform.
                Random exploration can help by diversifying the exploration
                space. Defaults to ``10``.
            ``n_iter``
                How many steps of bayesian optimization you want to perform.
                The more steps the more likely to find a good maximum you are.
                Defaults to ``10``.
            ``alpha`` 
                This parameters controls how much noise the GP can handle, so
                increase it whenever you think that extra flexibility is needed.
                Defaults to ``0.001``.
        loss : callable, default: :obj:`mbgdml.train.loss_f_rmse`
            Loss function for validation. The input of this function is the
            dictionary of :obj:`mbgdml.gdml.train.add_valid_errors` which
            contains force and energy MAEs and RMSEs.
        train_idxs : :obj:`numpy.ndarray`, default: ``None``
            The specific indices of structures to train the model on. If
            ``None`` will automatically sample the training data set.
        valid_idxs : :obj:`numpy.ndarray`, default: ``None``
            The specific indices of structures to validate models on.
            If ``None``, structures will be automatically determined.
        overwrite : :obj:`bool`, default: ``False``
            Overwrite existing files.
        write_json : :obj:`bool`, default: ``True``
            Write a JSON file containing information about the training job.
        write_idxs : :obj:`bool`, default: ``True``
            Write npy files for training, validation, and test indices.
        bo_verbose : :obj:`int`, default: ``2``
            ``bayes_opt`` verbosity. ``2`` prints out every trail, ``1``
            prints the most recent best model, and ``0`` prints nothing.
        
        Returns
        -------
        :obj:`dict`
            Optimal many-body GDML model.
        ``bayes_opt.BayesianOptimization``
            The Bayesian optimizer object.
        """
        from bayes_opt import BayesianOptimization

        if write_json:
            write_json = True
            job_json = {
                'model': {},
                'validation': {},
                'testing': {},
                'training': {'idxs': []},
            }
        else:
            write_json = False

        if train_idxs is not None:
            assert n_train == len(train_idxs)
            assert len(set(train_idxs)) == len(train_idxs)
        if valid_idxs is not None:
            assert n_valid == len(valid_idxs)
            assert len(set(valid_idxs)) == len(valid_idxs)
        
        dset_dict = dataset.asdict

        task_dir = os.path.join(save_dir, 'tasks')
        os.makedirs(task_dir, exist_ok=overwrite)

        losses = []
        valid_json = {
            'sigmas': [],
            'force': {'mae': [], 'rmse': []},
            'energy': {'mae': [], 'rmse': []},
            'idxs': [],
        }

        def opt_func(sigma):
            model_trial = self.train_model(
                dset_dict, n_train, dset_dict, n_valid, sigma,
                train_idxs=train_idxs, valid_idxs=valid_idxs
            )

            if len(valid_json['idxs']) == 0:
                valid_json['idxs'] = model_trial['idxs_valid'].tolist()

            valid_results, model_trial = add_valid_errors(
                model_trial, dset_dict, overwrite=True,
                max_processes=self.max_processes, use_torch=self.use_torch
            )

            l = loss(valid_results)

            valid_json['sigmas'].append(model_trial['sig'])
            valid_json['energy']['mae'].append(valid_results['energy']['mae'])
            valid_json['energy']['rmse'].append(valid_results['energy']['rmse'])
            valid_json['force']['mae'].append(valid_results['force']['mae'])
            valid_json['force']['rmse'].append(valid_results['force']['rmse'])

            model_trail_path = os.path.join(
                task_dir, f'model-{sigma}.npz'
            )

            save_model(model_trial, model_trail_path)

            return -l

        optimizer = BayesianOptimization(
            f=opt_func,
            pbounds={'sigma': sigma_bounds},
            verbose=bo_verbose,
        )
        optimizer.maximize(**gp_params)
        best_res = optimizer.max
        sigma_best = best_res['params']['sigma']
        model_best_path = os.path.join(
            task_dir, f'model-{sigma_best}.npz'
        )
        model_best = mbModel(model_best_path)

        # Testing model
        model_best, results_test = self.test_model(
            model_best, dset_dict, n_test=n_test
        )

        # Including final JSON stuff and writing.
        if write_json:
            job_json['training']['idxs'] = model_best.model['idxs_train'].tolist()
            job_json['testing']['n_test'] = int(model_best.model['n_test'][()])
            job_json['testing']['energy'] = results_test['energy']
            job_json['testing']['force'] = results_test['force']

            job_json['model']['sigma'] = int(model_best.model['sig'][()])
            job_json['model']['n_symm'] = len(model_best.model['perms'])
            job_json['validation'] = valid_json

            self.save_json(job_json, save_dir)

        if write_idxs:
            self.save_idxs(model_best, dset_dict, save_dir, n_test)
        
        shutil.rmtree(task_dir)

        # Saving model.
        model_best.save(model_name, model_best.model, save_dir)

        return model_best.model, optimizer

        
    def grid_search(
        self, dataset, model_name, n_train, n_valid,
        sigmas=list(range(2, 400, 30)), n_test=None, save_dir='.',
        loss=loss_f_rmse, train_idxs=None, valid_idxs=None,
        overwrite=False, write_json=False, write_idxs=True,
    ):
        """Train a GDML model using a grid search for sigma.

        Usually, the validation errors will decrease until an optimal sigma is
        found then start to increase (overfitting). We sort ``sigmas`` from
        lowest to highest and stop the search once the loss function starts
        increasing.

        Notes
        -----
        For higher-order models, the loss function could prematurely find
        a minimum at small sigmas. We recommend spanning a large search space
        (e.g., ``list(range(2, 500, 50))``), then refining the search grid by
        decreasing the step size.
        
        Parameters
        ----------
        dataset : :obj:`mbgdml.data.dataSet`
            Dataset to train, validate, and test a model on.
        model_name : :obj:`str`
            User-defined model name without the ``'.npz'`` file extension.
        n_train : :obj:`int`
            The number of training points to use.
        n_valid : :obj:`int`
            The number of validation points to use.
        sigmas : :obj:`list`, default: ``list(range(2, 400, 30))``
            Kernel length scales (i.e., hyperparameters) to train and validate
            GDML models. Note, more length scales usually mean longer training
            times. Two is the minimum value and can get as large as several
            hundred. Can be :obj:`int` or :obj:`float`.
        n_test : :obj:`int`, default: ``None``
            The number of test points to test the validated GDML model.
            Defaults to testing all available structures.
        save_dir : :obj:`str`, default: ``'.'``
            Path to train and save the mbGDML model. Defaults to current
            directory.
        loss : callable, default: :obj:`mbgdml.train.loss_f_rmse`
            Loss function for validation. The input of this function is the
            dictionary of :obj:`mbgdml.gdml.train.add_valid_errors` which
            contains force and energy MAEs and RMSEs.
        train_idxs : :obj:`numpy.ndarray`, default: ``None``
            The specific indices of structures to train the model on. If
            ``None`` will automatically sample the training data set.
        valid_idxs : :obj:`numpy.ndarray`, default: ``None``
            The specific indices of structures to validate models on.
            If ``None``, structures will be automatically determined.
        overwrite : :obj:`bool`, default: ``False``
            Overwrite existing files.
        write_json : :obj:`bool`, default: ``True``
            Write a JSON file containing information about the training job.
        write_idxs : :obj:`bool`, default: ``True``
            Write npy files for training, validation, and test indices.
        """
        if write_json:
            write_json = True
            job_json = {
                'model': {},
                'validation': {},
                'testing': {},
                'training': {'idxs': []},
            }
        else:
            write_json = False
        
        if train_idxs is not None:
            assert n_train == len(train_idxs)
            assert len(set(train_idxs)) == len(train_idxs)
        if valid_idxs is not None:
            assert n_valid == len(valid_idxs)
            assert len(set(valid_idxs)) == len(valid_idxs)
        
        dset_dict = dataset.asdict
        
        task_dir = os.path.join(save_dir, 'tasks')
        os.makedirs(task_dir, exist_ok=overwrite)

        # Starting grid search
        sigmas.sort()
        trial_model_paths = []
        losses = []
        valid_json = {
            'sigmas': [],
            'force': {'mae': [], 'rmse': []},
            'energy': {'mae': [], 'rmse': []},
            'idxs': [],
        }
        model_best_path = None
        for sigma in sigmas:
            model_trial = self.train_model(
                dset_dict, n_train, dset_dict, n_valid, sigma,
                train_idxs=train_idxs, valid_idxs=valid_idxs
            )

            if len(valid_json['idxs']) == 0:
                valid_json['idxs'] = model_trial['idxs_valid'].tolist()

            valid_results, model_trial = add_valid_errors(
                model_trial, dset_dict, overwrite=True,
                max_processes=self.max_processes, use_torch=self.use_torch
            )

            losses.append(loss(valid_results))
            valid_json['sigmas'].append(model_trial['sig'])
            valid_json['energy']['mae'].append(valid_results['energy']['mae'])
            valid_json['energy']['rmse'].append(valid_results['energy']['rmse'])
            valid_json['force']['mae'].append(valid_results['force']['mae'])
            valid_json['force']['rmse'].append(valid_results['force']['rmse'])

            model_trail_path = os.path.join(
                task_dir, f'model-trial-sig{model_trial["sig"]}.npz'
            )
            save_model(model_trial, model_trail_path)
            trial_model_paths.append(model_trail_path)
            
            if len(losses) > 2:
                if losses[-1] > losses[-2]:
                    log.info('Validation errors are rising')
                    log.info('Terminating grid search')
                    model_best_path = trial_model_paths[-2]
                    on_grid_bounds = False
                    break
        
        # Determine best model and checking optimal sigma.
        if model_best_path is None:
            model_best_path = trial_model_paths[-1]
            on_grid_bounds = True
            next_search_sign = '>'
        
        model_best = mbModel(model_best_path)
        sigma_best = model_best.model['sig'].item()
        if sigma_best == sigmas[0]:
            on_grid_bounds = True
            next_search_sign = '<'
        
        if on_grid_bounds:
            log.warning('Optimal sigma is on the bounds of grid search')
            log.warning('This model is not optimal')
            log.warning(
                f'Extend your grid search to be {next_search_sign} {sigma_best}'
            )
        
        # Testing model
        model_best, results_test = self.test_model(
            model_best, dset_dict, n_test=n_test
        )

        # Including final JSON stuff and writing.
        if write_json:
            job_json['training']['idxs'] = model_best.model['idxs_train'].tolist()
            job_json['testing']['n_test'] = int(model_best.model['n_test'][()])
            job_json['testing']['energy'] = results_test['energy']
            job_json['testing']['force'] = results_test['force']

            job_json['model']['sigma'] = int(model_best.model['sig'][()])
            job_json['model']['n_symm'] = len(model_best.model['perms'])
            job_json['validation'] = valid_json

            self.save_json(job_json, save_dir)
        
        if write_idxs:
            self.save_idxs(model_best, dset_dict, save_dir, n_test)
        
        shutil.rmtree(task_dir)

        # Saving model.
        model_best.save(model_name, model_best.model, save_dir)

        return model_best.model
