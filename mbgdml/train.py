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
import os
import shutil
from .data import mbModel, dataSet
from .analysis.problematic import prob_structures
from ._gdml.train import GDMLTrain, model_errors, add_valid_errors
from ._gdml.train import save_model, get_test_idxs
from .utils import save_json

import logging
log = logging.getLogger(__name__)

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
            The GDML solver to use. Currently the only option is
            ``'analytic'``.
        lam : float, default: ``1e-15``
            Hyper-parameter lambda (regularization strength). This generally
            does not need to change.
        solver_tol : float, default: ``1e-4``
           Solver tolerance.
        interact_cut_off : :obj:`float`, default: ``None``
            Untested option. Not recommended and turned off.
        use_torch : :obj:`bool`, default: ``False``
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

        self.GDMLTrain = GDMLTrain(
            max_processes=max_processes, use_torch=use_torch
        )
    
    def min_memory(self, n_train, n_atoms):
        """Minimum memory recommendation for training.

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
        mem = (8*((n_train * 3*n_atoms)**2))/1000000
        return mem
    
    def __del__(self):
        global glob

        if 'glob' in globals():
            del glob
    
    def create_task(
        self, train_dataset, n_train, valid_dataset, n_valid, sigma,
        train_idxs=None, valid_idxs=None
    ):
        """Create a single training task that can be used as a template for
        hyperparameter searches.

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
        """
        self.task = self.GDMLTrain.create_task(
            train_dataset, n_train, valid_dataset, n_valid, sigma, lam=self.lam,
            use_sym=self.use_sym, use_E=self.use_E, use_E_cstr=self.use_E_cstr,
            use_cprsn=self.use_cprsn, solver=self.solver,
            solver_tol=self.solver_tol, interact_cut_off=self.interact_cut_off,
            idxs_train=train_idxs, idxs_valid=valid_idxs
        )
        return self.task
    
    def train_model(self, task):
        """Trains a GDML model from a task.

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
        model = self.GDMLTrain.train(task)
        return model
    
    def test_model(self, model, dataset, n_test=None):
        """Test model and add mbGDML modifications.
        
        Parameters
        ----------
        model : :obj:`mbgdml.data.mbModel`
            Model to test.
        dataset : :obj:`dict`
            Test dataset.
        
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
        log.info('\n#   Writing train, valid, and test indices   #')
        train_idxs = model.model['idxs_train']
        valid_idxs = model.model['idxs_valid']

        np.save(os.path.join(save_dir, 'train_idxs'), train_idxs)
        np.save(os.path.join(save_dir, 'valid_idxs'), valid_idxs)

        test_idxs = get_test_idxs(model.model, dataset, n_test=n_test)
        np.save(os.path.join(save_dir, 'test_idxs'), test_idxs)
    
    def bayes_opt(
        self, dataset, model_name, n_train, n_valid,
        sigma_bounds=(2, 300), n_test=None, save_dir='.',
        gp_params={'init_points': 10, 'n_iter': 10, 'alpha': 0.001},
        loss=loss_f_rmse, train_idxs=None, valid_idxs=None,
        overwrite=False, write_json=True, write_idxs=True
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
            dictionary of :obj:`mbgdml._gdml.train.add_valid_errors` which
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
        
        Returns
        -------
        :obj:`dict`
            Optimal many-body GDML model.
        ``bayes_opt.BayesianOptimization``
            The Bayesian optimizer object.
        """
        from bayes_opt import BayesianOptimization

        t_job = log.t_start()

        log.info(
            '#############################\n'
            '#         Training          #\n'
            '#   Bayesian Optimization   #\n'
            '#############################\n'
        )

        if write_json:
            write_json = True
            job_json = {
                'model': {},
                'testing': {},
                'validation': {},
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
        
        dset_dict = dataset.asdict()

        task_dir = os.path.join(save_dir, 'tasks')
        os.makedirs(task_dir, exist_ok=overwrite)

        task = self.create_task(
            dset_dict, n_train, dset_dict, n_valid, None,
            train_idxs=train_idxs, valid_idxs=valid_idxs
        )

        losses = []
        valid_json = {
            'sigmas': [],
            'force': {'mae': [], 'rmse': []},
            'energy': {'mae': [], 'rmse': []},
            'idxs': [],
        }
        def opt_func(sigma):
            task['sig'] = sigma
            model_trial = self.train_model(task)

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
            verbose=0,
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

            job_json['model']['sigma'] = model_best.model['sig'][()]
            job_json['model']['n_symm'] = len(model_best.model['perms'])
            job_json['validation'] = valid_json

            save_json(os.path.join(save_dir, 'log.json'), job_json)

        if write_idxs:
            self.save_idxs(model_best, dset_dict, save_dir, n_test)
        
        shutil.rmtree(task_dir)

        # Saving model.
        model_best.save(model_name, model_best.model, save_dir)

        log.t_stop(t_job, message='\nJob duration : {time} s', precision=2)
        return model_best.model, optimizer

        
    def grid_search(
        self, dataset, model_name, n_train, n_valid,
        sigmas=list(range(2, 400, 30)), n_test=None, save_dir='.',
        loss=loss_f_rmse, train_idxs=None, valid_idxs=None,
        overwrite=False, write_json=True, write_idxs=True,
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
            dictionary of :obj:`mbgdml._gdml.train.add_valid_errors` which
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
        
        Returns
        -------
        :obj:`dict`
            GDML model with an optimal hyperparameter found via grid search.
        """
        t_job = log.t_start()
        log.info(
            '###################\n'
            '#    Training     #\n'
            '#   Grid Search   #\n'
            '###################\n'
        )
        if write_json:
            write_json = True
            job_json = {
                'model': {},
                'testing': {},
                'validation': {},
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
        
        dset_dict = dataset.asdict()
        
        task_dir = os.path.join(save_dir, 'tasks')
        os.makedirs(task_dir, exist_ok=overwrite)

        task = self.create_task(
            dset_dict, n_train, dset_dict, n_valid, sigmas[0],
            train_idxs=train_idxs, valid_idxs=valid_idxs
        )

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
        for next_sigma in sigmas[1:]:
            model_trial = self.train_model(task)

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
            
            if len(losses) > 1:
                if losses[-1] > losses[-2]:
                    log.info('\nValidation errors are rising')
                    log.info('Terminating grid search')
                    model_best_path = trial_model_paths[-2]
                    on_grid_bounds = False
                    break
            
            task['sig'] = next_sigma
        
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

            job_json['model']['sigma'] = model_best.model['sig'][()]
            job_json['model']['n_symm'] = len(model_best.model['perms'])
            job_json['validation'] = valid_json

            save_json(os.path.join(save_dir, 'log.json'), job_json)
        
        if write_idxs:
            self.save_idxs(model_best, dset_dict, save_dir, n_test)
        
        shutil.rmtree(task_dir)

        # Saving model.
        model_best.save(model_name, model_best.model, save_dir)

        log.t_stop(t_job, message='\nJob duration : {time} s', precision=2)
        return model_best.model
    
    def iterative_train(
        self, dataset, model_name, n_train_init, n_train_final, n_valid,
        model0=None, n_train_step=100, sigma_bounds=(2, 300), n_test=None,
        save_dir='.', gp_params={'init_points': 10, 'n_iter': 10, 'alpha': 0.001},
        loss=loss_f_rmse, overwrite=False, write_json=True, write_idxs=True
    ):
        """Iteratively trains a GDML model by using Bayesian optimization and
        adding problematic (high error) structures to the training set.

        Trains a GDML model with :meth:`mbgdml.train.mbGDMLTrain.bayes_opt`
        using ``sigma_bounds`` and ``gp_params``.

        Parameters
        ----------
        dataset : :obj:`mbgdml.data.dataSet`
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
        model0 : :obj:`dict`, default: ``None``
            Initial model to start iterative training with. Training indices
            will be taken from here.
        n_train_step : :obj:`int`, default: ``100``
            Number of problematic structures to add to the training set for
            each iteration.
        sigma_bounds : :obj:`tuple`, default: ``(2, 300)``
            Kernel length scale bounds for the Bayesian optimization.
        n_test : :obj:`int`, default: ``None``
            The number of test points to test the validated GDML model.
            Defaults to testing all available structures.
        save_dir : :obj:`str`, default: ``'.'``
            Path to train and save the mbGDML model.
        gp_params : :obj:`dict`
            Gaussian process kwargs. Others can be provided.

            ``init_points`` : (:obj:`int`, default: ``10``)
                How many steps of random exploration you want to perform.
                Random exploration can help by diversifying the exploration
                space. Defaults to ``10``.
            
            ``n_iter`` : (:obj:`int`, default: ``10``)
                How many steps of bayesian optimization you want to perform.
                The more steps the more likely to find a good maximum you are.

            ``alpha``  : (:obj:`float`, default: ``0.001``)
                This parameters controls how much noise the GP can handle, so
                increase it whenever you think that extra flexibility is needed.
        
        loss : callable, default: :obj:`mbgdml.train.loss_f_rmse`
            Loss function for validation. The input of this function is the
            dictionary of :obj:`mbgdml._gdml.train.add_valid_errors` which
            contains force and energy MAEs and RMSEs.
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
            '###################\n'
            '#    Iterative    #\n'
            '#    Training     #\n'
            '###################\n'
        )
        log.info(f'Initial training set size : {n_train_init}')
        if model0 is not None:
            log.info('Initial model was provided')
            log.info('Taking training indices from model')
            train_idxs = model0['idxs_train']
            log.log_array(train_idxs, level=10)
            prob_idxs = prob_structures([model0])
            train_idxs = np.concatenate((train_idxs, prob_idxs))
            log.info(f'Extended the training set to {len(train_idxs)}')
            n_stages = int((n_train_final-len(train_idxs))/n_train_step)
            n_train = len(train_idxs)
        else:
            log.info('An initial model will be trained')
            n_stages = int((n_train_final-n_train_init)/n_train_step)
            n_train = n_train_init
        
        i = 0
        while n_train <= n_train_final:
            
            save_dir_i = os.path.join(save_dir, str(i))
            model, _ = self.bayes_opt(
                dataset, model_name+f'-train{n_train}', n_train, n_valid,
                sigma_bounds=sigma_bounds, n_test=n_test, save_dir=save_dir_i,
                gp_params=gp_params, loss=loss, train_idxs=None,
                valid_idxs=None, overwrite=overwrite,
                write_json=write_json, write_idxs=write_idxs
            )

            # Check sigma bounds
            buffer = 5
            lower_buffer = min(sigma_bounds) + buffer
            upper_buffer = max(sigma_bounds) - buffer
            if (model['sig'] <= lower_buffer):
                log.warning(
                    f'WARNING: Optimal sigma is within {buffer} from the lower sigma bound'
                )
            elif (model['sig'] >= upper_buffer):
                log.warning(
                    f'WARNING: Optimal sigma is within {buffer} from the upper sigma bound'
                )

            train_idxs = model['idxs_train']
            prob_s = prob_structures([model])
            prob_idxs = prob_s.find(dataset, n_train_step, save_dir=save_dir_i)
            
            train_idxs = np.concatenate((train_idxs, prob_idxs))
            n_train = len(train_idxs)
            
            i += 1

        return model
