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
        solver='analytic', lam=1e-15, solver_tol=1e-4, interact_cut_off=None
    ):
        self.use_sym = use_sym
        self.use_E = use_E
        self.use_E_cstr = use_E_cstr
        self.use_cprsn = use_cprsn
        self.solver = solver
        self.lam = lam
        self.solver_tol = solver_tol
        self.interact_cut_off = interact_cut_off
    
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
    
    def grid_search(
        self, dataset, model_name, n_train, n_valid,
        sigmas=list(range(2, 400, 30)), n_test=None, save_dir='.',
        loss=loss_f_rmse, train_idxs=None, valid_idxs=None, max_processes=None,
        overwrite=False, use_torch=False, write_json=False, write_idxs=True,
    ):
        """Trains a GDML model using a grid search for sigma.

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
        sigmas : :obj:`list`, optional
            Kernel length scales (i.e., hyperparameters) to train and validate
            GDML models. Note, more length scales usually mean longer training
            times. Two is the minimum value and can get as large as several
            hundred. Defaults to ``list(range(2, 400, 30))``. Can be :obj:`int`
            or :obj:`float`.
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
        max_processes : :obj:`int`, default: ``None``
            The maximum number of cores to use for the training process. Will
            automatically calculate if not specified.
        overwrite : :obj:`bool`, default: ``False``
            Overwrite existing files.
        use_torch : :obj:`bool`, default: ``False``
            Use PyTorch to enable GPU acceleration.
        write_json : :obj:`bool`, default: ``True``
            Write a JSON file containing information about the training job.
        write_idxs : :obj:`bool`, default: ``True``
            Write npy files for training, validation, and test indices.
        """
        if write_json:
            import json
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
                max_processes=max_processes, use_torch=use_torch
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
        n_test, e_mae, e_rmse, f_mae, f_rmse = model_errors(
            model_best.model, dset_dict, is_valid=False, n_test=n_test,
            max_processes=None, use_torch=use_torch
        )
        model_best.model['n_test'] = np.array(n_test)
        
        # Adding mbGDML-specific modifications to model.
        model_best.add_modifications(dset_dict)
        if 'mb' in dset_dict.keys():
            model_best.model['mb'] = int(dset_dict['mb'][()])
        if 'mb_models_md5' in dset_dict.keys():
            model_best.model['mb_models_md5'] = dset_dict['mb_models_md5']

        # Including final JSON stuff and writing.
        if write_json:
            job_json['training']['idxs'] = model_best.model['idxs_train'].tolist()
            job_json['testing']['n_test'] = int(model_best.model['n_test'][()])
            job_json['testing']['energy_mae'] = e_mae
            job_json['testing']['energy_rmse'] = e_rmse
            job_json['testing']['forces_mae'] = f_mae
            job_json['testing']['forces_rmse'] = f_rmse

            job_json['model']['sigma'] = int(model_best.model['sig'][()])
            job_json['model']['n_symm'] = len(model_best.model['perms'])
            job_json['validation'] = valid_json

            from cclib.io.cjsonwriter import JSONIndentEncoder
            
            json_string = json.dumps(
                job_json, cls=JSONIndentEncoder, indent=4
            )

            json_path = os.path.join(save_dir, 'log.json')
            with open(json_path, 'w') as f:
                f.write(json_string)
        
        if write_idxs:
            train_idxs = model_best.model['idxs_train']
            valid_idxs = model_best.model['idxs_valid']

            np.save(os.path.join(save_dir, 'train-idxs'), train_idxs)
            np.save(os.path.join(save_dir, 'valid-idxs'), valid_idxs)

            test_idxs = get_test_idxs(model_best.model, dset_dict, n_test=n_test)
            np.save(os.path.join(save_dir, 'test-idxs'), test_idxs)
        
        shutil.rmtree(task_dir)

        # Saving model.
        model_best.save(model_name, model_best.model, save_dir)

        return model_best.model
