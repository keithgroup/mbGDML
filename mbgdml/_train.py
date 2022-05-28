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

"""
Slightly modified sGDML training routines coming from train.py and cli.py.
We import unmodified functions from sGDML to use here.

Changes are marked with ###   CHANGED   ### and terminated with
###   UNCHANGED   ###
"""

from functools import partial
import os
import shutil
import sys
import logging
import traceback
import numpy as np

import multiprocessing as mp
from sgdml.train import GDMLTrain
from sgdml.cli import _print_dataset_properties
from sgdml.cli import train as sgdml_train
from sgdml.cli import select as sgdml_select
from sgdml.cli import test as sgdml_test
from sgdml.utils import ui, io, perm
from sgdml import __version__ as sgdml_version

try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True

log = logging.getLogger('sgdml')

# Parameters from sgdml's __init__.py
MAX_PRINT_WIDTH = 100
LOG_LEVELNAME_WIDTH = 7
DONE = 1
NOT_DONE = 0
PACKAGE_NAME = 'sgdml'

class sGDMLTraining():
    """Customized sGDML training class.
    """

    def __init__(self):
        pass
    
    ###   train.py   ###
    def sgdml_create_task(  # create_task from train.py
        self,
        gdml_train,  # Different from sGDML. 
        train_dataset,
        n_train,
        valid_dataset,
        n_valid,
        sig,
        lam=1e-15,
        use_sym=True,
        use_E=True,
        use_E_cstr=False,
        use_cprsn=False,
        solver='analytic',
        solver_tol=1e-4,
        n_inducing_pts_init=25,
        interact_cut_off=None,
        callback=None,
        idxs_train=None,
        idxs_valid=None,
        use_frag_perms=False
    ):
        """
        Create a data structure of custom type `task`.

        These data structures serve as recipes for model creation,
        summarizing the configuration of one particular training run.
        Training and test points are sampled from the provided dataset,
        without replacement. If the same dataset if given for training
        and testing, the subsets are drawn without overlap.

        Each task also contains a choice for the hyper-parameters of the
        training process and the MD5 fingerprints of the used datasets.

        Parameters
        ----------
        gdml_train : :obj:`sgdml.train.GDMLTrain``
            The sGDML training object for this run. This is different from the
            sGDML function and needed to use the ``draw_strat_sample``
            function.
        train_dataset : :obj:`dict`
            Data structure of custom type :obj:`dataset` containing
            train dataset.
        n_train : :obj:`int`
            Number of training points to sample.
        valid_dataset : :obj:`dict`
            Data structure of custom type :obj:`dataset` containing
            validation dataset.
        n_valid : :obj:`int`
            Number of validation points to sample.
        sig : :obj:`int`
            Hyper-parameter (kernel length scale).
        lam : :obj:`float`, optional
            Hyper-parameter lambda (regularization strength).
        use_sym : :obj:`bool`, optional
            ``True``: include symmetries (sGDML), ``False``: GDML.
        use_E : :obj:`bool`, optional
            ``True``: reconstruct force field with corresponding potential
            energy surface.
            ``False``: ignore energy during training, even if energy labels are
            available in the dataset. The trained model will still be able to
            predict energies up to an unknown integration constant. Note, that
            the energy predictions accuracy will be untested.
        use_E_cstr : :obj:`bool`, optional
            ``True``: include energy constraints in the kernel,
            ``False``: default sGDML.
        use_cprsn : :obj:`bool`, optional
            ``True``: compress kernel matrix along symmetric degrees of freedom,
            ``False``: train using full kernel matrix.
        idxs_train : :obj:`numpy.ndarray`, optional
            The specific indices of structures to train the model on. If
            ``None`` will automatically sample the training data set.
        use_frag_perms : :obj:`float`, optional
            Find and use fragment permutations (experimental; not tested).
            Defaults to ``False``.

        Returns
        -------
        :obj:`dict`
            Data structure of custom type :obj:`task`.

        Raises
        ------
        ValueError
            If a reconstruction of the potential energy surface is requested,
            but the energy labels are missing in the dataset.
        """

        if use_E and 'E' not in train_dataset:
            raise ValueError(
                'No energy labels found in dataset!\n'
                + 'By default, force fields are always reconstructed including the\n'
                + 'corresponding potential energy surface (this can be turned off).\n'
                + 'However, the energy labels are missing in the provided dataset.\n'
            )

        use_E_cstr = use_E and use_E_cstr

        ###   CHANGED   ###
        # n_atoms = train_dataset['R'].shape[1]
        ###   UNCHANGED   ###

        if callback is not None:
            callback = partial(callback, disp_str='Hashing dataset(s)')
            callback(NOT_DONE)

        md5_train = io.dataset_md5(train_dataset)
        md5_valid = io.dataset_md5(valid_dataset)

        if callback is not None:
            callback(DONE)

        if callback is not None:
            callback = partial(
                callback, disp_str='Sampling training and validation subsets'
            )
            callback(NOT_DONE)

        ###   CHANGED   ###
        # Handles training indices
        if idxs_train is None:
            if 'E' in train_dataset:
                idxs_train = self.draw_strat_sample(
                    train_dataset['E'], n_train
                )
            else:
                idxs_train = np.random.choice(
                    np.arange(train_dataset['F'].shape[0]),
                    n_train - m0_n_train,
                    replace=False,
                )
                # TODO: m0 handling
        else:
            idxs_train = np.array(idxs_train)  # Ensures it is an array.

        # Handles validation indices.
        if idxs_valid is not None:
            idxs_valid = np.array(idxs_valid)
        else:
            excl_idxs = (
                idxs_train if md5_train == md5_valid else np.array([], dtype=np.uint)
            )

            if 'E' in valid_dataset:
                idxs_valid = gdml_train.draw_strat_sample(
                    valid_dataset['E'], n_valid, excl_idxs=excl_idxs,
                )
            else:
                idxs_valid_cands = np.setdiff1d(
                    np.arange(valid_dataset['F'].shape[0]), excl_idxs,
                    assume_unique=True
                )
                idxs_valid = np.random.choice(
                    idxs_valid_cands, n_valid, replace=False
                )
                # TODO: m0 handling, zero handling
        ###   UNCHANGED   ###

        if callback is not None:
            callback(DONE)

        ###   CHANGED   ###
        # Changes __version__ to sgdml_version
        R_train = train_dataset['R'][idxs_train, :, :]
        task = {
            'type': 't',
            'code_version': sgdml_version,
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
            'n_inducing_pts_init': n_inducing_pts_init,
            'interact_cut_off': interact_cut_off,
        }
        ###   UNCHANGED   ###

        if use_E:
            task['E_train'] = train_dataset['E'][idxs_train]
        
        lat_and_inv = None
        if 'lattice' in train_dataset:
            task['lattice'] = train_dataset['lattice']

            try:
                lat_and_inv = (task['lattice'], np.linalg.inv(task['lattice']))
            except np.linalg.LinAlgError:
                raise ValueError(  # TODO: Document me
                    'Provided dataset contains invalid lattice vectors (not invertible). Note: Only rank 3 lattice vector matrices are supported.'
                )

        if 'r_unit' in train_dataset and 'e_unit' in train_dataset:
            task['r_unit'] = train_dataset['r_unit']
            task['e_unit'] = train_dataset['e_unit']

        if use_sym:
            n_train = R_train.shape[0]
            R_train_sync_mat = R_train
            ###   CHANGED   ###
            if n_train > 1000:
                R_train_sync_mat = R_train[
                    np.random.choice(n_train, 1000, replace=False), :, :
                ]
                log.info(
                    'Symmetry search has been restricted to a random subset of 1000/{:d} training points for faster convergence.'.format(
                        n_train
                    )
                )
            ###   UNCHANGED   ###

            # TOOD: PBCs disabled when matching (for now).
            # task['perms'] = perm.find_perms(
            #    R_train_sync_mat, train_dataset['z'], lat_and_inv=lat_and_inv, max_processes=self._max_processes,
            # )
            task['perms'] = perm.find_perms(
                R_train_sync_mat,
                train_dataset['z'],
                lat_and_inv=None,
                callback=callback,
                max_processes=self._max_processes,
            )

            ###   CHANGED   ###
            if use_frag_perms:
                frag_perms = perm.find_frag_perms(
                    R_train_sync_mat,
                    train_dataset['z'],
                    lat_and_inv=None,
                    max_processes=self._max_processes,
                )
                task['perms'] = np.vstack((task['perms'], frag_perms))
                task['perms'] = np.unique(task['perms'], axis=0)

                print(
                    '| Keeping '
                    + str(task['perms'].shape[0])
                    + ' unique permutations.'
                )
            ###   UNCHANGED   ###

            # NEW

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

        return task
    
    def sgdml_create(  # create from cli.py.
        self,
        dataset,
        valid_dataset,
        n_train,
        n_valid,
        sigs,
        gdml,
        use_E,
        use_E_cstr,
        use_cprsn,
        overwrite,
        max_processes,
        task_dir=None,
        solver='analytic',
        n_inducing_pts_init=None,
        interact_cut_off=None,
        command=None,
        idxs_train=None,  # Added to specify which structures to train on
        idxs_valid=None  # Added to specify which structures to use for validation.
    ):
        """A slightly modified ``sgdml create`` function for additional
        functionality.

        Parameters
        ----------
        dataset : :obj:`tuple`
            The path to the data set and the ``npz`` data set object.
        n_train : :obj:`int`
            Number of training points for the model.
        n_valid : :obj:`int`
            Number of validation points to test each trial model.
        sigs : :obj:`list` [:obj:`int`]
            A list of sigma hyperparameters to train trial models on. Should
            be ordered from smallest (at minimum ``2``) to largest.
        use_E : :obj:`bool`
            Whether or not to reconstruct the potential energy surface
            (``True``) with or (``False``) without energy labels. Almost always
            should be ``True``.
        use_E_cstr : :obj:`bool`
            Whether or not to include energies as a part of the model training.
            Meaning ``True`` will add another column of alphas that will be
            trained to the energies. This is not necessary, but is found
            to be useful in higher order n-body models. Defaults to ``True``.
        use_cprsn : :obj:`bool`
            Compresses the kernel matrix along symmetric degrees of freedom to
            try to reduce training time. Usually does not provide significant
            benefits. Defaults to ``False``.
        overwrite : :obj:`bool`
            Whether or not to overwrite an already existing model and its
            training.
        max_processes : :obj:`int`
            The maximum number of cores to use for the training process. Will
            automatically calculate if not specified.
        """

        has_valid_dataset = not (valid_dataset is None or valid_dataset == dataset)

        dataset_path, dataset = dataset
        n_data = dataset['F'].shape[0]

        ###   CHANGED   ###
        _print_task_properties(
            use_sym=not gdml, use_cprsn=use_cprsn, use_E=use_E,
            use_E_cstr=use_E_cstr
        )
        print()
        ###   UNCHANGED   ###

        if n_data < n_train:
            raise ValueError(
                'Dataset only contains {} points, can not train on {}.'.format(
                    n_data, n_train
                )
            )

        if not has_valid_dataset:
            valid_dataset_path, valid_dataset = dataset_path, dataset
            if n_data - n_train < n_valid:
                raise ValueError(
                    'Dataset only contains {} points, can not train on {} and validate on {}.'.format(
                        n_data, n_train, n_valid
                    )
                )
        else:
            valid_dataset_path, valid_dataset = valid_dataset
            n_valid_data = valid_dataset['R'].shape[0]
            if n_valid_data < n_valid:
                raise ValueError(
                    'Validation dataset only contains {} points, can not validate on {}.'.format(
                        n_data, n_valid
                    )
                )

        if sigs is None:
            log.info(
                'Kernel hyper-parameter sigma was automatically set to range \'10:10:100\'.'
            )
            sigs = list(range(10, 100, 10))  # default range

        if task_dir is None:
            task_dir = io.train_dir_name(
                dataset,
                n_train,
                use_sym=not gdml,
                use_cprsn=use_cprsn,
                use_E=use_E,
                use_E_cstr=use_E_cstr,
            )

        task_file_names = []
        if os.path.exists(task_dir):
            if overwrite:
                log.info('Overwriting existing training directory.')
                shutil.rmtree(task_dir, ignore_errors=True)
                os.makedirs(task_dir)
            else:
                if io.is_task_dir_resumeable(
                    task_dir, dataset, valid_dataset, n_train, n_valid, sigs, gdml
                ):
                    log.info(
                        'Resuming existing hyper-parameter search in \'{}\'.'.format(
                            task_dir
                        )
                    )

                    # Get all task file names.
                    try:
                        _, task_file_names = io.is_dir_with_file_type(task_dir, 'task')
                    except Exception:
                        pass
                else:
                    raise ValueError(
                        'Unfinished hyper-parameter search found in \'{}\'.\n'.format(
                            task_dir
                        )
                        + 'Run \'%s %s -o %s %d %d -s %s\' to overwrite.'
                        % (
                            PACKAGE_NAME,
                            command,
                            dataset_path,
                            n_train,
                            n_valid,
                            ' '.join(str(s) for s in sigs),
                        )
                    )
        else:
            os.makedirs(task_dir)

        if task_file_names:

            with np.load(
                os.path.join(task_dir, task_file_names[0]), allow_pickle=True
            ) as task:
                tmpl_task = dict(task)
        else:
            if not use_E:
                log.info(
                    'Energy labels will be ignored for training.\n'
                    + 'Note: If available in the dataset file, the energy '
                    + 'labels will however still be used to generate stratified'
                    + ' training, test and validation datasets. Otherwise a '
                    + 'random sampling is used.'
                )

            if 'E' not in dataset:
                log.warning(
                    'Training dataset will be sampled with no guidance from energy labels (i.e. randomly)!'
                )

            if 'E' not in valid_dataset:
                log.warning(
                    'Validation dataset will be sampled with no guidance from energy labels (i.e. randomly)!\n'
                    + 'Note: Larger validation datasets are recommended due to slower convergence of the error.'
                )

            if ('lattice' in dataset) ^ ('lattice' in valid_dataset):
                log.error('One of the datasets specifies lattice vectors and one does not!')
                # TODO: stop program?

            if 'lattice' in dataset or 'lattice' in valid_dataset:
                log.info(
                    'Lattice vectors found in dataset: applying periodic boundary conditions.'
                )

            ###   CHANGED   ###
            gdml_train = GDMLTrain(max_processes=max_processes)
            try:
                if idxs_train is None:
                    # This is the default sGDML 
                    tmpl_task = gdml_train.create_task(
                        dataset,
                        n_train,
                        valid_dataset,
                        n_valid,
                        sig=1,
                        use_sym=not gdml,
                        use_E=use_E,
                        use_E_cstr=use_E_cstr,
                        use_cprsn=use_cprsn,
                        solver=solver,
                        n_inducing_pts_init=n_inducing_pts_init,
                        interact_cut_off=interact_cut_off,
                        callback=ui.callback,
                    )  # template task
                # Our modified procedure to allow more control over training.
                else:
                    tmpl_task = self.sgdml_create_task(
                        gdml_train,  # This is different from sGDML.
                        dataset,
                        n_train,
                        valid_dataset,
                        n_valid,
                        sig=1,
                        lam=1e-15,
                        use_sym=not gdml,
                        use_E=use_E,
                        use_E_cstr=use_E_cstr,
                        use_cprsn=use_cprsn,
                        solver=solver,
                        n_inducing_pts_init=n_inducing_pts_init,
                        interact_cut_off=interact_cut_off,
                        callback=ui.callback,
                        idxs_train=idxs_train,
                        idxs_valid=idxs_valid
                    )
            except BaseException:
                print()
                log.critical(traceback.format_exc())
                sys.exit()
            ###   UNCHANGED   ###

        n_written = 0
        for sig in sigs:
            tmpl_task['sig'] = sig
            task_file_name = io.task_file_name(tmpl_task)
            task_path = os.path.join(task_dir, task_file_name)

            if os.path.isfile(task_path):
                log.warning('Skipping existing task \'{}\'.'.format(task_file_name))
            else:
                np.savez_compressed(task_path, **tmpl_task)
                n_written += 1
        if n_written > 0:
            log.done(
                'Writing {:d}/{:d} task(s) with {} training points each.'.format(
                    n_written, len(sigs), tmpl_task['R_train'].shape[0]
                )
            )

        return task_dir
    
    def sgdml_select(  # select from cli.py
        self, model_dir, overwrite, max_processes, model_file=None,
        command=None, **kwargs
    ):  # noqa: C901

        func_called_directly = (
            command == 'select'
        )  # has this function been called from command line or from 'all'?
        if func_called_directly:
            ui.print_step_title('MODEL SELECTION')

        any_model_not_validated = False
        any_model_is_tested = False

        model_dir, model_file_names = model_dir
        if len(model_file_names) > 1:

            use_E = True

            rows = []
            data_names = ['sig', 'MAE', 'RMSE', 'MAE', 'RMSE']
            for i, model_file_name in enumerate(model_file_names):
                model_path = os.path.join(model_dir, model_file_name)
                _, model = io.is_file_type(model_path, 'model')

                use_E = model['use_E']

                if i == 0:
                    idxs_train = set(model['idxs_train'])
                    md5_train = model['md5_train']
                    idxs_valid = set(model['idxs_valid'])
                    md5_valid = model['md5_valid']
                else:
                    if (
                        md5_train != model['md5_train']
                        or md5_valid != model['md5_valid']
                        or idxs_train != set(model['idxs_train'])
                        or idxs_valid != set(model['idxs_valid'])
                    ):
                        raise AssistantError(
                            '{} contains models trained or validated on different datasets.'.format(
                                model_dir
                            )
                        )

                e_err = {'mae': 0.0, 'rmse': 0.0}
                if model['use_E']:
                    e_err = model['e_err'].item()
                f_err = model['f_err'].item()

                is_model_validated = not (np.isnan(f_err['mae']) or np.isnan(f_err['rmse']))
                if not is_model_validated:
                    any_model_not_validated = True

                is_model_tested = model['n_test'] > 0
                if is_model_tested:
                    any_model_is_tested = True

                rows.append(
                    [model['sig'], e_err['mae'], e_err['rmse'], f_err['mae'], f_err['rmse']]
                )

                ###   CHANGED   ###
                self.job_json['validation']['sigmas'].append(int(model['sig'][()]))
                self.job_json['validation']['energy_mae'].append(e_err['mae'])
                self.job_json['validation']['energy_rmse'].append(e_err['rmse'])
                self.job_json['validation']['forces_mae'].append(f_err['mae'])
                self.job_json['validation']['forces_rmse'].append(f_err['rmse'])
                ###   UNCHANGED   ###

                model.close()

            if any_model_not_validated:
                log.error(
                    'One or more models in the given directory have not been validated yet.\n'
                    + 'This is required before selecting the best performer.'
                )
                print()
                sys.exit()

            if any_model_is_tested:
                log.error(
                    'One or more models in the given directory have already been tested. This means that their recorded expected errors are test errors, not validation errors. However, one should never perform model selection based on the test error!\n'
                    + 'Please run the validation command (again) with the overwrite option \'-o\', then this selection command.'
                )
                return

            f_rmse_col = [row[4] for row in rows]
            best_idx = f_rmse_col.index(min(f_rmse_col))  # idx of row with lowest f_rmse
            best_sig = rows[best_idx][0]

            rows = sorted(rows, key=lambda col: col[0])  # sort according to sigma
            print(ui.white_bold_str('Cross-validation errors'))
            print(' ' * 7 + 'Energy' + ' ' * 6 + 'Forces')
            print((' {:>3} ' + '{:>5} ' * 4).format(*data_names))
            print(' ' + '-' * 27)
            format_str = ' {:>3} ' + '{:5.2f} ' * 4
            format_str_no_E = ' {:>3}     -     - ' + '{:5.2f} ' * 2
            for row in rows:
                if use_E:
                    row_str = format_str.format(*row)
                else:
                    row_str = format_str_no_E.format(*[row[0], row[3], row[4]])

                if row[0] != best_sig:
                    row_str = ui.gray_str(row_str)
                print(row_str)
            print()

            sig_col = [row[0] for row in rows]
            if best_sig == min(sig_col) or best_sig == max(sig_col):
                log.warning(
                    'The optimal sigma lies on the boundary of the search grid.\n'
                    + 'Model performance might improve if the search grid is extended in direction sigma {} {:d}.'.format(
                        '<' if best_idx == 0 else '>', best_sig
                    )
                )
                ###   mbGDML CHANGED   ###
                self.job_json['model']['sigma_on_boundary'] = True
            else:
                self.job_json['model']['sigma_on_boundary'] = False
                ###   sGDML RESUMED   ###

        else:  # only one model available
            log.warning(
                'Skipping model selection step as there is only one model to select.'
            )

            best_idx = 0

        best_model_path = os.path.join(model_dir, model_file_names[best_idx])

        if model_file is None:

            # generate model file name based on model properties
            best_model = np.load(best_model_path, allow_pickle=True)
            model_file = io.model_file_name(best_model, is_extended=True)
            best_model.close()

        model_exists = os.path.isfile(model_file)
        if model_exists and overwrite:
            log.info('Overwriting existing model file.')

        if not model_exists or overwrite:
            if func_called_directly:
                log.done('Writing model file \'{}\''.format(model_file))

            shutil.copy(best_model_path, model_file)
            shutil.rmtree(model_dir, ignore_errors=True)
        else:
            log.warning(
                'Model \'{}\' already exists.\n'.format(model_file)
                + 'Run \'{} select -o {}\' to overwrite.'.format(
                    PACKAGE_NAME, os.path.relpath(model_dir)
                )
            )

        if func_called_directly:
            _print_next_step('select', model_files=[model_file])

        return model_file
    
    def sgdml_all(  # all from cli.py.
        self,
        dataset,
        n_train,
        n_valid,
        n_test,
        sigs,
        solver='analytic',
        valid_dataset=None,
        test_dataset=None,
        use_sym=True,
        use_E=True,
        use_E_cstr=True,
        use_cprsn=False,
        overwrite=False,
        max_processes=None,
        use_torch=False,
        idxs_train=None,
        idxs_valid=None,
        model_file=None,
        task_dir=None,
    ):
        """Run all training procedures to generate an optimal sGDML model.

        Parameters
        ----------
        dataset : :obj:`tuple`
            The path to the data set and the ``npz`` data set object.
        n_train : :obj:`int`
            Number of training points for the model.
        n_valid : :obj:`int`
            Number of validation points to test each trial model.
        n_test : :obj:`int`
            Number of test points to test the final selected model.
        sigs : :obj:`list` [:obj:`int`]
            A list of sigma hyperparameters to train trial models on. Should
            be ordered from smallest (at minimum ``2``) to largest.
        solver : :obj:`str`, optional
            The sGDML solver to use. Currently the only option is
            ``'analytic'``.
        valid_dataset : :obj:`dict`, optional
            The data set you want to use to validate models from (if it is
            different from ``dataset``).
        test_dataset : :obj:`dict`, optional
            The data set you want to use to test the final model from (if it is
            different from ``dataset``).
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
            trained to the energies. This is not necessary, but is found
            to be useful in higher order n-body models. Defaults to ``True``.
        use_cprsn : :obj:`bool`, optional
            Compresses the kernel matrix along symmetric degrees of freedom to
            try to reduce training time. Usually does not provide significant
            benefits. Defaults to ``False``.
        overwrite : :obj:`bool`, optional
            Whether or not to overwrite an already existing model and its
            training.
        max_processes : :obj:`int`, optional
            The maximum number of cores to use for the training process. Will
            automatically calculate if not specified.
        use_torch : :obj:`bool`, optional
            Whether or not to use torch and GPUs to train.
        """
        ###   CHANGED   ###
        _print_splash(max_processes=max_processes, use_torch=use_torch)
        ###   UNCHANGED   ###

        # This prepares and prints training, validation, and test data set
        # information. This is just informative.
        print(
            '\n' + ui.white_back_str(' STEP 0 ') + ' Dataset(s)\n' + '-' * MAX_PRINT_WIDTH
        )

        _, dataset_extracted = dataset
        _print_dataset_properties(dataset_extracted, title_str='Properties')

        if valid_dataset is None:
            valid_dataset = dataset
        else:
            _, valid_dataset_extracted = valid_dataset
            print()
            _print_dataset_properties(
                valid_dataset, title_str='Properties (validation)'
            )

            if not np.array_equal(dataset_extracted['z'], dataset_extracted['z']):
                raise ValueError(
                    'Atom composition or order in validation dataset does not match the one in bulk dataset.'
                )

        if test_dataset is None:
            test_dataset = dataset
        else:
            _, test_dataset_extracted = test_dataset
            _print_dataset_properties(test_dataset_extracted, title_str='Properties (test)')

            if not np.array_equal(dataset['z'], test_dataset_extracted['z']):
                raise ValueError(
                    'Atom composition or order in test dataset does not match the one in bulk dataset.'
                )
        
        ###   CHANGED   ###
        # Handles max_processes.
        if max_processes is not None:
            n_cores = int(self.call_para("n_cores") or 1)
            if n_cores == 0:
                n_cores = 1
            elif n_cores < 0:
                n_cores = os.cpu_count() + n_cores
            max_processes = n_cores
        self._max_processes = max_processes  # Used in self.create
        
        # Do not see a reason for user specification of these variables.
        task_dir = None
        n_inducing_pts_init = None
        interact_cut_off = None

        ui.print_step_title('STEP 1', 'Cross-validation task creation')
        # Convert the no_sym bool to the gdml bool sGDML uses
        gdml = not use_sym
        ###   UNCHANGED   ###
        task_dir = self.sgdml_create(
            dataset,
            valid_dataset,
            n_train,
            n_valid,
            sigs,
            gdml,
            use_E,
            use_E_cstr,
            use_cprsn,
            overwrite,
            max_processes,
            task_dir=task_dir,
            solver=solver,
            n_inducing_pts_init=n_inducing_pts_init,
            interact_cut_off=interact_cut_off,
            command='all',
            idxs_train=idxs_train,  # Added to specify which structures to train on
            idxs_valid=idxs_valid  # Added to specify validation structures.
        )

        ui.print_step_title('STEP 2', 'Training and validation')
        task_dir_arg = io.is_dir_with_file_type(task_dir, 'task')
        model_dir_or_file_path = sgdml_train(
            task_dir_arg, valid_dataset, overwrite, max_processes, use_torch,
        )

        model_dir_arg = io.is_dir_with_file_type(
            model_dir_or_file_path, 'model', or_file=True
        )

        ###   CHANGED   ###
        ui.print_step_title('STEP 3', 'Hyper-parameter selection')
        if self.write_json:
            model_file_name = self.sgdml_select(
                model_dir_arg, overwrite, max_processes, model_file
            )
        else:
            model_file_name = sgdml_select(
                model_dir_arg, overwrite, max_processes, model_file
            )
        ###   UNCHANGED   ###

        ui.print_step_title('STEP 4', 'Testing')
        model_dir_arg = io.is_dir_with_file_type(model_file_name, 'model', or_file=True)
        sgdml_test(
            model_dir_arg,
            test_dataset,
            n_test,
            overwrite=False,
            max_processes=max_processes,
            use_torch=use_torch,
        )

        print(
            '\n'
            + ui.color_str(
                '  DONE  ', fore_color=ui.BLACK,
                back_color=ui.GREEN, bold=True
            )
            + ' Training assistant finished successfully.'
        )
        print('         This is your model file: \'{}\''.format(model_file_name))

def _print_task_properties(  # Unchanged _print_task_properties from cli.py
    use_sym, use_cprsn, use_E, use_E_cstr, title_str='Task properties'
):

    print(ui.white_bold_str(title_str))

    # print('  {:<18} {}'.format('Solver:', ui.unicode_str('[solver name]')))
    # print('    {:<16} {}'.format('Tolerance:', '[tol]'))

    energy_fix_str = (
        (
            'kernel constraints (+E)'
            if use_E_cstr
            else 'global integration constant recovery'
        )
        if use_E
        else 'none'
    )
    print('  {:<16} {}'.format('Energy handling:', energy_fix_str))

    print(
        '  {:<16} {}'.format(
            'Symmetries:', 'include (sGDML)' if use_sym else 'ignore (GDML)'
        )
    )
    print(
        '  {:<16} {}'.format(
            'Compression:', 'requested' if use_cprsn else 'not requested'
        )
    )

def _check_update():  # Unchanged _check_update from cli.py

    try:
        from urllib.request import urlopen
    except ImportError:
        from urllib2 import urlopen

    base_url = 'http://www.quantum-machine.org/gdml/'
    url = '%supdate.php?v=%s' % (base_url, sgdml_version)

    can_update, must_update = '0', '0'
    latest_version = ''
    try:
        response = urlopen(url, timeout=1)
        can_update, must_update, latest_version = response.read().decode().split(',')
        response.close()
    except BaseException:
        pass

    return can_update == '1', latest_version

def _print_splash(max_processes=None, use_torch=False):  # Unchanged _print_splash from cli.py

    logo_str = r"""         __________  __  _____
   _____/ ____/ __ \/  |/  / /
  / ___/ / __/ / / / /|_/ / /
 (__  ) /_/ / /_/ / /  / / /___
/____/\____/_____/_/  /_/_____/"""

    can_update, latest_version = _check_update()

    version_str = sgdml_version
    version_str += (
        ' ' + ui.yellow_back_str(' Latest: ' + latest_version + ' ')
        if can_update
        else ''
    )

    # TODO: does this import test work in python3?
    max_processes_str = (
        ''
        if max_processes is None or max_processes >= mp.cpu_count()
        else ' [using {}]'.format(max_processes)
    )
    hardware_str = 'found {:d} CPU(s){}'.format(mp.cpu_count(), max_processes_str)

    if use_torch and _has_torch and torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        if num_gpu > 0:
            hardware_str += ' / {:d} GPU(s)'.format(num_gpu)

    logo_str_split = logo_str.splitlines()
    print('\n'.join(logo_str_split[:-1]))
    ui.print_two_column_str(logo_str_split[-1] + '  ' + version_str, hardware_str)

    # Print update notice.
    if can_update:
        print(
            '\n'
            + ui.yellow_back_str(' UPDATE AVAILABLE ')
            + '\n'
            + '-' * MAX_PRINT_WIDTH
        )
        print(
            'A new stable release version {} of this software is available.'.format(
                latest_version
            )
        )
        print(
            'You can update your installation by running \'pip install sgdml --upgrade\'.'
        )