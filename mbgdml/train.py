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
import sys
import logging
import traceback
import numpy as np
from mbgdml.data import mbModel
from mbgdml.data import dataSet

import multiprocessing as mp
from sgdml.train import GDMLTrain
from sgdml.cli import _print_dataset_properties
from sgdml.cli import train as sgdml_train
from sgdml.cli import select as sgdml_select
from sgdml.cli import test as sgdml_test
from sgdml.utils import ui as sgdml_ui
from sgdml.utils import io as sgdml_io
from sgdml.utils import perm
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

class mbGDMLTrain():
    """

    Attributes
    ----------
    dataset_path : :obj:`str`
        Path to data set.
    dataset_name : :obj:`str`
        Data set file name without extension.
    dataset : :obj:`dict`
        mbGDML data set for training.
    """

    def __init__(self):
        pass

    def load_dataset(self, dataset_path):
        """Loads a GDML dataset from npz format from specified path.
        
        Parameters
        ----------
        dataset_path : :obj:`str`
            Path to a npz GDML dataset of a single cluster size (e.g., two
            water molecules).
        """
        self.dataset_path = dataset_path
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.dataset = dataSet(dataset_path).asdict
    
    def _sgdml_create_task(
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
        from functools import partial

        if use_E and 'E' not in train_dataset:
            raise ValueError(
                'No energy labels found in dataset!\n'
                + 'By default, force fields are always reconstructed including the\n'
                + 'corresponding potential energy surface (this can be turned off).\n'
                + 'However, the energy labels are missing in the provided dataset.\n'
            )

        use_E_cstr = use_E and use_E_cstr

        if callback is not None:
            callback = partial(callback, disp_str='Hashing dataset(s)')
            callback(NOT_DONE)

        md5_train = sgdml_io.dataset_md5(train_dataset)
        md5_valid = sgdml_io.dataset_md5(valid_dataset)

        if callback is not None:
            callback(DONE)

        if callback is not None:
            callback = partial(
                callback, disp_str='Sampling training and validation subsets'
            )
            callback(NOT_DONE)

        # This function is only called if we have specified idxs_train.
        # We just do a check that idxs_train is not None.
        assert idxs_train is not None
        idxs_train = np.array(idxs_train)  # Insures it is an array.

        # Allow specifying of validation indices.
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

        if callback is not None:
            callback(DONE)

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

        if use_E:
            task['E_train'] = train_dataset['E'][idxs_train]

        # lat_and_inv = None
        # if 'lattice' in train_dataset:
        #     task['lattice'] = train_dataset['lattice']
        # 
        #     try:
        #         lat_and_inv = (task['lattice'], np.linalg.inv(task['lattice']))
        #     except np.linalg.LinAlgError:
        #         raise ValueError(  # TODO: Document me
        #             'Provided dataset contains invalid lattice vectors (not invertible). Note: Only rank 3 lattice vector matrices are supported.'
        #         )

        if 'r_unit' in train_dataset and 'e_unit' in train_dataset:
            task['r_unit'] = train_dataset['r_unit']
            task['e_unit'] = train_dataset['e_unit']

        if use_sym:
            n_train = R_train.shape[0]
            R_train_sync_mat = R_train
            if n_train > 1000:
                R_train_sync_mat = R_train[
                    np.random.choice(n_train, 1000, replace=False), :, :
                ]
                self.log.info(
                    'Symmetry search has been restricted to a random subset of 1000/{:d} training points for faster convergence.'.format(
                        n_train
                    )
                )

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
    
    def _sgdml_create(
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

        _print_task_properties(
            use_sym=not gdml, use_cprsn=use_cprsn, use_E=use_E,
            use_E_cstr=use_E_cstr
        )
        print()

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
            task_dir = sgdml_io.train_dir_name(
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
                if sgdml_io.is_task_dir_resumeable(
                    task_dir, dataset, valid_dataset, n_train, n_valid, sigs, gdml
                ):
                    log.info(
                        'Resuming existing hyper-parameter search in \'{}\'.'.format(
                            task_dir
                        )
                    )

                    # Get all task file names.
                    try:
                        _, task_file_names = sgdml_io.is_dir_with_file_type(task_dir, 'task')
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
                        callback=sgdml_ui.callback,
                    )  # template task
                # Our modified procedure to allow more control over training.
                else:
                    tmpl_task = self._sgdml_create_task(
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
                        callback=sgdml_ui.callback,
                        idxs_train=idxs_train,
                        idxs_valid=idxs_valid
                    )
            except BaseException:
                print()
                log.critical(traceback.format_exc())
                sys.exit()

        n_written = 0
        for sig in sigs:
            tmpl_task['sig'] = sig
            task_file_name = sgdml_io.task_file_name(tmpl_task)
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
    
    def _sgdml_all(
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
        """A slightly modified ``sGDML all`` function to add functionality.

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
        _print_splash(max_processes=max_processes, use_torch=use_torch)

        # This prepares and prints training, validation, and test data set
        # information. This is just informative.
        print(
            '\n' + sgdml_ui.white_back_str(' STEP 0 ') + ' Dataset(s)\n' + '-' * MAX_PRINT_WIDTH
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
        
        # Handles max_processes.
        if max_processes is not None:
            n_cores = int(self.call_para("n_cores") or 1)
            if n_cores == 0:
                n_cores = 1
            elif n_cores < 0:
                n_cores = os.cpu_count() + n_cores
            max_processes = n_cores
        self._max_processes = max_processes  # Used in _sgdml_create
        
        # Do not see a reason for user specification of these variables.
        task_dir = None
        n_inducing_pts_init = None
        interact_cut_off = None

        sgdml_ui.print_step_title('STEP 1', 'Cross-validation task creation')

        # Convert the no_sym bool to the gdml bool sGDML uses
        gdml = not use_sym
        task_dir = self._sgdml_create(
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

        sgdml_ui.print_step_title('STEP 2', 'Training and validation')
        task_dir_arg = sgdml_io.is_dir_with_file_type(task_dir, 'task')
        model_dir_or_file_path = sgdml_train(
            task_dir_arg, valid_dataset, overwrite, max_processes, use_torch,
        )

        model_dir_arg = sgdml_io.is_dir_with_file_type(
            model_dir_or_file_path, 'model', or_file=True
        )

        sgdml_ui.print_step_title('STEP 3', 'Hyper-parameter selection')
        model_file_name = sgdml_select(
            model_dir_arg, overwrite, max_processes, model_file
        )

        sgdml_ui.print_step_title('STEP 4', 'Testing')
        model_dir_arg = sgdml_io.is_dir_with_file_type(model_file_name, 'model', or_file=True)
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
            + sgdml_ui.color_str(
                '  DONE  ', fore_color=sgdml_ui.BLACK,
                back_color=sgdml_ui.GREEN, bold=True
            )
            + ' Training assistant finished successfully.'
        )
        print('         This is your model file: \'{}\''.format(model_file_name))

    def train(
        self, model_name, n_train, n_validate, n_test, solver='analytic',
        sigmas=tuple(range(2, 110, 10)), save_dir='.', use_sym=True, use_E=True,
        use_E_cstr=True, use_cprsn=False, idxs_train=None, idxs_valid=None,
        max_processes=None, overwrite=False, torch=False,
    ):
        """Trains and saves a GDML model.
        
        Parameters
        ----------
        model_name : :obj:`str`
            User-defined model name without the ``'.npz'`` file extension.
        n_train : :obj:`int`
            The number of training points to sample.
        n_validate : :obj:`int`
            The number of validation points to sample, without replacement.
        n_test : :obj:`int`
            The number of test points to test the validated GDML model.
        solver : :obj:`str`, optional
            The sGDML solver to use. Currently the only option is
            ``'analytic'``.
        sigmas : :obj:`list`, optional
            Kernel length scales (i.e., hyperparameters) to train and validate
            GDML models. Note, more length scales usually mean longer training
            times. Two is the minimum value. Defaults to
            ``list(range(2, 110, 10))``.
        save_dir : :obj:`str`, optional
            Path to train and save the mbGDML model. Defaults to current
            directory (``'.'``).
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
        idxs_train : :obj:`numpy.ndarray`
            The specific indices of structures to train the model on. If
            ``None`` will automatically sample the training data set.
        idxs_valid : :obj:`numpy.ndarray`
            The specific indices of structures to validate models on.
            If ``None``, structures will be automatically determined.
        max_processes : :obj:`int`, optional
            The maximum number of cores to use for the training process. Will
            automatically calculate if not specified.
        overwrite : :obj:`bool`, optional
            Overwrite existing files. Defaults to ``False``.
        torch : :obj:`bool`, optional
            Use PyTorch to enable GPU acceleration.
        """
        if idxs_train is not None:
            assert n_train == len(idxs_train)
            assert len(set(idxs_train)) == len(idxs_train)
        if idxs_valid is not None:
            assert n_validate == len(idxs_valid)
            assert len(set(idxs_valid)) == len(idxs_valid)
            
        if save_dir[-1] != '/':
            save_dir += '/'
        os.chdir(save_dir)

        # sGDML training routine.
        self._sgdml_all(
            (self.dataset_path, self.dataset),
            n_train,
            n_validate,
            n_test,
            sigmas,
            solver=solver,
            valid_dataset=None,
            test_dataset=None,
            use_sym=use_sym,
            use_E=use_E,
            use_E_cstr=use_E_cstr,
            use_cprsn=use_cprsn,
            overwrite=overwrite,
            max_processes=max_processes,
            use_torch=torch,
            idxs_train=idxs_train,
            idxs_valid=idxs_valid,
            model_file=model_name + '.npz',
            task_dir=None,
        )
        
        ## mbGDML modifications.
        # Adding additional mbGDML info to the model.
        new_model = mbModel()
        new_model.load(model_name + '.npz')
        
        # Adding mbGDML-specific modifications to model.
        new_model.add_modifications(self.dataset)

        # Adding many-body information if present in dataset.
        if 'mb' in self.dataset.keys():
            new_model.model['mb'] = int(self.dataset['mb'][()])
        if 'mb_models_md5' in self.dataset.keys():
            new_model.model['mb_models_md5'] = self.dataset['mb_models_md5']

        # Saving model.
        new_model.save(model_name, new_model.model, '.')

def _print_task_properties(
    use_sym, use_cprsn, use_E, use_E_cstr, title_str='Task properties'
):

    print(sgdml_ui.white_bold_str(title_str))

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

def _check_update():

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

def _print_splash(max_processes=None, use_torch=False):

    logo_str = r"""         __________  __  _____
   _____/ ____/ __ \/  |/  / /
  / ___/ / __/ / / / /|_/ / /
 (__  ) /_/ / /_/ / /  / / /___
/____/\____/_____/_/  /_/_____/"""

    can_update, latest_version = _check_update()

    version_str = sgdml_version
    version_str += (
        ' ' + sgdml_ui.yellow_back_str(' Latest: ' + latest_version + ' ')
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
    sgdml_ui.print_two_column_str(logo_str_split[-1] + '  ' + version_str, hardware_str)

    # Print update notice.
    if can_update:
        print(
            '\n'
            + sgdml_ui.yellow_back_str(' UPDATE AVAILABLE ')
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