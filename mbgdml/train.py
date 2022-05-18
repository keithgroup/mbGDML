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
from mbgdml.data import mbModel
from mbgdml.data import dataSet
from mbgdml._train import sGDMLTraining

class mbGDMLTrain(sGDMLTraining):
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
    
    def train(
        self, model_name, n_train, n_validate, n_test, solver='analytic',
        sigmas=tuple(range(2, 400, 30)), save_dir='.', use_sym=True, use_E=True,
        use_E_cstr=True, use_cprsn=False, idxs_train=None, idxs_valid=None,
        max_processes=None, overwrite=False, torch=False, write_json=False
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
            times. Two is the minimum value and can get as large as several
            hundred. Defaults to ``tuple(range(2, 400, 30))``.
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
        write_json : :obj:`bool`, optional
            Write a JSON file containing information about the training job.
        """
        if write_json:
            import json
            self.write_json = True
            self.job_json = {
                'model': {},
                'testing': {},
                'validation': {
                    'sigmas': [],
                    'energy_mae': [],
                    'energy_rmse': [],
                    'forces_mae': [],
                    'forces_rmse': [],
                    'idxs': [],
                },
                'training': {'idxs': []},
            }
        else:
            self.write_json = False
        
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
        self.sgdml_all(  # TODO:
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
        if model_name[-4:] == '.npz':
            model_name = model_name[:-4]
        new_model.load(model_name + '.npz')
        
        # Adding mbGDML-specific modifications to model.
        new_model.add_modifications(self.dataset)

        # Including final JSON stuff and writing.
        if self.write_json:
            self.job_json['training']['idxs'] = new_model.model['idxs_train'].tolist()
            self.job_json['validation']['idxs'] = new_model.model['idxs_valid'].tolist()
            e_err = new_model.model['e_err'][()]
            f_err = new_model.model['f_err'][()]
            self.job_json['testing']['n_test'] = int(new_model.model['n_test'][()])
            self.job_json['testing']['energy_mae'] = e_err['mae']
            self.job_json['testing']['energy_rmse'] = e_err['rmse']
            self.job_json['testing']['forces_mae'] = f_err['mae']
            self.job_json['testing']['forces_rmse'] = f_err['rmse']

            self.job_json['model']['sigma'] = int(new_model.model['sig'][()])
            self.job_json['model']['n_symm'] = len(new_model.model['perms'])

            from cclib.io.cjsonwriter import JSONIndentEncoder
            
            json_string = json.dumps(
                self.job_json, cls=JSONIndentEncoder, indent=4
            )

            with open('log.json', 'w') as f:
                f.write(json_string)

        # Adding many-body information if present in dataset.
        if 'mb' in self.dataset.keys():
            new_model.model['mb'] = int(self.dataset['mb'][()])
        if 'mb_models_md5' in self.dataset.keys():
            new_model.model['mb_models_md5'] = self.dataset['mb_models_md5']

        # Saving model.
        new_model.save(model_name, new_model.model, '.')
