# MIT License
# 
# Copyright (c) 2020-2021, Alex M. Maldonado
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
import re
import subprocess
import numpy as np
from mbgdml import utils
from mbgdml.data import mbModel
from mbgdml.data import dataSet

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
            Path to NumPy npz file representing a GDML dataset of a single 
            cluster size (e.g., two water molecules).
        """
        self.dataset_path = dataset_path
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.dataset = dataSet(dataset_path).dataset

    def train_GDML(
        self, model_name, num_train, num_validate, num_test,
        sigma_range='2:10:100', save_dir='.', overwrite=False, torch=False,
        other_options=[]
    ):
        """Trains a GDML model through the command line interface.
        
        Parameters
        ----------
        model_name : :obj:`str`
            User-defined model name.
        num_train : :obj:`int`
            The number of training points to sample.
        num_validate : :obj:`int`
            The number of validation points to sample, without replacement.
        num_test : :obj:`int`
            The number of test points to test the validated GDML model.
        sigma_range : :obj:`str`, optional
            Range of kernel length scales to train and validate GDML values 
            one. Format is `<start>:<interval>:<stop>`. Note, the more length 
            scales the longer the training time. Two is the minimum value.
        save_dir : :obj:`str`, optional
            Path to train and save the mbGDML model. Defaults to current
            directory.
        overwrite : :obj:`bool`, optional
            Overwrite existing files. Defaults to false.
        torch : :obj:`bool`, optional
            Use PyTorch to enable GPU acceleration.
        other_options : :obj:`list` [:obj:`str`]
            Other command line arguments to control sGDML training. For example,
            ``['--E_cstr']``
        """
        # TODO add remaining options for CLI sGDML as optional arguments.

        if save_dir[-1] != '/':
            save_dir += '/'
        if save_dir != './':
            os.chdir(save_dir)

        # Makes log file name.
        log_name = model_name + '.log'

        # sGDML command to be executed.
        sGDML_command = [
            'sgdml', 'all',
            str(self.dataset_path),
            str(num_train), str(num_validate), str(num_test),
            '-s', sigma_range,
            '--model_file', model_name
        ] + other_options

        if overwrite:
            sGDML_command.append('-o')
        else:
            if os.path.exists(model_name + '.npz'):
                print(f'{model_name}.npz already exists, and overwrite is False.')
                return
        if torch:
            sGDML_command.append('--torch')
        
        print('Running sGDML training on ' + self.dataset_name + ' ...')
        
        completed_process = subprocess.run(
            sGDML_command,
            capture_output=True
        )
        
        cli_output = completed_process.stdout.decode('ascii')
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        log_output = ansi_escape.sub('', cli_output)

        if 'Training assistant finished sucessfully.' not in cli_output:
            print(completed_process.stderr.decode('ascii'))
            return
    
        print(cli_output)
        with open(log_name, 'w') as log:
            log.write(log_output)
        
        # Adding additional mbGDML info to the model.
        new_model = mbModel()
        new_model.load(model_name + '.npz')
        
        # Adding mbGDML-specific modifications to model.
        new_model.add_modifications(self.dataset)

        # Adding many-body information if present in dataset.
        if 'mb' in self.dataset.keys():
            new_model.model['mb'] = int(self.dataset['mb'][()])
            new_model.model['mb_models_md5'] = self.dataset['mb_models_md5']
        
        # Adding sigma selection to log (not apparent because of removed
        # formatting.
        model_sigma = new_model.model['sig'][()]
        with open(log_name, 'a') as log:
            log.write(f'Model sigma: {model_sigma}\n')

        # Saving model.
        new_model.save(model_name, new_model.model, save_dir)

