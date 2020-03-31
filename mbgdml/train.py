# MIT License
# 
# Copyright (c) 2020, Alex M. Maldonado
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

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
import subprocess
import sys
import re
import numpy as np

from cclib.io import ccread
from cclib.parser.utils import convertor

from periodictable import elements

from sgdml.train import GDMLTrain
import sgdml.cli as sGDML_cli

from mbgdml import utils
from mbgdml.data import mbGDMLModel

class MBGDMLTrain():


    def __init__(self):
        pass

    def load_dataset(self, dataset_path):
        """Loads a GDML dataset from npz format from specified path.
        
        Args:
            dataset_path (str): Path to stored numpy arrays representing a GDML
                dataset of a single solvent partition size.
        """
        self.dataset_path = dataset_path
        self.dataset = np.load(dataset_path)
    

    def _organization_dirs(self, model_dir):
        """Determines where a model should be saved.

        The model will be written in a directory that depends on the
        parent solvent cluster (e.g. 4MeOH).
        
        Args:
            model_dir (str): Path to a common directory for GDML models.

        Notes:
            Requires the 'dataset' attribute.
            Sets the 'model_dir' attribute for writing GDML models.
        """

        model_dir = utils.norm_path(model_dir)

        if not hasattr(self, 'dataset'):
            raise AttributeError('There is currently no dataset loaded.')
        
        # Parsing information from the dataset_name.
        parent_cluster, partition_size, _ = self.dataset.f.name[()].split('-')

        # Preparing directories.
        if str(self.dataset.f.system[()]) == 'solvent':
            model_solvent_dir = utils.norm_path(
                model_dir + parent_cluster
            )
            try:
                os.makedirs(model_solvent_dir)
            except FileExistsError:
                pass
            os.chdir(model_solvent_dir)
        
        self.model_dir = model_solvent_dir
    

    def train_GDML(self, model_dir, num_train, num_validate, num_test,
                   sigma_range='2:10:100'):
        """Trains a GDML model through the command line interface.
        
        Args:
            model_dir (str): Path to a common directory for GDML models.
            num_train (int): The number of training points to sample.
            num_validate (int): The number of validation points to sample,
                without replacement.
            num_test (int): The number of test points to test the validated
                GDML model.
            sigma_range (str, optional): Range of kernel length scales to train
                and validate GDML values one. Format is
                '<start>:<interval>:<stop>'. Note, the more length scales the
                longer the training time. Two is the minimum value.
        """
        # TODO add remaining options for CLI sGDML as optional arguments.

        # Makes log file name.
        dataset_name = self.dataset_path.split('/')[-1].split('.')[0]
        log_name = ''.join([dataset_name.replace('dataset', 'model'),
                            '-train.log'])

        # Prepares directory to save in
        self._organization_dirs(model_dir)

        sGDML_command = [
            'sgdml', 'all',
            str(self.dataset_path),
            str(num_train), str(num_test), str(num_validate),
            '-s', str(sigma_range)
        ]
        print('Running sGDML training on ' + dataset_name + ' ...')
        subprocess.run(
            sGDML_command,
            stdout=open(log_name, 'w'),
            encoding='unicode',
            bufsize=1
        ) # TODO Fix encoding

        # Adding input information to log file
        with open(log_name, 'a') as log:
            log.write('\n\n The following command was used for this file: ')
            log.write(' '.join(sGDML_command))
            log.write('\n Note, the model name will have "model" instead')
            log.write('of "dataset".')
        
        # Adding additional mbGDML info to the model.
        model = mbGDMLModel()
        model.get_model_name(log_name)
        model.load_model(model.name + '.npz')
        os.remove(model.name + '.npz')

        # Changing model name.
        model.name = model.name.replace('-dataset-', '-model-')

        # Adding system info.
        model.add_system_info(model.base_vars)

        # Adding many-body information if present in dataset.
        if 'mb' in self.dataset.keys():
            model.add_manybody_info(
                int(self.dataset.f.mb_order[()]), model.base_vars
            )

        # Saving model.
        model.save(model.name, model.base_vars, self.model_dir, False)

