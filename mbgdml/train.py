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
from mbgdml.solvents import solvent
from mbgdml.data import PartitionCalcOutput

gdml_partition_size_names = [
    'monomer', 'dimer', 'trimer', 'tetramer', 'pentamer'
]

class MBGDMLTrain():
    """[summary]

    Attributes:
    """

    def __init__(self, dataset_dir, model_dir):
        self.dataset_dir = utils.norm_path(dataset_dir)
        self.model_dir = utils.norm_path(model_dir)
        self.get_datasets()
    
    def get_datasets(self):
        """Finds all datasets with '.npz' extension in dataset directory.
        """
        self.dataset_paths = utils.get_files(self.dataset_dir, '.npz')
        # TODO load each dataset and change dataset_paths to all_datasets
        # as dict and add keys as partition size, values as path, fingerprint,
        # dataset size, etc.
    
    def load_dataset(self, dataset_path):
        """Loads a GDML dataset from npz format from specified path.
        
        Args:
            dataset_path (str): Path to stored numpy arrays representing a GDML
                dataset of a single solvent partition size.
        """
        self.dataset = np.load(dataset_path)
    
    def train_GDML(self, dataset_path, num_train, num_validate, num_test,
                   sigma_range='2:10:100'):
        """Trains a GDML model through the command line interface.
        
        Args:
            dataset_path (str): Path to GDML dataset to be used for training,
                validation, and testing. Total number of data points must be
                greater than the sum of training, validation, and testing data
                points.
            num_train (int): The number of training points to sample.
            num_validate (int): The number of validation points to sample,
                without replacement.
            num_test (int): The number of test points to test the validated
                GDML model.
            sigma_range (str, optional): Range of kernel length scales to train
                and validate GDML values one. Format is
                '<start>:<interval>:<stop>'. Note, the more length scales the
                longer the training time.
        """
        # TODO add remaining options for CLI sGDML as optional arguments.
        
        # Makes log file name.
        dataset_name = dataset_path.split('/')[-1].split('.')[0]
        log_name = ''.join([dataset_name, '-train.log'])

        # Prepares directory to save in
        save_dir = utils.norm_path(self.model_dir + 'gdml')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        os.chdir(save_dir)

        # Executing GDMLTrain via command line interface.
        # TODO look into switching to python interface
        sGDML_command = [
            'sgdml', 'all',
            str(dataset_path),
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
            log.write('\n\n The following input was used for this file:')
            log.write(' '.join(sGDML_command))

