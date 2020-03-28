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


def create_gdml_xyz(calc_output_dir, gdml_dataset_dir):
    """Writes xyz files for GDML datasets.

    Used as a driver for the _PartitionCalcOutput class that iterates over all
    partitions for a solvent. Writes and organizes GDML xyz files according
    to solvent, partition size, temperature, and MD iteration.
    
    Args:
        calc_output_dir (str): Path to folder that contains computational
            chemistry output files. Usually for an entire solvent.
        gdml_dataset_dir (str): Path that contains all GDML dataset files.
    """
    
    calc_output_dir = utils.norm_path(calc_output_dir)
    all_out_files = utils.get_files(calc_output_dir, 'out')

    for out_file in all_out_files:
        print('Writing the GDML file for ' + out_file.split('/')[-1] + ' ...')
        calc = PartitionCalcOutput(out_file)
        calc.write_gdml_data(gdml_dataset_dir)

def combine_gdml_xyz(gdml_partition_dir, write_dir):
    """Combines GDML xyz files.
    
    Finds all files labeled with 'gdml.xyz' (defined in 
    _PartitionCalcOutput._write_gdml_data) in a user specified directory
    and combines them. Typically used on a single partition size (e.g.,
    monomer, dimer, trimer, etc.) to represent the complete dataset of that
    partition size.

    Args:
        gdml_partition_dir (str): Path to directory containing GDML xyz files.
            Typically to a directory containing only a single partition size
            of a single solvent.
        write_dir (str): Path to the directory where partition-size GDML xyz
            files will be written. Usually the solvent directory in the 
            gdml-dataset directory.
    """

    gdml_partition_dir = utils.norm_path(gdml_partition_dir)
    write_dir = utils.norm_path(write_dir)

    # Gets all GDML files within a partition.
    all_gdml_files = utils.natsort_list(
        utils.get_files(gdml_partition_dir, 'gdml.xyz')
    )

    # Gets naming information from one of the GDML files.
    gdml_file_example = all_gdml_files[0].split('/')[-1][:-4].split('-')
    gdml_partition_size = gdml_partition_size_names[
        len(gdml_file_example[0]) - 1
    ]
    gdml_cluster = gdml_file_example[1]
    gdml_partition_file = write_dir \
                          + '-'.join([gdml_cluster, gdml_partition_size]) \
                          + '-gdml-dataset.xyz'
    
    # TODO check that each file has the same number of atoms.
    # TODO write data as npz instead of extended xyz file.
    # Writes all partitions to a single extended-xyz GDML file.
    open(gdml_partition_file, 'w').close()
    for partition in all_gdml_files:
        print('Writing the ' + partition.split('/')[-1] + ' file.')
        with open(partition, 'r') as partition_file:
            partition_data = partition_file.read()
        with open(gdml_partition_file, 'a+') as gdml_file:
            gdml_file.write(partition_data)

def gdml_xyz_datasets(gdml_solvent_dir):
    """Creates GDML xyz datasets for all solvent partitions.

    Finds all GDML xyz files (by searching for files containing 'gdml.xyz)
    and combines them into their respective partition-size GDML xyz file.
    These files are written for documentation purposes.
    
    Args:
        gdml_solvent_dir (str): Path to directory containing all GDML xyz files
            for a solvent. New files are written to this directory as well.
    """
    
    gdml_solvent_dir = utils.norm_path(gdml_solvent_dir)
    partition_dirs = os.listdir(gdml_solvent_dir)

    # Gets all directories which should contain all GDML files of a single
    # partition size.
    file_index = 0
    while file_index < len(partition_dirs):
        partition_dirs[file_index] = gdml_solvent_dir \
                                     + partition_dirs[file_index]
        file_index += 1
    partition_dirs = [item for item in partition_dirs if os.path.isdir(item)]
    
    # Combines and writes all partition-size GDML files.
    for size in partition_dirs:
        combine_gdml_xyz(size, gdml_solvent_dir)
