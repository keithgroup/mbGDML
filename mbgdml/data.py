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
import re
import numpy as np
from cclib.io import ccread
from cclib.parser.utils import convertor
from sgdml import __version__
from sgdml.utils import io as sgdml_io
from mbgdml import utils
import mbgdml.solvents as solvents
from mbgdml.predict import mbGDMLPredict

class _mbGDMLData():

    def __init__(self):
        pass

    def get_system_info(self, atoms):
        """Describes the dataset system.
        
        Args:
            atoms (list): Atomic numbers of all atoms in the system. The atoms
                are repeated; for example, water is ['H', 'H', 'O'].
        """
        self.system_info = solvents.system_info(atoms)
    
    def add_system_info(self, base_vars):
        """Adds information about the system to the model.
        
        Args:
            base_vars (dict): Custom data structure that contains all
                information for a GDML dataset.
        
        Returns:
            dict: An updated GDML dataset with additional information regarding
                the system.
        
        Notes:
            If the system is a solvent, the 'solvent' name and 'cluster_size'
            is included.
        """

        if not hasattr(self, 'system_info'):
            self.get_system_info(base_vars['z'].tolist())
        
        base_vars['system'] = self.system_info['system']
        if base_vars['system'] is 'solvent':
            base_vars['solvent'] = self.system_info['solvent_name']
            base_vars['cluster_size'] = self.system_info['cluster_size']
        
        return base_vars

    def save(self, name, base_vars, save_dir, dataset):
        """General save function for GDML datasets and models.
        
        Args:
            name (str): Name of the file to be saved not including the
                extension.
            base_vars (dict): Base variables for dataset or model.
            save_dir (str): Directory to save the file.
            dataset (bool): Is the file a dataset? Controls whether the md5 of
                the file is updated.
        """

        save_dir = utils.norm_path(save_dir)
        if dataset:
            base_vars['md5'] = sgdml_io.dataset_md5(base_vars)
        save_path = save_dir + name + '.npz'
        np.savez_compressed(save_path, **base_vars)

class mbGDMLModel(_mbGDMLData):
    
    def __init__(self):
        pass

    def load(self, model_path):
        """Loads GDML model.
        
        Args:
            model_path (str): Path to GDML model.
        """
        self.model = np.load(model_path, allow_pickle=True)
        self.base_vars = dict(self.model)
    
    def get_model_name(self, log_path):
        """Retrives GDML model's name from log file.
        
        Args:
            log_path (str): Path to the log file.
        """

        for line in reversed(list(open(log_path))):
            if 'This is your model file' in line:
                self.name = line.split(':')[-1][2:-6]
                break
    
    def add_manybody_info(self, mb_order):
        """Adds many-body (mb) information to GDML model.
        
        Args:
            mb_order (int): The max order of many-body predictions removed
                from the dataset.
        """
        if not hasattr(self, 'base_vars'):
            raise AttributeError('There is no model loaded.')

        self.base_vars['mb'] = mb_order
        

class mbGDMLDataset(_mbGDMLData):

    def __init__(self):
        pass
    

    def _organization_dirs(
        self,
        gdml_data_dir
    ):
        """Determines where a dataset should be saved.

        The dataset will be written in a directory that depends on the
        parent solvent cluster (e.g. 4MeOH), partition size (e.g., dimer),
        MD temperature (e.g. 300K), and MD iteration (e.g. 1).
        
        Args:
            gdml_data_dir (str): Path to a common directory for GDML datasets.

        Notes:
            Requires the 'dataset_name' attribute.
            Sets the 'gdml_file_path' attribute for writing GDML datasets.
        """

        gdml_data_dir = utils.norm_path(gdml_data_dir)

        if not hasattr(self, 'dataset_name'):
            raise AttributeError('There is no "dataset_name" attribute.')
        
        # Parsing information from the partition dataset_name.
        partition_label, parent_cluster_label, \
        md_temp, md_iter, _ = self.dataset_name.split('-')

        # Preparing directories.
        if self.system_info['system'] == 'solvent':
            # /path/to/gdml-datasets/solventlabel/partitionsize/temp/iteration
            gdml_solvent_dir = utils.norm_path(
                gdml_data_dir + parent_cluster_label
            )
            gdml_partition_size_dir = utils.norm_path(
                gdml_solvent_dir + str(self.system_info['cluster_size']) + 'mer'
            )
            gdml_temp_dir = utils.norm_path(
                gdml_partition_size_dir + str(md_temp)
            )
            gdml_iter_dir = utils.norm_path(
                gdml_temp_dir + str(md_iter)
            )
            all_dir = [gdml_solvent_dir, gdml_partition_size_dir,
                       gdml_temp_dir, gdml_iter_dir]
            for directory in all_dir:
                try:
                    os.chdir(directory)
                except:
                    os.mkdir(directory)
                    os.chdir(directory)
        
        # Writing GDML file.
        self.gdml_file_path = gdml_iter_dir + self.dataset_name

    
    def load(self, dataset_path):
        self.dataset = np.load(dataset_path)
        self.base_vars = dict(self.dataset)


    def partition_dataset_name(self, partition_label, cluster_label,
                               md_temp, md_iter):
        """Automates and standardizes partition datasets names.
        
        Args:
            partition_label (str): Identifies what solvent molecules are in
                the partition.
            cluster_label (str): The label identifying the parent solvent
                cluster of the partition.
            md_temp (str): Set point temperautre for the MD thermostat.
            md_iter (str): Identifies the iteration of the MD iteration.
        """
        
        self.dataset_name =  '-'.join([
            partition_label, cluster_label, md_temp, md_iter,
            'dataset'
        ])


    def create_dataset(
        self,
        gdml_data_dir,
        dataset_name,
        atoms,
        coords,
        energies,
        forces,
        e_units,
        e_units_calc,
        r_units_calc,
        theory='unknown',
        gdml_r_units='Angstrom',
        gdml_e_units='kcal/mol',
        write=True
    ):
        """Creates and writes GDML dataset.
        
        Args:
            gdml_data_dir (str): Path to common GDML dataset directory. This
                is above the solvent direcotry.
            dataset_name (str): The name to label the dataset.
            atoms (np.ndarray): A (n,) array containing n atomic numbers.
            coords (np.ndarray): A (m, n, 3) array containing the atomic
                coordinates of n atoms of m MD steps.
            energies (np.ndarray): A (m, n) array containing the energies of
                n atoms of m MD steps. cclib stores these in the units of eV.
            forces (np.ndarray): A (m, n, 3) array containing the atomic forces
                of n atoms of m MD steps. Simply the negative of grads.
            e_units (str): The energy units of the passed energies np.ndarray.
            e_units_calc (str): The units of energies reported in the partition
                calculation output file. This is used to converted energies and
                forces. Options are 'eV', 'hartree', 'kcal/mol', and 'kJ/mol'.
            r_units_calc (str): The units of the coordinates in the partition
                calculation output file. This is only used convert forces if
                needed. Options are 'Angstrom' or 'bohr'.
            theory (str, optional): The level of theory and basis set used
                for the partition calculations. For example, 'MP2.def2-TZVP.
                Defaults to 'unknown'.
            gdml_r_units (str, optional): Desired coordinate units for the GDML
                dataset. Defaults to 'Angstrom'.
            gdml_e_units (str, optional): Desired energy units for the GDML dataset.
                Defaults to 'kcal/mol'.
            write (bool, optional): Whether or not the dataset is written to
                disk. Defaults to True.
        """

        # Preparing energies in e_units.
        if e_units != gdml_e_units:
            energies = []
            for energy in energies:
                energies.append(
                    convertor(energy, e_units, gdml_e_units)
                )
            energies = np.array(energies)

        # Converting forces.
        # cclib does not convert gradients (or forces), this is where
        # the energy and coordinates units come into play.
        forces = forces * (convertor(1, e_units_calc, gdml_e_units) \
                           / convertor(1, r_units_calc, gdml_r_units))

        if not hasattr(self, 'system_info'):
            self.get_system_info(atoms.tolist())

        # sGDML variables.
        base_vars = {
            'type': 'd',  # Designates dataset or model.
            'code_version': __version__,  # sGDML version.
            'name': dataset_name,  # Name of the output file.
            'theory': theory,  # Theory used to calculate the data.
            'z': atoms,  # Atomic numbers of all atoms in system.
            'R': coords,  # Cartesian coordinates.
            'r_unit': gdml_r_units,  # Units for coordinates.
            'E': energies,  # Energy of the structure.
            'e_unit': gdml_e_units,  # Units of energy.
            'E_min': np.min(energies.ravel()),  # Energy minimum.
            'E_max': np.max(energies.ravel()),  # Energy maximum.
            'E_mean': np.mean(energies.ravel()),  # Energy mean.
            'E_var': np.var(energies.ravel()),  # Energy variance.
            'F': forces,  # Atomic forces for each atom.
            'F_min': np.min(forces.ravel()),  # Force minimum.
            'F_max': np.max(forces.ravel()),  # Force maximum.
            'F_mean': np.mean(forces.ravel()),  # Force mean.
            'F_var': np.var(forces.ravel())  # Force variance.
        }

        # mbGDML variables.
        base_vars = self.add_system_info(base_vars)

        base_vars['md5'] = sgdml_io.dataset_md5(base_vars)
        self.base_vars = base_vars

        # Writes dataset.
        if write:
            self._organization_dirs(gdml_data_dir)
            dataset_path = self.gdml_file_path + '.npz'
            np.savez_compressed(dataset_path, **base_vars)
    
    def print(self):

        if not hasattr(self, 'base_vars'):
            raise AttributeError('Please load a dataset first.')
        
        R = self.base_vars['R']
        E = self.base_vars['E']
        F = self.base_vars['F']

        num_config = R.shape[0]
        for config in range(num_config):
            print(f'-----Configuration {config}-----')
            print(f'Energy: {E[config][()]} kcal/mol')
            print(f'Forces:\n{F[config]}')

    
    def mb_dataset(self, nbody, models_dir):
        
        if not hasattr(self, 'base_vars'):
            raise AttributeError('Please load a dataset first.')

        nbody_index = 1
        while nbody_index < nbody:

            # Gets model.
            if nbody_index == 1:
                model_paths = utils.get_files(models_dir, '-1mer-')
            else:
                search_string = ''.join(['-', str(nbody_index), 'body-'])
                model_paths = utils.get_files(models_dir, search_string)

            # Removes logs that are found as well.
            model_paths = [path for path in model_paths if '.npz' in path]

            if len(model_paths) == 1:
                model_path = model_paths[0]
            else:
                raise ValueError(
                    f'There seems to be multiple {nbody_index}body models.'
                )
            
            # Removes n-body contributions.
            print(f'Removing {nbody_index}body contributions ...')
            nbody_model = mbGDMLModel()
            nbody_model.load(model_path)
            predict = mbGDMLPredict()
            self.base_vars = predict.remove_nbody(
                self.base_vars, nbody_model.model
            )

            nbody_index += 1

        # Removes old dataset attribute.
        if hasattr(self, 'dataset'):
            delattr(self, 'dataset')




class PartitionCalcOutput:
    """Quantum chemistry output file for all MD steps of a single partition.

    Output file that contains electronic energies and gradients of the same
    partition from a single MD trajectory. For a single dimer partition of a
    n step MD trajectory would have n coordinates, single point energies,
    and gradients.

    Args:
        output_path (str): Path to computational chemistry output file that
            contains energies and gradients (preferably ab initio) of all
            MD steps of a single partition.

    Attributes:
        output_file (str): Path to quantum chemistry output file for a
            partition.
        output_name (str): The name of the quantum chemistry output file
            (no extension).
        cluster (str): The label identifying the parent solvent cluster of
            the partition.
        temp (str): Set point temperautre for the MD thermostat.
        iter (str): Identifies the iteration of the MD iteration.
        partition (str): Identifies what solvent molecules are in the partition.
        partition_size (int): The number of solvent molecules in the partition.
        cclib_data (obj): Contains all data parsed from output file.
        atoms (np.array): A (n,) array containing n atomic numbers.
        coords (np.array): A (m, n, 3) array containing the atomic coordinates
            of n atoms of m MD steps.
        grads (np.array): A (m, n, 3) array containing the atomic gradients
            of n atoms of m MD steps. cclib does not change the unit, so the
            units are whatever is printed in the output file.
        forces (np.array): A (m, n, 3) array containing the atomic forces
            of n atoms of m MD steps. Simply the negative of grads.
        energies (np.array): A (m, n) array containing the energies of n atoms
            of m MD steps. cclib stores these in the units of eV.
        system (str): From Solvent class and designates the system. Currently
            only 'solvent' is implemented.
        solvent_info (dict): If the system is 'solvent', contains information
            about the system and solvent. 'solvent_name', 'solvent_label',
            'solvent_molec_size', and 'cluster_size'.
    """

    def __init__(self, output_path):
        self.output_file = output_path
        self._get_label_info()
        self.cclib_data = ccread(self.output_file)
        self._get_gdml_data()
        self.system_info = solvents.system_info(self.atoms.tolist())
    
    def _get_label_info(self):
        """Gets info from output file name.

        Output file should be labeled in the following manner:
        'out-NumSolv-temperature-iteration-partition.out'.
            'NumSolv' tells us the original solvent cluster size (e.g., 4MeOH);
            'temperature' is the set point for the thermostat (e.g., 300K);
            'iteration' identifies the MD simulation (e.g., 2);
            'partition' identifies what solvent molecules are in the partition
                (e.g., CD).
        A complete example would be 'out-4MeOH-300K-2-CD.out'.
        """
        self.output_name = self.output_file.split('/')[-1].split('.')[0]
        split_label = self.output_name.split('-')
        self.cluster = str(split_label[1])
        self.temp = str(split_label[2])
        self.iter = str(split_label[3])
        self.partition = str(split_label[4].split('.')[0])
        self.partition_size = int(len(self.partition))

    
    def _get_gdml_data(self):
        """Parses GDML-relevant data from partition output file.
        """
        try:
            self.atoms = self.cclib_data.atomnos
            self.coords = self.cclib_data.atomcoords
            self.grads = self.cclib_data.grads
            self.forces = np.negative(self.grads)
            if hasattr(self.cclib_data, 'mpenergies'):
                self.energies = self.cclib_data.mpenergies
            elif hasattr(self.cclib_data, 'scfenergies'):
                self.energies = self.cclib_data.scfenergies
            else:
                raise KeyError
        except:
            print('Something happened while parsing output file.')
            print('Please check ' + str(self.output_name) + ' output file.')
        
        # Reformats energies.
        energies = []
        for energy in self.energies:
            energies.append(convertor(energy[0], 'eV', 'kcal/mol'))
        self.energies = np.array(energies)
        self.energy_units = 'kcal/mol'


def create_datasets(calc_output_dir, dataset_dir, r_units_calc, e_units_calc,
                    theory='unknown', r_units='Angstrom', e_units='kcal/mol'):
    """Writes solvent partition datasets for GDML.

    Used as a driver for the PartitionCalcOutput class that iterates over all
    partitions for a solvent. Writes and organizes GDML xyz files according
    to solvent, partition size, temperature, and MD iteration.
    
    Args:
        calc_output_dir (str): Path to folder that contains computational
            chemistry output files. Usually for an entire solvent.
        dataset_dir (str): Path that contains all GDML dataset files.

    Notes:
        The partition calculation output files must have 'out' somewhere in
        the file name (or extentions).
    """
    
    calc_output_dir = utils.norm_path(calc_output_dir)
    all_out_files = utils.get_files(calc_output_dir, 'out')

    for out_file in all_out_files:
        print('Writing GDML dataset for ' + out_file.split('/')[-1] + ' ...')
        partition_calc = PartitionCalcOutput(out_file)
        dataset = mbGDMLDataset()
        dataset.partition_dataset_name(
            partition_calc.partition,
            partition_calc.cluster,
            partition_calc.temp,
            partition_calc.iter,
        )
        dataset.create_dataset(
            dataset_dir,
            dataset.dataset_name,
            partition_calc.atoms,
            partition_calc.coords,
            partition_calc.energies,
            partition_calc.forces,
            'kcal/mol',
            'hartree',
            'bohr',
            theory='MP2.def2-TZVP',
        )


def combine_datasets(partition_dir, write_dir):
    """Combines GDML datasets.
    
    Finds all files labeled with 'dataset.npz' (defined in 
    PartitionCalcOutput.create_dataset) in a user specified directory
    and combines them. Typically used on a single partition size (e.g.,
    monomer, dimer, trimer, etc.) to represent the complete dataset of that
    partition size.

    Args:
        partition_dir (str): Path to directory containing GDML datasets.
            Typically to a directory containing only a single partition size
            of a single solvent.
        write_dir (str): Path to the directory where the partition-size GDML
            dataset will be written. Usually the solvent directory in the 
            gdml-dataset directory.
    """

    # Normalizes directories.
    partition_dir = utils.norm_path(partition_dir)
    write_dir = utils.norm_path(write_dir)

    # Gets all GDML datasets within a partition directory.
    all_dataset_paths = utils.natsort_list(
        utils.get_files(partition_dir, 'dataset.npz')
    )

    # Prepares initial combined dataset from the first dataset found.
    dataset = np.load(all_dataset_paths[0])
    base_vars = dict(dataset)
    del dataset
    original_name = str(base_vars['name'][()])
    parent_label = original_name.split('-')[1]
    size_label = ''.join([str(base_vars['cluster_size']), 'mer'])
    base_vars['name'][()] = '-'.join([parent_label, size_label, 'dataset'])

    # Adds the remaining datasets to the new dataset.
    index_dataset = 1
    while index_dataset < len (all_dataset_paths):
        dataset_add = np.load(all_dataset_paths[index_dataset])
        print('Adding %s dataset to %s ...' % (dataset_add.f.name[()],
              base_vars['name'][()]))

        # Ensuring the datasets are compatible.
        try:
            assert np.array_equal(base_vars['type'], dataset_add.f.type)
            assert np.array_equal(base_vars['z'], dataset_add.f.z)
            assert np.array_equal(base_vars['r_unit'], dataset_add.f.r_unit)
            assert np.array_equal(base_vars['e_unit'], dataset_add.f.e_unit)
            assert np.array_equal(base_vars['system'], dataset_add.f.system)
            assert np.array_equal(base_vars['solvent'], dataset_add.f.solvent)
            assert np.array_equal(base_vars['cluster_size'],
                                  dataset_add.f.cluster_size)
        except AssertionError:
            base_filename = all_dataset_paths[0].split('/')[-1]
            incompat_filename = all_dataset_paths[index_dataset].split('/')[-1]
            raise ValueError('File %s is not compatible with %s' %
                             (incompat_filename, base_filename))

        # Seeing if the theory is always the same.
        # If theories are different, we change theory to 'unknown'.
        try:
            assert str(base_vars['theory'][()]) == str(dataset_add.f.theory[()])
        except AssertionError:
            base_vars['theory'][()] = 'unknown'

        # Concatenating relevant arrays.
        base_vars['R'] = np.concatenate(
            (base_vars['R'], dataset_add.f.R), axis=0
        )
        base_vars['E'] = np.concatenate(
            (base_vars['E'], dataset_add.f.E), axis=0
        )
        base_vars['F'] = np.concatenate(
            (base_vars['F'], dataset_add.f.F), axis=0
        )

        index_dataset += 1
    
    # Updating min, max, mean, an variances.
    base_vars['E_min'] = np.min(base_vars['E'].ravel())
    base_vars['E_max'] = np.max(base_vars['E'].ravel())
    base_vars['E_mean'] = np.mean(base_vars['E'].ravel())
    base_vars['E_var'] = np.var(base_vars['E'].ravel())
    base_vars['F_min'] = np.min(base_vars['F'].ravel())
    base_vars['F_max'] = np.max(base_vars['F'].ravel())
    base_vars['F_mean'] = np.mean(base_vars['F'].ravel())
    base_vars['F_var'] = np.var(base_vars['F'].ravel())

    # Writing combined dataset.
    base_vars['md5'] = sgdml_io.dataset_md5(base_vars)
    dataset_path = write_dir + str(base_vars['name'][()]) + '.npz'
    np.savez_compressed(dataset_path, **base_vars)