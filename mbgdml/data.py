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
from mbgdml.solvents import solvent

gdml_partition_size_names = [
    'monomer', 'dimer', 'trimer', 'tetramer', 'pentamer'
]

class PartitionCalcOutput():
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
        cluster (str): The label identifying the partition of the MD trajectory.
        temp (str): Set point temperautre for the MD thermostat.
        iter (int): Identifies the iteration of the MD iteration.
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
        self._get_solvent_info()
    
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
        self.iter = int(split_label[3])
        self.partition = str(split_label[4].split('.')[0])
        self.partition_size = int(len(self.partition))
    
    def _get_solvent_info(self):
        """Adds solvent information to object.
        """
        solvent_info = solvent(self.atoms.tolist())
        self.system = solvent_info.system
        if self.system is 'solvent':
            self.solvent_info = {
                'solvent_name': solvent_info.solvent_name,
                'solvent_label': solvent_info.solvent_label,
                'solvent_molec_size': solvent_info.solvent_molec_size,
                'cluster_size': solvent_info.cluster_size
            }

    
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
    
    def _organization_dirs(self, gdml_data_dir):
        """Determines where a dataset should be saved.

        The dataset will be written in a directory that depends on the
        solvent (e.g. MeOH), partition size (e.g., dimer), MD temperature
        (e.g. 300K), and MD iteration (e.g. 1).
        
        Args:
            gdml_data_dir (str): Path to a common directory for GDML datasets.

        Notes:
            Sets the 'gdml_file_path' attribute for writing GDML datasets.
        """

        gdml_data_dir = utils.norm_path(gdml_data_dir)

        # Preparing directories.
        # /path/to/gdml-datasets/solventlabel/partitionsize/temp/iteration
        gdml_solvent = self.solvent_info['solvent_label']
        gdml_solvent_dir = utils.norm_path(
            gdml_data_dir + gdml_solvent
        )
        gdml_partition_size_dir = utils.norm_path(
            gdml_solvent_dir + str(self.partition_size) + 'mer'
        )
        gdml_temp_dir = utils.norm_path(
            gdml_partition_size_dir + str(self.temp)
        )
        gdml_iter_dir = utils.norm_path(
            gdml_temp_dir + str(self.iter)
        )
        all_dir = [gdml_solvent_dir, gdml_partition_size_dir, gdml_temp_dir,
                   gdml_iter_dir]
        for directory in all_dir:
            try:
                os.chdir(directory)
            except:
                os.mkdir(directory)
                os.chdir(directory)
        
        # Writing GDML file.
        self.gdml_file_path = gdml_iter_dir + self.partition \
                              + '-' + self.cluster \
                              + '-' + self.temp \
                              + '-' + str(self.iter) \
                              + '-dataset'


    def create_dataset(self, gdml_dataset_dir, r_units_calc, e_units_calc,
                       theory='unknown', r_units='Angstrom',
                       e_units='kcal/mol', write=True):
        """Creates and writes GDML dataset.
        
        Args:
            gdml_dataset_dir (str): Path to common GDML dataset directory. This
                is above the solvent direcotry.
            r_units_calc (str): The units of the coordinates in the partition
                calculation output file. This is only used convert forces if
                needed. Options are 'Angstrom' or 'bohr'.
            e_units_calc (str): The units of energies reported in the partition
                calculation output file. This is used to converted energies and
                forces. Options are 'eV', 'hartree', 'kcal/mol', and 'kJ/mol'.
            theory (str, optional): The level of theory and basis set used
                for the partition calculations. Defaults to 'unknown'.
            r_units (str, optional): Desired coordinate units for the GDML
                dataset. Defaults to 'Angstrom'.
            e_units (str, optional): Desired energy units for the GDML dataset.
                Defaults to 'kcal/mol'.
            write (bool, optional): Whether or not the dataset is written to
                disk. Defaults to True.
        """

        # Preparing energies in e_units.
        # Note, cclib stores energies in eV.
        energies = []
        for energy in self.energies:
            energies.append(convertor(energy[0], 'eV', e_units))
        energies = np.array(energies)

        # Converting forces.
        # cclib does not convert gradients (or forces), this is where
        # the energy and coordinates units come into play.
        forces = self.forces * (convertor(1, e_units_calc, e_units) \
                                / convertor(1, r_units_calc, r_units))

        # Preparing dataset name.
        dataset_name = self.partition \
                       + '-' + self.cluster \
                       + '-' + self.temp \
                       + '-' + str(self.iter) \
                       + '-dataset'

        # sGDML variables.
        base_vars = {
            'type': 'd',  # Designates dataset or model.
            'code_version': __version__,  # sGDML version.
            'name': dataset_name,  # Name of the output file.
            'theory': theory,  # Theory used to calculate the data.
            'z': self.atoms,  # Atomic numbers of all atoms in system.
            'R': self.coords,  # Cartesian coordinates.
            'r_unit': r_units,  # Units for coordinates.
            'E': energies,  # Energy of the structure.
            'e_unit': e_units,  # Units of energy.
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
        if self.system is 'solvent':
            base_vars['system'] = 'solvent'
            base_vars['solvent'] = self.solvent_info['solvent_name']
            base_vars['cluster_size'] = self.solvent_info['cluster_size']

        base_vars['md5'] = sgdml_io.dataset_md5(base_vars)
        self.dataset = base_vars

        # Writes dataset.
        if write:
            self._organization_dirs(gdml_dataset_dir)
            dataset_path = self.gdml_file_path + '.npz'
            np.savez_compressed(dataset_path, **base_vars)
        

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
        calc = PartitionCalcOutput(out_file)
        calc.create_dataset(dataset_dir, r_units_calc, e_units_calc,
                            theory=theory, r_units=r_units,
                            e_units=e_units)


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