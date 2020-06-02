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

# pylint: disable=E1101

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
        
        Parameters
        ----------
            atoms : list
                Atomic numbers of all atoms in the system. The atoms
                are repeated; for example, water is ['H', 'H', 'O'].
        """
        self.system_info = solvents.system_info(atoms)
    
    def add_system_info(self, dataset):
        """Adds information about the system to the model.
        
        Parameters
        ----------
        dataset : dict
            Custom data structure that contains all information for a
            GDML dataset.
        
        Returns
        -------
        dict
            An updated GDML dataset with additional information regarding
            the system.
        
        Note
        ----
        If the system is a solvent, the 'solvent' name and 'cluster_size'
        is included.
        """

        if not hasattr(self, 'system_info'):
            self.get_system_info(dataset['z'].tolist())
        
        dataset['system'] = self.system_info['system']
        if dataset['system'] == 'solvent':
            dataset['solvent'] = self.system_info['solvent_name']
            dataset['cluster_size'] = self.system_info['cluster_size']
        
        return dataset

    def save(self, name, data, save_dir, is_dataset):
        """General save function for GDML data sets and models.
        
        Parameters
        ----------
        name : str
            Name of the file to be saved not including the
            extension.
        data : dict
            Base variables for dataset or model.
        save_dir : str
            Directory to save the file.
        is_dataset : bool
            Is the file a dataset? Controls whether the md5 of
            the file is saved.
        """

        save_dir = utils.norm_path(save_dir)
        if is_dataset:
            data['md5'] = sgdml_io.dataset_md5(data)
        save_path = save_dir + name + '.npz'
        np.savez_compressed(save_path, **data)


class mbGDMLModel(_mbGDMLData):
    """
    A class to load, inspect, and modify GDML models.

    Attributes
    ----------
    model : np.npzfile
        GDML model for predicting energies and forces.
        

    Methods
    -------
    load(model_path)
        Loads GDML model.
    get_model_name(log_path)
        Retrives GDML model's name from log file.
    add_manybody_info(mb_order)
        Adds many-body (mb) information to GDML model.
    """
    
    def __init__(self):
        pass

    @property
    def code_version(self):

        if hasattr(self, '_model_data'):
            return self._model_data['code_version'][()]
        else:
            raise AttributeError('No model is loaded.')

    def load(self, model_path):
        """Loads GDML model.
        
        Args:
            model_path (str): Path to GDML model.

        Raises:
            AttributeError: 
        """

        self.model = np.load(model_path, allow_pickle=True)
        self._model_data = dict(self.model_npz)

        if self._model_data['type'][()] != 'm':
            raise AttributeError('This npz is not a GDML model.')
    
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
        if not hasattr(self, 'model_data'):
            raise AttributeError('There is no model loaded.')

        self.model_data['mb'] = mb_order
        

class mbGDMLDataset(_mbGDMLData):

    def __init__(self):
        pass
    

    def _organization_dirs(
        self,
        gdml_data_dir,
        dataset_name
    ):
        """Determines where a dataset should be saved.

        The dataset will be written in a directory that depends on the
        partition size (e.g., dimer).
        
        Args:
            gdml_data_dir (str): Path to a common directory for GDML datasets.

        Notes:
            Requires the 'dataset_name' attribute.
            Sets the 'gdml_file_path' attribute for writing GDML datasets.
        """

        gdml_data_dir = utils.norm_path(gdml_data_dir)

        # Preparing directories.
        if self.system_info['system'] == 'solvent':
            gdml_partition_size_dir = utils.norm_path(
                gdml_data_dir + str(self.system_info['cluster_size']) + 'mer'
            )
            all_dir = [gdml_partition_size_dir]
            for directory in all_dir:
                try:
                    os.chdir(directory)
                except:
                    os.mkdir(directory)
                    os.chdir(directory)
        
        # Writing GDML file.
        self.gdml_file_path = gdml_partition_size_dir + dataset_name

    
    def load(self, dataset_path):
        self._dataset_npz = np.load(dataset_path)
        self.dataset = dict(self._dataset_npz)
        self._z = self.dataset['z']
        self._R = self.dataset['R']
        self._E = self.dataset['E']
        self._F = self.dataset['F']


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

    @property
    def z(self):

        if hasattr(self, '_user_data') or hasattr(self, 'dataset'):
            return self._z
        else:
            raise AttributeError('There is no data loaded.')
    
    @property
    def R(self):

        if hasattr(self, '_user_data') or hasattr(self, 'dataset'):
            return self._R
        else:
            raise AttributeError('There is no data loaded.')
    
    @property
    def F(self):

        if hasattr(self, '_user_data') or hasattr(self, 'dataset'):
            return self._F
        else:
            raise AttributeError('There is no data loaded.')
    
    @property
    def E(self):

        if hasattr(self, '_user_data') or hasattr(self, 'dataset'):
            if not hasattr(self, '_E'):
                raise AttributeError('No energies were provided in data set.')
            else:
                return self._E
        else:
            raise AttributeError('There is no data loaded.')

    def read_trajectory_xyz(self, trajectory_path):

        self._user_data = True

        parsed_data = ccread(trajectory_path)
        self._z = parsed_data.atomnos
        self._R = parsed_data.atomcoords
    
    def read_forces_xyz(self, trajectory_path):

        self._user_data = True
        
        parsed_data = ccread(trajectory_path)
        self._F = parsed_data.atomcoords


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
                for the partition calculations. For example, 'MP2.def2TZVP.
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
        dataset = {
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
        dataset = self.add_system_info(dataset)

        dataset['md5'] = sgdml_io.dataset_md5(dataset)
        self.dataset = dataset

        # Writes dataset.
        if write:
            self._organization_dirs(gdml_data_dir, dataset_name)
            dataset_path = self.gdml_file_path + '.npz'
            np.savez_compressed(dataset_path, **dataset)
    
    def print(self):

        if not hasattr(self, 'dataset'):
            raise AttributeError('Please load a data set first.')
        
        R = self.dataset['R']
        E = self.dataset['E']
        F = self.dataset['F']

        num_config = R.shape[0]
        for config in range(num_config):
            print(f'-----Configuration {config}-----')
            print(f'Energy: {E[config][()]} kcal/mol')
            print(f'Forces:\n{F[config]}')

    
    def mb_dataset(self, nbody, models_dir):
        
        if not hasattr(self, 'dataset'):
            raise AttributeError('Please load a data set first.')

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
            predict = mbGDMLPredict([nbody_model.model])
            self.dataset = predict.remove_nbody(self.dataset)

            nbody_index += 1

        # Removes old dataset attribute.
        if hasattr(self, 'dataset'):
            delattr(self, 'dataset')


class mbGDMLPredictset(_mbGDMLData):
    """
    A predict set is a data set with mbGDML predicted energy and forces instead
    of training data.

    When analyzing many structures using mbGDML it is easier (and faster) to
    predict all many-body contributions once and then analyze the stored data.
    The predict set accomplishes just this.
    """

    def __init__(self):
        pass
        
    
    def read(self, predictset_path):
        """Reads predict data set and loads data.
        
        Args:
            predictset_path (str): Path to predict data set.
        """
        predictset = np.load(predictset_path, allow_pickle=True)
        predictset = dict(predictset)
        for file in predictset:
            if type(predictset[file][()]) == np.str_:
                setattr(self, file, str(predictset[file][()]))
            else:
                setattr(self, file, predictset[file][()])

    
    def sum_contributions(self, struct_num, nbody_order):
        """
        Returns the energy and force of a structure at a
        specific many-body order.

        Predict sets have data that is broken down into many-body and 'total'
        contributions. Many-body contributions provide the total for that order;
        for example, 'E_3' gives you the total contribution (or correction) of
        all three bodies evaluated in the structure. This is not the total
        energy with one-body, two-body, and three-body corrections.

        This function returns the 'total energy' that includes the specified
        nbody_order and lower corrections.

        Args:
            struct_num (int): Specifies the index of the structure in the
                self.R array and the energy and force arrays.
            nbody_order (int): Highest many-body order corrections to include.
        
        Returns:
            tuple: Energy and force of the structure with all many-body
                corrections up to nbody_order.
        """

        if not hasattr(self, 'type'):
            raise AttributeError('Please read a predict set first.')
        
        nbody_index = 1
        while hasattr(self, f'E_{nbody_index}') and \
              hasattr(self, f'F_{nbody_index}') and \
              nbody_index <= nbody_order:

            E_cont = getattr(self, f'E_{nbody_index}')
            F_cont = getattr(self, f'F_{nbody_index}')

            if nbody_index == 1:
                E = E_cont['T'][struct_num]
                F = F_cont['T'][struct_num]
            else:
                E += E_cont['T'][struct_num]
                F += F_cont['T'][struct_num]

            nbody_index += 1
        
        return (E, F)
    

    def nbody_predictions(self, nbody_order):

        if not hasattr(self, 'R'):
            raise AttributeError('No coordinates;'
                                 'please read a predict set first.')
        else:
            num_structures = self.R.shape[0]

            for structure in range(0, num_structures):
                e, f = self.sum_contributions(structure, nbody_order)

                e = np.array([e])
                f = np.array([f])

                if structure == 0:
                    E = e
                    F = f
                else:
                    E = np.concatenate((E, e))
                    F = np.concatenate((F, f))

        return (E, F)

    def load_dataset(self, dataset_path):
        """
        Loads data set in preparation to create a predict set.
        """
        self.dataset_path = dataset_path
        self.dataset = dict(np.load(dataset_path))
    

    def load_models(self, model_paths):
        """
        Loads model(s) in preparation to create a predict set.
        """
        self.model_paths = model_paths
        self.mbgdml = mbGDMLPredict(model_paths)


    def create_predictset(self):
        """
        Creates a predict set from loaded data set and models.
        """

        if not hasattr(self, 'dataset') or not hasattr(self, 'mbgdml'):
            raise AttributeError('Please load a data set and mbGDML models.')

        num_config = self.dataset['R'].shape[0]
        name = str(self.dataset['name'][()]).replace(
            'dataset', 'prediction'
        )
        
        self.dataset = {
            'type': 'p',  # Designates predictions.
            'code_version': __version__,  # sGDML version.
            'name': name,
            'theory': self.dataset['theory'],
            'z': self.dataset['z'],
            'R': self.dataset['R'],
            'r_unit': self.dataset['r_unit'],
            'E_true': self.dataset['E'],
            'e_unit': self.dataset['e_unit'],
            'F_true': self.dataset['F'],
        }

        # Predicts and stores energy and forces.
        all_E = {}
        all_F = {}
        for i in range(num_config):
            print(f'Predicting structure {i} out of {num_config - 1} ...')
            e, f = self.mbgdml.decomposed_predict(
                self.dataset['z'].tolist(), self.dataset['R'][i]
            )

            for order in e:
                if i == 0:
                    all_E[order] = e[order]
                    all_F[order] = f[order]
                    
                    if order == 'T':
                        all_E[order] = all_E[order]
                        all_F[order] = np.array([all_F[order]])
                    else:
                        for combo in e[order]:
                            all_E[order][combo] = all_E[order][combo]
                            all_F[order][combo] = np.array(
                                [all_F[order][combo]]
                            )
                else:
                    if order == 'T':
                        all_E[order] = np.concatenate(
                            (all_E[order], e[order]),
                            axis=0
                        )
                        all_F[order] = np.concatenate(
                            (all_F[order], np.array([f[order]])),
                            axis=0
                        )
                    else:
                        for combo in e[order]:
                            all_E[order][combo] = np.concatenate(
                                (all_E[order][combo], e[order][combo]),
                                axis=0
                            )
                            all_F[order][combo] = np.concatenate(
                                (all_F[order][combo],
                                 np.array([f[order][combo]])),
                                 axis=0
                            )


        # Loop through all_E and all_F and add their keys to dataset
        for order in all_E:
            E_name = f'E_{order}'
            F_name = f'F_{order}'
            self.dataset[E_name] = all_E[order]
            self.dataset[F_name] = all_F[order]
        
        for data in self.dataset:
            setattr(self, data, self.dataset[data])





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


class structure:

    def __init__(self):
        pass

    def load_file(self, file_path):
        self._ccdata = ccread(file_path)
        self.z = self._ccdata.atomnos
        self.R = self._ccdata.atomcoords


    @property
    def quantity(self):
        """The number of structures.
        """
        if self.R.ndim == 2:
            return 1
        elif self.R.ndim == 3:
            if self.R.shape[0] == 1:
                return 1
            else:
                return self.R.shape[0]
        else:
            raise ValueError(
                f"The coordinates have an unusual dimension of {self.R.ndim}."
            )
    
    @property
    def z(self):
        """Array of atoms' atomic numbers in the same number as in
        the coordinate file.
        """

        return self._z
    
    @z.setter
    def z(self, atoms):
        #TODO check type and convert to list
        self._z = atoms
    
    @property
    def R(self):
        """The atomic positions of structure(s) in a np.ndarray of (m, 3) or
        (n, m, 3) shape where n is the number of structures and m is the
        number of atoms with 3 positional coordinates.
        """
        if self._R.ndim == 2:
            if self._R.shape[1] != 3:
                raise ValueError(
                    "GDML expects xyz coordinates in array's 2nd dimension"
                )

            return self._R
        elif self._R.ndim == 3:
            if self._R.shape[2] != 3:
                raise ValueError(
                    "GDML expects xyz coordinates in array's 3rd dimension"
                )

            if self._R.shape[0] == 1:
                return self._R[0]
            else:
                return self._R
        else:
            raise ValueError(
                f"The coordinates have an unusual dimension of {self._R.ndim}."
            )
    
    @R.setter
    def R(self, coords):
        self._R = coords
    



def create_datasets(calc_output_dir, dataset_dir, r_units_calc, e_units_calc,
                    theory='unknown', r_units='Angstrom', e_units='kcal/mol'):
    """Writes partition datasets for GDML.

    Used as a driver for the PartitionCalcOutput class that iterates over all
    partitions. Writes and organizes GDML xyz files according
    to partition size, temperature, and MD iteration.
    
    Args:
        calc_output_dir (str): Path to folder that contains computational
            chemistry output files. Usually for an entire solvent.
        dataset_dir (str): Path that contains all GDML dataset files.

    Notes:
        The partition calculation output files must have 'out' somewhere in
        the file name (or extension).
    """
    
    calc_output_dir = utils.norm_path(calc_output_dir)
    all_out_files = utils.get_files(calc_output_dir, 'out')

    for out_file in all_out_files:
        print('Writing GDML dataset for ' + out_file.split('/')[-1] + ' ...')
        partition_calc = PartitionCalcOutput(out_file)
        dataset = mbGDMLDataset()
        dataset.create_dataset(
            dataset_dir,
            partition_calc.output_name,
            partition_calc.atoms,
            partition_calc.coords,
            partition_calc.energies,
            partition_calc.forces,
            'kcal/mol',
            'hartree',
            'bohr',
            theory='MP2.def2TZVP',
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
        utils.get_files(partition_dir, '.npz')
    )

    # Prepares initial combined dataset from the first dataset found.
    dataset_npz = np.load(all_dataset_paths[0])
    dataset = dict(dataset_npz)
    del dataset_npz
    original_name = str(dataset['name'][()])
    parent_label = original_name.split('-')[1]
    size_label = ''.join([str(dataset['cluster_size']), 'mer'])
    dataset['name'][()] = '-'.join([parent_label, size_label, 'dataset'])

    # Adds the remaining datasets to the new dataset.
    index_dataset = 1
    while index_dataset < len (all_dataset_paths):
        dataset_add = np.load(all_dataset_paths[index_dataset])
        print('Adding %s dataset to %s ...' % (dataset_add.f.name[()],
              dataset['name'][()]))

        # Ensuring the datasets are compatible.
        try:
            assert np.array_equal(dataset['type'], dataset_add.f.type)
            assert np.array_equal(dataset['z'], dataset_add.f.z)
            assert np.array_equal(dataset['r_unit'], dataset_add.f.r_unit)
            assert np.array_equal(dataset['e_unit'], dataset_add.f.e_unit)
            assert np.array_equal(dataset['system'], dataset_add.f.system)
            assert np.array_equal(dataset['solvent'], dataset_add.f.solvent)
            assert np.array_equal(dataset['cluster_size'],
                                  dataset_add.f.cluster_size)
        except AssertionError:
            base_filename = all_dataset_paths[0].split('/')[-1]
            incompat_filename = all_dataset_paths[index_dataset].split('/')[-1]
            raise ValueError('File %s is not compatible with %s' %
                             (incompat_filename, base_filename))

        # Seeing if the theory is always the same.
        # If theories are different, we change theory to 'unknown'.
        try:
            assert str(dataset['theory'][()]) == str(dataset_add.f.theory[()])
        except AssertionError:
            dataset['theory'][()] = 'unknown'

        # Concatenating relevant arrays.
        dataset['R'] = np.concatenate(
            (dataset['R'], dataset_add.f.R), axis=0
        )
        dataset['E'] = np.concatenate(
            (dataset['E'], dataset_add.f.E), axis=0
        )
        dataset['F'] = np.concatenate(
            (dataset['F'], dataset_add.f.F), axis=0
        )

        index_dataset += 1
    
    # Updating min, max, mean, an variances.
    dataset['E_min'] = np.min(dataset['E'].ravel())
    dataset['E_max'] = np.max(dataset['E'].ravel())
    dataset['E_mean'] = np.mean(dataset['E'].ravel())
    dataset['E_var'] = np.var(dataset['E'].ravel())
    dataset['F_min'] = np.min(dataset['F'].ravel())
    dataset['F_max'] = np.max(dataset['F'].ravel())
    dataset['F_mean'] = np.mean(dataset['F'].ravel())
    dataset['F_var'] = np.var(dataset['F'].ravel())

    # Writing combined dataset.
    dataset['md5'] = sgdml_io.dataset_md5(dataset)
    dataset_path = write_dir + str(dataset['name'][()]) + '.npz'
    np.savez_compressed(dataset_path, **dataset)