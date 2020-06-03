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
import numpy as np
from cclib.io import ccread
from cclib.parser.utils import convertor
from sgdml import __version__
from sgdml.utils import io as sgdml_io
from mbgdml.data import mbGDMLData
from mbgdml.data import PartitionOutput
from mbgdml.data import mbGDMLModel
from mbgdml import utils
from mbgdml.predict import mbGDMLPredict
  

class mbGDMLDataset(mbGDMLData):
    """For creating, loading, manipulating, and using GDML data sets.

    Methods
    -------
    load(dataset_path)
        Use numpy.load to read information from data set.
    read_trajectory(file_path)
        Use cclib to read trajectory (usually xyz format) and assign z and R
        attributes.
    read_forces(file_path)
        Use cclib to read file (usually xyz format) to assign F attribute.
    create_dataset(gdml_data_dir, dataset_name, atoms, coords, energies, forces,
    e_units, e_units_calc, r_units_calc, theory='unknown',
    gdml_r_units='Angstrom', gdml_e_units='kcal/mol', write=True)
        Creates and writes a single GDML data set.

    Attributes
    ----------
    dataset : dict
        Contains all information and arrays stored in data set.
    """

    def __init__(self):
        pass

    
    def load(self, dataset_path):
        """Uses numpy.load to read data set

        Sets dataset, z, R, E, and F attributes.
        """

        self._dataset_npz = np.load(dataset_path)
        self.dataset = dict(self._dataset_npz)
        self._z = self.dataset['z']
        self._R = self.dataset['R']
        self._E = self.dataset['E']
        self._F = self.dataset['F']


    # TODO: write read_extended_xyz function


    def read_trajectory(self, file_path):
        """Use cclib to read trajectory (usually xyz format) and assign z and R
        attributes.

        Parameters
        ----------
        file_path : str
            Path to file.
        """

        self._user_data = True

        parsed_data = ccread(file_path)
        self._z = parsed_data.atomnos
        self._R = parsed_data.atomcoords
    

    def read_forces(self, file_path):
        """Use cclib to read file (usually xyz format) to assign F attribute.

        Parameters
        ----------
        file_path : str
            Path to file.
        """

        self._user_data = True
        
        parsed_data = ccread(file_path)
        self._F = parsed_data.atomcoords
    

    @property
    def z(self):
        """Atomic numbers of all atoms in data set structures.
        
        A (n,) shape array of type numpy.int32 containing atomic numbers of
        atoms in the structures in order as they appear.

        Raises
        ------
        AttributeError
            If there is no created or loaded data set.
        """

        if hasattr(self, '_user_data') or hasattr(self, 'dataset'):
            return self._z
        else:
            raise AttributeError('There is no data loaded.')
    

    @property
    def R(self):
        """The atomic positions of structure(s).
        
        A numpy.ndarray with shape of (m, n, 3) where m is the number of
        structures and n is the number of atoms with 3 positional coordinates.

        Raises
        ------
        AttributeError
            If there is no created or loaded data set.
        """

        if hasattr(self, '_user_data') or hasattr(self, 'dataset'):
            return self._R
        else:
            raise AttributeError('There is no data set.')
    

    @property
    def F(self):
        """The atomic forces of atoms in structure(s).
        
        A numpy.ndarray with shape of (m, n, 3) where m is the number of
        structures and n is the number of atoms with 3 positional coordinates.

        Raises
        ------
        AttributeError
            If there is no created or loaded data set.
        """

        if hasattr(self, '_user_data') or hasattr(self, 'dataset'):
            return self._F
        else:
            raise AttributeError('There is no data loaded.')
    

    @property
    def E(self):
        """The energies of structure(s).
        
        A numpy.ndarray with shape of (n,) where n is the number of atoms.

        Raises
        ------
        AttributeError
            If there is no created or loaded data set.
        """

        if hasattr(self, '_user_data') or hasattr(self, 'dataset'):
            if not hasattr(self, '_E'):
                raise AttributeError('No energies were provided in data set.')
            else:
                return self._E
        else:
            raise AttributeError('There is no data loaded.')


    def partition_dataset_name(
        self, cluster_label, partition_label, md_temp, md_iter
    ):
        """Automates and standardizes partition datasets names.
        
        Parameters
        ----------
        cluster_label : str
            The label identifying the parent cluster of the partition.
            For example, '4H2O.abc0'.
        partition_label : str
            Identifies what solvent molecules are in the partition.
            For example, 'AB'.
        md_temp : int
            Set point temperautre for the MD thermostat in Kelvin.
        md_iter : int
            Identifies the iteration of the MD iteration.
        
        Returns
        -------
        str
            Standardized data set name.
        """
        
        dataset_name =  '-'.join([
            cluster_label, partition_label, str(md_temp) + 'K', str(md_iter),
            'dataset'
        ])

        return dataset_name
    

    def _organization_dirs(self, gdml_data_dir, dataset_name):
        """Determines where a dataset should be saved.

        The dataset will be written in a directory that depends on the
        partition size (e.g., 2mer).
        
        Parameters
        ----------
        gdml_data_dir : str
            Path to a common directory for GDML datasets.

        Returns
        -------
        str
            Path to save directory of data set.
        """


        gdml_data_dir = utils.norm_path(gdml_data_dir)

        # Preparing directories.
        if self.system_info['system'] == 'solvent':
            partition_size_dir = utils.norm_path(
                gdml_data_dir + str(self.system_info['cluster_size']) + 'mer'
            )

            os.makedirs(partition_size_dir, exist_ok=True)
            
            return partition_size_dir


    def create_dataset(
        self,
        gdml_data_dir,
        dataset_name,
        z,
        R,
        E,
        F,
        e_units,
        r_units,
        theory='unknown',
        e_units_gdml='kcal/mol',
        r_units_gdml='Angstrom',
        e_units_calc=None,
        r_units_calc=None,
        write=True
    ):
        """Creates and writes GDML dataset.
        
        Parameters
        ----------
        gdml_data_dir : str
            Path to common GDML data set directory for a particular cluster.
        dataset_name : str
            The name to label the dataset.
        z : numpy.ndarray
            A (n,) array containing atomic numbers of n atoms.
        R : numpy.ndarray
            A (m, n, 3) array containing the atomic coordinates of n atoms of
            m MD steps.
        E : numpy.ndarray
            A (m,) array containing the energies of m MD steps.
        F : numpy.ndarray
            A (m, n, 3) array containing the atomic forces of n atoms of
            m MD steps. Simply the negative of grads.
        e_units : str
            The energy units of `energies`.
        r_units : str
            The distance units of `coords`.
        e_units_calc : str
            The units of energies reported in the partition calculation output
            file. This is used to convert forces. Options are 'eV', 'hartree',
            'kcal/mol', and 'kJ/mol'.
        r_units_calc : str
            The units of the coordinates in the partition calculation output
            file. This is only used convert forces if needed.
            Options are 'Angstrom' or 'bohr'.
        theory : str, optional
            The level of theory and basis set used for the partition
            calculations. For example, 'MP2.def2TZVP. Defaults to 'unknown'.
        r_units_gdml : str, optional
            Desired coordinate units for the GDML data set. Defaults to 'Angstrom'.
        e_units_gdml : str, optional
            Desired energy units for the GDML dataset. Defaults to 'kcal/mol'.
        write : bool, optional
            Whether or not the dataset is written to disk. Defaults to True.

        Raises
        ------
        ValueError
            If units do not match GDML units and no calculation units were
            provided.
        """

        # Converts energies and forces if units are not the same as GDML units.
        if e_units != e_units_gdml or r_units != r_units_gdml:
            if e_units_calc == None or r_units_calc == None:
                raise ValueError(
                    'The energy or coordinate units do not match GDML units.'
                    'Please specify calculation units for conversion.'
                )
            else:
                E = convertor(E, e_units, e_units_gdml)
                F = utils.convert_forces(
                    'unknown', F,
                    e_units, r_units,
                    e_units_calc=e_units_calc,
                    r_units_calc=r_units_calc
                )

        if not hasattr(self, 'system_info'):
            self.get_system_info(z.tolist())

        # sGDML variables.
        dataset = {
            'type': 'd',  # Designates dataset or model.
            'code_version': __version__,  # sGDML version.
            'name': dataset_name,  # Name of the output file.
            'theory': theory,  # Theory used to calculate the data.
            'z': z,  # Atomic numbers of all atoms in system.
            'R': R,  # Cartesian coordinates.
            'r_unit': r_units_gdml,  # Units for coordinates.
            'E': E,  # Energy of the structures.
            'e_unit': e_units_gdml,  # Units of energy.
            'E_min': np.min(E.ravel()),  # Energy minimum.
            'E_max': np.max(E.ravel()),  # Energy maximum.
            'E_mean': np.mean(E.ravel()),  # Energy mean.
            'E_var': np.var(E.ravel()),  # Energy variance.
            'F': F,  # Atomic forces for each atom.
            'F_min': np.min(F.ravel()),  # Force minimum.
            'F_max': np.max(F.ravel()),  # Force maximum.
            'F_mean': np.mean(F.ravel()),  # Force mean.
            'F_var': np.var(F.ravel())  # Force variance.
        }

        # mbGDML variables.
        dataset = self.add_system_info(dataset)

        dataset['md5'] = sgdml_io.dataset_md5(dataset)
        self.dataset = dataset

        # Writes dataset.
        if write:
            self.dataset_dir = self._organization_dirs(
                gdml_data_dir, dataset_name
            )
            os.chdir(self.dataset_dir)
            self.dataset_path = self.dataset_dir + dataset_name + '.npz'

            self.save(dataset_name, self.dataset, self.dataset_dir, True)
     

    def combine_datasets(self, partition_dir, write_dir):
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