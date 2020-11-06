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

# pylint: disable=E1101

import os
import numpy as np
from cclib.io import ccread
from cclib.parser.utils import convertor
from sgdml import __version__
from sgdml.utils import io as sgdml_io
from mbgdml.data import mbGDMLData
from mbgdml.data import mbGDMLModel
from mbgdml.parse import parse_stringfile
from mbgdml import utils
from mbgdml.predict import mbGDMLPredict
  

class mbGDMLDataset(mbGDMLData):
    """For creating, loading, manipulating, and using data sets.

    Parameters
    ----------
    dataset_path : :obj:`str`, optional
        Path to a saved :obj:`numpy.lib.npyio.NpzFile`.

    Attributes
    ----------
    dataset : :obj:`dict`
        Contains all data as :obj:`numpy.ndarray` objects.
    name : :obj:`str`
        Name of the data set. Defaults to ``'dataset'``.
    r_unit : :obj:`str`
        Units of distance. Options are ``'Angstrom'`` or ``'bohr'``.
    e_unit : :obj:`str`
        Units of energy. Options are ``'eV'``, ``'hartree'``,
            ``'kcal/mol'``, and ``'kJ/mol'``.
    code_verstion
        The version of :mod:`sgdml`.
    theory : :obj:`str`
        The level of theory used to compute energy and gradients of the data
        set.
    mb : :obj:`int`
        The order of n-body corrections this data set is intended for. For
        example, a tetramer solvent cluster with one-, two-, and three-body
        contributions removed will have a ``mb`` order of 4.
    """


    def __init__(self, dataset_path=None):
        self.type = 'd'
        self.name = 'dataset'
        if dataset_path is not None:
            self.load(dataset_path)


    def _update(self, dataset):
        """Updates object attributes.

        Parameters
        ----------
        dataset : :obj:`dict`
            Contains all information and arrays stored in data set.
        """
        self.dataset = dict(dataset)
        self._z = self.dataset['z']
        self._R = self.dataset['R']
        self._E = self.dataset['E']
        self._F = self.dataset['F']
        self.r_unit = str(self.dataset['r_unit'][()])
        self.e_unit = str(self.dataset['e_unit'][()])
        self.code_version = str(self.dataset['code_version'][()])
        self.name = str(self.dataset['name'][()])
        self.theory = str(self.dataset['theory'][()])
        # mbGDML added data set information.
        if self.z.ndim == 1:
            self.get_system_info(self.z.tolist())
        if 'mb' in self.dataset.keys():
            self.mb = str(self.dataset['mb'][()])

    def load(self, dataset_path):
        """Uses :func:``numpy.load`` to read data set.

        Parameters
        ----------
        dataset_path : :obj:`str`
            Path to NumPy ``npz`` file.
        """
        dataset_npz = np.load(dataset_path, allow_pickle=True)
        npz_type = str(dataset_npz.f.type[()])
        if npz_type != 'd':
            raise ValueError(f'{npz_type} is not a data set.')
        else:
            self._update(dict(dataset_npz))
    

    def read_xyz(
        self, file_path, xyz_type, r_unit=None, e_unit=None,
        energy_comments=False
    ):
        """Reads data from xyz files.

        Parameters
        ----------
        file_path : :obj:`str`
            Path to xyz file.
        xyz_type : :obj:`str`
            Type of data. Either ``'coords'``, ``'forces'``, ``'grads'``, or
            ``'extended'``.
        r_units : :obj:`str`, optional
            Units of distance. Options are ``'Angstrom'`` or ``'bohr'``.
        e_units : :obj:`str`, optional
            Units of energy. Options are ``'eV'``, ``'hartree'``,
            ``'kcal/mol'``, and ``'kJ/mol'``.
        energy_comments : :obj:`bool`, optional
            If there are comments specifying the energies of the structures.
            Defaults to ``False``.
        
        Notes
        -----
        If ``xyz_type`` is ``'grads'``, it will take the negative and store as
        forces.
        """
        self._user_data = True
        z, comments, data = parse_stringfile(file_path)
        z = [utils.atoms_by_number(i) for i in z]
        # If all the structures have the same order of atoms (as required by
        # sGDML), condense into a one-dimensional array.
        if len(set(tuple(i) for i in z)) == 1:
            z = np.array(z[0])
        else:
            z = np.array(z)
        self._z = z
        if self.z.ndim == 1:
            self.get_system_info(self.z.tolist())
        if energy_comments:
            try:
                E = np.array([float(i) for i in comments])
                self._E = E
            except ValueError as e:
                bad_comment = str(e).split(': ')[-1]
                raise ValueError(f'{bad_comment} should only contain a float.')
        data = np.array(data)
        if xyz_type == 'extended':
            self._R = data[:,:,3:]
            self._F = data[:,:,:3]
        elif xyz_type == 'coords':
            self._R = data
        elif xyz_type == 'grads':
            self._F = np.negative(data)
        elif xyz_type == 'forces':
            self._F = data
        else:
            raise ValueError(f'{xyz_type} is not a valid xyz data type.')
        if r_unit is not None:
            self.r_unit = r_unit
        if e_unit is not None:
            self.e_unit = e_unit
    

    def from_partitioncalc(self, partcalc):
        """Creates data set from partition calculations.

        Parameters
        ----------
        partcalc : :obj:`mbgdml.data.PartitionOutput`
            Data from energy and gradient calculations of same partition.
        """
        self._z = partcalc.z
        self._R = partcalc.R
        self._E = partcalc.E
        self._F = partcalc.F
        self.r_unit = partcalc.r_unit
        self.e_unit = partcalc.e_unit
        self.code_version = __version__
        self.theory = partcalc.theory
        self.create()


    @property
    def z(self):
        """Atomic numbers of all atoms in data set structures.
        
        A ``(n,)`` shape array of type :obj:`numpy.int32` containing atomic
        numbers of atoms in the structures in order as they appear.

        :type: :obj:`numpy.ndarray`
        """
        return self._z
    

    @z.setter
    def z(self, var):
        self._z = var
    

    @property
    def R(self):
        """Atomic coordinates of structure(s).
        
        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of calculations and ``n`` is the number of atoms with three 
        Cartesian components.

        :type: :obj:`numpy.ndarray`
        """
        return self._R
    

    @R.setter
    def R(self, var):
        self._R = var
    

    @property
    def F(self):
        """Atomic forces of atoms in structure(s).
        
        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of calculations and ``n`` is the number of atoms with three 
        Cartesian components.

        :type: :obj:`numpy.ndarray`
        """
        return self._F
    

    @F.setter
    def F(self, var):
        self._F = var
    

    @property
    def E(self):
        """The energies of structure(s).
        
        A :obj:`numpy.ndarray` with shape of ``(n,)`` where ``n`` is the number
        of atoms.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_E'):
            return self._E
        else:
            raise AttributeError('No energies were provided in data set.')
    

    @E.setter
    def E(self, var):
        self._E = var
    

    @property
    def E_min(self):
        """Minimum energy of all structures.

        :type: :obj:`float`
        """
        return float(np.min(self.E.ravel()))
    

    @property
    def E_max(self):
        """Maximum energy of all structures.

        :type: :obj:`float`
        """
        return float(np.max(self.E.ravel()))
    

    @property
    def E_var(self):
        """Energy variance.

        :type: :obj:`float`
        """
        return float(np.var(self.E.ravel()))
        
    
    @property
    def E_mean(self):
        """Mean of all energies.

        :type: :obj:`float`
        """
        return float(np.mean(self.E.ravel()))
    

    @property
    def F_min(self):
        """Minimum atomic force in all structures.

        :type: :obj:`float`
        """
        return float(np.min(self.F.ravel()))
    

    @property
    def F_max(self):
        """Maximum atomic force in all structures.

        :type: :obj:`float`
        """
        return float(np.max(self.F.ravel()))
    

    @property
    def F_var(self):
        """Force variance.

        :type: :obj:`float`
        """
        return float(np.var(self.F.ravel()))
    

    @property
    def F_mean(self):
        """Mean of all forces.

        :type: :obj:`float`
        """
        return float(np.mean(self.F.ravel()))
    

    @property
    def md5(self):
        """Unique MD5 hash of data set. Encoded with UTF-8.

        :type: :obj:`bytes`
        """
        return sgdml_io.dataset_md5(self.dataset)
    

    def convertE(self, E_units):
        """Convert energies and updates :attr:`e_unit`.

        Parameters
        ----------
        E_units : :obj:`str`
            Desired units of energy. Options are ``'eV'``, ``'hartree'``,
            ``'kcal/mol'``, and ``'kJ/mol'``.
        """
        self._E = convertor(self.E, self.e_unit, E_units)
        self.e_unit = E_units
    
    
    def convertR(self, R_units):
        """Convert coordinates and updates :attr:`r_unit`.

        Parameters
        ----------
        R_units : :obj:`str`
            Desired units of coordinates. Options are ``'Angstrom'`` or
            ``'bohr'``.
        """
        self._R = convertor(self.R, self.r_unit, R_units)
        self.r_unit = R_units


    def convertF(self, E_units, R_units):
        """Convert forces.

        Does not change :attr:`e_unit` or :attr:`r_unit`.

        Parameters
        ----------
        E_units : :obj:`str`
            Desired units of energy. Options are ``'eV'``, ``'hartree'``,
            ``'kcal/mol'``, or ``'kJ/mol'``.
        R_units : :obj:`str`
            Desired units of coordinates. Options are ``'Angstrom'`` or
            ``'bohr'``.
        """
        self._F = utils.convert_forces(
            'unknown', self.F, E_units, R_units, e_units_calc=self.e_units,
            r_units_calc=self.r_units
        )


    def name_partition(
        self, cluster_label, partition_label, md_temp, md_iter
    ):
        """Automates and standardizes partition datasets names.
        
        Parameters
        ----------
        cluster_label : :obj:`str`
            The label identifying the parent cluster of the partition.
            For example, ``'4H2O.abc0'``.
        partition_label : :obj:`str`
            Identifies what solvent molecules are in the partition.
            For example, ``'AB'``.
        md_temp : :obj:`int`
            Set point temperautre for the MD thermostat in Kelvin.
        md_iter : :obj:`int`
            Identifies the iteration of the MD simulation.
        """
        self.name =  '-'.join([
            cluster_label, partition_label, str(md_temp) + 'K', str(md_iter),
            'dataset'
        ])
    

    def _organization_dirs(self, save_dir, dataset_name):
        """Determines where a dataset should be saved.

        The dataset will be written in a directory that depends on the
        partition size (e.g., 2mer).
        
        Parameters
        ----------
        save_dir : :obj:`str`
            Path to a common directory for GDML datasets.

        Returns
        -------
        :obj:`str`
            Path to save directory of data set.
        """
        save_dir = utils.norm_path(save_dir)
        # Preparing directories.
        if self.system_info['system'] == 'solvent':
            partition_size_dir = utils.norm_path(
                save_dir + str(self.system_info['cluster_size']) + 'mer'
            )
            os.makedirs(partition_size_dir, exist_ok=True)
            return partition_size_dir


    def create(self):
        """Creates and writes a data set.
        """
        # sGDML variables.
        dataset = {
            'type': np.array('d'),  # Designates dataset or model.
            'code_version': np.array(__version__),  # sGDML version.
            'name': np.array(self.name),  # Name of the output file.
            'theory': np.array(self.theory),  # Theory used to calculate the data.
            'z': np.array(self.z),  # Atomic numbers of all atoms in system.
            'R': np.array(self.R),  # Cartesian coordinates.
            'r_unit': np.array(self.r_unit),  # Units for coordinates.
            'E': np.array(self.E),  # Energy of the structures.
            'e_unit': np.array(self.e_unit),  # Units of energy.
            'E_min': np.array(self.E_min),  # Energy minimum.
            'E_max': np.array(self.E_max),  # Energy maximum.
            'E_mean': np.array(self.E_mean),  # Energy mean.
            'E_var': np.array(self.E_var),  # Energy variance.
            'F': np.array(self.F),  # Atomic forces for each atom.
            'F_min': np.array(self.F_min),  # Force minimum.
            'F_max': np.array(self.F_max),  # Force maximum.
            'F_mean': np.array(self.F_mean),  # Force mean.
            'F_var': np.array(self.F_var)  # Force variance.
        }
        # mbGDML variables.
        if self.z.ndim == 1:
            self.get_system_info(self.z.tolist())
        dataset = self.add_system_info(dataset)
        dataset['md5'] = np.array(sgdml_io.dataset_md5(dataset))
        self.dataset = dataset
     

    def from_combined(self, dataset_dir, name=None):
        """Combines multiple data sets into one.
        
        Finds all files labeled with 'dataset.npz' (defined in 
        PartitionCalcOutput.create_dataset) in a user specified directory
        and combines them. Typically used on a single partition size (e.g.,
        monomer, dimer, trimer, etc.) to represent the complete dataset.

        Parameters
        ----------
        dataset_dir : :obj:`str`
            Path to directory containing GDML data sets. Typically to a
            directory containing only a single partition size
            of a single solvent.
        """
        # Normalizes directories.
        partition_dir = utils.norm_path(dataset_dir)
        # Gets all GDML datasets within a partition directory.
        all_dataset_paths = utils.natsort_list(
            utils.get_files(partition_dir, '.npz')
        )
        # Prepares initial combined dataset from the first dataset found.
        self.load(all_dataset_paths[0])
        if name is not None:
            self.name = name
        # Adds the remaining datasets to the new dataset.
        index_dataset = 1
        while index_dataset < len (all_dataset_paths):
            dataset_add = np.load(all_dataset_paths[index_dataset])
            print('Adding %s dataset to %s ...' % (dataset_add.f.name[()],
                self.name))
            # Ensuring the datasets are compatible.
            try:
                assert np.array_equal(self.type, dataset_add.f.type)
                assert np.array_equal(self.z, dataset_add.f.z)
                assert np.array_equal(self.r_unit, dataset_add.f.r_unit)
                assert np.array_equal(self.e_unit, dataset_add.f.e_unit)
                if hasattr(self, 'system'):
                    assert np.array_equal(self.system, dataset_add.f.system)
                if hasattr(self, 'solvent'):
                    assert np.array_equal(self.solvent, dataset_add.f.solvent)
                if hasattr(self, 'cluster_size'):
                    assert np.array_equal(self.cluster_size,
                                        dataset_add.f.cluster_size)
            except AssertionError:
                base_filename = all_dataset_paths[0].split('/')[-1]
                incompat_filename = all_dataset_paths[index_dataset].split('/')[-1]
                raise ValueError('File %s is not compatible with %s' %
                                (incompat_filename, base_filename))
            # Seeing if the theory is always the same.
            # If theories are different, we change theory to 'unknown'.
            try:
                assert self.theory == str(dataset_add.f.theory[()])
            except AssertionError:
                self.theory = 'unknown'
            # Concatenating relevant arrays.
            self._R = np.concatenate((self.R, dataset_add.f.R), axis=0)
            self._E = np.concatenate((self.E, dataset_add.f.E), axis=0)
            self._F = np.concatenate((self.F, dataset_add.f.F), axis=0)
            index_dataset += 1
        if self.z.ndim == 1:
            self.get_system_info(self.z.tolist())
        self.create()
        


    def print(self):
        """Prints all structure coordinates, energies, and force of a data set.
        """
        num_config = self.R.shape[0]
        for config in range(num_config):
            print(f'-----Configuration {config}-----')
            print(f'Energy: {self.E[config]} kcal/mol')
            print(f'Forces:\n{self.F[config]}')

    
    def create_mb(self, ref_dataset, model_paths):
        """Creates a many-body data set.

        Removes energy and force predictions from the reference data set using
        GDML models in ``model_paths``.

        Parameters
        ----------
        ref_dataset : :obj:`mbgdml.data.mbGDMLDataset`
            Reference data set of structures, energies, and forces. This is the
            data where mbGDML predictions will be subtracted from.
        model_paths : :obj:`list` [:obj:`str`]
            Paths to saved many-body GDML models in the form of
            :obj:`numpy.lib.npyio.NpzFile`.
        """
        if not hasattr(self, 'dataset'):
            raise AttributeError('Please load a data set first.')
        print(f'Removing /contributions ...')
        predict = mbGDMLPredict(model_paths)
        self.dataset = predict.remove_nbody(ref_dataset.dataset)
        self._update