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
from cclib.parser.utils import convertor
from sgdml import __version__
from sgdml.utils import io as sgdml_io
from mbgdml.data import mbGDMLData
import mbgdml.solvents as solvents
from mbgdml.parse import parse_stringfile
from mbgdml import utils
from mbgdml.predict import mbGDMLPredict
  

class mbGDMLDataset(mbGDMLData):
    """For creating, loading, manipulating, and using data sets.

    Parameters
    ----------
    dataset_path : :obj:`str`, optional
        Path to a saved :obj:`numpy.NpzFile`.

    Attributes
    ----------
    name : :obj:`str`
        Name of the data set. Defaults to ``'dataset'``.
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


    def __init__(self, path=None):
        self.type = 'd'
        self.name = 'dataset'
        if path is not None:
            self.load(path)
    

    @property
    def F(self):
        """Atomic forces of atoms in structure(s).
        
        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of structures and ``n`` is the number of atoms with three 
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


    def convertF(self, force_e_units, force_r_units, e_units, r_units):
        """Convert forces.

        Does not change :attr:`e_unit` or :attr:`r_unit`.

        Parameters
        ----------
        force_e_units : :obj:`str`
            Specifies package-specific energy units used in calculation.
            Available units are ``'eV'``, ``'hartree'``, ``'kcal/mol'``, and
            ``'kJ/mol'``.
        force_r_units : :obj:`str`
            Specifies package-specific distance units used in calculation.
            Available units are ``'Angstrom'`` and ``'bohr'``.
        e_units : :obj:`str`
            Desired units of energy. Available units are ``'eV'``,
            ``'hartree'``, ``'kcal/mol'``, and ``'kJ/mol'``.
        r_units : :obj:`str`
            Desired units of distance. Available units are ``'Angstrom'`` and
            ``'bohr'``.
        """
        self._F = utils.convert_forces(
            self.F, force_e_units, force_r_units, e_units, r_units
        )


    def _update(self, dataset):
        """Updates object attributes.

        Parameters
        ----------
        dataset : :obj:`dict`
            Contains all information and arrays stored in data set.
        """
        self._z = dataset['z']
        self._R = dataset['R']
        self._E = dataset['E']
        self._F = dataset['F']
        self._r_unit = str(dataset['r_unit'][()])
        self._e_unit = str(dataset['e_unit'][()])
        self.code_version = str(dataset['code_version'][()])
        self.name = str(dataset['name'][()])
        self.theory = str(dataset['theory'][()])
        # mbGDML added data set information.
        if 'mb' in dataset.keys():
            self.mb = int(dataset['mb'][()])

    def load(self, dataset_path):
        """Read data set.

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
        forces. If it is ``'extended'`` the three rightmost data will be stored
        as forces.
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
        partcalc : :obj:`~mbgdml.data.calculation.PartitionOutput`
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

    @property
    def dataset(self):
        """Contains all data as :obj:`numpy.ndarray` objects.

        :type: :obj:`dict`
        """
        # sGDML variables.
        dataset = {
            'type': np.array('d'),  # Designates dataset.
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
        dataset = self.add_system_info(dataset)
        dataset['md5'] = np.array(sgdml_io.dataset_md5(dataset))
        return dataset
     

    def from_combined(self, dataset_paths, name=None):
        """Combines multiple data sets into one.
        
        Typically used on a single partition size (e.g., monomer, dimer, trimer,
        etc.) to represent the complete dataset.

        Parameters
        ----------
        dataset_paths : :obj:`list` [:obj:`str`]
            Paths to data sets to combine.
        name : :obj:`str`, optional
            Name for the combined data set.
        """
        # Prepares initial combined dataset from the first dataset.
        self.load(dataset_paths[0])
        if name is not None:
            self.name = name
        # Adds the remaining datasets to the new dataset.
        for dataset_path in dataset_paths[1:]:
            dataset_add = np.load(dataset_path)
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
                base_filename = dataset_paths[0].split('/')[-1]
                incompat_filename = dataset_path.split('/')[-1]
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
        ref_dataset : :obj:`~mbgdml.data.dataset.mbGDMLDataset`
            Reference data set of structures, energies, and forces. This is the
            data where mbGDML predictions will be subtracted from.
        model_paths : :obj:`list` [:obj:`str`]
            Paths to saved many-body GDML models in the form of
            :obj:`numpy.NpzFile`.
        """
        print(f'Removing /contributions ...')
        predict = mbGDMLPredict(model_paths)
        dataset = predict.remove_nbody(ref_dataset.dataset)
        self._update(dataset)