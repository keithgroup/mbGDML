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
import numpy as np
from cclib.parser.utils import convertor
from mbgdml.data import mbGDMLData
import mbgdml.solvents as solvents
from mbgdml import __version__
from mbgdml.parse import parse_stringfile
from mbgdml import utils
from mbgdml.predict import mbGDMLPredict
  

class dataSet(mbGDMLData):
    """For creating, loading, manipulating, and using data sets.

    Parameters
    ----------
    dataset_path : :obj:`str`, optional
        Path to a saved :obj:`numpy.NpzFile`.

    Attributes
    ----------
    name : :obj:`str`
        Name of the data set. Defaults to ``'dataset'``.
    theory : :obj:`str`
        The level of theory used to compute energy and gradients of the data
        set.
    mb : :obj:`int`
        The order of n-body corrections this data set is intended for. For
        example, a tetramer solvent cluster with one-, two-, and three-body
        contributions removed will have a ``mb`` order of 4.
    """

    def __init__(self, *args):
        self.type = 'd'
        self.name = 'dataset'
        if len(args) == 1:
            self.load(args[0])

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
    def e_unit(self):
        """Units of energy. Options are ``'eV'``, ``'hartree'``,
        ``'kcal/mol'``, and ``'kJ/mol'``.

        :type: :obj:`str`
        """
        return self._e_unit

    @e_unit.setter
    def e_unit(self, var):
        self._e_unit = var

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
        return self.dataset['md5'][()]

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
        try:
            self.mbgdml_version = str(dataset['mbgdml_version'][()])
        except KeyError:
            # Some old data sets do not have this information.
            # This is unessential, so we will just ignore this.
            pass
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
        
    def forces_from_xyz(
        self, file_path, xyz_type, r_units, e_units, 
    ):
        """Reads data from xyz files.

        Parameters
        ----------
        file_path : :obj:`str`
            Path to xyz file.
        xyz_type : :obj:`str`
            Type of data. Either ``'coords'`` or ``'extended'``. Will discard
            any extended data.
        r_units : :obj:`str`, optional
            Units of distance. Options are ``'Angstrom'`` or ``'bohr'``.
        """
        self._user_data = True
        z, _, data = parse_stringfile(file_path)
        z = [utils.atoms_by_number(i) for i in z]

        # If all the structures have the same order of atoms (as required by
        # sGDML), condense into a one-dimensional array.
        if len(set(tuple(i) for i in z)) == 1:
            z = np.array(z[0])
        else:
            z = np.array(z)
        self._z = z

        # Stores Cartesian coordinates.
        data = np.array(data)
        if xyz_type == 'extended':
            self._R = data[:,:,3:]
        elif xyz_type == 'coords':
            self._R = data
        else:
            raise ValueError(f'{xyz_type} is not a valid xyz data type.')

        if r_unit is not None:
            self.r_unit = r_unit

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
        self.mbgdml_version = __version__
        self.theory = partcalc.theory

    @property
    def dataset(self):
        """Contains all data as :obj:`numpy.ndarray` objects.

        :type: :obj:`dict`
        """
        # Data always available for data sets.
        dataset = {
            'type': np.array('d'),
            'mbgdml_version': np.array(__version__),
            'name': np.array(self.name),
            'theory': np.array(self.theory),
            'z': np.array(self.z),
            'R': np.array(self.R),
            'r_unit': np.array(self.r_unit)
        }
        md5_properties = ['z', 'R']

        # When starting a new data set from a structure set, there will not be
        # any energy or force data. Thus, we try to add the data if available,
        # but will not error out if the data is not available.
        # Energies.
        try:
            dataset['E'] = np.array(self.E)
            dataset['e_unit'] = np.array(self.e_unit)
            dataset['E_min'] = np.array(self.E_min)
            dataset['E_max'] = np.array(self.E_max)
            dataset['E_mean'] = np.array(self.E_mean)
            dataset['E_var'] = np.array(self.E_var)
            md5_properties.append('E')
        except:
            pass
        
        # Forces.
        try:
            dataset['F'] = np.array(self.F)
            dataset['F_min'] = np.array(self.F_min)
            dataset['F_max'] = np.array(self.F_max)
            dataset['F_mean'] = np.array(self.F_mean)
            dataset['F_var'] = np.array(self.F_var)
            md5_properties.append('F')
        except:
            pass

        # mbGDML variables.
        dataset = self.add_system_info(dataset)
        dataset['md5'] = np.array(utils.md5_data(dataset, md5_properties))
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
        # pylint: disable=E1101
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
        """Prints all structure coordinates, energies, and forces of a data set.
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
        ref_dataset : :obj:`~mbgdml.data.dataset.dataSet`
            Reference data set of structures, energies, and forces. This is the
            data where mbGDML predictions will be subtracted from.
        model_paths : :obj:`list` [:obj:`str`]
            Paths to saved many-body GDML models in the form of
            :obj:`numpy.NpzFile`.
        """
        predict = mbGDMLPredict(model_paths)
        dataset = predict.remove_nbody(ref_dataset.dataset)
        self._update(dataset)