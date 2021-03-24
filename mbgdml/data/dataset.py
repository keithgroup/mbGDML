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
from random import randrange, sample
import numpy as np
from cclib.parser.utils import convertor
from mbgdml.data import mbGDMLData
import mbgdml.solvents as solvents
from mbgdml import __version__
from mbgdml.parse import parse_stringfile
from mbgdml import utils
from mbgdml.data import structureSet
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
    def Rset_md5(self):
        """Specifies structure sets (Rset) IDs/labels and MD5 hashes.

        Keys are the Rset IDs (:obj:`int`) and values are MD5 hashes
        (:obj:`str`) for this particular data set.

        :type: :obj:`dict`
        """
        if hasattr(self, '_Rset_md5'):
            return self._Rset_md5
        else:
            return {}
    
    @Rset_md5.setter
    def Rset_md5(self, var):
        self._Rset_md5 = var
    
    @property
    def Rset_info(self):
        """An array specifying where each structure in R originates from.

        A ``(n_R, 1 + n_z)`` array containing the Rset ID from ``Rset_md5`` in
        the first column and then the atom indices of the structure with respect
        to the full structure in the structure set, where ``n_R`` is the number
        of structures in R and ``n_z`` the number of atoms in each structure in
        this data set.

        If there has been no previous sampling, an array of shape (1, 0)
        is returned.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_Rset_info'):
            return self._Rset_info
        else:
            return np.array([[]])
    
    @Rset_info.setter
    def Rset_info(self, var):
        self._Rset_info = var

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
    
    def Rset_sample(
        self, structureset_path, num, size, always=[]
    ):
        """Adds structures from a structure set to the data set.

        Parameters
        ----------
        structureset_path : :obj:`str`
            Path to a mbGDML structure set.
        num : :obj:`int`
            Number of structures to sample from the structure set.
        size : :obj:`int`
            Desired number of molecules in each selection.
        always : :obj:`list` [:obj:`int`]
            Molecule indices that will be in every selection.
        """
        # Load structure set.
        Rset = structureSet(structureset_path)

        # Attempts to match any previous sampling to this structure set.
        # If not, adds this structure set to the Rset_md5 information.
        Rset_id = None
        if self.Rset_md5 != {}:
            new_k = 0  # New key should be one more than the last key.
            for k,v in self.Rset_md5.items():
                if v == Rset.md5:
                    Rset_id = k
                    break
                new_k += 1
            
            # If no matches.
            if Rset_id is None:
                Rset_id == new_k
        else:
            Rset_id = 0
        
        # Prepares sampling procedure.
        z = self.z
        R = self.R
        Rset_info = self.Rset_info
        Rset_z = Rset.z
        Rset_R = Rset.R
        max_n_R = Rset.R.shape[0]
        max_mol = Rset.mol_ids[-1] + 1
        mol_ids = Rset.mol_ids

        # New cluster routine.
        i = 0  # Successful samples from structure set.
        while i < num:
            # Generates our sample using random integers.
            struct_num_selection = randrange(max_n_R)
            mol_selection = sorted(sample(range(max_mol), size))
            Rset_selection = [Rset_id, struct_num_selection] + mol_selection

            # If this selection is already in Rset_info, then we will not include it
            # and try again.
            # If there has been no previous sampling, an array of shape (1, 0)
            # is returned.
            if Rset_info.shape[1] != 0:
                if (Rset_info[...]==Rset_selection).all(1).any():
                    continue

            # Adds new sampling into our Rset info.
            if Rset_info.shape[1] == 0:  # No previous Rset_info.
                Rset_axis = 1
            else:
                Rset_axis = 0
            Rset_info = np.concatenate(
                (Rset_info, np.array([Rset_selection])), axis=Rset_axis
            )

            # Adds selection's atomic coordinates to R.
            ## Gets atomic indices from molecule_ids in the Rset.
            atom_ids = []
            for mol_id in Rset_selection[2:]:
                atom_ids.extend(
                    [i for i,x in enumerate(mol_ids) if x == mol_id]
                )
            # Checks compatibility with atoms.
            if len(z) == 0:
                z = Rset_z[atom_ids]
            else:
                if not np.all([z, Rset_z[atom_ids]]):
                    print(f'z of data set: {z}')
                    print(f'Rset_info of selection: {Rset_selection}')
                    print(f'z of selection: {Rset_z[atom_ids]}')
                    raise ValueError(f'z of the selection is incompatible.')
            
            ## Adds selection's Cartesian coordinates to R.
            r_selection = np.array([Rset_R[struct_num_selection, atom_ids, :]])
            if R.shape[2] == 0:  # No previous R.
                R = r_selection
            else:
                R = np.concatenate(
                (R, r_selection),
                axis=0
            )
            
            i += 1
        
        # Stores all information only if sampling is successful.
        self.Rset_md5 = {**self.Rset_md5, **{Rset_id: Rset.md5}}
        self.Rset_info = Rset_info
        self.z = z
        self.R = R


        

        
    def forces_from_xyz(self, file_path):
        """Reads forces from xyz files.

        Parameters
        ----------
        file_path : :obj:`str`
            Path to xyz file.
        """
        self._user_data = True
        z, _, data = parse_stringfile(file_path)

        # Checks that z of this xyz file is compatible with the z from the data
        # set.
        z = [utils.atoms_by_number(i) for i in z]
        if len(set(tuple(i) for i in z)) == 1:
            z = np.array(z[0])
        else:
            z = np.array(z)
        assert np.all([z, self.z])

        # Stores Forces
        data = np.array(data)
        if data.shape[2] == 6:
            self._F = data[:,:,:3]
        elif data.shape[2] == 3:
            self._F = data
        else:
            raise ValueError(f'There was an issue parsing F from {file_path}.')

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
            'Rset_md5': np.array(self.Rset_md5),
            'z': np.array(self.z),
            'R': np.array(self.R),
            'r_unit': np.array(self.r_unit)
        }
        md5_properties = ['z', 'R']

        # When starting a new data set from a structure set, there will not be
        # any energy or force data. Thus, we try to add the data if available,
        # but will not error out if the data is not available.
        # Theory.
        try:
            dataset['theory'] = np.array(self.theory)
        except:
            pass

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