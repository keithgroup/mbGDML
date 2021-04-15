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
import itertools
from random import randrange, sample
import numpy as np
from cclib.parser.utils import convertor
from mbgdml.data import mbGDMLData
import mbgdml.solvents as solvents
from mbgdml import __version__ as mbgdml_version
from mbgdml.parse import parse_stringfile
from mbgdml import utils
from mbgdml.data import structureSet
from mbgdml.predict import mbPredict
from mbgdml.partition import partition_structures
  

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
    mb_models_md5 : :obj:`list` [:obj:`str`]
        The MD5 hash of the model used to remove n-body contributions from
        data set.
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
            return np.array([[]], dtype='int_')
    
    @Rset_info.setter
    def Rset_info(self, var):
        self._Rset_info = var
    
    @property
    def criteria(self):
        """Specifies structure criteria (if any) used to sample structures.

        The name of the function used.

        :type: :obj:`str`
        """
        if hasattr(self, '_criteria'):
            return self._criteria
        else:
            return ''
    
    @criteria.setter
    def criteria(self, var):
        self._criteria = var
    
    @property
    def z_slice(self):
        """Specifies z_slice used for criteria.

        The indices of atoms in each structure used by the criteria function.

        :type: :obj:`list`
        """
        if hasattr(self, '_z_slice'):
            return self._z_slice
        else:
            return []
    
    @z_slice.setter
    def z_slice(self, var):
        self._z_slice = var
    
    @property
    def cutoff(self):
        """Specifies cutoff(s) for the structure criteria.

        :type: :obj:`list`
        """
        if hasattr(self, '_cutoff'):
            return self._cutoff
        else:
            return []
    
    @cutoff.setter
    def cutoff(self, var):
        self._cutoff = var

    @property
    def F(self):
        """Atomic forces of atoms in structure(s).
        
        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of structures and ``n`` is the number of atoms with three 
        Cartesian components.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_F'):
            return self._F
        else:
            return np.array([[[]]])

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
            return np.array([])

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
        """Unique MD5 hash of data set.

        :type: :obj:`bytes`
        """
        return self.dataset['md5'][()].decode()

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
        self.name = str(dataset['name'][()])
        self._z = dataset['z']
        self._R = dataset['R']
        self._E = dataset['E']
        self._F = dataset['F']
        self._r_unit = str(dataset['r_unit'][()])
        try:
            self._criteria = str(dataset['criteria'][()])
            self._z_slice = dataset['z_slice'][()]
            self._cutoff = dataset['cutoff'][()]
        except KeyError:
            # Some old data sets will not have this information.
            pass
        try:
            self._e_unit = str(dataset['e_unit'][()])
        except KeyError:
            self._e_unit = 'N/A'
        try:
            self.mbgdml_version = str(dataset['mbgdml_version'][()])
        except KeyError:
            # Some old data sets do not have this information.
            # This is unessential, so we will just ignore this.
            pass
        try:
            self.theory = str(dataset['theory'][()])
        except KeyError:
            self.theory = 'N/A'
        # mbGDML added data set information.
        if 'mb' in dataset.keys():
            self.mb = int(dataset['mb'][()])
            self.mb_models_md5 = dataset['mb_models_md5'].astype('U32')
        try:
            self.Rset_info = dataset['Rset_info'][()]
            self.Rset_md5 = dataset['Rset_md5'][()]
        except KeyError:
            pass

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

    def _get_Rset_id(
        self, Rset
    ):
        """Determines the numerical Rset ID for this data set.

        Parameters
        ----------
        Rset : :obj:`str`
            A loaded :obj:`mbgdml.data.structureSet` object.
        
        Returns
        -------
        :obj:`int`
            Numerical ID for the Rset.
        """
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
                Rset_id = new_k
        else:
            Rset_id = 0

        return Rset_id
    
    def _Rset_sample_all(
        self, z, R, E, F, Rset, Rset_id, Rset_info, size
    ):
        """Selects all Rset structures for data set.

        Generally organized by adding all structures of a single mol_id one at
        a time. For example, if we were adding all monomers from a cluster with
        two water molecules, we would select the first molecule (``0``), add 
        all of its information, then add the second molecule (``1``). This
        method was chosen due to a previous methodology used to calculate
        energy and gradients of partitions.

        Parameters
        ----------
        z : :obj:`numpy.ndarray`
            Atomic numbers of the atoms in every structure prior to sampling.
        R : :obj:`numpy.ndarray`
            Cartesian atomic coordinates of data set structures prior to
            sampling.
        E : :obj:`numpy.ndarray`
            Energies of data set structures prior to sampling.
        F : :obj:`numpy.ndarray`
            Atomic forces of data set structures prior to sampling.
        Rset : :obj:`mbgdml.data.structureSet`
            A loaded structure set object.
        Rset_id : :obj:`int`
            The :obj:`int` that specifies the Rset (key in ``self.Rset_md5``).
        Rset_info : :obj:`int`
            An array specifying where each structure in R originates from.
        size : :obj:`int`
            Desired number of molecules in each selection.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            An array specifying where each structure in R originates from.
        :obj:`numpy.ndarray`
            Atomic coordinates of structure(s).
        :obj:`numpy.ndarray`
            The energies of structure(s). All are NaN.
        :obj:`numpy.ndarray`
            Atomic forces of atoms in structure(s). All are NaN.
        """
        Rset_z = Rset.z
        Rset_R = Rset.R

        # Getting all possible molecule combinations.
        mol_ids = Rset.mol_ids
        max_mol_i = max(mol_ids)
        comb_list = list(itertools.combinations(range(max_mol_i + 1), size))
        
        # Loops though every possible molecule combination.
        for comb in comb_list:
            
            # Adds Rset_id information for every structure in this combination.
            for struct_i in range(Rset.R.shape[0]):
                Rset_selection = [Rset_id, struct_i] + list(comb)
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
            
            # Adds selection's Cartesian coordinates to R.
            r_selection = np.array(Rset_R[:, atom_ids, :])
            if R.shape[2] == 0:  # No previous R.
                R = r_selection
            else:
                R = np.concatenate((R, r_selection), axis=0)

            # Adds NaN for energies.
            e_selection = np.empty((Rset.R.shape[0]))
            e_selection[:] = np.NaN
            if len(E.shape) == 0:  # No previous E.
                E = e_selection
            else:
                E = np.concatenate((E, e_selection), axis=0)

            # Adds NaN for forces.
            ## Force array will be the same shape as R.
            ## So we just take the r_selection array and make all values nan.
            f_selection = np.copy(r_selection)
            f_selection[:] = np.NaN
            if F.shape[2] == 0:  # No previous F.
                F = f_selection
            else:
                F = np.concatenate((F, f_selection), axis=0)
        

        return (Rset_info, z, R, E, F)
    
    def _Rset_sample_num(
        self, z, R, E, F, Rset, Rset_id, Rset_info, quantity, size,
        criteria=None, z_slice=[], cutoff=[]
    ):
        """Samples a structure set for data set.

        Parameters
        ----------
        z : :obj:`numpy.ndarray`
            Atomic numbers of the atoms in every structure prior to sampling.
        R : :obj:`numpy.ndarray`
            Cartesian atomic coordinates of data set structures prior to
            sampling.
        E : :obj:`numpy.ndarray`
            Energies of data set structures prior to sampling.
        F : :obj:`numpy.ndarray`
            Atomic forces of data set structures prior to sampling.
        Rset : :obj:`mbgdml.data.structureSet`
            A loaded structure set object.
        Rset_id : :obj:`int`
            The :obj:`int` that specifies the Rset (key in ``self.Rset_md5``).
        Rset_info : :obj:`int`
            An array specifying where each structure in R originates from.
        quantity : :obj:`int`
            Number of structures to sample from the structure set.
        size : :obj:`int`
            Desired number of molecules in each selection.
        criteria : :obj:`mbgdml.sample.sampleCritera`
            Structure criteria during the sampling procedure.
        z_slice : :obj:`numpy.ndarray`
            Indices of the atoms to be used for the cutoff calculation.
        cutoff : :obj:`list`
            Distance cutoff between the atoms selected by ``z_slice``. Must be
            in the same units (e.g., Angstrom) as ``R``.

        Returns
        -------
        :obj:`numpy.ndarray`
            An array specifying where each structure in R originates from.
        :obj:`numpy.ndarray`
            Atomic coordinates of structure(s).
        :obj:`numpy.ndarray`
            The energies of structure(s). All are NaN.
        :obj:`numpy.ndarray`
            Atomic forces of atoms in structure(s). All are NaN.
        """
        Rset_z = Rset.z
        Rset_R = Rset.R

        max_n_R = Rset.R.shape[0]
        max_mol = Rset.mol_ids[-1] + 1
        mol_ids = Rset.mol_ids

        # New cluster routine.
        i = 0  # Successful samples from structure set.
        while i < quantity:
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
            
            r_selection = np.array([Rset_R[struct_num_selection, atom_ids, :]])

            # Checks any structure criteria.
            if criteria is not None:
                # r_selection is 3 dimensions (to make it compatible to
                # concatenate). So we make need to select the first (and only)
                # structure.
                if not criteria(z, r_selection[0], z_slice, cutoff):
                    # If criteria is not met, will not include sample.
                    continue
            
            # Adds selection's Cartesian coordinates to R.
            if R.shape[2] == 0:  # No previous R.
                R = r_selection
            else:
                R = np.concatenate((R, r_selection), axis=0)

            # Adds NaN for energies.
            e_selection = np.array([np.NaN])
            if len(E.shape) == 0:  # No previous E.
                E = e_selection
            else:
                E = np.concatenate((E, e_selection), axis=0)

            # Adds NaN for forces.
            ## Force array will be the same shape as R.
            ## So we just take the r_selection array and make all values nan.
            f_selection = np.copy(r_selection)
            f_selection[:] = np.NaN
            if F.shape[2] == 0:  # No previous F.
                F = f_selection
            else:
                F = np.concatenate((F, f_selection), axis=0)

            i += 1
        
        return (Rset_info, z, R, E, F)
    
    def Rset_sample(
        self, Rset, quantity, size, always=[], criteria=None, z_slice=[],
        cutoff=[]
    ):
        """Adds structures from a structure set to the data set.

        Parameters
        ----------
        Rset : :obj:`mbgdml.data.structureSet`
            A loaded structure set object.
        quantity : :obj:`int`
            Number of structures to sample from the structure set. For example,
            ``'100'``, ``'452'``, or even ``'all'``.
        size : :obj:`str`
            Desired number of molecules in each selection.
        always : :obj:`list` [:obj:`int`], optional
            Molecule indices that will be in every selection.
        criteria : :obj:`mbgdml.sample.sampleCritera`
            Structure criteria during the sampling procedure.
        z_slice : :obj:`numpy.ndarray`
            Indices of the atoms to be used for the cutoff calculation.
        cutoff : :obj:`list`
            Distance cutoff between the atoms selected by ``z_slice``. Must be
            in the same units (e.g., Angstrom) as ``R``.
        """
        # Gets Rset_id for this new sampling.
        Rset_id = self._get_Rset_id(Rset)
        
        # Prepares sampling procedure.
        z = self.z
        R = self.R
        E = self.E
        F = self.F

        Rset_info = self.Rset_info

        if type(quantity) == 'int' or str(quantity).isdigit():
            quantity = int(quantity)
            Rset_info, z, R, E, F = self._Rset_sample_num(
                z, R, E, F, Rset, Rset_id, Rset_info, quantity, size,
                criteria=criteria, z_slice=z_slice, cutoff=cutoff
            )
            # Adds criteria information to the data set (only if sampling is 
            # successful).
            if criteria is not None:
                self.criteria = criteria.__name__
                self.z_slice = z_slice
                self.cutoff = cutoff
        elif quantity == 'all':
            Rset_info, z, R, E, F = self._Rset_sample_all(
                z, R, E, F, Rset, Rset_id, Rset_info, size
            )
        else:
            raise ValueError(f'{quantity} is not a valid selection.')
        
        # Stores all information only if sampling is successful.
        self.Rset_md5 = {**self.Rset_md5, **{Rset_id: Rset.md5}}
        self.Rset_info = Rset_info
        self.z = z
        self.R = R
        self.E = E
        self.F = F

        if hasattr(self, '_r_unit'):
            if self.r_unit != Rset.r_unit:
                raise ValueError('r_unit of Rset is not compatible with dset.')
        else:
            self._r_unit = Rset.r_unit
        
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
        self.mbgdml_version = mbgdml_version
        self.theory = partcalc.theory

    @property
    def dataset(self):
        """Contains all data as :obj:`numpy.ndarray` objects.

        :type: :obj:`dict`
        """
        # Data always available for data sets.
        dataset = {
            'type': np.array('d'),
            'mbgdml_version': np.array(mbgdml_version),
            'name': np.array(self.name),
            'Rset_md5': np.array(self.Rset_md5),
            'Rset_info': np.array(self.Rset_info),
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

        # mbGDML information.
        dataset = self.add_system_info(dataset)
        if hasattr(self, 'mb'):
            dataset['mb'] = np.array(self.mb)
            dataset['mb_models_md5'] = np.array(self.mb_models_md5, dtype='S32')

        try:
            dataset['criteria'] = np.array(self.criteria)
            dataset['z_slice'] = np.array(self.z_slice)
            dataset['cutoff'] = np.array(self.cutoff)
        except:
            pass
        
        # sGDML only works with S32 type MD5 hashes, so during training the 
        # data set MD5 mush be the same type (as they do comparisons).
        dataset['md5'] = np.array(
            utils.md5_data(dataset, md5_properties), dtype='S32'
        )
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
            rset_info = self.Rset_info[config]
            print(f'-----Configuration {config}-----')
            print(f'Rset_id: {int(rset_info[0])}     '
                  f'Structure index: {int(rset_info[1])}')
            print(f'Molecule indices: {rset_info[2:]}')
            print(f'Coordinates:\n{self.R[config]}')
            print(f'Energy: {self.E[config]}')
            print(f'Forces:\n{self.F[config]}\n')
    
    def write_xyz(self, save_dir):
        """Saves xyz file of all structures in data set.

        Parameters
        ----------
        save_dir : :obj:`str`
        """
        utils.write_xyz(self.z, self.R, save_dir, self.name)

    
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
        predict = mbPredict(model_paths)
        dataset = predict.remove_nbody(ref_dataset.dataset)
        self._update(dataset)