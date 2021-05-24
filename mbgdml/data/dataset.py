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
from mbgdml import __version__ as mbgdml_version
from mbgdml.parse import parse_stringfile
from mbgdml import utils
from mbgdml.data import structureSet
from mbgdml.predict import mbPredict
  

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
    mb_models_md5 : :obj:`numpy.ndarray`
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
            return np.array([])
    
    @z_slice.setter
    def z_slice(self, var):
        self._z_slice = np.array(var)
    
    @property
    def cutoff(self):
        """Specifies cutoff(s) for the structure criteria.

        :type: :obj:`list`
        """
        if hasattr(self, '_cutoff'):
            return self._cutoff
        else:
            return np.array([])
    
    @cutoff.setter
    def cutoff(self, var):
        self._cutoff = np.array(var)

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
        try:
            return self.dataset['md5'][()].decode()
        except:
            print('Not enough information in dset for MD5')
            raise
    
    @property
    def entity_ids(self):
        """An array specifying which atoms belong to what entities
        (e.g., molecules). Similar to PDBx/mmCIF ``_atom_site.label_entity_ids``
        data item.

        For example, a water and methanol molecule could be
        ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_entity_ids'):
            return self._entity_ids
        else:
            return np.array([])
    
    @entity_ids.setter
    def entity_ids(self, var):
        self._entity_ids = np.array(var)
    
    @property
    def comp_ids(self):
        """A 2D array relating ``entity_ids`` to a chemical component/species
        id or label (``comp_id``). The first column is the unique ``entity_id``
        and the second is a unique ``comp_id`` for that chemical species.
        Each ``comp_id`` is reused for the same chemical species.

        For example, two water and one methanol molecules could be
        ``[['0', 'h2o'], ['1', 'h2o'], ['2', 'meoh']]``.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_comp_ids'):
            return self._comp_ids
        else:
            return np.array([[]])
    
    @comp_ids.setter
    def comp_ids(self, var):
        self._comp_ids = np.array(var)

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
        if 'mb_models_md5' in dataset.keys():
            self.mb_models_md5 = dataset['mb_models_md5']
        if 'mb_dsets_md5' in dataset.keys():
            self.mb_dsets_md5 = dataset['mb_dsets_md5']

        try:
            self.Rset_info = dataset['Rset_info'][()]
            self.Rset_md5 = dataset['Rset_md5'][()]
            self.entity_ids = dataset['entity_ids']
            self.comp_ids = dataset['comp_ids']
        except KeyError:
            pass
        
        if 'centered' in dataset.keys():
            self.centered = dataset['centered'][()]

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
        self, data, selected_rset_id=None
    ):
        """Determines the numerical Rset ID for this data set.

        Parameters
        ----------
        Rset : :obj:`mbgdml.data`
            A loaded :obj:`mbgdml.data.structureSet` or
            :obj:`mbgdml.data.dataSet` object.
        selected_rset_id : obj:`int`, optional
            Currently dset sampling can only be done for one rset_id at a time.
            This specifies which rset structures in the data set to sample from.
            Defaults to ``None``.
        
        Returns
        -------
        :obj:`int`
            Numerical ID for the Rset.
        """
        # Attempts to match any previous sampling to this structure set.
        # If not, adds this structure set to the Rset_md5 information.
        Rset_id = None
        if self.Rset_md5 != {}:
            if data.type == 's':
                md5 = data.md5
            elif data.type == 'd':
                md5 = data.Rset_md5[selected_rset_id]
            new_k = 0  # New key should be one more than the last key.
            for k,v in self.Rset_md5.items():
                if v == md5:
                    Rset_id = k
                    break
                new_k += 1
            
            # If no matches.
            if Rset_id is None:
                Rset_id = new_k
        else:
            Rset_id = 0

        return Rset_id
    
    def _center_structures(self, z, R):
        """Centers each structure's center of mass to the origin.

        Previously centered structures should not be affected by this technique.

        Parameters
        ----------
        z : :obj:`numpy.ndarray`
            Atomic numbers of the atoms in every structure.
        R : :obj:`numpy.ndarray`
            Cartesian atomic coordinates of data set structures.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            Centered Cartesian atomic coordinates.
        """
        # Masses of each atom in the same shape of R.
        if R.ndim == 2:
            R = np.array([R])
        
        masses = np.empty(R[0].shape)
        
        for i in range(len(masses)):
            masses[i,:] = utils.z_to_mass[z[i]]
        
        for i in range(len(R)):
            r = R[i]
            cm_r = np.average(r, axis=0, weights=masses)
            R[i] = r - cm_r
        
        if R.shape[0] == 1:
            return R[0]
        else:
            return R
    
    def _check_entity_comp_ids(
        self, entity_ids, comp_ids, data_entity_ids, data_comp_ids,
        sampled_entity_ids_split
    ):
        """
        
        Parameters
        ----------
        entity_ids : :obj:`numpy.ndarray`
            Already sampled entity_ids of the data set. Could be an empty array.
        comp_ids : :obj:`numpy.ndarray`
            Already sampled comp_ids of the data set. Could be an empty array.
        data_entity_ids :obj:`numpy.ndarray`
            entity_ids of a data or structure set being sampled.
        data_comp_ids :obj:`numpy.ndarray`
            comp_ids of a data or structure set being sampled.
        sampled_entity_ids_split : :obj:`list` [:obj:`numpy.ndarray`]
            The unique data entity_ids of each new entity for this data set.
            For example, all the data entity_ids (from a structure set) that
            are included as entity_id = 0 in this data set.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            The correct entity_ids of this data set.
        :obj:`numpy.ndarray`
            The correct comp_ids of this data set.
        """
        if len(entity_ids) == 0 and comp_ids.shape == (1, 0):
            # If there is no previous sampling.
            # Just need to check that the new entities are self compatible.
            entity_ids = []  # Start a new entity id list.
            comp_ids = []  # Start a new component id list.
            
            # Loops through every column/entity that we sampled.
            for entity_id in range(len(sampled_entity_ids_split)):
                # Gets the size of the entity in the first structure.

                data_entity_id = sampled_entity_ids_split[entity_id][0]
                ref_entity_size = np.count_nonzero(
                    data_entity_ids == data_entity_id
                )
                # Adds the entity_ids of this entity
                entity_ids.extend([entity_id for _ in range(ref_entity_size)])
                # Adds comp_id
                comp_id = data_comp_ids[data_entity_id][1]
                comp_ids.append(
                    [str(entity_id), comp_id]
                )
                
                # We should not have to check the entities because we already
                # check z.

            entity_ids = np.array(entity_ids)
            comp_ids = np.array(comp_ids)
        else:
            # If there was previous sampling
            # Need to also check if compatible with the data set.
            # We should not have to check the entities because we already
            # check z.
            pass
        
        return entity_ids, comp_ids
    
    def _sample_all(
        self, z, R, E, F, sample_data, Rset_id, Rset_info, size, criteria=None,
        z_slice=[], cutoff=[], sampling_updates=False
    ):
        """Selects all Rset structures for data set.

        Generally organized by adding all structures of a single entity_id one at
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
        sample_data : :obj:`mbgdml.data`
            A loaded structure or data set object.
        Rset_id : :obj:`int`
            The :obj:`int` that specifies the Rset (key in ``self.Rset_md5``).
        Rset_info : :obj:`int`
            An array specifying where each structure in R originates from.
        size : :obj:`int`
            Desired number of molecules in each selection.
        criteria : :obj:`mbgdml.sample.sampleCritera`, optional
            Structure criteria during the sampling procedure. Defaults to
            ``None`` if no criteria should be used.
        z_slice : :obj:`numpy.ndarray`, optional
            Indices of the atoms to be used for the cutoff calculation. Defaults
            to ``[]`` is no criteria is selected or if it is not required for
            the selected criteria.
        cutoff : :obj:`list`, optional
            Distance cutoff between the atoms selected by ``z_slice``. Must be
            in the same units (e.g., Angstrom) as ``R``. Defaults to ``[]`` if
            no criteria is selected or a cutoff is not desired.
        sampling_updates : :obj:`bool`, optional
            Will print something for every 100 successfully sampled structures.
            Defaults to ``False``.
        
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
        sample_data_z = sample_data.z
        sample_data_R = sample_data.R

        # Getting all possible molecule combinations.
        entity_ids = sample_data.entity_ids
        max_entity_i = max(entity_ids)
        comb_list = list(itertools.combinations(range(max_entity_i + 1), size))
        n_comb = len(comb_list)

        # Adds Rset_id information for every structure in this combination.
        num_R = sample_data_R.shape[0]
        for struct_i in range(num_R):
            if sampling_updates and (struct_i+1)%500 == 0:
                print(
                    f'Sampled clusters from {struct_i+1} out of {num_R} structures'
                )
            if sampling_updates and struct_i+1 == num_R:
                print(f'Sampled all {num_R*n_comb} possible structures')
            
            # Loops though every possible molecule combination.
            for comb in comb_list:

                Rset_selection = [Rset_id, struct_i] + list(comb)
                # Adds new sampling into our Rset info.
                if Rset_info.shape[1] == 0:  # No previous Rset_info.
                    Rset_axis = 1
                else:
                    Rset_axis = 0
                    # Checks to see if combination is already in data set.
                    if (Rset_info[...]==Rset_selection).all(1).any():
                        # Does not add the combination.
                        continue

                # Gets atomic indices from entity_ids in the Rset.
                atom_ids = []
                for entity_id in Rset_selection[2:]:
                    atom_ids.extend(
                        [i for i,x in enumerate(entity_ids) if x == entity_id]
                    )
                
                # Checks compatibility with atoms.
                if len(z) == 0:
                    z = sample_data_z[atom_ids]
                else:
                    if not np.all([z, sample_data_z[atom_ids]]):
                        print(f'z of data set: {z}')
                        print(f'Rset_info of selection: {Rset_selection}')
                        print(f'z of selection: {sample_data_z[atom_ids]}')
                        raise ValueError(f'z of the selection is incompatible.')
                
                # Checks any structure criteria.
                r_selection = np.array([sample_data_R[struct_i, atom_ids, :]])
                r_entity_ids = entity_ids[atom_ids]
                if criteria is not None:
                    # r_selection is 3 dimensions (to make it compatible to
                    # concatenate). So we make need to select the first (and only)
                    # structure.
                    accept_r, _ = criteria(
                        z, r_selection[0], z_slice, r_entity_ids, cutoff
                    )
                    if not accept_r:
                        # If criteria is not met, will not include sample.
                        continue
                
                # SUCCESSFUL SAMPLE

                Rset_info = np.concatenate(
                    (Rset_info, np.array([Rset_selection])), axis=Rset_axis
                )
                
                if R.shape[2] == 0:  # No previous R.
                    R = r_selection
                else:
                    R = np.concatenate((R, r_selection), axis=0)

                # Adds NaN for energies.
                e_selection = np.empty((1,))
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
    
    def _sample_num(
        self, z, R, E, F, Rset, Rset_id, Rset_info, quantity, size,
        criteria=None, z_slice=[], cutoff=[], sampling_updates=False
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
        criteria : :obj:`mbgdml.sample.sampleCritera`, optional
            Structure criteria during the sampling procedure. Defaults to
            ``None`` if no criteria should be used.
        z_slice : :obj:`numpy.ndarray`, optional
            Indices of the atoms to be used for the cutoff calculation. Defaults
            to ``[]`` is no criteria is selected or if it is not required for
            the selected criteria.
        cutoff : :obj:`list`, optional
            Distance cutoff between the atoms selected by ``z_slice``. Must be
            in the same units (e.g., Angstrom) as ``R``. Defaults to ``[]`` if
            no criteria is selected or a cutoff is not desired.
        sampling_updates : :obj:`bool`, optional
            Will print something for every 100 successfully sampled structures.
            Defaults to ``False``.

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
        entity_ids = Rset.entity_ids
        max_entity_id = max(entity_ids)

        # New cluster routine.
        i = 0  # Successful samples from structure set.
        while i < quantity:
            # Generates our sample using random integers.
            struct_num_selection = randrange(max_n_R)
            comp_selection = sorted(sample(range(max_entity_id + 1), size))
            Rset_selection = [Rset_id, struct_num_selection] + comp_selection

            # If this selection is already in Rset_info, then we will not include it
            # and try again.
            # If there has been no previous sampling, an array of shape (1, 0)
            # is returned.
            if Rset_info.shape[1] != 0:
                if (Rset_info[...]==Rset_selection).all(1).any():
                    continue

            # Adds selection's atomic coordinates to R.
            ## Gets atomic indices from molecule_ids in the Rset.
            atom_idx = utils.get_R_slice(Rset_selection[2:], entity_ids)
            
            # Checks compatibility with atoms.
            if len(z) == 0:
                z = Rset_z[atom_idx]
            else:
                if not np.array_equal(z, Rset_z[atom_idx]):
                    print(f'Rset_info of selection: {Rset_selection}')
                    print(f'z of data set: {z}')
                    print(f'z of selection: {Rset_z[atom_idx]}')
                    raise ValueError(f'z of the selection is incompatible.')
            
            r_selection = np.array([Rset_R[struct_num_selection, atom_idx, :]])
            r_entity_ids = entity_ids[atom_idx]

            # Checks any structure criteria.
            if criteria is not None:
                # r_selection is 3 dimensions (to make it compatible to
                # concatenate). So we make need to select the first (and only)
                # structure.
                accept_r, _ = criteria(
                    z, r_selection[0], z_slice, r_entity_ids, cutoff
                )
                if not accept_r:
                    # If criteria is not met, will not include sample.
                    continue
            
            # SUCCESSFUL SAMPLE
            if Rset_info.shape[1] == 0:  # No previous Rset_info.
                Rset_axis = 1
            else:
                Rset_axis = 0
            Rset_info = np.concatenate(
                (Rset_info, np.array([Rset_selection])), axis=Rset_axis
            )
            
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

            if sampling_updates:
                if i%100 == 0:
                    print(f'Successfully found {i} structures')
        
        return (Rset_info, z, R, E, F)
    
    def Rset_sample(
        self, Rset, quantity, size, always=[], criteria=None, z_slice=[],
        cutoff=[], center_structures=False, sampling_updates=False
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
            Molecule indices that will be in every selection. Not implemented
            yet.
        criteria : :obj:`mbgdml.sample.sampleCritera`, optional
            Structure criteria during the sampling procedure. Defaults to
            ``None`` if no criteria should be used.
        z_slice : :obj:`numpy.ndarray`, optional
            Indices of the atoms to be used for the cutoff calculation. Defaults
            to ``[]`` is no criteria is selected or if it is not required for
            the selected criteria.
        cutoff : :obj:`list`, optional
            Distance cutoff between the atoms selected by ``z_slice``. Must be
            in the same units (e.g., Angstrom) as ``R``. Defaults to ``[]`` if
            no criteria is selected or a cutoff is not desired.
        center_structures : :obj:`bool`, optional
            Move the center of mass of each structure to the origin thereby 
            centering each structure (and losing the actual coordinates of
            the structure). While not required for correct use of mbGDML this
            can be useful for other analysis or data set visualization. Defaults
            to ``False``.
        sampling_updates : :obj:`bool`, optional
            Will print something for every 100 successfully sampled structures.
            Defaults to ``False``.
        """
        # Gets Rset_id for this new sampling.
        Rset_id = self._get_Rset_id(Rset)
        
        # Prepares sampling procedure.
        z = self.z
        R = self.R
        E = self.E
        F = self.F

        if R.shape[2] == 0:
            n_R_initial = 0
        else:
            n_R_initial = R.shape[0]

        Rset_info = self.Rset_info
        if isinstance(quantity, int) or str(quantity).isdigit():
            quantity = int(quantity)
            Rset_info, z, R, E, F = self._sample_num(
                z, R, E, F, Rset, Rset_id, Rset_info, quantity, size,
                criteria=criteria, z_slice=z_slice, cutoff=cutoff, 
                sampling_updates=sampling_updates
            )
            # Adds criteria information to the data set (only if sampling is 
            # successful).
            if criteria is not None:
                self.criteria = criteria.__name__
                self.z_slice = z_slice
                self.cutoff = cutoff
        elif quantity == 'all':
            Rset_info, z, R, E, F = self._sample_all(
                z, R, E, F, Rset, Rset_id, Rset_info, size,
                sampling_updates=sampling_updates
            )
        else:
            raise ValueError(f'{quantity} is not a valid selection.')
        
        # Ensures there are no duplicate structures.
        if not Rset_info.shape == np.unique(Rset_info, axis=0).shape:
            raise ValueError(
                'There are duplicate structures in the data set'
            )
        
        # Checks entity and comp ids.
        Rset_info_idx_new = np.array(range(n_R_initial, len(Rset_info)))
        Rset_entity_ids_sampled_split = np.split(  # A list of column arrays from entity_ids
            Rset_info[Rset_info_idx_new][:, 2:],
            Rset_info.shape[1] - 2,  # Rset_id and structure number not included.
            axis=1
        )
        Rset_entity_ids_sampled_split = [
            np.unique(i) for i in Rset_entity_ids_sampled_split
        ]
        entity_ids, comp_ids = self._check_entity_comp_ids(
            self.entity_ids, self.comp_ids, Rset.entity_ids, Rset.comp_ids,
            Rset_entity_ids_sampled_split
        )
        self.entity_ids = entity_ids
        self.comp_ids = comp_ids
        
        # Moves the center of mass of every structure to the origin.
        if center_structures:
            R = self._center_structures(z, R)
        
        # Checks r_unit.
        if hasattr(self, '_r_unit'):
            if self.r_unit != Rset.r_unit:
                raise ValueError('r_unit of is not compatible with dset.')
        else:
            self._r_unit = Rset.r_unit
        
        # Stores all information only if sampling is successful.
        self.Rset_md5 = {**self.Rset_md5, **{Rset_id: Rset.md5}}
        self.Rset_info = Rset_info
        self.z = z
        self.R = R
        self.E = E
        self.F = F

    def sample_structures(
        self, data, quantity, size, selected_rset_id=None, always=[], criteria=None,
        z_slice=[], cutoff=[], center_structures=False, sampling_updates=False
    ):
        """Samples all possible combinations from a data set.

        Adds NaN to energies and forces.

        Parameters
        ----------
        data : :obj:`mbgdml.data`
            A loaded structure or data set object to sample from.
        quantity : :obj:`int`
            Number of structures to sample from the structure set. For example,
            ``'100'``, ``'452'``, or even ``'all'``.
        size : :obj:`str`
            Desired number of molecules in each selection.
        selected_rset_id : obj:`int`, optional
            Currently sampling can only be done for one rset_id at a time. This
            specifies which rset structures in the data set to sample from and
            is required.
        always : :obj:`list` [:obj:`int`], optional
            Molecule indices that will be in every selection. Not implemented
            yet.
        criteria : :obj:`mbgdml.sample.sampleCritera`, optional
            Structure criteria during the sampling procedure. Defaults to
            ``None`` if no criteria should be used.
        z_slice : :obj:`numpy.ndarray`, optional
            Indices of the atoms to be used for the cutoff calculation. Defaults
            to ``[]`` is no criteria is selected or if it is not required for
            the selected criteria.
        cutoff : :obj:`list`, optional
            Distance cutoff between the atoms selected by ``z_slice``. Must be
            in the same units (e.g., Angstrom) as ``R``. Defaults to ``[]`` if
            no criteria is selected or a cutoff is not desired.
        center_structures : :obj:`bool`, optional
            Move the center of mass of each structure to the origin thereby 
            centering each structure (and losing the actual coordinates of
            the structure). While not required for correct use of mbGDML this
            can be useful for other analysis or data set visualization. Defaults
            to ``False``.
        sampling_updates : :obj:`bool`, optional
            Will print something for every 100 successfully sampled structures.
            Defaults to ``False``.
        """
        data_type = data.type

        # Gets Rset_id for this new sampling.
        if data_type == 's':
            Rset_id = self._get_Rset_id(data)
            Rset_md5 = data.md5
        elif data_type == 'd':
            assert selected_rset_id is not None
            Rset_id = self._get_Rset_id(data, selected_rset_id=selected_rset_id)
            Rset_md5 = data.Rset_md5[selected_rset_id]

        # Prepares sampling procedure.
        z = self.z
        R = self.R
        E = self.E
        F = self.F
        Rset_info = self.Rset_info

        if R.shape[2] == 0:
            n_R_initial = 0
        else:
            n_R_initial = R.shape[0]

        if isinstance(quantity, int) or str(quantity).isdigit():
            quantity = int(quantity)
            Rset_info, z, R, E, F = self._sample_num(
                z, R, E, F, data, Rset_id, Rset_info, quantity, size,
                criteria=criteria, z_slice=z_slice, cutoff=cutoff, 
                sampling_updates=sampling_updates
            )
            # Adds criteria information to the data set (only if sampling is 
            # successful).
            if criteria is not None:
                self.criteria = criteria.__name__
                self.z_slice = z_slice
                self.cutoff = cutoff
        elif quantity == 'all':
            Rset_info, z, R, E, F = self._sample_all(
                z, R, E, F, data, Rset_id, Rset_info, size,
                criteria=criteria, z_slice=z_slice, cutoff=cutoff,
                sampling_updates=sampling_updates
            )
        else:
            raise ValueError(f'{quantity} is not a valid selection.')
        
        # Ensures there are no duplicate structures.
        if not Rset_info.shape == np.unique(Rset_info, axis=0).shape:
            raise ValueError(
                'There are duplicate structures in the data set'
            )
        
        # Checks entity and comp ids.
        Rset_info_idx_new = np.array(range(n_R_initial, len(Rset_info)))
        if len(Rset_info_idx_new) > 0:
            Rset_entity_ids_sampled_split = np.split(  # A list of column arrays from entity_ids
                Rset_info[Rset_info_idx_new][:, 2:],
                Rset_info.shape[1] - 2,  # Rset_id and structure number not included.
                axis=1
            )
            Rset_entity_ids_sampled_split = [
                np.unique(i) for i in Rset_entity_ids_sampled_split
            ]
            entity_ids, comp_ids = self._check_entity_comp_ids(
                self.entity_ids, self.comp_ids, data.entity_ids, data.comp_ids,
                Rset_entity_ids_sampled_split
            )
            self.entity_ids = entity_ids
            self.comp_ids = comp_ids
        else:
            # No new structures were sampled
            return

        # Moves the center of mass of every structure to the origin.
        if center_structures:
            R = self._center_structures(z, R)
            self.centered = True
        
        # Checks r_unit.
        if hasattr(self, '_r_unit'):
            if self.r_unit != data.r_unit:
                raise ValueError('r_unit of is not compatible with dset.')
        else:
            self._r_unit = data.r_unit

        # Stores all information only if sampling is successful.
        self.Rset_md5 = {**self.Rset_md5, **{Rset_id: Rset_md5}}
        self.Rset_info = Rset_info
        self.z = z
        self.R = R
        self.E = E
        self.F = F
        
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
            'r_unit': np.array(self.r_unit),
            'entity_ids': self.entity_ids,
            'comp_ids': self.comp_ids
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
        if hasattr(self, 'mb'):
            dataset['mb'] = np.array(self.mb)
        if hasattr(self, 'mb_models_md5'):
            dataset['mb_models_md5'] = np.array(self.mb_models_md5, dtype='S32')
        if hasattr(self, 'mb_dsets_md5'):
            dataset['mb_dsets_md5'] = np.array(self.mb_dsets_md5, dtype='S32')

        try:
            dataset['criteria'] = np.array(self.criteria)
            dataset['z_slice'] = np.array(self.z_slice)
            dataset['cutoff'] = np.array(self.cutoff)
        except:
            pass
        
        if hasattr(self, 'centered'):
            dataset['centered'] = np.array(self.centered)
        
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
    
    def create_mb_from_models(self, ref_dset, model_paths):
        """Creates a many-body data set using mbGDML predictions.

        Removes energy and force predictions from the reference data set using
        GDML models in ``model_paths``.

        Parameters
        ----------
        ref_dset : :obj:`~mbgdml.data.dataset.dataSet`
            Reference data set of structures, energies, and forces. This is the
            data where mbGDML predictions will be subtracted from.
        model_paths : :obj:`list` [:obj:`str`]
            Paths to saved many-body GDML models in the form of
            :obj:`numpy.NpzFile`.
        """
        predict = mbPredict(model_paths)
        dataset = predict.remove_nbody(ref_dset.dataset)
        self._update(dataset)
    
    def create_mb_from_dsets(self, ref_dset, dset_lower_paths):
        """Creates a many-body data set from lower-order data sets.

        If ``ref_dset`` has n > 1 molecules, ``lower_dset_paths`` is a list
        (of size n-1) data set paths with all possible lower-order contributions
        removed. For example, if n is 3, paths to a monomer data set with
        original energies and forces along with a dimer data set with monomer
        contributions removed are required.

        Parameters
        ----------
        ref_dset : :obj:`~mbgdml.data.dataset.dataSet`
            Reference data set of structures, energies, and forces. This is the
            data where mbGDML predictions will be subtracted from.
        dset_lower_paths : :obj:`list` [:obj:`str`]
            Paths to many-body data set contributions to be removed from
            ``ref_dset``.
        """
        dsets_lower = [dataSet(dset_path) for dset_path in dset_lower_paths]
        dset_n_z = [len(i.z) for i in dsets_lower]  # Number of atoms
        n_body_order = [dset_n_z.index(i) for i in sorted(dset_n_z)]
        dsets_lower = [dsets_lower[i] for i in n_body_order]  # Smallest to largest molecules.

        # Remove energy and force contributions from data set.
        ref_dset = utils.e_f_contribution(
            ref_dset, dsets_lower, 'remove'
        )

        ref_dset.mb = len(set(ref_dset.entity_ids))

        mb_dsets_md5 = []
        for dset_lower in dsets_lower:
            if hasattr(dset_lower, 'md5'):
                mb_dsets_md5.append(dset_lower.md5)
        ref_dset.mb_dsets_md5 = mb_dsets_md5
        
        self._update(ref_dset.dataset)
