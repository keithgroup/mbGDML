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
import json
import itertools
from random import randrange, sample, choice
import numpy as np
from cclib.parser.utils import convertor
from mbgdml.data import mbGDMLData
from mbgdml import __version__ as mbgdml_version
from mbgdml.parse import parse_stringfile
from mbgdml import utils
from mbgdml.predict import mbPredict
  

class dataSet(mbGDMLData):
    """For creating, loading, manipulating, and using data sets.

    Parameters
    ----------
    dataset_path : :obj:`str`, optional
        Path to a `npz` file.
    """

    def __init__(self, *args):
        self.type = 'd'
        self.name = 'dataset'
        if len(args) == 1:
            self.load(args[0])
    
    @property
    def name(self):
        """Human-redable label for the data set.

        :type: :obj:`str`
        """
        if hasattr(self, '_name'):
            return self._name
        else:
            return None
    
    @name.setter
    def name(self, var):
        self._name = str(var)

    @property
    def Rset_md5(self):
        """Specifies structure sets IDs/labels and corresponding MD5 hashes.

        Keys are the Rset IDs (:obj:`int`) and values are MD5 hashes
        (:obj:`str`) for the particular structure set.
        
        This is used as a bredcrumb trail that specifies where each structure
        in the data set originates from.

        Examples
        --------
        >>> dset.Rset_md5
        {0: '2339670ad87a606cb11a72191dfd9f58'}

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
        if hasattr(self, '_e_unit'):
            return self._e_unit
        else:
            return 'n/a'

    @e_unit.setter
    def e_unit(self, var):
        self._e_unit = var
    
    @property
    def theory(self):
        """The level of theory used to compute energy and gradients of the data
        set.

        :type: :obj:`str`
        """
        if hasattr(self, '_theory'):
            return self._theory
        else:
            return 'n/a'

    @theory.setter
    def theory(self, var):
        self._theory = var

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

        Notes
        -----
        :obj:`mbgdml.data.basedata.mbGDMLData.z` and
        :obj:`mbgdml.data.basedata.mbGDMLData.R` are always used to generate the
        MD5 hash. If available, :obj:`mbgdml.data.dataset.dataSet.E` and
        :obj:`mbgdml.data.dataset.dataSet.F` are used.

        :type: :obj:`str`
        """
        try:
            return self.dataset['md5'][()].decode()
        except:
            print('Not enough information in dset for MD5')
            raise
    
    @property
    def entity_ids(self):
        """1D array specifying which atoms belong to which entities.
        
        An entity represents a related set of atoms such as a single molecule,
        several molecules, or a functional group. For mbGDML, an entity usually
        corresponds to a model trained to predict energies and forces of those
        atoms. Each ``entity_id`` is an :obj:`int` starting from ``0``.
        
        It is conceptually similar to PDBx/mmCIF ``_atom_site.label_entity_ids``
        data item.

        Examples
        --------
        A single water molecule would be ``[0, 0, 0]``. A water (three atoms)
        and methanol (six atoms) molecule in the same structure would be 
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
        """2D array relating ``entity_ids`` to a chemical component/species
        id or label (``comp_id``).
        
        The first column is the unique ``entity_id`` and the second is a unique
        ``comp_id`` for that specific chemical species. Each ``comp_id`` is then
        reused for entities of the same chemical species.

        Examples
        --------
        Suppose we have a structure containing a water and methanol molecule.
        We can use the labels of ``h2o`` and ``meoh`` (which could be
        anything): ``[['0', 'h2o'], ['1', 'meoh']]``. Note that the
        ``entity_id`` is a :obj:`str`.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_comp_ids'):
            return self._comp_ids
        else:
            return np.array([[]])
    
    @comp_ids.setter
    def comp_ids(self, var):
        self._comp_ids = np.array(var)
    
    @property
    def mb(self):
        """Many-body expansion order of this data set. This is ``None`` if the
        data set does not contain many-body energies and forces.

        :type: :obj:`int`
        """
        if hasattr(self, '_mb'):
            return self._mb
        else:
            return None
    
    @mb.setter
    def mb(self, var):
        self._mb = int(var)
    
    @property
    def mb_dsets_md5(self):
        """All MD5 hash of data sets used to remove n-body contributions from
        data sets.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_mb_dsets_md5'):
            return self._mb_dsets_md5
        else:
            return None
    
    @mb_dsets_md5.setter
    def mb_dsets_md5(self, var):
        self._mb_dsets_md5 = var.astype(str)
    
    @property
    def mb_models_md5(self):
        """All MD5 hash of models used to remove n-body contributions from
        models.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_mb_models_md5'):
            return self._mb_models_md5
        else:
            return None
    
    @mb_models_md5.setter
    def mb_models_md5(self, var):
        self._mb_models_md5 = var.astype(str)

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
            self._e_unit = 'n/a'
        try:
            self.mbgdml_version = str(dataset['mbgdml_version'][()])
        except KeyError:
            # Some old data sets do not have this information.
            # This is unessential, so we will just ignore this.
            pass
        try:
            self.theory = str(dataset['theory'][()])
        except KeyError:
            self.theory = 'n/a'
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
                ref_entity_size = np.count_nonzero(
                    data_entity_ids == entity_id
                )
                # Adds the entity_ids of this entity
                entity_ids.extend([entity_id for _ in range(ref_entity_size)])
                # Adds comp_id
                comp_id = data_comp_ids[entity_id][1]
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
    
    def _generate_structure_samples(
        self, quantity, size, data_ids, structure_idxs, max_sample_entity_ids
    ):
        """

        Parameters
        ----------
        sample_type : :obj:`str`
            ``'num'`` or ``'all'`` depending on ``quantity``.
        size : :obj:`str`

        data_ids : :obj:`numpy.ndarray`

        structure_idxs : :obj:`numpy.ndarray`
            All structure indices that we could sample from.
        max_sample_entity_ids : :obj:`int`
            The largest ``entity_id`` from the data or structure set we are
            sampling from.
        """
        if isinstance(quantity, int) or str(quantity).isdigit():
            while True:
                struct_num_selection = randrange(len(structure_idxs))
                data_id = choice(data_ids)
                comp_selection = sorted(sample(range(max_sample_entity_ids + 1), size))
                Rset_selection = [data_id, struct_num_selection] + comp_selection
                yield Rset_selection
        elif quantity == 'all':
            comb_list = list(itertools.combinations(range(max_sample_entity_ids + 1), size))
            for struct_i in structure_idxs:
                for comb in comb_list:
                    for data_id in data_ids:
                        data_selection = [data_id, struct_i] + list(comb)
                        yield data_selection
    
    def _sample(
        self, z, R, E, F, sample_data, quantity, data_ids, Rset_id, Rset_info,
        size, criteria=None, z_slice=[], cutoff=[], sampling_updates=False, 
        copy_EF=True
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
        quantity : :obj:`str`
            Number of structures to sample from the structure set. For example,
            ``'100'``, ``'452'``, or even ``'all'``.
        data_ids : :obj:`numpy.ndarray`
            Array of :obj:`int` of the structure set used to sampled data
            from. For example, if you are sampling from a data set this would be
            the ``Rset_id`` in that data set not the new ``Rset_id`` for this
            current data set.
        Rset_id : :obj:`int`
            The :obj:`int` that specifies the Rset (key in ``self.Rset_md5``) in
            this current data set.
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
        sample_data_type = sample_data.type
        sample_data_z = sample_data.z
        sample_data_R = sample_data.R

        sample_entity_ids = sample_data.entity_ids
        max_sample_entity_ids = max(sample_entity_ids)

        if sample_data_type == 'd':
            sample_data_Rset_info = sample_data.Rset_info
            sample_data_E = sample_data.E
            sample_data_F = sample_data.F

            structure_idxs = np.array([], dtype=np.int64)
            for data_id in data_ids:
                structure_idxs = np.concatenate(
                    (
                        structure_idxs,
                        np.where(sample_data_Rset_info[:,0] == data_id)[0]
                    ),
                    axis=0
                )
        elif sample_data_type == 's':
            structure_idxs = np.arange(0, len(sample_data_R))
        
        num_accepted_r = 0
        for data_selection in self._generate_structure_samples(
            quantity, size, data_ids, structure_idxs, max_sample_entity_ids
        ):
            # Ends sampling for number quantities.
            # The generator does not stop.
            if isinstance(quantity, int) or str(quantity).isdigit():
                if num_accepted_r == quantity:
                    break
            
            # Sampling updates
            if sampling_updates:
                if isinstance(quantity, int) or str(quantity).isdigit():
                    if num_accepted_r%500 == 0:
                        print(f'Successfully found {num_accepted_r} structures')
                elif quantity == 'all':
                    if (num_accepted_r+1)%500 == 0:
                        print(
                            f'Successfully sampled {num_accepted_r+1} clusters'
                        )
            
            i_r_sample = data_selection[1]
            # Gets Rset_selection instead of data_selection
            if sample_data_type == 'd':
                i_r_rset = sample_data_Rset_info[i_r_sample][1]
                i_r_rset_entity_ids = sample_data_Rset_info[i_r_sample][2:][data_selection[2:]]
                Rset_selection = [Rset_id, i_r_rset] + list(i_r_rset_entity_ids)
            elif sample_data_type == 's':
                Rset_selection = data_selection

            # Checks if Rset_info is already present.
            if Rset_info.shape[1] == 0:  # No previous Rset_info.
                Rset_axis = 1
            else:
                Rset_axis = 0
                # Checks to see if combination is already in data set.
                if (Rset_info[...]==Rset_selection).all(1).any():
                    # Does not add the combination.
                    continue

            # Gets atomic indices from entity_ids in the Rset.
            atom_idx = []
            for entity_id in data_selection[2:]:
                atom_idx.extend(
                    [i for i,x in enumerate(sample_entity_ids) if x == entity_id]
                )
            
            # Checks compatibility with atoms.
            if len(z) == 0:
                z = sample_data_z[atom_idx]
            else:
                if not np.all([z, sample_data_z[atom_idx]]):
                    print(f'z of data set: {z}')
                    print(f'Rset_info of selection: {Rset_selection}')
                    print(f'z of selection: {sample_data_z[atom_idx]}')
                    raise ValueError(f'z of the selection is incompatible.')
            
            # Checks any structure criteria.
            r_selection = np.array([sample_data_R[i_r_sample, atom_idx, :]])
            r_entity_ids = sample_entity_ids[atom_idx]
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
            
            ###   SUCCESSFUL SAMPLE   ###
            Rset_info = np.concatenate(
                (Rset_info, np.array([Rset_selection])), axis=Rset_axis
            )
            
            if R.shape[2] == 0:  # No previous R.
                R = r_selection
            else:
                R = np.concatenate((R, r_selection), axis=0)

            # Handles selecting energies and forces.
            ## If sampling from a data set, we assume if the size is the
            ## same then we will transfer over the data.
            ### Energies
            if sample_data_type == 'd' and copy_EF and size == (max_sample_entity_ids+1):
                e_selection = np.array([sample_data_E[i_r_sample]])
            else:
                # Adds NaN for energies.
                e_selection = np.empty((1,))
                e_selection[:] = np.NaN
            ### Forces
            if sample_data_type == 'd' and copy_EF and size == (max_sample_entity_ids+1):
                f_selection = np.array([sample_data_F[i_r_sample]])
            else:
                # Adds NaN for forces.
                ## Force array will be the same shape as R.
                f_selection = np.copy(r_selection)
                f_selection[:] = np.NaN

            # Adding energies and forces.
            if len(E.shape) == 0:  # No previous E in this data set.
                E = e_selection
            else:
                E = np.concatenate((E, e_selection), axis=0)
            
            if F.shape[2] == 0:  # No previous F.
                F = f_selection
            else:
                F = np.concatenate((F, f_selection), axis=0)

            num_accepted_r += 1
        
        # Adds theory information if we already have it.
        if sample_data_type == 'd' and size == (max_sample_entity_ids+1):
            self.theory = sample_data.theory
            self.e_unit = sample_data.e_unit
        
        return (Rset_info, z, R, E, F)
    
    def sample_structures(
        self, data, quantity, size, consistent_entities=[], criteria=None,
        z_slice=[], cutoff=[], center_structures=False, sampling_updates=False, 
        copy_EF=True):
        """Randomly samples a ``quantity`` of geometries of a specific
        ``size`` from data or structure sets.

        When sampling from :class:`~mbgdml.data.structureset.structureSet`,
        :obj:`numpy.nan` is added to :attr:`E` and :attr:`F` for each structure.
        PES data is added if available with ``copy_EF = True`` when sampling
        from :class:`~mbgdml.data.dataset.dataSet` and the requested ``size``
        is the same as the :class:`~mbgdml.data.dataset.dataSet`.

        Currently, you have to set ``quantity = 'all'`` when sampling from
        :class:`~mbgdml.data.dataset.dataSet`.

        Parameters
        ----------
        data : :obj:`mbgdml.data`
            A loaded structure or data set object to sample from.
        quantity : :obj:`str` or :obj:`int`
            Number of structures to sample from the data. For example,
            ``'100'``, ``452``, or ``'all'``.
        size : :obj:`int`
            Desired number of entities in each selected geometry.
        consistent_entities : :obj:`list` [:obj:`int`], optional
            Molecule indices that will be in every selection. Not implemented
            yet.
        criteria : :obj:`mbgdml.criteria`, optional
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
        copy_EF : :obj:`bool`, optional
            If sampling from a data set, copy over the energies and forces of
            the sampled structures to the new data set. Defaults to ``True``.
        """
        data_type = data.type
        
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
        
        # Gets Rset_id for this new sampling.
        if data_type == 's':
            Rset_id = self._get_Rset_id(data)
            data_ids = np.array([Rset_id])
            Rset_md5 = data.md5

            Rset_info, z, R, E, F = self._sample(
                z, R, E, F, data, quantity, data_ids, Rset_id, Rset_info, size,
                criteria=criteria, z_slice=z_slice, cutoff=cutoff,
                sampling_updates=sampling_updates, copy_EF=copy_EF
            )

            self.Rset_md5 = {**self.Rset_md5, **{Rset_id: Rset_md5}}
        elif data_type == 'd':
            data_ids = np.array([i for i in data.Rset_md5.keys()])
            if quantity == 'all':
                for data_id in data_ids:
                    Rset_id = self._get_Rset_id(data, selected_rset_id=data_id)
                    Rset_md5 = data.Rset_md5[Rset_id]

                    Rset_info, z, R, E, F = self._sample(
                        z, R, E, F, data, quantity, np.array([data_id]), Rset_id,
                        Rset_info, size, criteria=criteria, z_slice=z_slice,
                        cutoff=cutoff, sampling_updates=sampling_updates, 
                        copy_EF=copy_EF
                    )

                    self.Rset_md5 = {**self.Rset_md5, **{Rset_id: Rset_md5}}
            elif isinstance(quantity, int) or str(quantity).isdigit():
                raise ValueError(
                    f'This is not implemented for data set sampling'
                )
        
        # Ensures there are no duplicate structures.
        if not Rset_info.shape == np.unique(Rset_info, axis=0).shape:
            raise ValueError(
                'There are duplicate structures in the data set'
            )
        
        # Adds criteria information to the data set (only if sampling is 
        # successful).
        if criteria is not None:
            self.criteria = criteria.__name__
            self.z_slice = z_slice
            self.cutoff = cutoff
        
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
            R = utils.center_structures(z, R)
            self.centered = True
        
        # Checks r_unit.
        if hasattr(self, '_r_unit'):
            if self.r_unit != data.r_unit:
                raise ValueError('r_unit of is not compatible with dset.')
        else:
            self._r_unit = data.r_unit

        # Stores all information only if sampling is successful.
        self.Rset_info = Rset_info
        self.z = z
        self.R = R
        self.E = E
        self.F = F
    
    def add_pes_data(
        self, calc_dir, theory_label, e_unit_dset, e_unit_calc, dtype='qcjson',
        allow_remaining_nan=True, center_calc_R=False, r_match_atol=5.1e-07,
        r_match_rtol=0.0
    ):
        """Add potential energy surface (PES) data (i.e., energies and/or
        forces) to the data set (:attr:`E` and :attr:`F`).

        Assumes that ``r_unit`` of the calculation is the same as the data set.
        `QCJSON <https://github.com/keithgroup/qcjson>`_ files are currently
        the only way to import energy and gradient data.

        Energies and forces are added by matching Cartesian coordinates of the
        calculation to a structure in the data set within some tolerance.

        Parameters
        ----------
        calc_dir : :obj:`str`
            Path to a directory containing potential energy surface data to add
            to this data set.
        theory_label : :obj:`str`
            Defines the level of theory and crucial aspects of calculations.
            For example, ``'RI-MP2/def2-TZVP'``, ``'GFN2-xTB'``, or
            ``'CCSD(T)/CBS'``.
        e_unit_dset : :obj:`str`
            The current (or desired) energy units of the data set. Will update
            :attr:`e_unit` to this value.
        e_unit_calc : :obj:`str`
            The energy unit of the data. Will be converted to ``e_unit_dset`` if
            necessary.
        dtype : :obj:`str`, optional
            Specifies the PES data format and implicitly sets the file extension
            search string. Options are ``'qcjson'``. Defaults to ``'qcjson'``.
        allow_remaining_nan : :obj:`bool`, optional
            If :attr:`E` and :attr`F` should, or could, have ``np.nan`` as one
            or more elements after adding all possible data. Pretty much only
            serves as a sanity check. Defaults to ``True``.
        center_calc_R : :obj:`bool`, optional
            Center the cartessian coordinates of the parsed data from the
            calculations in order to match correctly with :attr:`R`.
        r_match_atol : :obj:`float`, optional
            Absolute tolerance for matching the coordinates of a calculation to
            a structure in the data set. Defaults to ``6.8e-07``.
        r_match_rtol : :obj:`float`, optional
            Relative tolerance for matching the coordinates of a calculation to
            a structure in the data set. Defaults to ``0.0``.
        """

        if dtype == 'qcjson':
            search_string = 'json'
        else:
            raise ValueError(f'{dtype} is not a valid selection.')
        
        # Gets all engrad output quantum chemistry json files.
        # For more information: https://github.com/keithgroup/qcjson
        engrad_calc_paths = utils.get_files(calc_dir, '.'+search_string)
        if len(engrad_calc_paths) == 0:
            raise AssertionError(
                f'No calculation files ending with {search_string} were found.'
            )
        
        # Organizing engrad calculations so earlier structures are loaded first.
        # This does multiple things. Reduces runtime as fewer iterations need to 
        # be made. Also, some data sets may involve trajectories that start from
        # the same structure, so multiple structures might be the same.
        engrad_calc_paths = utils.natsort_list(engrad_calc_paths)

        missing_engrad_indices = np.argwhere(np.isnan(self.E))[:,0].tolist()
        if len(missing_engrad_indices) == 0:
            raise AssertionError(
                f'There are no energies or forces missing in the data set.'
            )
        
        z_dset = self.z
        # Loops thorugh engrad calculations and adds energies and forces for each
        # structure.
        for engrad_calc_path in engrad_calc_paths:
            engrad_name = utils.get_filename(engrad_calc_path)
            

            # Gets energies and gradients from qcjson.
            with open(engrad_calc_path) as f:
                engrad_data = json.load(f)

            # Loops through all structures that are missing engrads.
            for engrad_i in range(len(engrad_data)):
                _dset_i_to_remove = []
                can_remove = False

                if isinstance(engrad_data, list):
                    data = engrad_data[engrad_i]
                else:
                    data = engrad_data

                engrad_i_r = np.array(
                    data['molecule']['geometry']
                )
                if center_calc_R:
                    engrad_i_r = utils.center_structures(z_dset, engrad_i_r)

                for dset_i in missing_engrad_indices:
                    dset_i_r = self.R[dset_i]

                    # If the atomic coordinates match we assume this is the
                    # originating engrad structure.
                    # ORCA output files will only include six significant figures
                    # for Cartesian coordinates. Sometimes the data sets have more
                    # significant figures and the default tolerances for allclose were too
                    # high, so I had to lower them a little.
                    # Because we used natsort for the engrad calculations they
                    # should be in order, but we check anyway.
                    if np.allclose(engrad_i_r, dset_i_r, atol=r_match_atol, rtol=r_match_rtol):
                        # Get, convert, and add energies and forces to data set.
                        energy = data['properties']['return_energy']
                        energy = convertor(energy, e_unit_calc, e_unit_dset)
                        self.E[dset_i] = energy

                        forces = np.negative(
                            np.array(data['return_result'])
                        )
                        forces = utils.convert_forces(
                            forces, e_unit_calc, 'Angstrom', e_unit_dset, 'Angstrom'
                        )
                        self.F[dset_i] = forces

                        # Found the correct structure, so we terminate looking for
                        # the dset structure match.
                        can_remove = True
                        break
                    
                # Removes all NaN indices from missing_engrad_indices that have
                # already been matched.
                if can_remove:
                    missing_engrad_indices.remove(dset_i)
        
        still_missing = len(np.argwhere(np.isnan(self.E))[:,0].tolist())
        if still_missing > 0 and not allow_remaining_nan:
            raise AssertionError(
                f'There are still nan values leftover when allow_remaining_nan is False.'
            )
        
        # Finalizes data set and saves.
        self.e_unit = e_unit_dset
        self.theory = theory_label
        
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
        # data set MD5 must be the same type (as they do comparisons).
        dataset['md5'] = np.array(
            utils.md5_data(dataset, md5_properties), dtype='S32'
        )
        return dataset

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
            `npz`.
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
        ref_dset.mb_dsets_md5 = np.array(mb_dsets_md5)
        
        self._update(ref_dset.dataset)
