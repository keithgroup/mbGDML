# MIT License
# 
# Copyright (c) 2020-2022, Alex M. Maldonado
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

"""Analyses for structures."""

import itertools
import numpy as np
from .._gdml.desc import Desc
from .. import utils
import umap.umap_ as umap

class structureEmbedding:
    """Uses the uniform manifold approximation and projection to embed R
    descriptors.

    For more information see https://umap-learn.readthedocs.io/en/latest/.

    Attributes
    ----------
    reducer : `umap.umap`
        A UMAP object used to compute embedding.
    embedding : :obj:`numpy.ndarray`
        A 2D array with rows being each structure (in the order they are
        provided in data) with their reduced dimension coordinates being
        the columns.
    """

    def __init__(self, n_neighbors=9, min_dist=0.1, random_state=None):
        """
        Parameters
        ----------
        n_neighbors : :obj:`int`, optional
            The size of the local neighborhood. Smaller numbers will push data
            into their own regions whereas larger numbers promote a more
            uniform, global picture. Defaults to 9.
        min_dist : :obj:`float`, optional
            The minimum distance between two points. Lower values enable UMAP to
            place points closer to each other. For these descriptors, lower
            values are generally recommended to represent similar structures.
            Defaults to 0.1.
        random_state : :obj:`int` or :obj:`None`, optional
            If :obj:`int`, the seed is used by the random number generator. If
            :obj:`None`, a random number is generated and passed into UMAP.
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        if random_state is None:
            self.random_state = np.random.randint(2**32 - 1)
        else:
            self.random_state = random_state
    
    @property
    def data_info(self):
        """Information about the data included in the UMAP npz.

        The order of data information needs to be in the same order as the data
        in ``R``, ``R_desc``, etc. was added. Each item in the list provides
        information about the data source in a :obj:`dict` with the following
        keys:

        ``'name'``
            Human-friendly name specifying the data source.
        ``'md5'``
            The MD5 hash of the data source.
        ``'type'``
            The type of data. Two common ones are ``'d'`` for data set or
            ``'m'`` for model.
        ``'quantity'``
            The number of structures included in the UMAP npz. This is used for
            slicing and separating data for plots.

        :type: :obj:`list` [:obj:`dict`]
        """
        if hasattr(self, '_data_info'):
            return self._data_info
        else:
            return []
    
    @data_info.setter
    def data_info(self, var):
        self._data_info = var
    
    @property
    def z(self):
        """Atomic numbers of all atoms in data set structures.
        
        A ``(n,)`` shape array of type :obj:`numpy.int32` containing atomic
        numbers of atoms in the structures in order as they appear.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_z'):
            return self._z
        else:
            return np.array([])
    
    @z.setter
    def z(self, var):
        self._z = var
    
    @property
    def R(self):
        """Atomic coordinates of structure(s).
        
        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of structures and ``n`` is the number of atoms with three 
        Cartesian components.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_R'):
            return self._R
        else:
            return np.array([[[]]])

    @R.setter
    def R(self, var):
        self._R = var
    
    @property
    def R_desc(self):
        """Inverse pairwise distance descriptors of R from sGDML.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_R_desc'):
            return self._R_desc
        else:
            return np.array([[]])

    @R_desc.setter
    def R_desc(self, var):
        self._R_desc = var
    
    @property
    def E_true(self):
        """True (or reference) energies of the structures in R.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_E_true'):
            return self._E_true
        else:
            return np.array([])

    @E_true.setter
    def E_true(self, var):
        self._E_true = var
    
    @property
    def E_pred(self):
        """Predicted energies of the structures in R from a mbGDML model.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_E_pred'):
            return self._E_pred
        else:
            return np.array([])

    @E_pred.setter
    def E_pred(self, var):
        self._E_pred = var
    
    @property
    def F_true(self):
        """True (or reference) atomic forces of atoms in structure(s).
        
        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of structures and ``n`` is the number of atoms with three 
        Cartesian components.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_F_true'):
            return self._F_true
        else:
            return np.array([[[]]])

    @F_true.setter
    def F_true(self, var):
        self._F_true = var
    
    @property
    def F_pred(self):
        """Predicted atomic forces of atoms in structures from a model.
        
        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of structures and ``n`` is the number of atoms with three 
        Cartesian components.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_F_pred'):
            return self._F_pred
        else:
            return np.array([[[]]])

    @F_pred.setter
    def F_pred(self, var):
        self._F_pred = var

    def R_desc_from_model(self, a_model):
        """Returns the inverse pairwise sGDML descriptor and its Jacobian from
        a mbGDML model.

        Parameters
        ----------
        a_model : :obj:`mbgdml.data.model`
            A loaded model object.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            Inverse pairwise distance descriptors.
        """
        return a_model.model['R_desc'].T

    def get_R_desc(self, z, R):
        """Calculates the inverse pairwise sGDML descriptor and its Jacobian.

        Parameters
        ----------
        z : :obj:`numpy.ndarray`
            Atomic numbers of all atoms in data set structures. A ``(n,)`` shape
            array of type containing atomic numbers of atoms in the structures
            in order as they appear.
        R : :obj:`numpy.ndarray`
            Atomic coordinates of structure(s). A :obj:`numpy.ndarray` with
            shape of ``(m, n, 3)`` where ``m`` is the number of structures and
            ``n`` is the number of atoms with three Cartesian components.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            Inverse pairwise distance descriptors.
        :obj:`numpy.ndarray`
            Descriptor Jacobian.
        """
        desc = Desc(len(z))
        R_desc, R_d_desc = desc.from_R(R)
        return (R_desc, R_d_desc)

    def embed(self):
        """Embed the descriptors and derivatives in two dimensions.

        Will set the ``reducer`` attribute if it does not exist or update it
        if ``n_neighbors``, ``min_dist``, or ``random_state`` have changed.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            A 2D array with rows being each structure (in the order they are
            provided in data) with their reduced dimension coordinates being
            the columns.
        
        Notes
        -----
        We recommend first tuning ``n_neighbors`` to provide a balance of
        clustering and overlap/uniformness. Then tune ``min_dist`` to be the
        smallest number that allows you to qualitatively determine number of
        points in a clustered region.

        For more information on these parameters see
        https://umap-learn.readthedocs.io/en/latest/parameters.html.
        """
        # pylint: disable=undefined-variable
        # We check if we need to update the reducer attribute. We always
        # reinitialize reducer as it takes very little time, but we still have
        # to check the random_state for reproducibility.
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors, min_dist=self.min_dist,
            random_state=self.random_state
        )
        self.reducer = reducer
        
        data = self.R_desc
        self.embedding = reducer.fit_transform(data)
        return self.embedding

    @property
    def umap_data(self):
        """

        :type: :obj:`dict`
        """
        data = {
            'z': self.z,
            'R': self.R,
            'R_desc': self.R_desc,
            'E_true': self.E_true,
            'F_true': self.F_true,
            'E_pred': self.E_pred,
            'F_pred': self.F_pred,
            'data_info': np.array(self.data_info),
            'n_neighbors': np.array(self.n_neighbors),
            'min_dist': np.array(self.min_dist),
            'random_state': np.array(self.random_state)
        }

        # Checks embedding.
        if not hasattr(self, 'embedding'):
            self.embed()
        
        if self.n_neighbors != self.reducer.n_neighbors \
           or self.min_dist != self.reducer.min_dist \
           or self.random_state != self.reducer.random_state:
            self.embed()

        data['embedding'] = self.embedding

        return data

    def save(self, name, data, save_dir):
        """Save UMAP data into an npz file.
        
        Parameters
        ----------
        name : :obj:`str`
            Name of the file to be saved not including the ``npz`` extension.
        data : :obj:`dict`
            Data to be saved to ``npz`` file.
        save_dir : :obj:`str`
            Directory to save the file (with or without the ``'/'`` suffix).
        """
        save_dir = utils.norm_path(save_dir)
        save_path = save_dir + name + '.npz'
        np.savez_compressed(save_path, **data)
    
    def _update(self, umap_data):
        """Updates UMAP object properties from ``umap_data`` :obj:`dict`.

        Parameters
        ----------
        umap_data : :obj:`dict`

        """
        # pylint: disable=undefined-variable

        self.z = umap_data['z']
        self.R = umap_data['R']
        self.R_desc = umap_data['R_desc']
        self.E_true = umap_data['E_true']
        self.F_true = umap_data['F_true']
        self.E_pred = umap_data['E_pred']
        self.F_pred = umap_data['F_pred']
        self.data_info = umap_data['data_info'].tolist()
        self.n_neighbors = umap_data['n_neighbors'][()]
        self.min_dist = umap_data['min_dist'][()]
        self.random_state = umap_data['random_state'][()]
        
        self.reducer = umap.UMAP(
            n_neighbors=self.n_neighbors, min_dist=self.min_dist,
            random_state=self.random_state
        )
        self.embedding = umap_data['embedding']

    
    def load(self, umap_path):
        """Load a UMAP npz file.

        Parameters
        ----------
        umap_path : :obj:`str`
            Path to UMAP data npz file.
        """
        umap_data = dict(np.load(umap_path, allow_pickle=True))
        self._update(umap_data)

class mbExpansion:
    """Predicts energies and forces of a structure using a many-body expansion.

    Uses data sets with energy and force predictions up to and including
    n molecules from a single structure. Can provide a theoretical maximum
    accuracy of a many-body force field.
    """

    def __init__(self):
        pass

    def _contribution(self, E, F, mol_info, dsets, operation):
        """Adds or removes energy and force contributions in data sets.

        Forces are currently not updated but still returned.

        Parameters
        ----------
        E : :obj:`numpy.ndarray`
            Initial energy. Can be zero or nonzero.
        F : :obj:`numpy.ndarray`
            Initial forces. Can be zero or nonzero.
        dsets : :obj:`list` [:obj:`mbgdml.data.dataset.dataSet`]
            Data set contributions to be added or removed from ``E`` and ``F``.
        operation : :obj:`str`
            ``'add'`` or ``'remove'`` the contributions.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            Updated energies.
        :obj:`numpy.ndarray`
            Updated forces.
        """
        # Loop through every lower order n-body data set.
        for dset_lower in dsets:
            # Lower order information.
            mol_info_lower = dset_lower.r_prov_specs[:,2:]
            n_mol_lower = len(mol_info_lower[0])

            # Loop through every structure.
            for i_r in range(len(E)):
                # We have to match the molecule information for each structure to
                # remove the right information.
                r_mol_info = mol_info[i_r]  # The molecules in this structure.
                mol_combs = list(  # Molecule combinations to be removed
                    itertools.combinations(r_mol_info, n_mol_lower)
                )
                mol_combs = np.array(mol_combs)

                # Loop through every molecular combination.
                for mol_comb in mol_combs:
                    
                    i_r_lower = np.where(  # Index of the structure in the lower data set.
                        np.all(mol_info_lower == mol_comb, axis=1)
                    )[0][0]

                    e_r_lower = dset_lower.E[i_r_lower]
                    # f_r_lower = dset_lower.F[i_r_lower]

                    # Removing energy contributions.
                    if operation == 'add':
                        E[i_r] += e_r_lower
                    elif operation == 'remove':
                        E[i_r] -= e_r_lower
                    else:
                        raise ValueError(f'{operation} is not "add" or "remove".')

                    # Removing force contributions.
                    # TODO: Force contributions.
        
        return E, F

    def create_nbody_dset(self, dset, nbody_dsets):
        """Creates a n-body data set with < n contributions removed from other data
        sets.

        Parameters
        ----------
        dset : :obj:`mbgdml.data.dataset.dataSet`
            The n-body data set with total energies and forces.
        nbody_dsets : :obj:`list` [:obj:`mbgdml.data.dataset.dataSet`]
            A list of lower order n-body data sets.
        
        Returns
        -------
        :obj:`mbgdml.data.dataset.dataSet`
            The same data set with the energies and forces being the n-body
            contributions.
        """
        # Local variables to work with.
        E = dset.E
        F = dset.F
        mol_info = dset.r_prov_specs[:, 2:]  # Molecule numbers for each structure.

        # Loop through every lower order n-body data set.
        E, F = self._contribution(E, F, mol_info, nbody_dsets, 'remove')
        
        # Updating energies and forces.
        dset.E = E
        dset.F = F
        return dset

    def nbody_contribution(
        self, dset, n_molecules, n_atoms_per_molecule
    ):
        """Calculates contributions for n-body structures where n > 1.

        Only works with data sets containing the same chemical species.

        Parameters
        ----------
        dset : :obj:`mbgdml.data.dataset.dataSet`

        n_molecules : :obj:`int`
            The total number of molecules in the parent structure.
        n_atoms_per_molecule : :obj:`int`
            Number of atoms per molecule.
        """
        #pylint: disable=no-member
        # Contributions to the parent structure.
        E = np.zeros(1,)
        F = np.zeros(dset.F.shape)
        mol_info = dset.r_prov_specs[:, 2:]

        for i in range(len(E)):
            e, f = self._contribution(E, F, mol_info, [dset], 'add')
            E[i] = e[0]
            #F[i] = f

        return E, F

    def predict(self, nmer_dsets):
        """Predict the energy and force of some structure using a many-body
        decomposition.

        Only works with data sets containing the same chemical species.

        Parameters
        ----------
        nmer_dsets : :obj:`list` [:obj:`mbgdml.data.dataset.dataSet`]
            All data sets of nmer clusters from a single structure. At the very
            least, a 1mer data set must be provided.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            Theoretically predicted energy using a many-body expansion.
        :obj:`numpy.ndarray`
            Theoretically predicted forces using a many-body expansion. Not 
            currently working.
        """
        # pylint: disable=not-an-iterable
        # pylint: disable=unsubscriptable-object
        # Determines increasing order n-body cluster size.
        dset_n_z = [len(i.z) for i in nmer_dsets]  # Number of atoms
        n_body_order = [dset_n_z.index(i) for i in sorted(dset_n_z)]
        nmer_dsets = [nmer_dsets[i] for i in n_body_order]  # Smallest to largest molecules.
        
        # Creates many-body data sets.
        nbody_dsets = [nmer_dsets[0]]
        for i in range(1, len(nmer_dsets)):
            nbody_dsets.append(
                self.create_nbody_dset(nmer_dsets[i], nbody_dsets[:i])
            )
        
        # Sets up energy and force variables.
        n_molecules = nbody_dsets[0].n_R
        n_atoms_per_molecule = nbody_dsets[0].n_z
        E = np.zeros((1,))
        F = np.zeros((n_molecules, n_atoms_per_molecule, 3))
        
        # Loop through every data set and add its contribution to E and F.
        for i_dset in range(len(nbody_dsets)):
            nbody_dset = nbody_dsets[i_dset]

            # Total contributions of all structures in many-body data set.
            e = np.array([np.sum(nbody_dset.E)])
            f = np.array([np.sum(nbody_dset.F)])
            
            # Adds n-body contribution to total energy and forces.
            E = np.add(E, e)
            F = np.add(F, f)
        
        # Reshape F to match mbGDML data sets and predictions.
        F = np.reshape(F, (n_molecules*n_atoms_per_molecule, 3))  # TODO: Maybe need 3D?

        return E, F
