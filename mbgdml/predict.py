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

import itertools

import numpy as np
from sgdml.predict import GDMLPredict
import mbgdml.solvents as solvents

class mbGDMLPredict():
    """Predict energies and forces of structures using GDML models.
    """

    def __init__(self, models):
        """Sets GDML models to be used for many-body predictions.
        
        Parameters
        ----------
        models : :obj:`list` [:obj:`str`]
            Contains paths to either standard or many-body GDML models.
        """
        self._load_models(models)
    

    def _load_models(self, models):
        """Loads models and preprares GDMLPredict.
        
        Parameters
        ----------
        models : :obj:`list` [:obj:`str`]
            Contains paths to either standard or many-body GDML models.
        """
        self.gdmls = []
        self.models = []
        for model in models:
            loaded = np.load(model, allow_pickle=True)
            self.models.append(loaded)
            gdml = GDMLPredict(loaded)
            self.gdmls.append(gdml)



    def _calculate(self, R, system_size, molecule_size, gdml):
        """The actual calculate/predict function for a single GDML model.

        Predicts the energy and force contribution from a single many-body
        (or standard) GDML model.
        
        Parameters
        ----------
        R : :obj:`numpy.ndarray`
            Cartesian coordinates of all atoms of the structure specified in 
            the same order as z. The array should have shape (n, 3) where n is 
            the number of atoms.
        system_size : :obj:`int`
            Number of molecules of the defined chemical species in the 
            structure.
        molecule_size : :obj:`int`
            Number of atoms in the defined chemical species.
        gdml : :obj:`sgdml.predict.GDMLPredict`
            Object used to predict energies and forces of the structure defined 
            in ``R``.
        
        Returns
        -------
        :obj:`dict`
            Energy contributions of the structure. Dictionary keys specify the
            molecule combination. For example, ``'0,2'`` contains the
            predictions of two-body energy of the dimer containing molecules
            ``0`` and ``2``. Also contains a ``'T'`` key representing the total
            n-body contribution for the whole system.
        :obj:`dict`
            Force contributions of the structure. Dictionary keys specify the
            molecule combination. For example, ``'1,2,5'`` contains the
            predictions of three-body energy of the trimer containing molecules
            ``1``, ``2``, and ``5``. Also contains a ``'T'`` key representing
            the total n-body contribution for the whole system.
        """

        # 'T' is for total
        E_contributions = {'T': 0.0}
        F_contributions = {'T': np.zeros(R.shape)}

        nbody_order = int(gdml.n_atoms/molecule_size)
        
        # Getting list of n-body combinations (int).
        nbody_combinations = list(
            itertools.combinations(list(range(0, system_size)), nbody_order)
        )

        # Getting all contributions for each molecule combination.
        for comb in nbody_combinations:

            # Gets indices of all atoms in the combination of molecules.
            atoms = []
            for molecule in comb:
                atoms += list(range(
                    molecule * molecule_size, (molecule + 1) * molecule_size
                ))
            
            # Predicts energies
            e, f = gdml.predict(R[atoms].flatten())

            # Adds contributions prediced from model.
            comb_label = ','.join(str(molecule) for molecule in comb)
            E_contributions[comb_label] = e
            F_contributions[comb_label] = f.reshape(len(atoms), 3)

            # Adds contributions to total energy and forces.
            E_contributions['T'] += E_contributions[comb_label]
            F_contributions['T'][atoms] += F_contributions[comb_label]
        
        return E_contributions, F_contributions


    def decomposed_predict(self, z, R):
        """Computes predicted total energy and atomic forces decomposed by
        many-body order.
        
        Parameters
        ----------
        z : :obj:`numpy.ndarray`
            A ``(n,)`` shape array of type :obj:`numpy.int32` containing atomic
            numbers of atoms in the structures in order as they appear.
        R : :obj:`numpy.ndarray`
            A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is
            the number of structures and ``n`` is the number of atoms with three 
            Cartesian components.
        
        Returns
        -------
        tuple
            Dictionaries of many-body contributions of total energies
            (float) and forces (np.ndarray) of the structure. Note that
            forces has the same shape as R. Each key of the dictionary is
            the order of the many-body model. Within this dictionary is a
            total, 'T', and molecule combinations identified in the system
            and their contributions.
        
        Examples
        --------
        >>> atoms = [8, 1, 1, 8, 1, 1]
        >>> coords = array([[ 1.530147,  1.121901,  0.396232],
                            [ 2.350485,  1.1297  , -0.085235],
                            [ 0.823842,  1.311077, -0.2846  ],
                            [-1.464849, -1.155867, -0.449863],
                            [-1.41447 , -1.99605 , -0.917981],
                            [-0.868584, -1.275659,  0.32329 ]])
        >>> E_predicted, F_predicted = decomposed_predict(atoms, coords)
        >>> E_predicted['T']  # Units here are kcal/mol
        -95775.24328734782
        >>> F_predicted['T']  # Units here are kcal/(mol A)
        array([[-4.06668790e+01,  5.97564871e+00, -1.63273803e+01],
               [ 1.10787347e+01,  6.38884805e-01, -9.01790818e+00],
               [ 2.82554119e+01, -8.11473944e+00,  2.53684238e+01],
               [ 1.45845554e+01, -2.53514310e+00,  1.76065017e+01],
               [-3.53967582e-02,  2.91676403e-01,  2.61011071e-01],
               [-1.32164262e+01,  3.74366076e+00, -1.78906481e+01]])
        """
        # TODO run tests and add more information

        # Gets system information from dataset.
        # This assumes the system is only solvent.
        if type(z) != list:
            z = z.tolist()
        dataset_info = solvents.system_info(z)
        system_size = dataset_info['cluster_size']
        molecule_size = dataset_info['solvent_molec_size']

        # 'T' is for total
        E_contributions = {'T': 0.0}
        F_contributions = {'T': np.zeros(R.shape)}

        # Adds contributions from all models.
        for gdml in self.gdmls:
            
            nbody_order = int(gdml.n_atoms/molecule_size)
            E_contributions[nbody_order], F_contributions[nbody_order] = \
                self._calculate(R, system_size, molecule_size, gdml)

            # Adds contributions to total energy and forces.
            E_contributions['T'] += E_contributions[nbody_order]['T']
            F_contributions['T'] += F_contributions[nbody_order]['T']
        
        return E_contributions, F_contributions


    def predict(self, z, R):
        """Predicts total energy and atomic forces using many-body GDML models.
        
        Parameters
        ----------
        z : :obj:`numpy.ndarray`
            Atomic numbers of all atoms in the system.
        R : :obj:`numpy.ndarray`
            Cartesian coordinates of all atoms of the structure specified in 
            the same order as ``z``. The array should have shape ``(n, 3)``
            where ``n`` is the number of atoms.
        
        Returns
        -------
        :obj:`float`
            Total energy of the system.
        :obj:`numpy.ndarray
            Atomic forces of the system in the same shape as ``R``.
        """
        e, f = self.decomposed_predict(z, R)
        
        return e['T'], f['T']


    def remove_nbody(self, ref_dataset):
        """Removes mbGDML prediced energies and forces from a reference data
        set.

        Parameters
        ----------
        ref_dataset : :obj:`dict`
            Contains all data as :obj:`numpy.ndarray` objects.
        """
        nbody_dataset = ref_dataset
        z = nbody_dataset['z']
        R = nbody_dataset['R']
        E = nbody_dataset['E']
        F = nbody_dataset['F']
        num_config = R.shape[0]
        system = str(nbody_dataset['system'][()])
        if system == 'solvent':
            dataset_info = solvents.system_info(ref_dataset['z'].tolist())
            system_size = dataset_info['cluster_size']
        # Removing all n-body contributions for every configuration.
        for config in range(num_config):
            if z.ndim == 1:
                z_predict = z
            else:
                z_predict = z[config]
            e, f = self.predict(z_predict, R[config])
            F[config] -= f
            E[config] -= e

        # Updates dataset.
        nbody_dataset['E'] = np.array(E)
        nbody_dataset['E_min'] = np.array(np.min(E.ravel()))
        nbody_dataset['E_max'] = np.array(np.max(E.ravel()))
        nbody_dataset['E_mean'] = np.array(np.mean(E.ravel()))
        nbody_dataset['E_var'] = np.array(np.var(E.ravel()))
        nbody_dataset['F'] = np.array(F)
        nbody_dataset['F_min'] = np.array(np.min(F.ravel()))
        nbody_dataset['F_max'] = np.array(np.max(F.ravel()))
        nbody_dataset['F_mean'] = np.array(np.mean(F.ravel()))
        nbody_dataset['F_var'] = np.array(np.var(F.ravel()))
        nbody_dataset['mb'] = np.array(int(system_size))

        # Tries to add model md5 hashes to data set
        mb_models_md5 = []
        for model in self.models:
            model = dict(model)
            if 'md5' in model.keys():
                mb_models_md5.append(str(model['md5'][()]))
        nbody_dataset['mb_models_md5'] = mb_models_md5
        
        # Generating new data set name
        name_old = str(nbody_dataset['name'][()])
        nbody_label = str(int(system_size)) + 'body'
        name = '-'.join([name_old, nbody_label])
        nbody_dataset['name'] = np.array(name)

        return nbody_dataset
