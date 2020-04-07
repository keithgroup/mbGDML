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

import itertools

import numpy as np
from sgdml.predict import GDMLPredict
import mbgdml.solvents as solvents

class mbGDMLPredict():


    def __init__(self, mb_gdml):
        """Sets GDML models to be used for many-body prediction.
        
        Args:
            mb_gdml (list): Contains sgdml.GDMLPredict objects for all models
                to be used for prediction.
        """
        self.gdmls = mb_gdml


    def _calculate(self, z, R, system_size, molecule_size, gdml):
        """The actual calculate/predict function for a single GDML model.

        Predicts the energy and forces of a structure from a single GDML model.
        In other words, it predicts the energy and force contribution from a
        single many-body (or standard) GDML model.
        
        Args:
            z (list): Atomic numbers of all atoms in the system.
            R (np.ndarray): Cartesian coordinates of all atoms of the structure
                specified in the same order as z. The array should have shape
                (n, 3) where n is the number of atoms.
            system_size (int): Number of molecules of the defined chemical
                species in the structure.
            molecule_size (int): Number of atoms in the defined chemical
                species.
            gdml (sgdml.GDMLPredict): Object used to predict energies
                and forces of the structure defined in R.
        
        Returns:
            tuple: Dictionaries of contributions and total energies (float) and
                forces (np.ndarray) of the structure. Note that forces has
                the same shape as R. Each key of the dictionary is the molecule
                combination. For example, '01' contains the predictions of
                two-body energy and forces of the pair containing molecules 0
                and 1. Additionally, each dict has 'T' representing the total
                energy or forces.
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
            comb_label = ''.join(str(molecule) for molecule in comb)
            E_contributions[comb_label] = e
            F_contributions[comb_label] = f.reshape(len(atoms), 3)

            # Adds contributions to total energy and forces.
            E_contributions['T'] += E_contributions[comb_label]
            F_contributions['T'][atoms] += F_contributions[comb_label]
        
        return E_contributions, F_contributions


    def decomposed_predict(self, z, R):
        """Computes predicted total energy and atomic forces decomposed by
        many body order.
        
        Args:
            z (list): Atomic numbers of all atoms in the system.
            R (np.ndarray): Cartesian coordinates of all atoms of the structure
                specified in the same order as z. The array should have shape
                (n, 3) where n is the number of atoms.
        
        Returns:
            tuple: Dictionaries of many-body contributions of total energies
                (float) and forces (np.ndarray) of the structure. Note that
                forces has the same shape as R. Each key of the dictionary is
                the order of the many-body model. Within this dictionary is a
                total, 'T', and molecule combinations identified in the system
                and their contributions.
        
        Examples:
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
        dataset_info = solvents.system_info(z.tolist())
        system_size = dataset_info['cluster_size']
        molecule_size = dataset_info['solvent_molec_size']

        # 'T' is for total
        E_contributions = {'T': 0.0}
        F_contributions = {'T': np.zeros(R.shape)}

        # Adds contributions from all models.
        for gdml in self.gdmls:
            
            nbody_order = int(gdml.n_atoms/molecule_size)
            E_contributions[nbody_order], F_contributions[nbody_order] = \
                self._calculate(z, R, system_size, molecule_size, gdml)

            # Adds contributions to total energy and forces.
            E_contributions['T'] += E_contributions[nbody_order]['T']
            F_contributions['T'] += F_contributions[nbody_order]['T']
        
        return E_contributions, F_contributions


    def predict(self, z, R):
        """Predicts total energy and atomic forces using many-body GDML models.
        
        Args:
            z (list): Atomic numbers of all atoms in the system.
            R (np.ndarray): Cartesian coordinates of all atoms of the structure
                specified in the same order as z. The array should have shape
                (n, 3) where n is the number of atoms.
        
        Returns:
            tuple: Total energy of the system (float) and atomic forces
                (np.ndarray) in the units specified by the GDML models.
                Note that the forces have the same shape as R.
        """
        e, f = self.decomposed_predict(z, R)
        
        return e['T'], f['T']


    def remove_nbody(self, dataset):
        """Creates GDML dataset with GDML predicted n-body predictions
        removed.

        To employ the many body expansion, we need GDML models that predict
        n-body corrections/contributions. This provides the appropriate dataset
        for training an (n+1)-body mbGDML model.

        Args:
            base_vars (dict): GDML dataset converted to a dict containing n-body
                contributions to be removed.
        """

        if len(self.gdmls) != 1:
            raise ValueError('N-body contributions can only be removed one '
                             'at time. Please load only one GDML.')
        else:
            nbody_model = self.gdmls[0]
        
        # Assinging variables from dataset to be updated or used.
        base_vars = dict(dataset)
        z = base_vars['z']
        R = base_vars['R']
        E = base_vars['E']
        F = base_vars['F']

        # Getting information from the dataset.
        num_config = R.shape[0]

        # Getting system information from dataset and model.
        system = str(base_vars['system'][()])

        if system == 'solvent':
            dataset_info = solvents.system_info(z.tolist())
            model_info = solvents.system_info(nbody_model.f.z.tolist())
        
        system_size = dataset_info['cluster_size']
        nbody_order = model_info['cluster_size']
        molecule_size = model_info['solvent_molec_size']
        
        # Getting list of n-body combinations.
        nbody_combinations = list(
            itertools.combinations(list(range(0, system_size)), nbody_order)
        )

        # Removing all n-body contributions for every configuration.
        gdml = GDMLPredict(nbody_model)
        for config in range(num_config):

            for comb in nbody_combinations:
                # Gets indices of all atoms in the
                # n-body combination of molecules.
                atoms = []
                for molecule in comb:
                    atoms += list(range(
                        molecule * molecule_size, (molecule + 1) * molecule_size
                    ))

                # Removes n-body contributions prediced from nbody_model
                # from the dataset.
                e, f = gdml.predict(R[config, atoms].flatten())
                F[config, atoms] -= f.reshape(len(atoms), 3)
                E[config] -= e

        # Updates dataset.
        base_vars['E'] = E
        base_vars['E_min'] = np.min(E.ravel())
        base_vars['E_max'] = np.max(E.ravel())
        base_vars['E_mean'] = np.mean(E.ravel())
        base_vars['E_var'] = np.var(E.ravel())
        base_vars['F'] = F
        base_vars['F_min'] = np.min(F.ravel())
        base_vars['F_max'] = np.max(F.ravel())
        base_vars['F_mean'] = np.mean(F.ravel())
        base_vars['F_var'] = np.var(F.ravel())

        if 'mb' in base_vars.keys():
            if type(base_vars['mb']) is not int:
                o_mb = base_vars['mb'][()]
            else:
                o_mb = base_vars['mb']
            n_nb = int(nbody_order) + 1
            if o_mb > n_nb:
                mb = o_mb
            else:
                mb = n_nb
            base_vars['mb'] = mb
        else:
            base_vars['mb'] = int(nbody_order + 1)

        cluster_label = str(base_vars['name'][()]).split('-')[0]

        nbody_label = str(base_vars['mb']) + 'body'

        name = '-'.join([cluster_label, nbody_label, 'dataset'])
        base_vars['name'][()] = name

        return base_vars
