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

    def __init__(self):
        pass


    def remove_nbody(self, base_vars, nbody_model):
        """Updates GDML dataset with GDML predicted n-body predictions
        removed.

        To employ the many body expansion, we need GDML models that predict
        n-body corrections/contributions. This provides the appropriate dataset
        for training an (n+1)-body mbGDML model.

        Args:
            base_vars (dict): GDML dataset converted to a dict containing n-body
                contributions to be removed.
            nbody_model (np.NpzFile): Loaded mbGDML model that predicts n-body
                contributions.
        """

        # Assinging variables from dataset to be updated or used.
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

            '''
            # DEBUG START
            if config > 380 and config < 385:
                o_energy = base_vars['E'][config][()]
                o_forces = base_vars['F'][config]
                print(f'\n\n-----Configuration {config}-----')
                print(f'Original Energy: {o_energy} kcal/mol')
                print(f'Original Forces:\n{o_forces}')
            # DEBUG END
            '''
            for comb in nbody_combinations:
                # Gets indices of all atoms in the
                # n-body combination of molecules.
                atoms = []
                for molecule in comb:
                    atoms += list(range(
                        molecule * molecule_size, (molecule + 1) * molecule_size
                    ))
                '''
                o_energy = E[config][()]
                o_forces = F[config, atoms]
                '''
                # Removes n-body contributions prediced from nbody_model
                # from the dataset.
                e, f = gdml.predict(R[config, atoms].flatten())
                F[config, atoms] -= f.reshape(len(atoms), 3)
                E[config] -= e

                '''
                # DEBUG
                if config > 380 and config < 385:
                    print(f'\nMolecule combination: {comb}')
                    print(f'Atoms: {atoms}')

                    print(f'Original energy: {o_energy}')
                    print(f'Predicted energy: {e[0]}')
                    print(f'New energy: {E[config]}')

                    print(f'Original forces:\n{o_forces}')
                    print(f'Predicted forces:\n{f.reshape(len(atoms), 3)}')
                    print(f'New forces:\n{F[config, atoms]}')
                # DEBUG END
                '''

        
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
