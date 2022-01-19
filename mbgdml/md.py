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

import numpy as np
from cclib.parser.utils import convertor
from ase.calculators.calculator import Calculator
from ase.io import read
from ase.optimize import QuasiNewton
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, Stationary, ZeroRotation)
from ase.md.verlet import VelocityVerlet
from ase import units
from mbgdml import utils
from mbgdml.predict import mbPredict
from mbgdml.data import mbGDMLData


class mbGDMLCalculator(Calculator):
    """Initializes mbGDML calculator with models and units.
    
    Parameters
    -----------
    model_paths : :obj:`list`
        Paths of all models to be used for GDML prediction.
    e_unit_model : :obj:`str`, optional
        Specifies the units of energy prediction for the GDML models. Defaults
        to 'kcal/mol'.
    r_unit_model : :obj:`str`, optional
        Specifies the distance units for GDML models. Defaults to 'Angstrom'.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(
        self, elements, model_paths, entity_ids, comp_ids,
        e_unit_model='kcal/mol', r_unit_model='Angstrom', *args, **kwargs
    ):
        """

        Parameters
        ----------
        """

        # TODO logging?
        self.atoms = None
        self.elements = elements
        self.entity_ids = entity_ids
        self.comp_ids = comp_ids

        #self.load_models(model_paths)
        self.gdml_predict = mbPredict(model_paths)

        self.e_unit_model = e_unit_model
        self.r_unit_model = r_unit_model
        
        self.results = {}  # calculated properties (energy, forces, ...)
        self.parameters = None  # calculational parameters

        if self.parameters is None:
            # Use default parameters if they were not read from file:
            self.parameters = self.get_default_parameters()

        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()

    def calculate(self, atoms=None, *args, **kwargs):
        """Predicts energy and forces using many-body GDML models.
        """

        super(mbGDMLCalculator, self).calculate(
            atoms, *args, **kwargs
        )

        r = np.array(atoms.get_positions())
        e, f = self.gdml_predict.predict(
            self.elements, r, self.entity_ids, self.comp_ids
        )
        e = e[0]

        # convert model units to ASE default units (eV and Ang)
        if self.e_unit_model != 'eV':
            e *= convertor(1, self.e_unit_model, 'eV')
            f *= convertor(1, self.e_unit_model, 'eV')

        if self.r_unit_model != 'Angstrom':
            f /= convertor(1, self.r_unit_model, 'Angstrom')

        self.results = {'energy': e, 'forces': f.reshape(-1, 3)}

class mbGDMLMD(mbGDMLData):
    """Molecular dynamics through ASE with many-body GDML models.
        
    Parameters
    ----------
    structure_name : :obj:`str`
        Name of the structure. Mainly for file naming.
    structure_path : :obj:`str`
        Path to the structure file.
    """

    def __init__(self, structure_name, structure_path):
        self._load_structure(structure_name, structure_path)

    def _load_structure(self, structure_name, structure_path):
        """Sets the appropriate attributes for structure information.
        
        Parameters
        ----------
        structure_name : :obj:`str`
            Name of the structure. Mainly for file naming.
        structure_path : :obj:`str`
            Path to the structure file.
        """
        self.structure_name = structure_name
        self.structure_path = structure_path
        self.structure = read(structure_path) # mol
        self.atoms = self.structure.numbers
    
    def load_calculator(self, model_paths):
        """Loads the many-body GDML ASE calculator.
        
        Parameters
        ----------
        model_paths : :obj:`list`
            Paths to all many-body GDML models to be used.
        """

        self.calc = mbGDMLCalculator(self.atoms, model_paths)
        self.structure.set_calculator(self.calc)

    def relax(self, max_force=1e-4, steps=100):
        """Short relaxation of structure.
        
        Parameters
        ----------
        max_force : :obj:`float`, optional
            Maximum force. Defaults to ``1e-4``.
        steps : :obj:`int`, optional
            Maximum allowable steps. Defaults to ``100``.
        
        Raises
        ------
        AttributeError
            Requires a calculator to be loaded first.
        """
        if not hasattr(self, 'calc'):
            raise AttributeError('Please load a calculator first.')

        # do a quick geometry relaxation
        qn = QuasiNewton(self.structure)
        qn.run(max_force, steps)
        
    def printenergy(self, a):
        """Quick function to print MD information during simulation.
        
        Parameters
        ----------
        a : :obj:`ase.atoms`
            Atoms object from ASE.
        """
        # function to print the potential, kinetic and total energy
        e_pot = a.get_potential_energy() / len(a)
        e_kin = a.get_kinetic_energy() / len(a)
        print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
              'Etot = %.3feV' % (e_pot, e_kin, e_kin / (1.5 * units.kB),
              e_pot + e_kin))
    
    def run(self, steps, t_step, temp):
        """Runs a MD simulation using the Verlet algorithm in ASE.
        
        Parameters
        ----------
        steps : :obj:`int`
            Number of steps for the MD simulation.
        t_step : :obj:`float`
            Time step in femtoseconds.
        temp : :obj:`float`
            Temperature in Kelvin used for initializing velocities.
        
        Raises
        ------
        AttributeError
            Requires a calculator to be loaded first.
        """
        
        if not hasattr(self, 'calc'):
            raise AttributeError('Please load a calculator first.')

        # Initialize momenta corresponding to temp
        MaxwellBoltzmannDistribution(self.structure, temp * units.kB)
        Stationary(self.structure) # zero linear momentum
        ZeroRotation(self.structure) # zero angular momentum

        self.name = f'{self.structure_name}-{t_step}fs-{steps}steps-{temp}K'
        # run MD with constant energy using the VelocityVerlet algorithm
        dyn = VelocityVerlet(
            self.structure, t_step * units.fs,
            trajectory=f'{self.name}.traj'
        )

        # now run the dynamics
        self.printenergy(self.structure)
        for i in range(steps):
            dyn.run(10)
            self.printenergy(self.structure)
