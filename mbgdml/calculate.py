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
import subprocess

import numpy as np
from mako.template import Template
from periodictable import elements
from sgdml.predict import GDMLPredict
from cclib.parser.utils import convertor

from ase.calculators.calculator import Calculator
from ase.io import read
from ase.optimize import QuasiNewton
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, Stationary, ZeroRotation)
from ase.md.verlet import VelocityVerlet
from ase import units

from mbgdml import utils
from mbgdml import parse
from mbgdml import partition
from mbgdml.predict import mbGDMLPredict
from mbgdml.data import mbGDMLData


class mbGDMLCalculator(Calculator):
    """Initializes mbGDML calculator with models and units.
    
    Parameteres
    -----------
    model_paths : list
        Paths of all models to be used for GDML prediction.
    e_unit_model : str, optional
        Specifies the units of energy prediction for the GDML models. Defaults
        to 'kcal/mol'.
    r_unit_model : str, optional
        Specifies the distance units for GDML models. Defaults to 'Angstrom'.
    
    Attributes
    ----------


    Methods
    -------
    load_models(model_paths)
        Loads models for GDML prediction.
    calculate(atoms=None, *args, **kwargs)
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, elements, model_paths, e_unit_model='kcal/mol',
                 r_unit_model='Angstrom', *args, **kwargs):

        # TODO logging?
        self.atoms = None
        self.elements = elements

        self.load_models(model_paths)
        self.gdml_predict = mbGDMLPredict(self.gdmls)

        self.e_unit_model = e_unit_model
        self.r_unit_model = r_unit_model
        
        self.results = {}  # calculated properties (energy, forces, ...)
        self.parameters = None  # calculational parameters

        if self.parameters is None:
            # Use default parameters if they were not read from file:
            self.parameters = self.get_default_parameters()

        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()

    def load_models(self, model_paths):
        """Loads models for GDML prediction.
        
        Parameteres
        -----------
        model_paths : list
            Contains paths of all GDML models to be loaded. All of these models 
            will be used during the MD run. 
        """
        gdmls = []
        model_index = 0
        while model_index < len(model_paths):
            loaded_model = np.load(model_paths[model_index])
            predict_model = GDMLPredict(loaded_model)
            gdmls.append(predict_model)

            model_index += 1
        
        self.gdmls = gdmls

    def calculate(self, atoms=None, *args, **kwargs):
        """Predicts energy and forces using many-body GDML models.

        """

        super(mbGDMLCalculator, self).calculate(
            atoms, *args, **kwargs
        )

        r = np.array(atoms.get_positions())
        e, f = self.gdml_predict.predict(self.elements, r)

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
    structure_name : str
        Name of the structure. Mainly for file naming.
    structure_path : str
        Path to the structure file.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, structure_name, structure_path):
        self._load_structure(structure_name, structure_path)

    def _load_structure(self, structure_name, structure_path):
        """Sets the appropriate attributes for structure information.
        
        Parameteres
        -----------
        structure_name : str
            Name of the structure. Mainly for file naming.
        structure_path : str
            Path to the structure file.
        """
        self.structure_name = structure_name
        self.structure_path = structure_path
        self.structure = read(structure_path) # mol
        self.atoms = self.structure.numbers
    
    def load_calculator(self, model_paths):
        """Loads the many-body GDML ASE calculator.
        
        Parameteres
        -----------
        model_paths : list
            Paths to all many-body GDML models to be used.
        """

        self.calc = mbGDMLCalculator(self.atoms, model_paths)
        self.structure.set_calculator(self.calc)

    def relax(self, max_force=1e-4, steps=100):
        """Short relaxation of structure.
        
        Parameteres
        -----------
        max_force : float, optional
            Maximum force. Defaults to 1e-4.
        steps : int, optional
            Maximum allowable steps. Defaults to 100.
        
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
        
        Parameteres
        -----------
        a : `ase.atoms`
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
        
        Parameteres
        -----------
        steps : `int`
            Number of steps for the MD simulation.
        t_step : `float`
            Time step in femtoseconds.
        temp : `float`
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

class CalcTemplate:
    """Contains all quantum chemistry calculations templates for mako.
    """

    def __init__(self):
        self.orca_input_template = \
'\
# ${nameJob}\n\
${commandSignal} ${theoryLevel} ${basisSet} ${calcType} ${options}\n\
\n\
${control_signal}pal\n\
nprocs ${numCores}\n\
end\n\
\n\
${controlBlocks}\n\
\n\
*xyz ${charge} ${multiplicity}\n\
${coords}\
*\n\
'

        self.orca_submit_template = \
'\
#!/bin/bash\n\
#SBATCH --job-name=${nameJob}\n\
#SBATCH --output=${nameOutput}.out\n\
#SBATCH --nodes=${numNodes}\n\
#SBATCH --ntasks-per-node=${numCores}\n\
#SBATCH --time=${timeDays}-${timeHours}:00:00\n\
#SBATCH --cluster=${nameCluster}\n\
\n\
cd $SBATCH_O_WORKDIR\n\
module purge\n\
module load openmpi/3.1.4\n\
module load orca/4.2.0\n\
\n\
cp $SLURM_SUBMIT_DIR/*.inp $SLURM_SCRATCH\n\
\n\
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH\n\
# Suppresses OpenFabrics (openib) absence nonfatal error.\n\
export OMPI_MCA_btl_base_warn_component_unused=0\n\
# Makes all error messages print.\n\
export OMPI_MCA_orte_base_help_aggregate=0\n\
\n\
cd $SLURM_SCRATCH\n\
$(which orca) *.inp\n\
\n\
'

        self.orca_add_job='\
\n\n$new_job\n\n\
'

class ORCA:
    """Prepares, writes, and submits ORCA 4.2.0 calculations.
    """

    def __init__(self):
        
        templates = CalcTemplate()
        self.commandSignal = '!'  # Command for ORCA
        self.control_signal = '%'
        self.fileExtension = 'inp'  # Defines file extension for ORCA input files
        self.progression_parameters = ''
        self.template_orca_string = templates.orca_input_template
        self.template_orca_submit_string = templates.orca_submit_template

        # Calculation properties
        # Required properties are None type, whereas optionals are blank strings
        self.nameJob = None  # Name of the job for record keeping
        self.nameOutput = None  # Name of the output file (for verification and
                                # progression).
        self.nameInput = None  # Name of the input file
        self.inputFiles = None  # Name of all input files
        self.theoryLevel = None  # Level of theory
        self.basisSet = None  # Basis sets
        self.calcType = None  # ORCA calculation type
        self.options = ''  # ORCA calculation options e.g. solvation models
        self.nameCluster = 'smp'  # CRC cluster; default is smp
        self.numNodes = '1'  # Number of nodes; smp default is 1
        self.numCores = '12'   # Number of cores to use; default is 12
        self.timeDays = None  # Requested days of runtime
        self.timeHours = None  # Requested hours of runtime
        self.controlBlocks = ''  # Additional control for different 

        # System Properties
        self.charge = None  # Overall charge of the system
        self.multiplicity = None  # Multiplicity of the system
        self.coordsString = None  # String of coordinates; each atom coordinate separated by '\n'


    def write_input(self):
        
        templateOrca = Template(self.template_orca_string)
        
        self.input_corrections()

        renderedOrca = templateOrca.render(
            nameJob=self.nameJob, commandSignal=self.commandSignal,
            control_signal=self.control_signal, theoryLevel=self.theoryLevel,
            basisSet=self.basisSet, calcType=self.calcType,
            options=self.options, numCores=self.numCores, charge=self.charge,
            multiplicity=self.multiplicity, controlBlocks=self.controlBlocks,
            coords=self.coordsString
        )

        with open(str(self.nameJob).replace(' ', '-') + '.' + self.fileExtension, 'w') as inputFile:
            inputFile.write(renderedOrca)

        return None
    
    def write_submit(self):
        
        templateOrcaSubmit = Template(self.template_orca_submit_string)

        # Guesses default calculation run time based on type is not already specified
        if self.timeDays == None or self.timeHours == None:
            if str(self.calcType).lower() == 'opt':
                self.timeDays = '1'
                self.timeHours = '00'
            elif str(self.calcType).lower() == 'spe':
                self.timeDays = '0'
                self.timeHours = '12'
            elif str(self.calcType).lower() == 'freq':
                self.timeDays = '3'
                self.timeHours = '00'
            else:
                self.timeDays = '3'
                self.timeHours = '00'

        self.input_corrections()

        renderedOrcaSubmit = templateOrcaSubmit.render(
            nameJob = self.nameJob, nameOutput = self.nameOutput,
            nameCluster = self.nameCluster, numNodes = self.numNodes,
            numCores = self.numCores, timeDays = self.timeDays,
            timeHours = self.timeHours, varBash = '$',
            nameInput = self.nameInput, calcType=self.calcType,
            inputFiles = self.nameInput + '.' + self.fileExtension,
            progression_parameters = self.progression_parameters
        )

        slurm_file = 'submit-' + str(self.nameJob).replace(' ', '-') + '.slurm'

        with open(slurm_file, 'w') as inputFile:
            inputFile.write(renderedOrcaSubmit)

        return slurm_file
    

    def input_corrections(self):
        
        # Ensures correct spe keyword for ORCA.
        if self.calcType.lower() == 'spe':
            self.calcType = 'SP'

        # Forces numerical frequencies if using frozencore and MP2.
        if ' frozencore' in self.options.lower() and self.calcType == 'freq' \
           and 'mp2' in self.theoryLevel.lower():
            self.calcType = 'NumFreq'
        
        self.nameInput = self.nameJob
        # File corrections.

        return None


def partition_engrad(
    package,
    partition_dict,
    temperature,
    md_iteration,
    calc_dir='.',
    calc_name='partition-engrad',
    theory_level_engrad='MP2',
    basis_set_engrad='def2-TZVP',
    options_engrad='TightSCF FrozenCore',
    control_blocks_engrad='%scf\n    ConvForced true\nend\n%maxcore 8000\n',
    submit=False
):
    """Sets up a partition ORCA 4.2.0 EnGrad calculation for trajectory.
    
    Can be submitted or just have the input files prepared.
    
    Parameteres
    -----------
    package : `str`
        Specifies the quantum chemistry program to be used. ORCA is currently
        the only package directly supported.
    partition_dict : `dict`
        Contains all information for the partition including 'solvent_label',
        'partition_size', 'atoms', and 'coords'.
    temperature : `int`
        Used for labeling and identifying the thermostat temperature for the 
        molecular dynamics simulation.
    md_iteration : `int`
        Used for labeling and identifying the iteration of the molecular 
        dynamics simulation.
    calc_dir : `str`, optional
        Path to write calculation. Defaults to current directory ('./').
    calc_name : `str`, optional
        Name for the calculation. Defaults to 'partition-engrad'.
    theory_level_engrad : `str`, optional
        Keword that specifies the level of theory (e.g., MP2, BP86, B3LYP, 
        etc.) used for calculations. Defaults to 'MP2' for ORCA 4.2.0.
    basis_set_engrad : `str`, optional
        Keyword that specifies the basis set (e.g., def2-SVP, def2-TZVP, etc.). 
        Defaults to 'def2-TZVP'.
    options_engrad : `str`, optional
        All options specifically for the EnGrad calculation (e.g., SCF 
        convergence criteria, algorithms, etc.) Defaults to 'TightSCF 
        FrozenCore'.
    control_blocks_engrad : `str`, optional
        All options that control the calculation. Defaults to
        '%scf    ConvForced true end'.
    submit : `bool`, optional
        Controls whether the calculation is submitted. Defaults to False.
    
    Examples
    --------
    >>> calculate.partition_engrad('orca', '/path/to/dir/', partition_dict, 
    temp, iteration)
    """


    # Normalizes path
    calc_path = utils.norm_path(calc_dir)

    # Gets solvent information
    #solvent_info = solvents.system_info(parsed_traj['atoms'])

    # Moves into MD step calculation folder.
    os.makedirs(calc_path, exist_ok=True)
    os.chdir(calc_path)
    
    # Creates calculation object
    if package.lower() == 'orca':
        engrad = ORCA()

    # Calculation properties
    engrad.theoryLevel = theory_level_engrad
    engrad.basisSet = basis_set_engrad
    engrad.calcType = 'engrad'
    engrad.options = options_engrad
    engrad.controlBlocks = control_blocks_engrad
    engrad.charge = '0'
    engrad.multiplicity = '1'
    engrad.numCores = '6'
    engrad.timeDays = '01'
    engrad.timeHours = '00'

    # We want to skip the first step because this will be the same 
    # for every temperature and iteration.
    templates = CalcTemplate()
    step_index = 1
    while step_index < partition_dict['coords'].shape[0]:

        # If this is the first step, we need to write the submission file.
        if step_index == 1:
            engrad.nameJob = calc_name
            output_file = 'out-' + engrad.nameJob + '.out'
            engrad.nameOutput = output_file[:-4]
            engrad.template_orca_submit_string = templates.orca_submit_template
            slurm_file = engrad.write_submit()
        else:
            engrad.template_orca_string = templates.orca_add_job \
                                        + templates.orca_input_template

        engrad.nameJob = calc_name + '-step' + str(step_index)
        engrad.coordsString = utils.string_coords(
            partition_dict['atoms'],
            partition_dict['coords'][step_index]
        )
        engrad.write_input()
        
        
        with open(calc_name + '.inp', 'a+') as outfile:
            with open(engrad.nameJob + '.inp') as infile:
                for line in infile:
                    outfile.write(line)
        os.remove(engrad.nameJob + '.inp')
        
        step_index += 1
    
    bash_command = 'sbatch ' + slurm_file
    if submit:
        submit_process = subprocess.Popen(
            bash_command.split(),
            stdout=subprocess.PIPE
        )
        output, error = submit_process.communicate()
