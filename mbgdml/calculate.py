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
from sgdml.predict import GDMLPredict
from cclib.parser.utils import convertor
from ase.calculators.calculator import Calculator
from ase.io import read
from ase.optimize import QuasiNewton
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, Stationary, ZeroRotation)
from ase.md.verlet import VelocityVerlet
from ase import units
from mbgdml import utils
from mbgdml import partition
from mbgdml.predict import mbGDMLPredict
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
        
        Parameters
        ----------
        model_paths : :obj:`list` [:obj:`str`]
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

class CalcTemplate:
    """Contains all quantum chemistry templates for mako.

    Parameters
    ----------
    package : :obj:`str`
        Computational chemistry package to perform calculations.
    """

    def __init__(self, package):
        if package.lower() == 'orca':
            self.input = (
                "# ${job_name}\n"
                "${command_signal} ${theory} ${basis_set} ${calc_type} ${options}\n"
                "\n"
                "${control_signal}pal\n"
                "    nprocs ${cores}\n"
                "end\n"
                "\n"
                "${control_blocks}\n"
                "\n"
                "*xyz ${charge} ${multiplicity}\n"
                "${coords}"
                "*\n"
            )


            self.submit = (
                "#!/bin/bash\n"
                "#SBATCH --job-name=${job_name}\n"
                "#SBATCH --output=${output_name}.out\n"
                "#SBATCH --nodes=${nodes}\n"
                "#SBATCH --ntasks-per-node=${cores}\n"
                "#SBATCH --time=${days}-${hours}:00:00\n"
                "#SBATCH --cluster=${cluster}\n"
                "\n"
                "${submit_script}\n"
            )

            self.add_job="\n\n$new_job\n\n"

class ORCA:
    """Prepares, writes, and submits ORCA 4 calculations.
    """

    def __init__(
        self,
        job_name,
        input_name,
        output_name
    ):
        """
        Parameters
        ----------
        job_name : :obj:`str`
            Name of the job for SLURM input file.
        input_name : :obj:`str`
            File name for the input file.
        output_name : :obj:`str`
            File name for the output file.
        """
        templates = CalcTemplate('orca')

        # ORCA properties
        self.command_signal = '!'
        self.control_signal = '%'
        self.input_extension = 'inp'
        self.progression_parameters = ''
        self.template_input = templates.input
        self.template_submit = templates.submit

        # Calculation properties
        self.job_name = job_name
        self.input_name = input_name
        self.output_name = output_name


    def input(
        self,
        calc_type,
        coords,
        theory,
        basis_set,
        charge,
        multiplicity,
        cores,
        options='',
        control_blocks='',
        write=True,
        write_dir='.',
        input_extension='inp'
    ):
        """Rendered input file as string.

        Parameters
        ----------
        calc_type : :obj:`str`
            Type of calculation. Options are ``'SP'``, ``'ENGRAD'``, ``'OPT'``,
            ``'FREQ'``, ``'NUMFREQ'``. Note that analytical frequency
            calculation is called with ``'FREQ'``.
        coords : :obj:`str`
            XYZ atomic coordinates as a string. A water molecule for example,
            ``'O          0.00000        0.00000        0.11779\nH         
            0.00000        0.75545       -0.47116\nH          0.00000      
            -0.75545       -0.47116'``.
        theory : :obj:`str`
            The level of theory for the calculations. For example, ``'B3LYP'``
            or ``'MP2'``.
        basis_set : :obj:`str`
            The basis set to be used in the calculations. For example,
            ``'def2-TZVP'``.
        charge : :obj:`int`
            System charge.
        multiplicity : :obj:`int`
            System multiplicity.
        cores : :obj:`int`
            Number of requested cores per node.
        options : :obj:`str`
            Other calculations options such as implicit solvents, convergence 
            criteria, etc. For example, ``'CPCM(water) Grid4 TightSCF'``.
        control_blocks : :obj:`str`, optional
            All options that control the calculation. For example
            ``'%scf\n    ConvForced true\nend\n%maxcore 8000\n'``.
        write : :obj:`bool`, optional
            Whether or not to write the file. Defaults to ``True``.
        write_dir : :obj:`str`, optional
            Directory to write the input file. Defaults to ``'.'``
        input_extension: :obj:`str`, optional
            File extension for ORCA input. Defaults to ``'inp'``.
        """
        self.calc_type = calc_type
        self.charge = charge
        self.multiplicity = multiplicity
        self.coords = coords

        self.theory = theory
        self.basis_set = basis_set
        if calc_type.lower() in ['sp', 'engrad', 'opt', 'freq', 'numfreq']:
            self.calc_type = calc_type
        else:
            raise ValueError(f'{calc_type} is unsupported.')
        self.options = options
        self.control_blocks = control_blocks
        self.cores = cores
        templateOrca = Template(self.template_input)
        self.input_check()

        rendered = templateOrca.render(
            job_name=self.job_name, command_signal=self.command_signal,
            control_signal=self.control_signal, theory=self.theory,
            basis_set=self.basis_set, calc_type=self.calc_type,
            options=self.options, cores=str(self.cores), charge=str(self.charge),
            multiplicity=str(self.multiplicity),
            control_blocks=self.control_blocks, coords=self.coords
        )
        filename = str(self.input_name).replace(' ', '-') \
                   + '.' + self.input_extension
        if write:
            if write_dir[-1] != '/':
                write_dir += '/'
            with open(write_dir + filename, 'w') as inputFile:
                inputFile.write(rendered)

        return filename, rendered
    
    def submit(
        self,
        cluster,
        nodes,
        cores,
        days,
        hours,
        submit_script,
        write=True,
        write_dir='.'
    ):
        """Prepare submission script.

        Parameters
        ----------
        cluster : :obj:`str`
            Name of cluster for calculations. For example, ``'smp'``.
        nodes : :obj:`int`
            Number of requested nodes.
        cores : :obj:`int`
            Number of requested cores per node.
        days : :obj:`int`
            Requested run time days.
        hours : :obj:`int`
            Requested run time hours.
        write : :obj:`bool`, optional
            Whether or not to write the file. Defaults to ``True``.
        write_dir : :obj:`str`, optional
            Directory to write the input file. Defaults to ``'.'``
        
        Returns
        -------
        :obj:`str`
            File name (with extension) of the submission script.
        :obj:`str`
            Rendered submission script.
        """
        # Run options
        self.cluster = cluster
        self.nodes = nodes
        self.cores = cores
        self.days = days
        self.hours = hours
        self.submit_script = submit_script
        
        templateOrcaSubmit = Template(self.template_submit)

        rendered = templateOrcaSubmit.render(
            job_name = self.job_name, output_name = self.output_name,
            cluster = self.cluster, nodes = self.nodes,
            cores = self.cores, days = str(self.days),
            hours = str(self.hours), input_name = self.input_name,
            submit_script=self.submit_script
        )

        file_name = 'submit-' + str(self.job_name).replace(' ', '-') + '.slurm'

        if write:
            if write_dir[-1] != '/':
                write_dir += '/'
            with open(write_dir + file_name, 'w') as inputFile:
                inputFile.write(rendered)

        return file_name, rendered
    

    def input_check(self):
        """Performs checks on input specifications.
        """

        # ORCA requires numerical frequencies if using frozencore and MP2.
        if ' frozencore' in self.options.lower() and self.calc_type == 'freq' \
           and 'mp2' in self.theory.lower():
            self.calc_type = 'NumFreq'


def partition_engrad(
    package,
    z,
    R,
    job_name,
    input_name,
    output_name,
    theory,
    basis_set,
    charge,
    multiplicity,
    cluster,
    nodes,
    cores,
    days,
    hours,
    calc_dir='.',
    options='',
    control_blocks='',
    submit_script='',
    write=True,
    submit=False
):
    """Partition ORCA 4 EnGrad calculation for trajectory.
    
    Can be submitted using ``sbatch`` or just have the input files prepared.
    Default submission script is set for University of Pittsburgh's Center for
    Research Computing cluster.
    
    Parameters
    ----------
    package : :obj:`str`
        Specifies the quantum chemistry program to be used. ``'ORCA'`` is
        currently the only package directly supported.
    z : :obj:`numpy.ndarray`
        A ``(n,)`` or ``(m, n)`` shape array of type :obj:`numpy.int32`
        containing atomic numbers of atoms in the structures in order as they
        appear for every ``m`` structure.
    R : :obj:`numpy.ndarray`
        A :obj:`numpy.ndarray` with shape of ``(n, 3)`` or ``(m, n, 3)`` where
        ``m`` is the number of structures and ``n`` is the number of atoms with
        three Cartesian components.
    job_name : :obj:`str`
            Name of the job for SLURM input file.
    input_name : :obj:`str`
        File name for the input file.
    output_name : :obj:`str`
        File name for the output file.
    theory : :obj:`str`, optional
        Keword that specifies the level of theory (e.g., MP2, BP86, B3LYP, 
        etc.) used for calculations. Defaults to 'MP2' for ORCA 4.2.0.
    basis_set : :obj:`str`, optional
        Keyword that specifies the basis set (e.g., def2-SVP, def2-TZVP, etc.). 
        Defaults to 'def2-TZVP'.
    charge : :obj:`int`, optional
        System charge. Defaults to ``0``.
    multiplicity : :obj:`int`, optional
        System multiplicity. Defaults to ``1``.
    cluster : :obj:`str`
        Name of cluster for calculations. For example, ``'smp'``.
    nodes : :obj:`int`
        Number of requested nodes.
    cores : :obj:`int`
        Number of processing cores for the calculation.
    days : :obj:`int`
        Requested run time days.
    hours : :obj:`int`
        Requested run time hours.
    calc_dir : :obj:`str`, optional
        Path to write calculation. Defaults to current directory (``'.'``).
    options : :obj:`str`, optional
        All options specifically for the EnGrad calculation (e.g., SCF 
        convergence criteria, algorithms, etc.).
    control_blocks : :obj:`str`, optional
        All options that control the calculation.
    submit_script : :obj:`str`, optional
        The SLURM submission script content. Defaults to
        ``pitt_crc_orca_submit``.
    write : :obj:`bool`, optional
        Whether or not to write the file. Defaults to ``True``.
    submit : :obj:`bool`, optional
        Controls whether the calculation is submitted. Defaults to ``False``.
    
    Returns
    -------
    :obj:`str`
        The SLURM submission script.
    :obj:`str`
        The input file.
    """
    if calc_dir[-1] != '/':
        calc_dir += '/'
    os.makedirs(calc_dir, exist_ok=True)
    os.chdir(calc_dir)

    if z.ndim == 1:
        z = np.array([z])
    if R.ndim == 2:
        R = np.array([R])
    
    # Prepares calculation
    if package.lower() == 'orca':
        engrad = ORCA(
            job_name,
            input_name,
            output_name
        )
        templates = CalcTemplate('orca')

    # Writes initial files
    input_file_string = ''
    if submit_script == '':
        submit_script = pitt_crc_orca_submit
    slurm_file_name, slurm_file = engrad.submit(
        cluster,
        nodes,
        cores,
        days,
        hours,
        submit_script,
        write=write,
        write_dir=calc_dir
    )
    for step_index in range(0, R.shape[0]):
        if z.shape[0] == 1:
            step_z = z[0]
        else:
            step_z = z[step_index]
        step_R = R[step_index]
        engrad.template_input = templates.add_job \
                                    + templates.input

        step_R_string = utils.string_coords(step_z, step_R)
        _, calc_string = engrad.input(
            'EnGrad',
            step_R_string,
            theory,
            basis_set,
            charge,
            multiplicity,
            cores,
            options=options,
            control_blocks=control_blocks,
            write=False,
        )
        if input_file_string == '':
            input_file_string = calc_string
        else:
            input_file_string += calc_string
    
    if write:
        with open(input_name + '.' + engrad.input_extension, 'w') as f:
            f.write(input_file_string)
    
    if submit:
        bash_command = 'sbatch ' + slurm_file_name
        submit_process = subprocess.Popen(
            bash_command.split(),
            stdout=subprocess.PIPE
        )
        _, _ = submit_process.communicate()

    return slurm_file, input_file_string


pitt_crc_orca_submit = (
    "cd $SBATCH_O_WORKDIR\n"
    "module purge\n"
    "module load openmpi/3.1.4\n"
    "module load orca/4.2.0\n"
    "\n"
    "cp $SLURM_SUBMIT_DIR/*.inp $SLURM_SCRATCH\n"
    "\n"
    "export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH\n"
    "# Suppresses OpenFabrics (openib) absence nonfatal error.\n"
    "export OMPI_MCA_btl_base_warn_component_unused=0\n"
    "# Makes all error messages print.\n"
    "export OMPI_MCA_orte_base_help_aggregate=0\n"
    "\n"
    "cd $SLURM_SCRATCH\n"
    "$(which orca) *.inp\n"
)