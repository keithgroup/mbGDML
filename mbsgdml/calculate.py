import os
import subprocess

from periodictable import elements
from mbsgdml import utils, parse, partition
from wacc.calculations.packages.orca import ORCA
import wacc.calculations.packages.templates as templates


def partition_engrad(
    package, calc_path, partition_dict, temperature, md_iteration,
    theory_level_engrad='MP2', basis_set_engrad='def2-TZVP',
    options_engrad='TightSCF FrozenCore',
    control_blocks_engrad='%scf\n    ConvForced true\nend\n', submit=False
):
    """ Sets up a partition ORCA 4.2.0 EnGrad calculation for trajectory.
    
    Can be submitted or just have the input files prepared.
    
    Args:
        package (str): specifies the quantum chemistry program to be used. ORCA
            is currently the only package directly supported.
        calc_path (str): path to the parent directory for the calculation
            directory.
        partition_dict (dict): contains all information for the partition
            including 'solvent_label', 'partition_size', 'atoms', and 'coords'.
        temperature (int): used for labeling and identifying the thermostat
            temperature for the molecular dynamics simulation.
        md_iteration (int): used for labeling and identifying the iteration of
            the molecular dynamics simulation.
        theory_level_engrad (str, optional): keword that specifies the
            level of theory (e.g., MP2, BP86, B3LYP, etc.) used for calculations.
            Defaults to 'MP2' for ORCA 4.2.0.
        basis_set_engrad (str, optional): keyword that specifies the basis set
            (e.g., def2-SVP, def2-TZVP, etc.). Defaults to 'def2-TZVP'.
        options_engrad (str, optional): all options specifically for the EnGrad
            calculation (e.g., SCF convergence criteria, algorithms, etc.)
            Defaults to 'TightSCF FrozenCore'.
        control_blocks_engrad (str, optional): all options that control the
            calculation. Defaults to '%scf    ConvForced true end'.
        submit (bool, optional): controls whether the calculation is submitted.
            Defaults to False.
    
    Example:
        calculate.partition_engrad(
            'orca', '/path/to/dir/',
            partition_dict, temp, iteration
        )
    """


    # Normalizes path
    calc_path = utils.norm_path(calc_path)
    
    # Creates calc folder name, e.g. '4H2O-300-1'.
    calc_name_base = str(partition_dict['cluster_size']) \
                     + partition_dict['solvent_label'] \
                     + '-' + str(temperature) \
                     + '-' + str(md_iteration) \
                     + '-' + partition_dict['partition_label']

    # Moves into MD step calculation folder.
    try:
        os.chdir(calc_path)
        os.mkdir(calc_name_base)
        os.chdir(calc_name_base)
    except:
        print('This folder already exists.')
        return None
    
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
    step_index = 1
    while step_index < partition_dict['coords'].shape[0]:

        # If this is the first step, we need to write the submission file.
        if step_index == 1:
            engrad.nameJob = calc_name_base
            output_file = 'out-' + engrad.nameJob + '.out'
            engrad.nameOutput = output_file[:-4]
            engrad.template_orca_submit_string = templates.orca_submit_template
            slurm_file = engrad.write_submit()
        else:
            engrad.template_orca_string = templates.orca_add_job \
                                        + templates.orca_input_template

        engrad.nameJob = calc_name_base + '-step' + str(step_index)
        engrad.coordsString = utils.string_coords(
            partition_dict['atoms'],
            partition_dict['coords'][step_index]
        )
        engrad.write_input()
        
        
        with open(calc_name_base + '.inp', 'a+') as outfile:
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
