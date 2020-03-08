import os
import subprocess

import mbsgdml.utils as utils
import mbsgdml.parse as parse
from wacc.calculations.packages.orca import ORCA
import wacc.calculations.packages.templates as templates


def partition_engrad(
    package, partition_dict, path_calcs, temperature, md_iteration,
    theory_level_engrad='MP2', basis_set_engrad='Def2-TZVP',
    options_engrad='TightSCF FrozenCore',
    control_blocks_engrad='%scf\n    ConvForced true\nend\n', submit=False
):
    
    # Creates calc folder name, e.g. '4H2O-300K-1-step0'.
    calc_name_base = str(int(num_molecules)) + solvent_label \
                         + '-' + str(temperature) + 'K-' + str(md_iteration) \
                         + '-step' + str(md_step)

    # Makes sure path ends with forward slash.
    if path_calcs[-1] != '/':
        path_calcs = path_calcs + '/'
    os.chdir(path_calcs)
    
    # Moves into MD step calculation folder.
    try:
        os.mkdir(calc_name_base)
        os.chdir(calc_name_base)
    except:
        os.chdir(calc_name_base)
    
    # Creates calculation object
    if package.lower() == 'orca':
        engrad_calc = ORCA()

    # Calculation properties
    engrad_calc.theoryLevel = theory_level_engrad
    engrad_calc.basisSet = basis_set_engrad
    engrad_calc.calcType = 'engrad'
    engrad_calc.options = options_engrad
    engrad_calc.controlBlocks = control_blocks_engrad
    engrad_calc.multiplicity = '1'
    engrad_calc.numCores = '4'
    engrad_calc.timeDays = '0'
    engrad_calc.timeHours = '1'
    engrad_calc.template_orca_submit_string = templates.orca_submit_template \
                                              + templates.orca_verification

    # Puts coordinates of each segment into a dictionary labeled by numbers
    structure_coords = parse.struct_dict('gdml', xyz_segments)

    
    for structure in structure_coords:
        
        engrad_calc.nameJob = structure + '-engrad'
        output_file = 'out-' + engrad_calc.nameJob + '.out'
        engrad_calc.nameOutput = output_file[:-4]
        engrad_calc.charge = '0'
        engrad_calc.coordsString = structure_coords[structure]

        os.mkdir(engrad_calc.nameJob)
        os.chdir(engrad_calc.nameJob)
        engrad_calc.write_input()
        slurm_file = engrad_calc.write_submit()
        bash_command = 'sbatch ' + slurm_file
        if submit:
            process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
        os.chdir(os.pardir)

    return None