import os
import subprocess

from periodictable import elements
import mbsgdml.utils as utils
import mbsgdml.parse as parse
import mbsgdml.partition as partition
from wacc.calculations.packages.orca import ORCA
import wacc.calculations.packages.templates as templates

def string_coords(atoms_array, coords_array):
    
    combined = ''
    atom_index = 0
    while atom_index < len(atoms_array):
        element = str(atoms_array[atom_index])
        coords = np.array2string(
            coords_array[atom_index],
            suppress_small=True, separator='   ',
            formatter={'float_kind':'{:0.9f}'.format}
        )[1:-1] + '\n'
        
        combined += (element + '   ' + coords).replace(' -', '-')

        atom_index += 1


def partition_engrad(
    package, calc_path, partition_dict, temperature, md_iteration,
    theory_level_engrad='MP2', basis_set_engrad='Def2-TZVP',
    options_engrad='TightSCF FrozenCore',
    control_blocks_engrad='%scf\n    ConvForced true\nend\n', submit=False
):


    # Normalizes path
    calc_path = utils.norm_path(calc_path)
    
    # Creates calc folder name, e.g. '4H2O-300K-1'.
    calc_name_base = partition_dict['partition_label'] \
                     + '-' + str(partition_dict['partition_size']) \
                     + partition_dict['solvent_label'] \
                     + '-' + str(temperature) + 'K-' \
                     + str(md_iteration)

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


    step_index = 0
    while step_index < partition_dict['coords'].shape[0]:

        if step_index == 0:
            engrad.nameJob = calc_name_base
            engrad.coordsString = partition_dict['coords'][step_index]  # Where we convert the coords
            output_file = 'out-' + engrad.nameJob + '.out'
            engrad.nameOutput = output_file[:-4]
            engrad.template_orca_submit_string = templates.orca_submit_template

            engrad.write_input()
            slurm_file = engrad.write_submit()
        else:
            engrad.nameJob = calc_name_base + '-step' + str(step_index)
            engrad.coordsString = partition_dict['coords'][step_index]  # Where we convert the coords

            engrad.template_orca_string = templates.orca_add_job \
                                          + templates.orca_input_template
            
            engrad.write_input()

            combine_command = calc_name_base + '-step' + str(step_index) \
                              + '.inp >> ' \
                              + calc_name_base + '.inp'
            
            filenames = ['file1.txt', 'file2.txt', ...]
            with open(calc_name_base + '.inp', 'a') as outfile:
                with open(engrad.nameJob + '.inp') as infile:
                    for line in infile:
                        outfile.write(line)
        
        step_index += 1
    
    print(os.getcwd())
    os.system('rm A-1MeOH-300K-1-step1.inp')
    
    bash_command = 'sbatch ' + slurm_file
    if submit:
        process = subprocess.Popen(
            bash_command.split(),
            stdout=subprocess.PIPE
        )
        output, error = process.communicate()
    

    return None

test_partition = partition.partition_trajectory('/home/alex/repos/MB-sGDML/tests/4MeOH-300K-1-md-trajectory.xyz')
partition_engrad(
    'orca', '/home/alex/repos/MB-sGDML/tests', test_partition['A'], 300, 1
)