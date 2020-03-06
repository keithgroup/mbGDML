import itertools
import os

import numpy as np
from cclib.io import ccread
from cclib.parser.utils import convertor
from periodictable import elements
from mbsgdml import utils, parse, solvents
                
# TODO remove this function after switching to numpy arrays instead of strings
def split_trajectory(trajectory_path, processing_path, solvent, temperature,
                     iteration):
    """Separates each MD step in a trajectory as its own xyz file in a
    specified location.

    Naming scheme is: numbersolventmolecule-tempt-iteration-traj-struct; e.g.
    4H2O-100K-1-traj-struct.
    
    Args:
        trajectory_path (str): Specifies the MD trajectory to separate into
        steps.
        processing_path (str): Specifies folder where processed xyz file folder
        will be saved.
        solvent (lst): Specifies solvents to determine the number of atoms
        included in a molecule and labeling.
        temperature (int): Provided information of MD thermostat set
        temperature; used for labeling.
        iteration (int): Provided information of MD trajectory number;
        used for labeling.
    
    Returns:
        str: Path to step xyz files.
    """

    # Determines labels for file names and directories.
    
    if solvent[0] == 'water':
        solvent_label = 'H2O'
    elif solvent[0] == 'methanol':
        solvent_label = 'MeOH'
    elif solvent [0] == 'acetonitrile':
        solvent_label = 'acn'
    
    num_molecules = parse.cluster_size(trajectory_path, solvent)
    # Creates cluster naming basis, e.g. '4H2O-300K-1'.
    cluster_base_name = str(int(num_molecules)) + solvent_label \
                         + '-' + str(temperature) + 'K-' + str(iteration)

    # Normalizes processing_path, assigns folder name, creates folder.
    # Will append an integer if there is already a folder with the same name
    # present.
    processing_folder = utils.norm_path(processing_path) + cluster_base_name \
                      + '-traj-struct'
    processing_folder = utils.make_folder(processing_folder)
    

    # Starts parsing the trajectory.
    with open(trajectory_path, 'r') as traj:

        structure_file = []
        md_step = '0,'
        line = traj.readline()

        # Loops through each line in trajectory file.
        while line:
            
            split_line = line.split(' ')
            
            # Writes the xyz file for the step and resets variables.
            # Occurs when the current line is the number of atoms.
            # Makes sure to not write file if there is a blank line.
            # Makes sure to not write file in the first line.
            if len(split_line) == 1 and split_line[0] != '\n' \
               and len(structure_file) != 0:

                with open(processing_folder + '/' + cluster_base_name \
                          + '-step' + md_step[:-1] + '.xyz', 'w') as f:
                    f.write(''.join(structure_file))
                structure_file = [line]

            # Reformats the comment line to only include the potential energy
            # in kcal/mol.
            elif split_line[0] == '#':

                energy = float(split_line[8][6:]) * 627.509474 
                md_step = split_line[5]
                structure_file.append('# ' + str(energy) + '\n')

            # Skips new lines in trajectory.
            elif split_line[0] == '\n':
                pass

            # Any line containing atomic coordinates gets added to xyz file.
            else:
                structure_file.append(line)

            line = traj.readline()
        
        # Writes last trajectory step.
        # Previous while loop does not trigger write conditional statement for
        # the last step.
        with open(processing_folder + '/' + cluster_base_name \
                  + '-step' + md_step[:-1] + '.xyz', 'w') as f:
            
            f.write(''.join(structure_file))
    
    return processing_folder


def partition_cluster(cluster, nbody):
    """Creates dictionary with all possible n-body combinations of solvent
    cluster.
    
    Args:
        cluster (dict): Dictionary of solvent molecules in cluster from
                        parse_cluster
        nbody (int):    Desired number of solvent molecules in combination.
    
    Returns:
        dict:   All nonrepeating solvent molecule combinations from cluster with
                keys being uppercase concatenations of molecule labels and
                values being the string of coordinates.

        ABC is considered the same as BAC and only included once.
    """

    segments = {}

    # Creates list of combinations of cluster dictionary keys
    # comb_list is a list of tuples,
    # e.g. [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')].
    comb_list = list(itertools.combinations(cluster, nbody))
    
    # Adds each solvent molecule coordinates in each combination
    # to a dictionary.
    for combination in comb_list:
        # Combination is tuple of solvent solvent molecules, e.g. ('B', 'C').
        
        for molecule in combination:
            # Molecule is the letter assigned to the solvent molecule, e.g. 'A'.

            # Tries to concatenate two molecules; if it fails it initializes
            # the variable.
            try:
                combined_atoms = np.concatenate((combined_atoms,
                                                 cluster[molecule]['atoms']))
                combined_coords = np.concatenate((combined_coords,
                                                 cluster[molecule]['coords']))
            except UnboundLocalError:
                combined_atoms = cluster[molecule]['atoms']
                combined_coords = cluster[molecule]['coords']

        # Adds the segment to the dict.
        segments[''.join(combination)] = {'atoms': combined_atoms,
                                          'coords': combined_coords}

        # Clears segment variables for the next one.
        del combined_atoms, combined_coords

    return segments

test_cluster = parse.parse_stringfile('/home/alex/repos/MB-sGDML/tests/4MeOH-300K-1-md-trajectory.xyz')
test_parse = parse.parse_cluster({'atoms': test_cluster['atoms'], 'coords': test_cluster['coords'][0]})
test_partition = partition_cluster(test_parse, 2)
print(test_partition)


def separate_cluster(xyzFile, proc_path, solvent, temperature, iteration, step):
    """Creates folder in specified directory that includes all 1, 2, 3, ...
    solvent molecule combinations.

    This is to be used on a single MD snapshot.
    
    Args:
        xyzFile (str): Path to xyz file of a single MD snapshot.
        proc_path (str): Directory to save a folder containing all solvent combinations.
        solvent (list): Specifies solvents to determine the number of atoms included in a molecule and labeling.
        temperature (int): Provided information of MD thermostate set temperature; used for labeling.
        iteration (int): Provided information of MD trajectory number; used for labeling.
        step (int): Provided information of MD step for consistent labeling and tracking.
    """

    # Specifies number of atoms in each molecule and solvent label.
    if solvent[0] == 'water':
        solvent_label = 'H2O'
        numAtoms = 3
    
    # Gets number of molecules in solvent cluster.
    labelNumMolecules = parse.numberMolecules(xyzFile, solvent)

    # Creates trajectory naming basis, e.g. '4H2O-300K-1'.
    cluster_base_name = str(int(labelNumMolecules)) + solvent_label \
                         + '-' + str(temperature) + 'K-' + str(iteration)

    # Creates folder to store each molecules combination xyz files.
    if proc_path[-1:] == '/':
        nameFolderAttempt = proc_path + cluster_base_name + '-segments/'
    else:
        nameFolderAttempt = proc_path + '/' + cluster_base_name + '-segments/'

    # Creates the folder if not present.
    try:
        os.mkdir(nameFolderAttempt)
    except:
        pass
    
    # Refreshes folder name and creates base path for a steps molecule combination.
    nameFolder = nameFolderAttempt
    nameFolderSegment = utils.make_folder(nameFolder + 'step' + str(step))

    # Gets all solvent molecules from the xyz file into a dictionary.
    parsedCluster = parse.parse_cluster(xyzFile)

    # Creates xyz files of all possible combinations involving 1, 2, 3, ... solvent molecules.
    segmentSize = 1
    while segmentSize <= labelNumMolecules:

        # Gets all combinations of current combination size.
        segmentedCluster = partition_cluster(parsedCluster, segmentSize)

        # Writes all combinations of a specific size to individual xyz files.
        for segment, coordinates in segmentedCluster.items():
            with open(nameFolderSegment + str(segment) \
                      + '-' + cluster_base_name \
                      + '-step' + str(step) + '.xyz', 'w') as f:
                f.write(str(len(segment) * numAtoms) + '\n') 
                f.write('#\n')
                f.write(coordinates)

        segmentSize += 1


def process_trajectory(traj_path, proc_path, solvent, temperature, iteration):
    """Processes MD trajectory into individual xyz files for
    each snapshot and further segments each snapshot into all possible n-body combinations.
    
    Args:
        traj_path (str): Path to MD trajectory.
        proc_path (str): Directory to save a folder containing all solvent combinations.
        solvent (list): Specifies solvents to determine the number of atoms included in a molecule and labeling.
        temperature (int): Provided information of MD thermostate set temperature; used for labeling.
        iteration (int): Provided information of MD trajectory number; used for labeling.
    """

    # Separates trajectory into individual xyz files.
    trajectoryStepsPath = split_trajectory(
            traj_path, proc_path, solvent, temperature, iteration
    )

    # Gets list of all individual xyz files (not in order),
    # e.g. ['4H2O-300K-1-step56.xyz', '4H2O-300K-1-step40.xyz', '4H2O-300K-1-step71.xyz']
    trajectorySteps = [f for f in os.listdir(trajectoryStepsPath) \
                       if os.path.isfile(os.path.join(trajectoryStepsPath, f))]

    # Creates folder of all combinations for each step in MD trajectory.
    indexStep = 0
    while indexStep < len(trajectorySteps):
        # Determines xyz file string of step indexStep (increasing order).
        fileStepString = [step for step in trajectorySteps \
                          if ('step' + str(indexStep) + '.xyz') in step][0]
        # Concatenates path of step xyz file.
        stepPath = trajectoryStepsPath + \
                   trajectorySteps[trajectorySteps.index(fileStepString)]
        # Creates the folder with all combinations.
        separate_cluster(
            stepPath, proc_path, solvent, temperature, iteration, indexStep
        )

        indexStep += 1
    
    
'''
process_trajectory(
   '/home/alex/Dropbox/keith/projects/gdml/data/md/4H2O-md/4H2O-100K-3-md/4H2O-100K-3-md-trajectory.xyz',
   '/home/alex/Dropbox/keith/projects/gdml/data/segments/', ['water'], 100, 3
)
'''


def convert_gradients(gradients, number_atoms):
    # Eh / bohr to kcal / (angstrom mol)
    atom_index = 0
    while atom_index < number_atoms:
        coord_index = 0
        while coord_index < 3:
            gradients[atom_index][coord_index] = gradients[atom_index][coord_index] \
                * ( 627.50947414 / 0.5291772109)

            coord_index += 1
        atom_index += 1

    return gradients

def prepare_training(
    segment_calc_folder, training_folder, solvent, temperature, iteration
):
    
    if training_folder[-1] != '/':
        training_folder = training_folder + '/'

    all_out_files = utils.get_files(segment_calc_folder, 'out')

    # Organizes segment type into dictionary.
    out_files_dict = {}
    while len(all_out_files) > 0:
        # Start with last file.
        ref_file = all_out_files[0]

        # Grabs molecule label.
        label_ref = ref_file.split('/')[-1].split('-')[1]
        # Adding all output files matching reference label into list.
        collected_out_files = [file for file in all_out_files if \
                               ('-' + label_ref + '-') in file]
        out_files_dict[label_ref] = collected_out_files


        # Removes all occurrences.
        all_out_files = [file for file in all_out_files if \
                         ('-' + label_ref + '-') not in file]
    
    # Specifies number of atoms in each molecule and solvent label.
    if solvent[0] == 'water':
        solvent_label = 'H2O'
        num_atoms = 3
    
    # Determines max number of solvent number for labeling.
    segment_list = list(out_files_dict.keys())
    largest_segment = max(segment_list, key=len)
    max_cluster_data = ccread(out_files_dict[str(largest_segment)][0])
    max_num_molecules = int(len(max_cluster_data.atomnos) / num_atoms)

    for segment in out_files_dict:

        # Naming segment file
        gdml_file_name = training_folder \
                                 + str(segment) + '-' \
                                 + str(max_num_molecules) + solvent_label \
                                 + '-' + str(temperature) + 'K-' \
                                 + str(iteration) + '-gdml.txt'

        # Loops through each MD step for segment and adds to gdml file.
        step_index = 0
        while step_index < len(out_files_dict[segment]):
            segment_step = [file for file in out_files_dict[segment] if \
                           ('-step' + str(step_index) + '-') in file]
            
            if len(segment_step) == 1:
                segment_data = ccread(segment_step[0])
                atoms = list(segment_data.atomnos)
                num_atoms = len(atoms)
                coords = segment_data.atomcoords[-1]
                energy = convertor(segment_data.mpenergies[0][-1], 'eV', 'kcal/mol')
                forces = convert_gradients(segment_data.grads[0], num_atoms)
                

                # Combining atoms, coordinates, and forces.
                gdml_data = []
                atom_index = 0
                while atom_index < len(atoms):
                    atom_string = '  ' + str(elements[atoms[atom_index]])
                    coord_string = array2string(
                                       coords[atom_index],
                                       suppress_small=True, separator='   ',
                                       formatter={'float_kind':'{:0.6f}'.format}
                                   )[1:-1].replace(' -', '-') + '    '
                    force_string = array2string(
                                       forces[atom_index],
                                       suppress_small=True, separator='   ',
                                       formatter={'float_kind':'{:0.8f}'.format}
                                   )[1:-1].replace(' -', '-') + '\n'
                    
                    # Adds proper spacing between atom and first coordinate.
                    element_space = len(atom_string) - 1
                    if coords[atom_index][0] < 0:
                        neg_space = 1
                    else:
                        neg_space = 0
                    num_spaces = 10 - element_space - neg_space
                    spaces = '{:<' + str(num_spaces) + '}'
                    atom_string = spaces.format(atom_string)

                    # Adds proper spacing between coordinates and forces.
                    # Negative numbers.
                    if forces[atom_index][0] > 0:
                        coord_string = coord_string + ' '
                    # Greater than 10.
                    if abs(forces[atom_index][0]) >= 10:
                        coord_string = coord_string[:-1]
                    

                    gdml_data.append(atom_string + coord_string + force_string)
                    
                    atom_index += 1
                            
                lines = [str(num_atoms) + '\n', '# ' + str(energy) + '\n'] + gdml_data

                # Writing or appending gdml file (will overwrite).
                if step_index == 0:
                    file_mode = 'w'
                else:
                    file_mode = 'a'
                
                with open(gdml_file_name, file_mode) as gdml_file:
                    gdml_file.writelines(lines)
                

            else:
                print('There is not a unique output file for MD step '\
                       + str(step_index) + ' for segment ' + str(segment))
                return None
            
            step_index += 1
        
    
    return None
                

# TODO format data for numbers larger than 10 (reduce spacing)

'''
prepare_training(
    '/home/alex/Dropbox/keith/projects/gdml/data/segment-calculations/4H2O-300K-1',
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-files/4H2O-300K-1',
    ['water'], '300', '1'
)
'''