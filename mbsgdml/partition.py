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


def partition_trajectory(traj_path):
    """Partitions MD trajectory into separate trajectories for each possible
    segment.
    
    Args:
        traj_path (str): Path to trajectory with xyz coordinates.
    """

    # Parses trajectory.
    parsed_traj = parse.parse_stringfile(traj_path)

    # Gets length of trajectory.
    traj_steps = parsed_traj['coords'].shape[0]

    # Gets solvent information.
    solvent_info = solvents.identify_solvent(parsed_traj['atoms'])

    # Directory containing all partitions atoms and coords.
    traj_partition = {}  
    # Loops through each step in the MD trajectory.
    step_index = 0
    while step_index < traj_steps:
        cluster = parse.parse_cluster(
            {'atoms': parsed_traj['atoms'],
             'coords': parsed_traj['coords'][step_index]}
        )

        # Loops through all possible n-body partitions and adds the atoms once
        # and adds each step traj_partition.
        nbody_index = 1
        while nbody_index <= solvent_info['cluster_size']:
            partitions = partition_cluster(cluster, nbody_index)
            partition_labels = list(partitions.keys())
            
            # Tries to add the next trajectory step to 'coords'; if it fails it
            # initializes 'atoms' and 'coords' for that partition.
            for label in partition_labels:
                try:
                    traj_partition[label]['coords'] = np.append(
                        traj_partition[label]['coords'],
                        np.array([partitions[label]['coords']]),
                        axis=0
                    )
                except KeyError:
                    traj_partition[label] = {
                        'atoms': partitions[label]['atoms'],
                        'coords': np.array([partitions[label]['coords']])
                    }
            
            nbody_index += 1
        step_index += 1
    
    return traj_partition

                

# TODO format data for numbers larger than 10 (reduce spacing)

'''
prepare_training(
    '/home/alex/Dropbox/keith/projects/gdml/data/segment-calculations/4H2O-300K-1',
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-files/4H2O-300K-1',
    ['water'], '300', '1'
)
'''