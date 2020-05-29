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
import os

import numpy as np
from cclib.io import ccread
from cclib.parser.utils import convertor
from periodictable import elements
from mbgdml import utils
from mbgdml import parse
from mbgdml import solvents

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


def partition_trajectory(traj_path):
    """Partitions MD trajectory into separate trajectories for each possible
    segment.
    
    Args:
        traj_path (str): Path to trajectory with xyz coordinates.
    
    Returns:
        dict: All nonrepeating solvent molecule combinations from cluster with
              keys being uppercase concatenations of molecule labels and
              values being the string of coordinates.
    """

    # Parses trajectory.
    parsed_traj = parse.parse_coords(traj_path)

    # Gets length of trajectory.
    traj_steps = parsed_traj['coords'].shape[0]

    # Gets solvent information.
    solvent_info = solvents.system_info(parsed_traj['atoms'])

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
                partition_info = solvents.system_info(
                    partitions[label]['atoms']
                )

                try:
                    traj_partition[label]['coords'] = np.append(
                        traj_partition[label]['coords'],
                        np.array([partitions[label]['coords']]),
                        axis=0
                    )
                except KeyError:
                    traj_partition[label] = {
                        'solvent_label': partition_info['solvent_label'],
                        'cluster_size': solvent_info['cluster_size'],
                        'partition_label': label,
                        'partition_size': partition_info['cluster_size'],
                        'atoms': partitions[label]['atoms'],
                        'coords': np.array([partitions[label]['coords']])
                    }
            
            nbody_index += 1
        step_index += 1
    
    return traj_partition
