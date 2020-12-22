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

import itertools
import numpy as np
from mbgdml import parse
from mbgdml import utils
from mbgdml import solvents

def partition_cluster(cluster, nbody):
    """All possible n-body combinations from a cluster.
    
    Parameters
    ----------
    cluster : :obj:`dict`
        Dictionary of solvent molecules in cluster from
        :func:`~mbgdml.parse.parse_cluster`.
    nbody : :obj:`int`
        Desired number of solvent molecules in combination.
    
    Returns
    -------
    :obj:`dict`
        All nonrepeating solvent molecule combinations from cluster with keys 
        being ``'#,#,#'`` where each ``#`` represents a molecule index. For
        example, ``'0,11,19'`` would be a cluster with molecules 0, 11, and 19
        from the super cluster. Each value is a dictionary containing

        ``'z'``
            A ``(n,)`` shape array of type :obj:`numpy.int32` containing atomic
            numbers of atoms in the structures in order as they appear.
        
        ``'R'``
            A :obj:`numpy.ndarray` with shape of ``(n, 3)`` where ``n`` is the
            number of atoms with three Cartesian components.

    Notes
    -----
    0,1,2 is considered the same as 2,1,0 and only included once.
    """
    segments = {}

    # Creates list of combinations of cluster dictionary keys.
    # comb_list is a list of tuples, 
    # e.g. [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)].
    comb_list = list(itertools.combinations(cluster, nbody))
    # Adds each solvent molecule coordinates in each combination
    # to a dictionary.
    for combination in comb_list:
        # Combination is tuple of solvent solvent molecules, e.g. (0, 1).
        
        for molecule in combination:
            # Molecule is the letter assigned to the solvent molecule, e.g. 'A'.

            # Tries to concatenate two molecules; if it fails it initializes
            # the variable.
            try:
                combined_atoms = np.concatenate(
                    (combined_atoms, cluster[molecule]['z'])
                )
                combined_coords = np.concatenate(
                    (combined_coords, cluster[molecule]['R'])
                )
            except UnboundLocalError:
                combined_atoms = cluster[molecule]['z']
                combined_coords = cluster[molecule]['R']

        # Adds the segment to the dict.
        comb_key = ','.join([str(i) for i in combination])
        segments[comb_key] = {
            'z': combined_atoms, 'R': combined_coords
        }

        # Clears segment variables for the next one.
        del combined_atoms, combined_coords

    return segments


def partition_stringfile(file_path, max_nbody=4):
    """Partitions an XYZ file.

    Takes a cluster of molecules and separates into individual partitions. A
    partition being a monomer, dimer, trimer, etc. from the original cluster.
    For example, a structure of three solvent molecules will provide three
    individual partitions of monomers, three dimers, and one trimer.
    
    Parameters
    ----------
    traj_path : :obj:`str`
        Path to trajectory with xyz coordinates.
    max_nbody: :obj:`int`, optional
        Highest order of n-body structure to include.
    
    Returns
    -------
    :obj:`dict`
        All nonrepeating molecule combinations from original cluster with keys 
        being uppercase concatenations of molecule labels and values being the 
        string of coordinates.
    
    Raises
    ------
    ValueError
        If any structure has a different order of atoms.
    """

    # Parses trajectory.
    z_all, _, R_list = parse.parse_stringfile(file_path)
    try:
        assert len(set(tuple(i) for i in z_all)) == 1
    except AssertionError:
        raise ValueError(f'{file_path} contains atoms in different order.')
    z_elements = z_all[0]
    R = np.array(R_list)
    # Gets system information
    z = np.array(utils.atoms_by_number(z_elements))
    sys_info = solvents.system_info(z.tolist())

    # Gets 
    all_partitions = {}  
    for coords in R:
        cluster = parse.parse_cluster(z, coords)
        # Loops through all possible n-body partitions and adds the atoms once
        # and adds each step traj_partition.
        i_nbody = 1
        while i_nbody <= sys_info['cluster_size'] and i_nbody <= max_nbody:
            partitions = partition_cluster(cluster, i_nbody)
            partition_labels = list(partitions.keys())
            # Tries to add the next trajectory step to 'coords'; if it fails it
            # initializes 'atoms' and 'coords' for that partition.
            for label in partition_labels:
                partition_info = solvents.system_info(
                    partitions[label]['z'].tolist()
                )

                try:
                    all_partitions[label]['R'] = np.append(
                        all_partitions[label]['R'],
                        np.array([partitions[label]['R']]),
                        axis=0
                    )
                except KeyError:
                    all_partitions[label] = {
                        'solvent_label': partition_info['solvent_label'],
                        'cluster_size': sys_info['cluster_size'],
                        'partition_label': label,
                        'partition_size': partition_info['cluster_size'],
                        'z': partitions[label]['z'],
                        'R': np.array([partitions[label]['R']])
                    }
            i_nbody += 1

    return all_partitions
