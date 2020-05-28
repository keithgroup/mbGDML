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

import os
import numpy as np
import cclib
from periodictable import elements
from mbgdml import solvents, utils

def parse_coords(fileName):
    
    try:
        data = cclib.io.ccread(fileName)

        atoms = data.atomnos
        coords = data.atomcoords

    except:
        print('Something happened while parsing xyz coordinates.')
    
    return {'atoms': atoms, 'coords': coords}

def parse_gdml_data(out_file):
    """Parses GDML-relevant data from partition output file.
    
    Args:
        out_file (str): path to computational chemistry output file. This should
            contain all MD steps for the partition.
    
    Returns:
        dict: contains all information needed to build GDML data sets.
            'atoms' is a (n) numpy array containing the atomic numbers of the
                atoms in the partition.
            'coords' is a (m, n, 3) numpy array containing the m MD step
                coordinates of the n atoms in the partition.
            'grads' is a (m, n, 3) numpy array containing the gradients of the
                m MD steps containing the n atoms in the partition.
            'energies' is a (m, 1) numpy array containing the electronic 
                energies of each m MD steps for the partition.
    """

    try:
        data = cclib.io.ccread(out_file)

        atoms = data.atomnos
        coords = data.atomcoords
        grads = data.grads
        if hasattr(data, 'mpenergies'):
            energies = data.mpenergies
        elif hasattr(data, 'scfenergies'):
            energies = data.scfenergies
        else:
            raise KeyError
    except:
        print('Something happened while parsing output file.')
        return None

    parsed_data = {
        'atoms': atoms,
        'coords': coords,
        'grads': grads,
        'energies': energies
    }

    return parsed_data

def cluster_size(xyz_path, solvent):
    """Determines number of solvent molecules in a xyz file.
    
    Args:
        xyz_path (str): Path to xyz file of interest.
        solvent (lst): Specifies solvents to determine the number of atoms included in a molecule.
    
    Returns:
        int: Number of solvent molecules in specified xyz file.
    """

    with open(xyz_path, 'r') as xyz_file:

        line = xyz_file.readline()

        while line:
           
            split_line = line.split(' ')
            
            # Grabs number of atoms in xyz file.
            if len(split_line) == 1 and split_line[0] != '\n':

                atom_num = int(split_line[0])

                if solvent[0] == 'water':
                    molecule_num = atom_num / 3
                    return molecule_num
                elif solvent[0] == 'acetonitrile' or solvent[0] == 'acn':
                    molecule_num = atom_num / 6
                    return molecule_num

            line = xyz_file.readline()

def parse_stringfile(stringfile_path):
    """Parses xyz stringfile into list containing xyz coordinates of each
    structure in order. A stringfile is a text file with a list of xyz
    structures. This also could be a trajectory.
    
    Args:
        stringfile_path (str):  Specifies path to stringfile.
    
    Returns:
        dict:   Contains 'atoms' which is a list of atoms identified by their
                element symbol and 'coords' which is a numpy array with shape
                (n, m, 3) where n is the number of structures and m is the
                number of atoms in each structure.
    """
    
    stringfile_data = parse_coords(stringfile_path)
   
    return stringfile_data


def parse_cluster(cluster_data):
    """Creates dictionary of all solvent molecules in solvent cluster from a
    cluster_data dictionary.

    Notes:
        Molecules are specified by specified by uppercase letters, and their
        values the xyz coordinates.
    
    Args:
        cluster_data (dict): Contains 'atoms' that is a list of elements
    organized by molecule and matches the order of the numpy array
    containing atomic coordinates
    
    Returns:
        dict: Contains solvent molecules with keys of uppercase characters
    with dicts as values containing 'atoms' and 'coords'.
    
    {
        'A':    {
                 'atoms':   array([8, 1, 6, 1, 1, 1], dtype=int32),
                 'coords':  array([[ 1.45505901, -0.880818  ,  0.851331  ],
                                   [ 0.505216  , -1.15252401,  0.844885  ],
                                   [ 2.22036801, -1.94217701,  0.2977    ],
                                   [ 3.25924901, -1.61877101,  0.264909  ],
                                   [ 1.89693001, -2.18833701, -0.717112  ],
                                   [ 2.15446301, -2.84056201,  0.915723  ]])
                },
        'B':    {
                 'atoms':   array([8, 1, 6, 1, 1, 1], dtype=int32),
                 'coords':  array([[ 1.14600801,  1.44243601, -0.473102  ],
                                  [ 1.38561801,  0.614295  ,  0.00943   ],
                                  [ 1.47489701,  2.54700101,  0.357836  ],
                                  [ 1.16626501,  3.45362102, -0.159587  ],
                                  [ 2.55100901,  2.59826701,  0.538796  ],
                                  [ 0.957941  ,  2.49831901,  1.31983201]])
                }
    }
    """

    # Identifies the solvent and size of a cluster.
    solvent_info = solvents.solvent(list(cluster_data['atoms']))

    # Partitions solvent cluster into individual solvent molecules.
    cluster_molecules = {}
    molecule_index = 1
    while molecule_index <= solvent_info.cluster_size:
        # Grabs index positions of atomic coordinates for the solvent molecule
        atom_start = molecule_index * solvent_info.solvent_molec_size \
                     - solvent_info.solvent_molec_size
        atom_end = molecule_index * solvent_info.solvent_molec_size
        
        # Creates molecule label and grabs atoms and atomic coordinates for
        # the molecule.
        molecule_label = chr(ord('@')+molecule_index)
        molecule_atoms = cluster_data['atoms'][atom_start:atom_end]
        molecule_coords = cluster_data['coords'][atom_start:atom_end,:]

        cluster_molecules[molecule_label] = {'atoms': molecule_atoms,
                                             'coords': molecule_coords}

        molecule_index += 1

    return cluster_molecules


def struct_dict(origin, struct_list):
    
    structure_coords = {}
    index_struct = 0

    while index_struct < len(struct_list):
        parsed_coords = parse_coords(struct_list[index_struct])
        coord_string = utils.string_coords(parsed_coords['atoms'],
                                            parsed_coords['coords'])
        
        # Naming scheme for ABCluster minima
        if origin.lower() == 'abcluster':
            structure_coords[str(index_struct)] = coord_string
        # Naming scheme for GDML produced segments
        elif origin.lower() == 'gdml':
            split_structure_path = struct_list[index_struct].split('/')
            structure_name = split_structure_path[-1][:-4]
            structure_coords[structure_name] = coord_string
        else:
            structure_coords[str(index_struct)] = coord_string

        index_struct += 1

    return structure_coords