# MIT License
# 
# Copyright (c) 2020-2022, Alex M. Maldonado
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

import cclib
from . import utils

def parse_coords(fileName):
    
    try:
        data = cclib.io.ccread(fileName)

        atoms = data.atomnos
        coords = data.atomcoords

    except BaseException:
        print('Something happened while parsing xyz coordinates.')
    
    return {'atoms': atoms, 'coords': coords}

def parse_engrad(out_file):
    """Parses GDML-relevant data (coordinates, energies, and gradients)
    from partition output file.

    Uses ``cclib`` to parse data from computational chemistry calculations
    involving multiple calculations of structures containing same atoms in 
    different configurations.
    
    Parameters
    ----------
    out_file : :obj:`str`
        Path to computational chemistry output file. This should contain all MD 
        steps for the partition.
    
    Returns
    -------
    :obj:`dict`
        All information needed to build GDML data sets. Contains the following
        keys:

            ``'z'``
                ``(n,)`` :obj:`numpy.ndarray` of atomic numbers.
            ``'R'``
                ``(m, n, 3)`` :obj:`numpy.ndarray` containing the coordinates of
                ``m`` calculations of the ``n`` atoms in the structure.
            ``'E'``
                ``(m, 1)`` :obj:`numpy.ndarray` containing the energies of
                ``m`` calculations.
            ``'G'``
                ``(m, n, 3)`` :obj:`numpy.ndarray` containing the gradients of
                ``m`` calculations of the ``n`` atoms in the structure.
    """
    try:
        data = cclib.io.ccread(out_file)
        atoms = data.atomnos
        coords = data.atomcoords
        grads = data.grads
        if hasattr(data, 'mpenergies'):
            energies = data.mpenergies[:,0]
        elif hasattr(data, 'scfenergies'):
            energies = data.scfenergies[:,0]
        else:
            raise KeyError('cclib energies were not found.')
        parsed_data = {'z': atoms, 'R': coords, 'E': energies, 'G': grads}
        return parsed_data
    except BaseException:
        print('Something happened while parsing output file.')
        raise 

def cluster_size(xyz_path, solvent):
    """Determines number of solvent molecules in a xyz file.
    
    Parameters
    ----------
    xyz_path : :obj:`str`
        Path to xyz file of interest.
    solvent : :obj:`list`
        Specifies solvents to determine the number of atoms included in a 
        molecule.
    
    Returns
    -------
    int
        Number of solvent molecules in specified xyz file.
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
    """Parses data from string file.

    A string file is data presented as consecutive xyz data. The data could be
    three Cartesian coordinates for each atom, three atomic force vector
    components, or both coordinates and atomic forces in one line (referred to
    as extended xyz).
    
    Parameters
    ----------
    stringfile_path : :obj:`str`
        Path to string file.
    
    Returns
    -------
    :obj:`tuple` [:obj:`list`]
        Parsed atoms (as element symbols :obj:`str`), comments, and data as
        :obj:`float` from string file.
    """
    z, comments, data = [], [], []
    with open(stringfile_path, 'r') as f:
        for _, line in enumerate(f):
            line = line.strip()
            if not line:
                # Skips blank lines
                pass
            else:
                line_split = line.split()
                if len(line_split) == 1 \
                    and float(line_split[0]) % int(line_split[0]) == 0.0:
                    # Skips number of atoms line, adds comment line, and
                    # prepares next z and data item.
                    comment_line = next(f)
                    comments.append(comment_line.strip())
                    z.append([])
                    data.append([])
                else:
                    # Grabs z and data information.
                    z[-1].append(line_split[0])
                    data[-1].append([float(i) for i in line_split[1:]])
    return z, comments, data

def struct_dict(origin, struct_list):
    
    structure_coords = {}
    index_struct = 0

    while index_struct < len(struct_list):
        parsed_coords = parse_coords(struct_list[index_struct])
        coord_string = utils.string_xyz_arrays(parsed_coords['atoms'],
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