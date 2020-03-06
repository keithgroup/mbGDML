import os
import cclib
from periodictable import elements

def parse_coords(fileName):
    
    try:
        data = cclib.io.ccread(fileName)

        atoms = data.atomnos
        coords = data.atomcoords

    except:
        print('Something happened while parsing xyz coordinates.')
    
    return {'atoms': atoms, 'coords': coords}

def string_coords(atoms, coords):
    """Puts atomic coordinates into a Python string. Typically used for 
    writing to an input file.
    
    Args:
        atoms (list): Contains elements as strings.
        coords (list): Contains atomic positions in the same order as the
        the atoms
    
    Returns:
        string: all atomic positions ready to be written to a file.
    """

    coord_string = ''

    atom_count = 0
    while atom_count < len(atoms):
        
        coord_string = coord_string + str(elements[atoms[atom_count]]) \
                        + ' ' + '  '.join([str(atom_coord) \
                            for atom_coord in coords[atom_count]]) \
                        + '\n'

        atom_count += 1
    
    return coord_string


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

    # Replaces 'atoms' with elemental symbols
    element_list = []
    for atom in stringfile_data['atoms']:
        element_list.append(str(elements[atom]))
    stringfile_data['atoms'] = element_list

    '''
    # Starts parsing the stringfile.
    with open(stringfile_path, 'r') as stringfile:

        structures = ['']
        line = stringfile.readline()
        structure_number = 0

        # Loops through each line in string file and adds coordinates to
        # structures.
        while line:

            # Splits line into list
            # e.g. ['C', '1.208995', '-0.056447', '2.319079\n']
            split_line = [item for item in line.split(' ') if item != '']

            # Length of split coordinate will have length of four
            if len(split_line) == 4:
                # First item should be either a character or string
                try:
                    int(split_line[0])
                except ValueError:
                    # Remainder of items should be floats.
                    try:
                        float(split_line[1])
                        float(split_line[2])
                        float(split_line[3][:-2])
                        structures[structure_number] = structures[structure_number] \
                                                        + line
                    except:
                        pass
            # If it is a line that doesn't have a coordinate we start a new
            # structure in the list.
            else:
                if len(structures) != 0 and structures[structure_number] != '':
                    structure_number += 1
                    structures.append('')

            line = stringfile.readline()  # Next line
    '''     
    return stringfile_data


def struct_dict(origin, struct_list):
    
    structure_coords = {}
    index_struct = 0

    while index_struct < len(struct_list):
        parsed_coords = parsing.parse_coords(struct_list[index_struct])
        coord_string = parsing.string_coords(parsed_coords['atoms'],
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