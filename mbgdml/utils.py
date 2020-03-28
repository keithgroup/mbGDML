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
from natsort import natsorted, ns

def norm_path(path):
    """Normalizes directory paths to be consistent.
    
    Args:
        path (string): Path to a directory.
    
    Returns:
        normd_path: Normalized path.
    """

    normd_path = path  # Initializes path variable.

    # Makes sure path ends with forward slash.
    if normd_path[-1] != '/':
        normd_path = path + '/'
    
    return normd_path

def get_files(path, expression):
    """Returns paths to all files in a given directory that matches a provided
    expression in the file name. Commonly used to find all files of a certain
    type, e.g. output or xyz files.
    
    Args:
        path (str): Specifies the directory to search.
        expression (str): Expression to be tested against all file names in
        'path'.
    
    Returns:
        list: all absolute paths to files matching the provided expression.
    """
    path = norm_path(path)
    
    all_files = []
    for (dirpath, _, filenames) in os.walk(path):
        index = 0
        while index < len(filenames):
            filenames[index] = norm_path(dirpath) + filenames[index]
            index += 1

        all_files.extend(filenames)
        
    files = []
    for file in all_files:
        if expression in file:
            files.append(file)

    return files

def make_folder(folder):
    """Creates folder at specified path.
    If the current folder exists, it creates another folder
    with an added number.
    
    Args:
        pathFolder (str): Path to desired folder.
    
    Returns:
        str: Final path of new folder; ends in '/'.
    """
    
    # First tries to create the desired directory.
    try:

        os.mkdir(folder)
        return folder + '/'

    # If there is already a directory with the same name,
    # append a positive integer until there is no previously existing directory.
    except FileExistsError:

        indexDir = 1
        dirExists = True
        while dirExists:
            try:
                pathFolderIteration = folder + '-' + str(indexDir)
                os.mkdir(pathFolderIteration)

                dirExists = False

                return pathFolderIteration + '/'

            # Increments number by 1 until it finds the lowest number.
            except FileExistsError:
                indexDir += 1


def natsort_list(unsorted_list):
    """Basic function that organizes a list based on human (or natural) sorting
    methodology.
    
    Args:
        unsorted_list (list): List of strings.
    
    Returns:
        list: Sorted list of string.
    """
    sorted_list = natsorted(unsorted_list, alg=ns.IGNORECASE)

    return sorted_list

def string_coords(atoms, coords):
    """Puts atomic coordinates into a Python string. Typically used for 
    writing to an input file.
    
    Args:
        atoms (np.array): A (n, 1) numpy array containing all n elements labled
            by their atomic number.
        coords (np.array): Contains atomic positions in a (n, 3) numpy array
            where the x, y, and z Cartesian coordinates in Angstroms are given
            for the n atoms.
    
    Returns:
        str: all atomic coordinates contained in a string.
    """

    atom_coords_string = ''

    atom_index = 0
    while atom_index < len(atoms):
        atom_element = str(elements[atoms[atom_index]])
        coords_string = np.array2string(
            coords[atom_index],
            suppress_small=True, separator='   ',
            formatter={'float_kind':'{:0.9f}'.format}
        )[1:-1] + '\n'
        
        atom_coords_string += (atom_element + '   ' \
                               + coords_string).replace(' -', '-')

        atom_index += 1
    
    return atom_coords_string

def convert_gradients(gradients, number_atoms):
    # Eh/bohr to kcal/(angstrom mol)
    atom_index = 0
    while atom_index < number_atoms:
        coord_index = 0
        while coord_index < 3:
            gradients[atom_index][coord_index] = gradients[atom_index][coord_index] \
                * ( 627.50947414 / 0.5291772109)

            coord_index += 1
        atom_index += 1

    return gradients

def atoms_by_element(atom_list):
    """Converts a list of atoms identified by their atomic number to their
    elemental symbol.
    
    Args:
        atom_list (list): Contains numbers that represent atomic numbers.
    
    Returns:
        list: Contains strings of elemental symbols matching atom_list.
    """

    atom_list_elements = []
    for atom in atom_list:
        atom_list_elements.append(str(elements[atom]))

    return atom_list_elements