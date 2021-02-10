# MIT License
# 
# Copyright (c) 2020-2021, Alex M. Maldonado
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


import os
import numpy as np
import cclib
from natsort import natsorted, ns
import hashlib

element_to_z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
    'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23,
    'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31,'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
    'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44,
    'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51,
    'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
    'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72,
    'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79,
    'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
    'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93,
    'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106,
    'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
    'Uuq': 114, 'Uuh': 116,
}
z_to_element = {v: k for k, v in element_to_z.items()}

def norm_path(path):
    """Normalizes directory paths to be consistent.
    
    Parameters
    ----------
    path : str
        Path to a directory.
    
    Returns
    -------
    str
        Normalized path.
    """

    normd_path = path  # Initializes path variable.

    # Makes sure path ends with forward slash.
    if normd_path[-1] != '/':
        normd_path = path + '/'
    
    return normd_path


def get_files(path, expression, recursive=True):
    """Returns paths to all files in a given directory that matches a provided
    expression in the file name. Commonly used to find all files of a certain
    type, e.g. output or xyz files.
    
    Parameters
    ----------
    path : :obj:`str`
        Specifies the directory to search.
    expression : :obj:`str`
        Expression to be tested against all file names in 'path'.
    recursive :obj:`bool`, optional
        Recursively find all files in all subdirectories.
    
    Returns
    -------
    :obj:`list` [:obj:`str`]
        All absolute paths to files matching the provided expression.
    """
    if path[-1] != '/':
        path += '/'
    if recursive:
        all_files = []
        for (dirpath, _, filenames) in os.walk(path):
            index = 0
            while index < len(filenames):
                if dirpath[-1] != '/':
                    dirpath += '/'
                filenames[index] = dirpath + filenames[index]
                index += 1
            all_files.extend(filenames)
        files = []
        for f in all_files:
            if expression in f:
                files.append(f)
    else:
        files = []
        for f in os.listdir(path):
            filename = os.path.basename(f)
            if expression in filename:
                files.append(path + f)
    return files

def natsort_list(unsorted_list):
    """Basic function that organizes a list based on human (or natural) sorting
    methodology.
    
    Parameters
    -----------
    unsorted_list : list
        List of strings.
    
    Returns
    -------
    list
        Sorted list of string.
    """
    sorted_list = natsorted(unsorted_list, alg=ns.IGNORECASE)

    return sorted_list

def string_coords(z, R):
    """Puts atomic coordinates into a Python string. Typically used for 
    writing to an input file.
    
    Parameters
    atoms : :obj:`numpy.ndarray`
        A (n,) numpy array containing all ``n`` elements labled by their atomic 
        number.
    coords : :obj:`numpy.array`
        Contains atomic positions in a (n, 3) numpy array where the x, y, and z 
        Cartesian coordinates in Angstroms are given for the n atoms.
    
    Returns
    -------
    :obj:`str`
        XYZ atomic coordinates as a string.
    """
    atom_coords_string = ''
    atom_index = 0
    while atom_index < len(z):
        atom_element = str(z_to_element[z[atom_index]])
        coords_string = np.array2string(
            R[atom_index],
            suppress_small=True, separator='   ',
            formatter={'float_kind':'{:0.9f}'.format}
        )[1:-1] + '\n'
        atom_coords_string += (atom_element + '   ' \
                               + coords_string).replace(' -', '-')
        atom_index += 1
    
    return atom_coords_string

def write_xyz(z, R, save_dir, file_name):
    """Write XYZ file given atomic numbers and Cartesian coordinates.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        A ``(n,)`` shape array of type :obj:`numpy.int32` containing atomic
        numbers of atoms in the structures in order as they appear.
    R : :obj:`numpy.ndarray`
        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of structures and ``n`` is the number of atoms with three 
        Cartesian components.
    save_dir : :obj:`str`
        Path to directory to save XYZ file.
    file_name : :obj:`str`
        Name to save the file.
    """
    if R.ndim == 2:
        R = np.array([R])
    
    if save_dir[-1] != '/':
        save_dir += '/'
    
    num_atoms = len(z)
    with open(f'{save_dir}{file_name}.xyz', 'w') as f:
        for structure in R:
            f.writelines(
                [f'{num_atoms}\n', '\n', string_coords(z, structure)]
            )

def convert_forces(
    forces, e_units_calc, r_units_calc, e_units, r_units
):
    """Converts forces (or gradients) to specified units.

    Parameters
    ----------
    forces : :obj:`numpy.ndarray`
        An array with units of energy and distance matching `e_units_calc`
        and `r_units_calc`.
    e_units_calc : :obj:`str`
        Specifies package-specific energy units used in calculation. Available
        units are ``'eV'``, ``'hartree'``, ``'kcal/mol'``, and ``'kJ/mol'``.
    r_units_calc : :obj:`str`
        Specifies package-specific distance units used in calculation. Available
        units are ``'Angstrom'`` and ``'bohr'``.
    e_units : :obj:`str`
        Desired units of energy. Available units are ``'eV'``, ``'hartree'``,
        ``'kcal/mol'``, and ``'kJ/mol'``.
    r_units : obj:`str`
        Desired units of distance. Available units are ``'Angstrom'`` and
        ``'bohr'``.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        Forces converted into the desired units.
    """
    #'ORCA': {'e_unit': 'hartree', 'r_unit': 'bohr'}
    if e_units not in ['eV', 'hartree', 'kcal/mol', 'kJ/mol']:
        raise ValueError(f'{e_units} is not an available energy unit.')
    if r_units not in ['Angstrom', 'bohr']:
        raise ValueError(f'{r_units} is not an available distance unit.')
    forces_conv = forces
    if e_units_calc != e_units:
        forces_conv = cclib.parser.utils.convertor(
            forces_conv, e_units_calc, e_units
        )
    if r_units_calc != r_units:
        forces_conv = cclib.parser.utils.convertor(
            forces_conv, r_units, r_units_calc
        )
    return forces_conv


def atoms_by_element(atom_list):
    """Converts a list of atoms identified by their atomic number to their
    elemental symbol in the same order.
    
    Parameters
    ----------
    atom_list : :obj:`list` [:obj:`int`]
        Atomic numbers of atoms within a structure.
    
    Returns
    -------
    :obj:`list` [:obj:`str`]
        Element symbols of atoms within a structure.
    """

    atom_list_elements = []
    for atom in atom_list:
        atom_list_elements.append(str(z_to_element[atom]))

    return atom_list_elements

def atoms_by_number(atom_list):
    """Converts a list of atoms identified by their elemental symbol to their
    atomic number.
    
    Parameters
    ----------
    atom_list : :obj:`list` [:obj:`str`]
        Element symbols of atoms within a structure.
    
    Returns
    -------
    :obj:`list` [:obj:`int`]
        Atomic numbers of atoms within a structure.
    """
    return [int(element_to_z[i]) for i in atom_list]

def md5_data(data, info=['z', 'R']):
    """Creates MD5 hash for a set of data.

    Parameters
    ----------
    set : :obj:`mbGDML.mbGDMLData`
        Any supported mbGDML data type. Includes structure sets, data sets, and
        models.
    info : :obj:`list`, optional
        List of set keys to include in the MD5 hash. At minimum it defaults to
        ``z`` (atomic numbers) and ``R`` (atomic coordinates). Other information
        such as ``E`` (energies) and ``F`` (forces) are highly encourged.
    
    Returns
    -------
    :obj:`str`
        MD5 hash of the data.
    """
    md5_hash = hashlib.md5()
    for i in info:
        d = data[i]
        if type(d) is np.ndarray:
            d = d.ravel()
        md5_hash.update(hashlib.md5(d).digest())
    return md5_hash.hexdigest()
