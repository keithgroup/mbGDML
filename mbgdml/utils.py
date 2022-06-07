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
import hashlib
import itertools
import json
import numpy as np
import os

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

# Standard atomic weight from version 4.1 of
# https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
# If lower and upper bounds were provided the lower bound was selected.
z_to_mass = (
    None, 1.00784, 4.002602, 6.938, 9.0121831, 10.806, 12.0096, 14.00643,  # N
    15.99903, 18.998403163, 20.1797, 22.98976928, 24.304, 26.9815385, 28.084,  # Si
    30.973761998, 32.059, 35.446, 39.948, 39.0983, 40.078, 44.955908,  # Sc
    47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546,  # Cu
    65.38, 69.723, 72.630, 74.921595, 78.971, 79.901, 83.798, 85.4678, 87.62,  # Sr
    88.90584, 91.224, 92.90637, 95.95, 98.0, 101.07, 102.90550, 106.42, 107.8682,  # Ag
    112.414, 114.818, 118.710, 121.760, 127.60, 126.90447, 131.293,  # Xe
    132.90545196, 137.327, 138.90547, 140.116, 140.90766, 144.242, 145.0, 150.36,  # Sm
    151.964, 157.25, 158.92535, 162.500, 164.93033, 167.259, 168.93422, 173.054,  # Yb
    174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084,  # Pt
    196.966569, 200.592, 204.382, 207.2, 208.98040, 209.0, 210.0, 222.0, 223.0, 226.0,  # Ra
    227.0, 232.0377, 231.03588, 238.02891, 237.0, 244.0  # Pu
)

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
    expression in the file name.
    
    Parameters
    ----------
    path : :obj:`str`
        Specifies the directory to search.
    expression : :obj:`str`
        Expression to be tested against all file names in ``path``.
    recursive : :obj:`bool`, optional
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

def get_filename(path):
    """The name of the file without the extension from a path.

    If there are periods in the file name with no file extension, will always
    remove the last one.

    Parameters
    ----------
    path : :obj:`str`
        Path to file.

    Returns
    -------
    :obj:`str`
        The file name without an extension.
    """
    return os.path.splitext(os.path.basename(path))[0]

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
    from natsort import natsorted, ns
    sorted_list = natsorted(unsorted_list, alg=ns.IGNORECASE)

    return sorted_list

def string_xyz_arrays(z, R, *args, precision=10):
    """Create string of array data in XYZ format for a single structure.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`, int, ndim=1
        Atomic numbers of all atoms in the system.
    R : :obj:`numpy.ndarray`, float, ndim=2
        Cartesian coordinates of all atoms in the same order as ``Z``.
    args
        Other :obj:`numpy.ndarray` (ndim>=1) to add where it's assumed the
        zero axis is with respect to ``R``. For example, if we have atomic
        forces the array shape would be ``(n, 3)`` where ``n`` is the number of
        atoms in the structure.
    precision : :obj:`int`, optional
        Number of decimal points for printing array data. Defaults to ``13``.
    
    Returns
    -------
    :obj:`str`
        The XYZ string for a single structure. This does not include the number
        of atoms 
    """
    struct_string = ''
    for i in range(len(z)):
        atom_string = str(z_to_element[z[i]])
        for arr in (R, *args):
            if arr is not None:
                atom_string += '    '
                atom_string += np.array2string(
                    arr[i], suppress_small=True, separator='    ',
                    formatter={'float_kind': lambda x: f'%.{precision}f' % x}
                )[1:-1]
        atom_string = atom_string.replace(' -', '-')
        atom_string += '\n'
        struct_string += atom_string
    return struct_string

def write_xyz(
    xyz_path, z, R, comments=None, data_precision=10
):
    """Write standard XYZ file.
    
    Parameters
    ----------
    xyz_path : :obj:`str`
        Path to XYZ file to write.
    Z : :obj:`numpy.ndarray`
        Atomic numbers of all atoms in the system.
    R : :obj:`numpy.ndarray`
        Cartesian coordinates of all structures in the same order as ``Z``.
    comments : :obj:`list`, optional
        Comment lines for each XYZ structure.
    data_precision : :obj:`int`, optional
        Number of decimal points for printing array data. Default is ``13``.
    """
    n_atoms = len(z)
    with open(xyz_path, 'w') as f:
        for i in range(len(R)):
            f.write(f'{n_atoms}\n')
            if comments is not None:
                comment = comments[i]
                if comment[-2:] != '\n':
                    comment += '\n'
            else:
                comment = '\n'
            f.write(comment)
            f.write(string_xyz_arrays(z, R[i], precision=data_precision))

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
    r_units : :obj:`str`
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

def md5_data(data, keys):
    """Creates MD5 hash for a set of data.

    Parameters
    ----------
    data : :obj:`dict`
        Any supported mbGDML data type as a dictionary. Includes structure sets,
        data sets, and models.
    keys : :obj:`list` of :obj:`str`
        List of keys in ``data`` to include in the MD5 hash.
    
    Returns
    -------
    :obj:`str`
        MD5 hash of the data.
    
    Notes
    -----
    We sort ``keys`` because the MD5 hash depends on the order we digest the
    data.
    """
    keys.sort()
    md5_hash = hashlib.md5()
    for key in keys:
        d = data[key]
        if type(d) is np.ndarray:
            d = d.ravel()
        md5_hash.update(hashlib.md5(d).digest())
    return md5_hash.hexdigest()

def e_f_contribution(dset, dsets_lower, operation):
    """Adds or removes energy and force contributions from data sets.

    Forces are currently not updated but still returned.

    Parameters
    ----------
    dset : :obj:`mbgdml.data.dataset.dataSet`
        The reference data set.
    dsets_lower : :obj:`list` [:obj:`mbgdml.data.dataset.dataSet`]
        Data set contributions to be added or removed from ``E`` and ``F``.
    operation : :obj:`str`
        ``'add'`` or ``'remove'`` the contributions.
    
    Returns
    -------
    :obj:`mbgdml.data.dataset.dataSet`
    """
    E = dset.E
    F = dset.F

    # Loop through every lower order n-body data set.
    for dset_lower in dsets_lower:
        # Checks MD5 hashes
        for Rset_id, Rset_md5 in dset.Rset_md5.items():
            assert Rset_md5 == dset_lower.Rset_md5[Rset_id]

        # Loop through every structure in the reference data set.
        for i in range(len(dset.R)):
            # We have to match the molecule information for each reference dset
            # structure to the molecules in this lower dset to remove the right
            # information.
            r_info = dset.r_prov_specs[i]  # r_prov_specs of this structure.
            mol_combs = list(  # Molecule combinations to be removed from structure.
                itertools.combinations(
                    r_info[2:], len(set(dset_lower.entity_ids))  # Number of molecules in lower dset
                )
            )
            mol_combs = np.array(mol_combs)

            # Loop through every molecular combination.
            for mol_comb in mol_combs:
                r_info_lower_comb = np.block([r_info[:2], mol_comb])
                
                # Index of the molecule combination in the lower data set.
                i_r_lower = np.where(
                    np.all(dset_lower.r_prov_specs == r_info_lower_comb, axis=1)
                )[0][0]

                e_r_lower = dset_lower.E[i_r_lower]
                f_r_lower = dset_lower.F[i_r_lower]

                # Adding or removing contributions.
                if operation == 'add':
                    E[i] += e_r_lower
                elif operation == 'remove':
                    E[i] -= e_r_lower
                else:
                    raise ValueError(f'{operation} is not "add" or "remove".')
                
                for Rset_id in r_info_lower_comb[2:]:
                    entity_id = np.where(r_info[2:] == Rset_id)[0][0]
                    entity_idx = np.where(dset.entity_ids == entity_id)[0]

                    entity_id_lower = np.where(
                        r_info_lower_comb[2:] == Rset_id
                    )[0][0]
                    entity_idx_lower = np.where(
                        dset_lower.entity_ids == entity_id_lower
                    )[0]
                    
                    if operation == 'add':
                        F[i][entity_idx] += f_r_lower[entity_idx_lower]
                    elif operation == 'remove':
                        F[i][entity_idx] -= f_r_lower[entity_idx_lower]

    dset.E = E
    dset.F = F
    return dset

def get_entity_ids(
    atoms_per_mol, num_mol, starting_idx=0, add_to=None
):
    """Generates entity ids for a single species.

    Note that all of the atoms in each molecule must occur in the same order and
    be grouped together.

    Parameters
    ----------
    atoms_per_mol : :obj:`int`
        Number of atoms in the molecule.
    num_mol : :obj:`int`
        Number of molecules of this type in the system.
    starting_idx : :obj:`int`
        Number to start entity_id labels.
    add_to : :obj:`list`
        Entity ids to append new ids to.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        Entity ids for a structure.
    """
    entity_ids = []
    for i in range(starting_idx, num_mol+starting_idx):
        entity_ids.extend([i for _ in range(0, atoms_per_mol)])
    
    if add_to is not None:
        if isinstance(add_to, np.ndarray):
            add_to = add_to.tolist()
        return np.array(add_to + entity_ids)
    else:
        return np.array(entity_ids)

def get_comp_ids(label, num_mol, entity_ids, add_to=None):
    """Prepares the list of component ids for a system with only one species.

    Parameters
    ----------
    label : :obj:`int`
        Species label.
    num_mol : :obj:`int`
        Number of molecules of this type in the system.
    entity_ids : :obj:`int`
        A uniquely identifying integer specifying what atoms belong to
        which entities. Entities can be a related set of atoms, molecules,
        or functional group. For example, a water and methanol molecule
        could be ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.
    add_to : :obj:`list`
        Component ids to append new ids to.

    Returns
    -------
    :obj:`numpy.ndarray`
        Component ids for a structure.
    """
    comp_ids = [label for _ in range(0, num_mol)]
    if add_to is not None:
        if isinstance(add_to, np.ndarray):
            add_to = add_to.tolist()
        return np.array(add_to + comp_ids)
    else:
        return np.array(comp_ids)

def get_R_slice(entities, entity_ids):
    """Retrives R slice for specific entities.

    Parameters
    ----------
    entities : :obj:`numpy.ndarray`
        Desired entities from R. For example, ``np.array([2, 5])``.
    entity_ids : :obj:`numpy.ndarray`
        A uniquely identifying integer specifying what atoms belong to
        which entities. Entities can be a related set of atoms, molecules,
        or functional group. For example, a water and methanol molecule
        could be ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        The indices of all atoms of all entities.
    """
    atom_idx = []
    for entity_id in entities:
        atom_idx.extend(
            [i for i,x in enumerate(entity_ids) if x == entity_id]
        )
    return np.array(atom_idx)

def center_structures(z, R):
    """Centers each structure's center of mass to the origin.

    Previously centered structures should not be affected by this technique.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        Atomic numbers of the atoms in every structure.
    R : :obj:`numpy.ndarray`
        Cartesian atomic coordinates of data set structures.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        Centered Cartesian atomic coordinates.
    """
    # Masses of each atom in the same shape of R.
    if R.ndim == 2:
        R = np.array([R])
    
    masses = np.empty(R[0].shape)
    
    for i in range(len(masses)):
        masses[i,:] = z_to_mass[z[i]]
    
    for i in range(len(R)):
        r = R[i]
        cm_r = np.average(r, axis=0, weights=masses)
        R[i] = r - cm_r
    
    if R.shape[0] == 1:
        return R[0]
    else:
        return R

def save_json(json_path, json_dict):
    """Save JSON file.

    Parameters
    ----------
    json_path : :obj:`str`
        JSON file path to save.
    json_dict : :obj:`dict`
        JSON dictionary to be saved.
    """ 
    json_string = json.dumps(
        json_dict, cls=cclib.io.cjsonwriter.JSONIndentEncoder, indent=4
    )
    with open(json_path, 'w') as f:
        f.write(json_string)