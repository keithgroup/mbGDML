# MIT License
#
# Copyright (c) 2020-2023, Alex M. Maldonado
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
import hashlib
import itertools
import json
import numpy as np
import cclib
from qcelemental import periodictable as ptable
from .descriptors import get_center_of_mass
from .logger import GDMLLogger

log = GDMLLogger(__name__)


def get_files(path, expression, recursive=True):
    r"""Returns paths to all files in a given directory that matches a provided
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
    if path[-1] != "/":
        path += "/"
    if recursive:
        all_files = []
        for (dirpath, _, filenames) in os.walk(path):
            index = 0
            while index < len(filenames):
                if dirpath[-1] != "/":
                    dirpath += "/"
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
    r"""The name of the file without the extension from a path.

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


def string_xyz_arrays(Z, R, *args, precision=10):
    r"""Create string of array data in XYZ format for a single structure.

    Parameters
    ----------
    Z : :obj:`numpy.ndarray`, int, ndim=1
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
    struct_string = ""
    atom_strings = atoms_by_element(Z)
    for i, atom_string in enumerate(atom_strings):
        for arr in (R, *args):
            if arr is not None:
                atom_string += "    "
                atom_string += np.array2string(
                    arr[i],
                    suppress_small=True,
                    separator="    ",
                    formatter={"float_kind": lambda x: f"%.{precision}f" % x},
                )[1:-1]
        atom_string = atom_string.replace(" -", "-")
        atom_string += "\n"
        struct_string += atom_string
    return struct_string


def write_xyz(xyz_path, Z, R, comments=None, data_precision=10):
    r"""Write standard XYZ file.

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
    n_atoms = len(Z)
    with open(xyz_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(R):
            f.write(f"{n_atoms}\n")
            if comments is not None:
                comment = comments[i]
                if comment[-2:] != "\n":
                    comment += "\n"
            else:
                comment = "\n"
            f.write(comment)
            f.write(string_xyz_arrays(Z, r, precision=data_precision))


def parse_xyz(xyz_path):
    r"""Parses data from xyz file.

    An xyz file is data presented as consecutive xyz structures. The data could be
    three Cartesian coordinates for each atom, three atomic force vector
    components, or both coordinates and atomic forces in one line (referred to
    as extended xyz).

    Parameters
    ----------
    xyz_path : :obj:`str`
        Path to xyz file.

    Returns
    -------
    :obj:`tuple` [:obj:`list`]
        Parsed atoms (as element symbols :obj:`str`), comments, and data as
        :obj:`float` from string file.
    """
    Z, comments, data = [], [], []
    with open(xyz_path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            line = line.strip()
            if not line:
                # Skips blank lines
                pass
            else:
                line_split = line.split()
                if (
                    len(line_split) == 1
                    and float(line_split[0]) % int(line_split[0]) == 0.0
                ):
                    # Skips number of atoms line, adds comment line, and
                    # prepares next z and data item.
                    comment_line = next(f)
                    comments.append(comment_line.strip())
                    Z.append([])
                    data.append([])
                else:
                    # Grabs z and data information.
                    Z[-1].append(line_split[0])
                    data[-1].append([float(i) for i in line_split[1:]])
    return Z, comments, data


def convert_forces(forces, e_units_calc, r_units_calc, e_units, r_units):
    r"""Converts forces (or gradients) to specified units.

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
    if e_units not in ["eV", "hartree", "kcal/mol", "kJ/mol"]:
        raise ValueError(f"{e_units} is not an available energy unit.")
    if r_units not in ["Angstrom", "bohr"]:
        raise ValueError(f"{r_units} is not an available distance unit.")
    forces_conv = forces
    if e_units_calc != e_units:
        forces_conv = cclib.parser.utils.convertor(forces_conv, e_units_calc, e_units)
    if r_units_calc != r_units:
        forces_conv = cclib.parser.utils.convertor(forces_conv, r_units, r_units_calc)
    return forces_conv


def atoms_by_element(atom_list):
    r"""Converts a list of atoms identified by their atomic number to their
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
    return [ptable.to_symbol(z) for z in atom_list]


def atoms_by_number(atom_list):
    r"""Converts a list of atoms identified by their elemental symbol to their
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
    return [ptable.to_atomic_number(symbol) for symbol in atom_list]


def md5_data(data, keys):
    r"""Creates MD5 hash for a set of data.

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
        if isinstance(d, np.ndarray):
            d = d.ravel()
        md5_hash.update(hashlib.md5(d).digest())
    return md5_hash.hexdigest()


def get_entity_ids(atoms_per_mol, num_mol, starting_idx=0, add_to=None):
    r"""Generates entity ids for a single species.

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
    for i in range(starting_idx, num_mol + starting_idx):
        entity_ids.extend([i for _ in range(0, atoms_per_mol)])

    if add_to is not None:
        if isinstance(add_to, np.ndarray):
            add_to = add_to.tolist()
        return np.array(add_to + entity_ids)

    return np.array(entity_ids)


def get_comp_ids(label, num_mol, add_to=None):
    r"""Prepares the list of component ids for a system with only one species.

    Parameters
    ----------
    label : :obj:`int`
        Species label.
    num_mol : :obj:`int`
        Number of molecules of this type in the system.
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

    return np.array(comp_ids)


def get_R_slice(entities, entity_ids):
    r"""Retrieves R slice for specific entities.

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
        atom_idx.extend([i for i, x in enumerate(entity_ids) if x == entity_id])
    return np.array(atom_idx)


def center_structures(Z, R):
    r"""Centers each structure's center of mass to the origin.

    Previously centered structures should not be affected by this technique.

    Parameters
    ----------
    Z : :obj:`numpy.ndarray`, ndim: ``1``
        Atomic numbers of the atoms in every structure.
    R : :obj:`numpy.ndarray`, ndim: ``3``
        Cartesian atomic coordinates of data set structures.

    Returns
    -------
    :obj:`numpy.ndarray`
        Centered Cartesian atomic coordinates.
    """
    # Masses of each atom in the same shape of R.
    if R.ndim == 2:
        R = np.array([R])
    n_atoms = R.shape[1]

    R -= np.repeat(get_center_of_mass(Z, R), n_atoms, axis=0).reshape(R.shape)

    if R.shape[0] == 1:
        R = R[0]

    return R


def save_json(json_path, json_dict):
    r"""Save JSON file.

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
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json_string)


def gen_combs(sets, replacement=False):
    r"""Generate combinations from multiple sets.

    Parameters
    ----------
    sets : :obj:`list` or :obj:`tuple`, ndim: ``2``
        An iterable that contains multiple sets.
    replacement : :obj:`bool`, default: ``False``
        Allows repeated combinations in different order. If ``False``,
        ``(0, 1)`` and ``(1, 0)`` could be possible if there is overlap
        in the sets.

    Yields
    ------
    :obj:`tuple`
        Combination of one element per set in ``sets``.

    Examples
    --------
    >>> sets = ((0,) (1, 2), (1, 2, 3))
    >>> combs = gen_combs(sets)
    >>> for comb in combs:
    ...     print(comb)
    ...
    (0, 1, 2)
    (0, 1, 3)
    (0, 2, 3)
    """
    combs = itertools.product(*sets)
    # Excludes combinations that have repeats (e.g., (0, 0) and (1, 1. 2)).
    combs = itertools.filterfalse(lambda x: len(set(x)) < len(x), combs)
    # At this point, there are still duplicates in this iterator.
    # For example, (0, 1) and (1, 0) are still included.
    for comb in combs:
        # Sorts options is to avoid duplicate structures.
        # For example, if combination is (1, 0) the sorted version is not
        # equal and will not be included.
        if not replacement:
            if sorted(comb) != list(comb):
                continue
        yield comb


def chunk_iterable(iterable, n):
    r"""Chunk an iterable into ``n`` objects.

    Parameters
    ----------
    iterable : ``iterable``
        Iterable to chunk.
    n : :obj:`int`
        Size of each chunk.

    Yields
    ------
    :obj:`tuple`
        ``n`` objects.
    """
    iterator = iter(iterable)
    for first in iterator:
        yield tuple(itertools.chain([first], itertools.islice(iterator, n - 1)))


def chunk_array(array, n):
    r"""Chunk an array.

    Parameters
    ----------
    array : :obj:`numpy.ndarray`
        Array to chunk.
    n : :obj:`int`
        Size of each chunk.
    """
    for i in range(0, len(array), n):
        array_worker = array[i : i + n]
        yield array_worker
