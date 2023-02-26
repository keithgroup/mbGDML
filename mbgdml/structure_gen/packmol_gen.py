# MIT License
#
# Copyright (c) 2022-2023, Alex M. Maldonado
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

"""Packmol routines"""

import os
import subprocess
import numpy as np
from ..utils import parse_xyz, atoms_by_number
from ..logger import GDMLLogger

log = GDMLLogger(__name__)


def get_packmol_input(
    shape,
    length_scale,
    mol_numbers,
    mol_paths,
    output_path=None,
    periodic=False,
    dist_tolerance=2.0,
    filetype="xyz",
    seed=-1,
    periodic_shift=1.0,
):
    r"""Packmol input file lines for a box containing one or more species.

    Parameters
    ----------
    shape : :obj:`str`
        Desired packmol shape. Supported options: ``sphere``, ``box``.
    length_scale : :obj:`float`
        Relevant length scale in Angstroms for the packmol shape.

        - ``sphere``: diameter;
        - ``box``: side length.
    mol_numbers : :obj:`numpy.ndarray`, ndim: ``1``
        Number of molecules for each species.
    mol_paths : :obj:`list` of :obj:`str`
        Paths to xyz files for each species in the same order as ``mol_numbers``.
    output_path : :obj:`str`, default: ``None``
        Path to save the xyz file. If ``None``, then no output line is included.
    periodic : :obj:`bool`, default: ``False``
        Will periodic boundary conditions be used?
    dist_tolerance : :obj:`bool`, default: ``2.0``
        The minimum distance between pairs of atoms of different molecules.
    filetype : :obj:`str`, default: ``xyz``
        Packmol output format.
    seed : :obj:`int`, default: ``-1``
        Random number generator seed. If equal to ``-1`` then a random seed is
        generated.
    periodic_shift : :obj:`float`, default: ``1.0``
        Reduce the length scale by this much in Angstroms on all sides. This means
        periodic images will be 2.0 Angstroms apart (with the default value).

    Examples
    --------
    >>> shape = "box"
    >>> length_scale = 10.0
    >>> num_mols = np.array([33], dtype=np.uint16)
    >>> mol_paths = "./1h2o.xyz"
    >>> packmol_box_input(
    ... shape, length_scale, num_mols, mol_paths, periodic=True
    ... )
    ['tolerance 2.0\n', 'filetype xyz\n\n', 'structure ./1h2o.xyz\n',
    '    number 33\n', '    inside box 1.0 1.0 1.0 9.0 9.0 9.0\n', 'end structure\n\n']
    """
    if shape not in ["sphere", "box"]:
        raise ValueError(f"{shape} is not a valid selection.")
    if shape == "sphere":
        if periodic:
            raise ValueError()

    # Handling periodic lengths for packmol.
    length_scale = float(length_scale)
    if periodic:
        length_scale = length_scale - periodic_shift
    else:
        periodic_shift = 0.0
    shape_input = (
        f"    inside box {periodic_shift} {periodic_shift} {periodic_shift} "
        f"{length_scale} {length_scale} {length_scale}"
    )

    packmol_input_lines = [
        f"tolerance {dist_tolerance}",
        f"seed {seed}",
        f"filetype {filetype}\n",
    ]

    if isinstance(mol_paths, str):
        mol_paths = [
            mol_paths,
        ]

    for num_mol, mol_path in zip(mol_numbers, mol_paths):
        packmol_input_lines.extend(
            [
                f"structure {mol_path}",
                f"    number {num_mol}",
                shape_input,
                "end structure\n",
            ]
        )

    if output_path is not None:
        packmol_input_lines.append(f"output {output_path}")

    packmol_input_lines = [i + "\n" for i in packmol_input_lines]
    return packmol_input_lines


def run_packmol(work_dir, packmol_input_lines, packmol_path="packmol"):
    r"""Generate structure by running packmol.

    Output must be in the xyz format.

    Parameters
    ----------
    work_dir : :obj:`str`
        Work directory to write input file and run packmol.
    packmol_input_lines : :obj:`list`
        Lines to a packmol input file.
    packmol_path : :obj:`str`, default: ``packmol``
        Path to packmol binary. Default value assumes it can be located in your
        ``PATH``.

    Returns
    -------
    :obj:`numpy.ndarray`
        Atomic numbers of the structure
    :obj:`numpy.ndarray`
        Cartesian coordinates of the structure.
    """
    if not os.path.exists(work_dir):
        raise RuntimeError(f"The working directory, {work_dir}, does not exist")

    input_path = os.path.join(work_dir, "packmol.in")
    with open(input_path, mode="w+", encoding="utf-8") as f:
        f.writelines(packmol_input_lines)

    subprocess.run(
        f"{packmol_path} < {input_path}", capture_output=False, shell=True, check=True
    )

    for line in packmol_input_lines:
        if "output " in line:
            _, output_path = line.strip().split(" ")

    elements, _, R = parse_xyz(output_path)
    Z = np.array(atoms_by_number(elements[0]), dtype=np.uint8)
    R = np.array(R[0], dtype=np.float64)
    return Z, R
