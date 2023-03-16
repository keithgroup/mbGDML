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

"""Drive MD simulations and sample high-error, n-body structures"""

import os
import shutil
import uuid
from ase import Atoms
from ase.optimize import BFGS
import numpy as np
import ray
from ..interfaces.ase import mbeCalculator
from ..structure_gen.packmol_gen import run_packmol
from ..utils import get_entity_ids, get_comp_ids
from ..logger import GDMLLogger

log = GDMLLogger(__name__)


@ray.remote(num_cpus=2)
class MDDriver:
    r"""Run MD simulations using the ASE interface."""

    def __init__(
        self,
        work_dir,
        packmol_input,
        mol_n_atoms,
        mol_comp_ids,
        mbe_pred,
        label=None,
        ase_e_conv=1.0,
        ase_f_conv=1.0,
        opt_f_max=0.4,
        opt_steps=50,
        periodic_cell=None,
        keep_files=False,
    ):
        r"""
        Parameters
        ----------
        work_dir : :obj:`str`
            Path to working directory. This can be possibly shared with other Actors.
        packmol_input : :obj:`str`
            Input file for packmol structure generation.
        mol_n_atoms : :obj:`tuple`
            Number of atoms for each "structure" line in the order they appear in
            ``packmol_input``. Used to create ``entity_ids``
        mbe_pred : :obj:`mbgdml.mbe.mbePredict`
            Predictor function for driving MD simulations.
        label : :obj:`str`, default: ``None``
            Unique label for this MD simulation. Otherwise, a random label will be
            generated.
        keep_files : :obj:`bool`, default: ``False``
            Keep all working files of this Actor.
        """
        self.log = GDMLLogger(__name__)
        self.log.info("Initializing MDDriver")
        self.work_dir = work_dir
        if not os.path.exists(work_dir):
            raise RuntimeError(f"The working directory, {work_dir}, does not exist")

        self.packmol_input = packmol_input

        self.mol_n_atoms = mol_n_atoms
        self.mol_comp_ids = mol_comp_ids
        self.mbe_pred = mbe_pred

        if label is None:
            label = str(uuid.uuid4().hex)
        self.label = label
        self.log.debug("Label is %s", label)
        self.temp_dir = os.path.join(work_dir, label)
        os.mkdir(self.temp_dir)

        self.keep_files = keep_files

        self.ase_e_conv = ase_e_conv
        self.ase_f_conv = ase_f_conv
        self.opt_f_max = opt_f_max
        self.opt_steps = opt_steps
        self.periodic_cell = periodic_cell
        self.opt_traj_name = "geom-opt.traj"

        self.log.info("Done initializing MDDriver")

    def getattr(self, attr):
        r"""Get attribute.

        Parameters
        ----------
        attr : :obj:`str`
            Attribute name.

        Returns
        -------
        Attribute
        """
        return getattr(self, attr)

    # pylint: disable-next=invalid-name
    def create_starting_R(self):
        r"""Create the starting structure for a MD simulation using packmol.

        Parameters
        ----------
        packmol_input : :obj:`str`
            Input file for packmol structure generation.
        mol_n_atoms : :obj:`tuple`
            Number of atoms for each "structure" line in the order they appear in
            ``packmol_input``. Used to create ``entity_ids``

        """
        self.log.info("Generating structure with packmol")

        # Determine entity_ids and comp_ids for predictions.
        i = 0
        entity_ids = np.array([])
        comp_ids = np.array([])
        for line in self.packmol_input:
            if " number " in line:
                # This is a new species.
                num_mol = int(line.strip().split()[-1])
                entity_ids = get_entity_ids(
                    self.mol_n_atoms[i], num_mol, add_to=entity_ids
                )
                comp_ids = get_comp_ids(self.mol_comp_ids[i], num_mol, add_to=comp_ids)
                i += 1
        self.entity_ids = entity_ids
        self.comp_ids = comp_ids

        output_path = os.path.join(self.temp_dir, "packmol-output.xyz")
        self.packmol_input.append(f"output {output_path}\n")
        # pylint: disable-next=invalid-name
        self.Z_packmol, self.R_packmol = run_packmol(self.temp_dir, self.packmol_input)

    def setup_ase(self):
        """

        Parameters
        ----------
        e_conv : :obj:`float`, default: ``1.0``
            Model energy conversion factor to eV (required by ASE).
        f_conv : :obj:`float`, default: ``1.0``
            Model forces conversion factor to eV/A (required by ASE).
        """
        self.log.info("Setting up ASE calculator")
        pbc = False
        cell_v = None
        if self.periodic_cell:
            self.log.debug("Periodic boundary conditions will be used")
            pbc = True
            cell_v = self.periodic_cell.cell_v

        mbe_calc = mbeCalculator(
            self.mbe_pred, e_conv=self.ase_e_conv, f_conv=self.ase_f_conv
        )
        mbe_calc.set(entity_ids=self.entity_ids, comp_ids=self.comp_ids)
        self.ase_atoms = Atoms(
            numbers=self.Z_packmol, positions=self.R_packmol, cell=cell_v, pbc=pbc
        )
        self.ase_atoms.calc = mbe_calc

    def optimize(self, traj_path="geom-opt.traj"):
        r"""Perform geometry optimization on ASE Atoms object."""
        self.log.info("Performing geometry optimization")
        traj_path = os.path.join(self.temp_dir, self.opt_traj_name)
        dyn = BFGS(atoms=self.ase_atoms, trajectory=traj_path)
        dyn.run(fmax=self.opt_f_max, steps=self.opt_steps)

    def check_md_stability(self, R, E):
        r"""Determines if the MD simulation is unstable and needs to be terminated.

        Parameters
        ----------
        R : :obj:`numpy.ndarray`
            Cartesian coordinates of the last :math:`N` steps.
        E : :obj:`numpy.ndarray`
            Energies of the last :math:`N` steps. Can be potential, kinetic, or total
            energies.
        """

    def step(self):
        """Perform a single MD step."""

    def cleanup(self):
        r"""Cleanup Actor files and such."""
        if not self.keep_files:
            shutil.rmtree(self.temp_dir)

    def run(self):
        """ """
        self.create_starting_R()
        self.setup_ase()
        self.optimize()
        self.cleanup()
        return True
