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

from abc import ABC, abstractmethod
import os
import shutil
import uuid
import numpy as np
from ..structure_gen.packmol_gen import run_packmol
from ..utils import get_entity_ids, get_comp_ids
from ..logger import GDMLLogger

log = GDMLLogger(__name__)


class MDDriver(ABC):
    r"""Run MD simulations using the ASE interface."""

    def __init__(
        self,
        work_dir,
        packmol_input,
        mol_n_atoms,
        mol_comp_ids,
        mbe_pred,
        label=None,
        periodic_cell=None,
        opt_max_steps=50,
        md_n_steps=1000,
        md_t_step=1.0,
        md_traj_interval=1,
        md_stability_interval=5,
        keep_files=True,
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
            ``packmol_input``. Used to create ``entity_ids``.
        mol_comp_ids : :obj:`tuple`
            Component IDs of each species in the order that they appear in
            ``packmol_input``. Used to create ``comp_ids``.
        mbe_pred : :obj:`mbgdml.mbe.mbePredict`
            Predictor function for driving MD simulations.
        label : :obj:`str`, default: ``None``
            Unique label for this MD simulation. Otherwise, a random label will be
            generated.
        periodic_cell : :obj:`mbgdml.periodic.Cell`, default: :obj:`None`
            Use periodic boundary conditions defined by this object. If this
            is not :obj:`None` only :meth:`~mbgdml.mbe.mbePredict.predict` can be used.
        opt_max_steps : :obj:`int`, default: ``50``
            Maximum number of geometry optimization steps.
        md_n_steps : :obj:`int`, default: ``1000``
            Number of MD steps to take after optimization.
        md_t_step : :obj:`float`, default: ``1.0``
            MD time step in femtoseconds.
        md_traj_interval : :obj:`int`, default: ``1``
            Steps between logging in the trajectory.
        md_stability_interval : :obj:`int`, default: ``5``
            Number of previous MD steps to include in MD stability analysis.
        keep_files : :obj:`bool`, default: ``True``
            Keep all working files of this Actor.
        """
        self.log = GDMLLogger("MDDriver")
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

        self.periodic_cell = periodic_cell
        self.opt_max_steps = opt_max_steps
        self.md_n_steps = md_n_steps
        self.md_t_step = md_t_step
        self.md_traj_interval = md_traj_interval
        self.md_stability_interval = md_stability_interval

        self.opt_traj_name = "opt.traj"
        self.md_traj_name = "md.traj"

        self.keep_files = keep_files

        self.e_change_threshold = 3.0  # eV

    def getattr(self, attr):
        r"""Get attribute.

        Parameters
        ----------
        attr : :obj:`str`
            Attribute name.
        """
        return getattr(self, attr)

    # pylint: disable-next=invalid-name
    def create_starting_R(self):
        r"""Create the starting structure for a MD simulation using packmol."""
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

    @abstractmethod
    def setup(self):
        r"""Prepares MD driver."""

    @abstractmethod
    def optimize(self):
        r"""Perform geometry optimization on ASE Atoms object."""

    @abstractmethod
    def get_energies(self):
        r"""Retrieves energies for checking MD stability."""

    def check_md_stability(self):
        r"""Determines if the MD simulation is stable. If not, it will be terminated.

        Returns
        -------
        :obj:`bool`
            If the MD simulation is stable.
        """
        self.log.debug("Checking MD simulations")
        energies = self.get_energies()
        energies -= energies[0]
        energy_change_abs_mean = np.abs(np.mean(energies[1:]))
        self.log.debug("Average absolute energy change: %d", energy_change_abs_mean)
        if energy_change_abs_mean > self.e_change_threshold:
            self.log.debug("This is above the threshold of %d", self.e_change_threshold)
            self.log.debug("MD simulation is considered unstable")
            return False
        self.log.debug("MD simulation is considered stable")
        return True

    @abstractmethod
    def md(self):  # pylint: disable=invalid-name
        r"""Perform MD simulation."""

    def cleanup(self):
        r"""Cleanup temporary files and such."""
        if not self.keep_files:
            shutil.rmtree(self.temp_dir)

    def run(self):
        r"""Prepare and run MD simulation."""
        self.create_starting_R()
        self.setup()
        self.optimize()
        self.md()
        self.cleanup()
