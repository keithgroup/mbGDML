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
import sys
import uuid
from ase import Atoms
from ase import units as ase_units
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
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
        periodic_cell=None,
        ase_e_conv=1.0,
        ase_f_conv=1.0,
        opt_f_max=0.4,
        opt_max_steps=50,
        md_n_steps=1000,
        md_t_step=1.0 * ase_units.fs,
        md_init_temp=100.0,
        md_temp=300.0,
        nose_hoover_ttime=50.0 * ase_units.fs,
        md_external_stress=101325 * ase_units.Pascal,
        md_traj_interval=1,
        md_stability_interval=5,
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
        ase_e_conv : :obj:`float`, default: ``1.0``
            Model energy conversion factor to eV (required by ASE).
        ase_f_conv : :obj:`float`, default: ``1.0``
            Model forces conversion factor to eV/A (required by ASE).
        opt_f_max : :obj:`float`, default: ``0.4``
            Max force in eV/A to be considered converged.
        opt_max_steps : :obj:`int`, default: ``50``
            Maximum number of geometry optimization steps.
        md_n_steps : :obj:`int`, default: ``1000``
            Number of MD steps to take after optimization.
        md_t_step : :obj:`float`, default: ``1.0*ase_units.fs``
            MD time step in ASE units.
        md_init_temp : :obj:`float`, default: ``100``
            Temperature used to initialize velocities for MD simulations using the
            Maxwell-Boltzmann distribution.
        md_temp : :obj:`float`, default: ``300``
            Temperature set point for thermostat.
        nose_hoover_ttime : :obj:`float`, default: ``50.0*ase_units.fs``
            Characteristic timescale of the thermostat in ASE internal units.
        md_external_stress : :obj:`float`, default: ``101325*ase_units.Pascal``
            External stress to the system in ASE internal units. The value is not
            actually used.
        md_traj_interval : :obj:`int`, default: ``1``
            Steps between logging in the trajectory.
        keep_files : :obj:`bool`, default: ``True``
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

        self.periodic_cell = periodic_cell
        self.ase_e_conv = ase_e_conv
        self.ase_f_conv = ase_f_conv
        self.opt_f_max = opt_f_max
        self.opt_max_steps = opt_max_steps
        self.md_n_steps = md_n_steps
        self.md_t_step = md_t_step
        self.md_init_temp = md_init_temp
        self.md_temp = md_temp
        self.nose_hoover_ttime = nose_hoover_ttime
        self.md_external_stress = md_external_stress
        self.md_traj_interval = md_traj_interval
        self.md_stability_interval = md_stability_interval

        self.opt_traj_name = "opt.traj"
        self.md_traj_name = "nvt.traj"

        self.keep_files = keep_files

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

    def setup_ase(self):
        """Prepares ASE atoms with calculator."""
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

    def optimize(self):
        r"""Perform geometry optimization on ASE Atoms object."""
        self.log.info("Performing geometry optimization")
        traj_path = os.path.join(self.temp_dir, self.opt_traj_name)
        dyn = BFGS(atoms=self.ase_atoms, trajectory=traj_path)
        dyn.run(fmax=self.opt_f_max, steps=self.opt_max_steps)

    def check_md_stability(self):
        r"""Determines if the MD simulation is stable. If not, it will be terminated.
        
        Returns
        -------
        :obj:`bool`
            If the MD simulation is stable.
        """
        self.log.info("Checking MD simulations")
        traj_path = os.path.join(self.temp_dir, self.md_traj_name)
        md_traj_read = Trajectory(traj_path, mode="r")
        energies = []
        for atoms in md_traj_read[-self.md_stability_interval :]:
            energies.append(atoms.get_potential_energy())
        energies = np.array(energies)
        energies -= energies[0]
        energy_change_abs_mean = np.abs(np.mean(energies[1:]))
        self.log.debug("Average absolute energy change: %d", energy_change_abs_mean)
        threshold = 3.0
        if energy_change_abs_mean > threshold:
            self.log.debug("This is above the threshold of %d", threshold)
            self.log.debug("MD simulation is considered unstable")
            return False
        self.log.debug("MD simulation is considered stable")
        return True

    def nvt(self):
        """Perform MD simulation."""
        self.log.info("Setting up MD")
        ase_atoms = self.ase_atoms
        self.log.debug("Initializing velocities using Maxwell-Boltzmann distribution")
        MaxwellBoltzmannDistribution(ase_atoms, temperature_K=self.md_init_temp)
        self.log.debug("Setting up NPT object")
        self.md = NPT(  # pylint: disable=invalid-name
            atoms=ase_atoms,
            timestep=self.md_t_step,
            externalstress=self.md_external_stress,
            ttime=self.nose_hoover_ttime,
            pfactor=None,
            temperature_K=self.md_temp,
        )
        self.log.debug("Setting up loggers")
        md_traj_path = os.path.join(self.temp_dir, self.md_traj_name)
        md_traj = Trajectory(md_traj_path, mode="w", atoms=ase_atoms)

        self.md_step = 0

        def print_ase_step(
            a=ase_atoms,
        ):  # store a reference to atoms in the definition.
            """Function to print status of MD simulation."""
            epot = a.get_potential_energy()
            ekin = a.get_kinetic_energy()
            ekin_per_atom = ekin / len(a)
            temp_step = ekin_per_atom / (1.5 * ase_units.kB)

            print(
                f"Step {self.md_step : >5}/{self.md_n_steps : <5}     "
                f"E_pot: {epot : >15.8f} eV     "
                f"T: {temp_step : >7.2f} K"
            )
            self.md_step += 1
            if self.md_step > self.md_stability_interval:
                if not self.check_md_stability():
                    # Raises caught system exit to terminate MD simulation.
                    sys.exit("MD simulation is unstable")

        self.md.attach(print_ase_step, interval=1)
        self.md.attach(md_traj.write, interval=self.md_traj_interval)
        self.log.info("Starting simulation")
        try:
            self.md.run(self.md_n_steps)
        except SystemExit as e:
            if "MD simulation is unstable" != str(e):
                raise e

    def cleanup(self):
        r"""Cleanup temporary files and such."""
        if not self.keep_files:
            shutil.rmtree(self.temp_dir)

    def run(self):
        r"""Prepare and run MD simulation."""
        self.create_starting_R()
        self.setup_ase()
        self.optimize()
        self.nvt()
        self.cleanup()
