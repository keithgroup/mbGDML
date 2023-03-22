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
import sys
import time
from ase import Atoms
from ase import units as ase_units
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
import numpy as np
from .drive_md import MDDriver
from ..interfaces.ase import mbeCalculator
from ..logger import GDMLLogger

log = GDMLLogger(__name__)


class ASEMDDriver(MDDriver):
    r"""Run MD simulations using the ASE interface.

    .. tip::

        If you want to use this class as an Actor with ray, import ``ASEMDDriver``
        and use ``ASEMDDriver = ray.remote(ASEMDDriver)``. Make sure to set the
        environmental variable ``OMP_NUM_THREADS`` to ``1`` if starting ray with the
        cli.
    """

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
        md_t_step=1.0,
        md_init_temp=100.0,
        md_temp=300.0,
        nose_hoover_ttime=50.0,
        md_external_stress=101325,
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
        ase_e_conv : :obj:`float`, default: ``1.0``
            Model energy conversion factor to eV (required by ASE).
        ase_f_conv : :obj:`float`, default: ``1.0``
            Model forces conversion factor to eV/Å (required by ASE).
        opt_f_max : :obj:`float`, default: ``0.4``
            Max force in eV/Å to be considered converged.
        opt_max_steps : :obj:`int`, default: ``50``
            Maximum number of geometry optimization steps.
        md_n_steps : :obj:`int`, default: ``1000``
            Number of MD steps to take after optimization.
        md_t_step : :obj:`float`, default: ``1.0``
            MD time step in femtoseconds.
        md_init_temp : :obj:`float`, default: ``100``
            Temperature (Kelvin) used to initialize velocities for MD simulations using
            the Maxwell-Boltzmann distribution.
        md_temp : :obj:`float`, default: ``300``
            Temperature (Kelvin) set point for thermostat.
        nose_hoover_ttime : :obj:`float`, default: ``50.0``
            Characteristic timescale of the thermostat in femtoseconds.

            .. warning::

                This parameter is specific to your system (and potential). The default
                value is likely not appropriate and an optimal value should be
                investigated.
        md_external_stress : :obj:`float`, default: ``101325``
            External stress to the system in Pascals. The value is not actually used
            in the NVT simulations performed here.
        md_traj_interval : :obj:`int`, default: ``1``
            Steps between logging in the trajectory.
        md_stability_interval : :obj:`int`, default: ``5``
            Number of previous MD steps to include in MD stability analysis.
        keep_files : :obj:`bool`, default: ``True``
            Keep all working files of this Actor.
        """
        # Perform any conversion into internal ASE units.
        md_t_step *= ase_units.fs
        nose_hoover_ttime *= ase_units.fs
        md_external_stress *= ase_units.Pascal

        super().__init__(
            work_dir,
            packmol_input,
            mol_n_atoms,
            mol_comp_ids,
            mbe_pred,
            label=label,
            periodic_cell=periodic_cell,
            opt_max_steps=opt_max_steps,
            md_t_step=md_t_step,
            md_n_steps=md_n_steps,
            md_traj_interval=md_traj_interval,
            md_stability_interval=md_stability_interval,
            keep_files=keep_files,
        )

        self.ase_e_conv = ase_e_conv
        self.ase_f_conv = ase_f_conv
        self.opt_f_max = opt_f_max
        self.md_init_temp = md_init_temp
        self.md_temp = md_temp
        self.nose_hoover_ttime = nose_hoover_ttime
        self.md_external_stress = md_external_stress

        self.log.info("Done initializing MDDriver")

    def setup(self):
        r"""Prepares ASE atoms with calculator."""
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

    def get_energies(self):
        r"""Retrieves energies for checking MD stability.

        Returns
        -------
        :obj:`numpy.ndarray`
            Energies to use for checking MD simulation.
        """
        traj_path = os.path.join(self.temp_dir, self.md_traj_name)
        md_traj_read = Trajectory(traj_path, mode="r")
        energies = []
        for atoms in md_traj_read[-self.md_stability_interval :]:
            energies.append(atoms.get_potential_energy())
        energies = np.array(energies)
        return energies

    def md(self):
        """Perform MD simulation."""
        self.log.info("Setting up MD")
        ase_atoms = self.ase_atoms
        self.log.debug("Initializing velocities using Maxwell-Boltzmann distribution")
        MaxwellBoltzmannDistribution(ase_atoms, temperature_K=self.md_init_temp)
        self.log.debug("Setting up NVT object")
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

        self.md_step = 1

        def print_ase_step(
            a=ase_atoms,
        ):  # store a reference to atoms in the definition.
            """Function to print status of MD simulation."""
            epot = a.get_potential_energy()
            ekin = a.get_kinetic_energy()
            ekin_per_atom = ekin / len(a)
            temp_step = ekin_per_atom / (1.5 * ase_units.kB)

            Time = time.localtime()  # pylint: disable=invalid-name
            time_string = f"{Time[3]:02}:{Time[4]:02}:{Time[5]:02}"
            if self.md_step == 1:
                print(" Step        Time         E_pot (eV)      T (K)")
            print(
                f"{self.md_step:>5}    {time_string:>8}"
                f"    {epot:>15.8f}    {temp_step:>7.2f}"
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
