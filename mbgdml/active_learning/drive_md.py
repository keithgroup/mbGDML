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
import ray
from ..logger import GDMLLogger

log = GDMLLogger(__name__)


@ray.remote
class MDDriver:
    r"""Run MD simulations using the ASE interface."""

    def __init__(self, work_dir, label=None, keep_files=False):
        r"""
        Parameters
        ----------
        work_dir : :obj:`str`
            Path to working directory. This can be possibly shared with other Actors.
        label : :obj:`str`, default: ``None``
            Unique label for this MD simulation. Otherwise, a random label will be
            generated.
        keep_files : :obj:`bool`, default: ``False``
            Keep all working files of this Actor.
        """
        self.work_dir = work_dir
        if not os.path.exists(work_dir):
            raise RuntimeError(f"The working directory, {work_dir}, does not exist")

        if label is None:
            label = str(uuid.uuid4().hex)
        self.label = label
        self.temp_dir = os.path.join(work_dir, label)
        os.mkdir(self.temp_dir)

        self.keep_files = keep_files

    def getattr(self, attr):
        """Get attribute.

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
    def create_starting_R(self, packmol_input):
        r"""Create the starting structure for a MD simulation using packmol.

        Parameters
        ----------
        packmol_input : :obj:`str`
            Input file for packmol structure generation.
        """

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

    def cleanup(self):
        r"""Cleanup Actor files and such."""
        if not self.keep_files:
            shutil.rmtree(self.temp_dir)
