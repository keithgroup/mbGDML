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

import itertools
import numpy as np
from qcelemental import periodictable as ptable
from .logger import GDMLLogger

log = GDMLLogger(__name__)


class Criteria:
    r"""Descriptor criteria for accepting a structure based on a descriptor
    and cutoff.
    """

    def __init__(self, desc, desc_kwargs, cutoff, bound="upper"):
        """
        Parameters
        ----------
        desc : ``callable``
            Computes the descriptor. First two arguments must be ``Z`` and
            ``R``.
        desc_kwargs : :obj:`dict`
            Keyword arguments for the descriptor function after ``Z`` and ``R``.
            This can be an empty tuple.
        cutoff : :obj:`float`, :obj:`None`, or :obj:`tuple`
            Cutoff to accept or reject a structure. If :obj:`None`, all structures
            are accepted. If a :obj:`tuple` is provided, structures that are
            within these cutoffs will be accepted.
        bound : :obj:`str`, default: ``'upper'``
            What bound does the cutoff represent? ``'upper'`` means any
            descriptor that is equal to or larger than the cutoff will be
            rejected. ``'lower'`` means anything equal to or smaller than the
            cutoff. If ``cutoff`` is a tuple, we ignore this.
        """
        self.desc = desc
        self.desc_kwargs = desc_kwargs
        self.cutoff = cutoff
        if isinstance(self.cutoff, (tuple, list)):
            self.cutoff = sorted(self.cutoff)  # Will always be a list.
        bound = bound.lower()
        assert bound in ["upper", "lower"]
        self.bound = bound

    def accept(self, Z, R, **kwargs):
        r"""Determine if we accept the structure.

        Parameters
        ----------
        Z : :obj:`numpy.ndarray`
            Atomic numbers of the structure.
        R : :obj:`numpy.ndarray`, ndim: ``2`` or ``3``
            Cartesian coordinates of the structure.
        kwargs
            Additional keyword arguments to pass into the descriptor function.

        Returns
        -------
        :obj:`bool` or :obj:`numpy.ndarray`
            If the descriptor is less than the cutoff.
        :obj:`float` or :obj:`numpy.ndarray`
            The value of the descriptor.
        """
        if R.ndim == 2:
            R = R[None, ...]
        n_R = R.shape[0]

        if self.cutoff is None:
            accept_r = np.full(n_R, True)
            return accept_r, None

        desc_v = self.desc(Z, R, **self.desc_kwargs, **kwargs)
        if isinstance(self.cutoff, list):
            accept_r = (self.cutoff[0] < desc_v) & (desc_v < self.cutoff[1])
        else:
            if self.bound == "upper":
                accept_r = desc_v < self.cutoff
            else:  # lower
                accept_r = desc_v > self.cutoff
        if n_R == 1:
            accept_r = accept_r[0]
            desc_v = desc_v[0]
        return accept_r, desc_v


def get_center_of_mass(Z, R):
    r"""Compute the center of mass.

    Parameters
    ----------
    Z : :obj:`numpy.ndarray`, ndim: ``1``
        Atomic numbers of all atoms in the system.
    R : :obj:`numpy.ndarray`, ndim: ``3``
        Cartesian coordinates.

    Returns
    -------
    :obj:`float`
        The center of mass cartesian coordinates.
    """
    if R.ndim == 2:
        R = np.array([R])
    masses = np.empty(R[0].shape)
    for i in range(len(masses)):
        masses[i, :] = ptable.to_mass(Z[i])
    R_masses = np.full(R.shape, masses)
    cm_structure = np.average(R, axis=1, weights=R_masses)
    return cm_structure


def max_atom_pair_dist(Z, R):
    r"""The largest atomic pairwise distance.

    Parameters
    ----------
    Z : :obj:`numpy.ndarray`, ndim: ``1``
        Atomic numbers of all atoms in the system.
    R : :obj:`numpy.ndarray`, ndim: ``3``
        Cartesian coordinates.

    Returns
    -------
    :obj:`numpy.ndarray`, ndim: ``1``
        The largest pairwise distance.
    """
    if R.ndim == 2:
        R = np.array([R])
    # Finds all atom pairs.
    all_pairs = np.array(list(itertools.combinations(range(len(Z)), 2)))
    # Creates arrays of all the points for all structures.
    pair0_points = np.array([[R[i, j] for j in all_pairs[:, 0]] for i in range(len(R))])
    pair1_points = np.array([[R[i, j] for j in all_pairs[:, 1]] for i in range(len(R))])
    # Computes the distance, then largest distances of all structures.
    distances = np.linalg.norm(pair0_points - pair1_points, axis=2)
    max_distances = np.amax(distances, axis=1)
    return max_distances


def com_distance_sum(Z, R, entity_ids):
    r"""The sum of pairwise distances from each entity's center of mass to
    the total structure center of mass.

    This descriptor, :math:`L`, is defined as

    .. math::

        L = \sum_i^N l_i = \sum_i^N \Vert \mathbf{CM}_{i} - \mathbf{CM} \Vert_2

    where :math:`\mathbf{CM}` is the center of mass of the structure and
    :math:`\mathbf{CM}_i` is the center of mass of monomer :math:`i`.

    Parameters
    ----------
    Z : :obj:`numpy.ndarray`, ndim: ``1``
        Atomic numbers of all atoms in the system.
    R : :obj:`numpy.ndarray`, ndim: ``3``
        Cartesian coordinates.
    entity_ids : :obj:`numpy.ndarray`, ndim: ``1``
        A uniquely identifying integer specifying what atoms belong to
        which entities. Entities can be a related set of atoms, molecules,
        or functional group. For example, a water and methanol molecule
        could be ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.

    Returns
    -------
    :obj:`numpy.ndarray`
        (ndim: ``1``) Calculated distance metric.
    """
    if R.ndim == 2:
        R = R[None, ...]
    cm_structures = get_center_of_mass(Z, R)

    if not isinstance(entity_ids, np.ndarray):
        entity_ids = np.array(entity_ids)

    d_sum = np.zeros(R.shape[0])
    for entity_id in set(entity_ids):
        atom_idxs = np.where(entity_ids == entity_id)[0]
        cm_entity = get_center_of_mass(Z, R[:, atom_idxs])
        d_sum += np.linalg.norm(cm_structures - cm_entity, axis=1)

    return d_sum
