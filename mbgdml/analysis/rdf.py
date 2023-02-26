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

"""Compute RDF curves under periodic boundary conditions."""

import ray
import numpy as np
from ..periodic import Cell
from ..utils import gen_combs, chunk_iterable
from ..logger import GDMLLogger

log = GDMLLogger(__name__)


# Possible ray task.
def _bin_distances(R, R_idxs, atom_pairs, rdf_settings, cell_vectors):
    r"""Compute relevant RDF data for one or more structures.

    Parameters
    ----------
    R : :obj:`numpy.ndarray` or :obj:`numpy.memmap`, ndim: ``3``
        Atomic coordinates of one or more structures.
    R_idxs : :obj:`int` or :obj:`list`
        Indices of ``R`` to compute RDF contributions.
    atom_pairs : :obj:`numpy.ndarray`, ndim: ``2``
        Indices of all atom pairs to consider for each structure.
    rdf_settings : :obj:`dict`
        Keyword arguments for :func:`numpy.histogram` to bin distances.
    cell_vectors : :obj:`numpy.ndarray`
        The three periodic cell vectors. For example, a cube of length 16.0 would
        be ``[[16.0, 0.0, 0.0], [0.0, 16.0, 0.0], [0.0, 0.0, 16.0]]``.

    Returns
    -------
    :obj:`numpy.ndarray`
        Histogram count of distances.
    :obj:`float`
        Cumulative volume for this set of structures.
    :obj:`int`
        Number of structures computed here.
    """
    R = R[R_idxs]
    if R.ndim == 2:
        R = R[None, ...]
    n_R = R.shape[0]

    cell = Cell(cell_vectors, cell_vectors[0][0] / 2, True)

    # Compute histogram of distances for structure(s)
    D = R[:, atom_pairs[:, 1]] - R[:, atom_pairs[:, 0]]
    new_shape = (np.prod(D.shape[:2]), 3)
    D = D.reshape(new_shape)
    D = cell.d_mic(D, check_cutoff=False)  # Should check_cutoff be false?
    dists = np.linalg.norm(D, ord=2, axis=1).flatten()
    count, _ = np.histogram(dists, **rdf_settings)

    # Determine volume contribution.
    vol_contrib = n_R * cell.volume

    return count, vol_contrib, n_R


class RDF:
    r"""Handles calculating the radial distribution function (RDF),
    :math:`g(r)`, of a constant volume simulation.
    """

    def __init__(
        self,
        Z,
        entity_ids,
        comp_ids,
        cell_vectors,
        bin_width=0.05,
        rdf_range=(0.0, 15.0),
        inter_only=True,
        use_ray=False,
        ray_address="auto",
        n_workers=1,
    ):
        r"""
        Parameters
        ----------
        Z : :obj:`numpy.ndarray`, ndim: ``1``
            Atomic numbers of all atoms in the system.
        entity_ids : :obj:`numpy.ndarray`, ndim: ``1``
            Integers that specify which fragment each atom belongs to for all
            structures.
        comp_ids : :obj:`numpy.ndarray`, ndim: ``1``
            Labels for each ``entity_id`` used to determine the desired entity
            for RDF computations.
        cell_vectors : :obj:`numpy.ndarray`
            The three cell vectors.
        inter_only : :obj:`bool`, default: ``True``
            Only intermolecular distances are allowed. If ``True``, atoms that
            have the same ``entity_id`` are ignored.
        use_ray : :obj:`bool`, default: ``False``
            Use `ray <https://docs.ray.io/en/latest/>`__ to parallelize
            computations.
        n_workers : :obj:`int`, default: ``1``
            Total number of workers available for ray. This is ignored if ``use_ray``
            is ``False``.
        ray_address : :obj:`str`, default: ``"auto"``
            Ray cluster address to connect to.
        """
        # Store data
        self.Z = Z
        self.entity_ids = entity_ids
        self.comp_ids = comp_ids
        self.cell_vectors = cell_vectors
        self.bin_width = bin_width
        self.rdf_range = rdf_range
        self.inter_only = inter_only
        self.use_ray = use_ray
        self.n_workers = n_workers

        # Setup ray
        if use_ray:
            if not ray.is_initialized():
                log.debug("ray is not initialized")
                # Try to connect to already running ray service (from ray cli).
                try:
                    log.debug("Trying to connect to ray at address %r", ray_address)
                    ray.init(address=ray_address)
                except ConnectionError:
                    log.debug("Failed to connect to ray at %r", ray_address)
                    log.debug("Trying to initialize ray with %d cores", n_workers)
                    ray.init(num_cpus=n_workers)
                log.debug("Successfully initialized ray")
            else:
                log.debug("Ray was already initialized")
            self._max_chunk_size = 300

    def _setup(self, comp_id_pair, entity_idxs):
        r"""Prepare to do RDF computation.

        Parameters
        ----------
        comp_id_pair : :obj:`tuple`
            The component ID of the entities to consider.
        entity_idxs : :obj:`tuple`, ndim: ``1`` or ``2``
            The atom indices in each component to compute distances from.
        """
        # Setup histogram
        rdf_span = self.rdf_range[-1] - self.rdf_range[0]
        nbins = int(rdf_span / self.bin_width)
        self._hist_settings = {"bins": nbins, "range": self.rdf_range}
        count, edges = np.histogram([-1], **self._hist_settings)
        count = count.astype(np.float64)
        count *= 0.0
        self._count = count
        self.edges = edges
        self.bins = 0.5 * (edges[:-1] + edges[1:])

        # Cumulative volume for rdf normalization.
        self._cuml_volume = 0.0  # Cumulative volume.
        self._n_analyzed = 0  # Number of structures analyzed

        # Compute atom pairs indices.
        atom_sets = []
        for i, comp_id in enumerate(comp_id_pair):
            entity_idx = entity_idxs[i]  # Could contain an int or tuple
            avail_entities = np.where(comp_id == self.comp_ids)[0]
            # Convert entity_ids into atom indices
            sets = []
            for entity_id in avail_entities:
                if isinstance(entity_idx, (tuple, list)):
                    entity_idx = list(entity_idx)
                elif isinstance(entity_idx, int):
                    entity_idx = [entity_idx]
                sets.extend(
                    np.argwhere(entity_id == self.entity_ids).T[0][entity_idx].tolist()
                )
            atom_sets.append(sets)
        self._atom_sets = atom_sets

        # Invalid or unwanted values will be labeled with -1 to drop later.
        atom_pairs = np.empty(
            (len(tuple(gen_combs(atom_sets))), len(atom_sets)), dtype=np.int32
        )
        self._n_pairs, self._n_sets = atom_pairs.shape
        i = 0
        for comb in gen_combs(atom_sets):
            # Check if the pair is on the same entity if requested.
            if self.inter_only:
                if len(set(self.entity_ids[list(comb)])) == 1:
                    atom_pairs[i] = -1
                    i += 1
                    continue
            atom_pairs[i] = comb
            i += 1
        atom_pairs = atom_pairs[atom_pairs >= 0]
        self._atom_pairs = atom_pairs.reshape(
            (int(len(atom_pairs) / self._n_sets), self._n_sets)
        )

    def run(self, R, comp_id_pair, entity_idxs, step=1):
        r"""Perform the RDF computation.

        Parameters
        ----------
        R : :obj:`numpy.ndarray`, ndim: ``3``
            Cartesian coordinates of all atoms in the system under periodic
            boundary conditions.

            .. tip::
                We recommend a memory-map for large ``R`` to reduce memory
                requirements.

        comp_id_pair : :obj:`tuple`
            The component ID of the entities to consider. For example,
            ``('h2o', 'h2o')`` or ``('h2o', 'meoh')``.
        entity_idxs : :obj:`tuple`, ndim: ``1`` or ``2``
            The atom indices in each component to compute distances from.
        step : :obj:`int`
            Number of structures/frames to skip between each analyzed frame.

        Returns
        -------
        :obj:`numpy.ndarray`
            ``self.bins``: :math:`r` as the midpoint of each histogram bin.
        :obj:`numpy.ndarray`
            ``self.results``: :math:`g(r)` with respect to ``bins``.

        Examples
        --------
        Suppose we want to compute the :math:`O_{w}`-:math:`O_{m}` RDF where
        :math:`O_{w}` and :math:`O_{m}` are the oxygen atoms of water and
        methanol, respectively. We can define our system as such.

        >>> import numpy as np
        >>> Z = np.array([8, 1, 1, 8, 1, 6, 1, 1, 1]*25)
        >>> entity_ids = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]*25)
        >>> comp_ids = np.array(['h2o', 'meoh']*25)
        >>> cell_vectors = np.array([[16., 0., 0.], [0., 16., 0.], [0., 0., 16.]])

        .. note::
            The above information is an arbitrary system made up of 25 water and
            25 methanol molecules. They are contained in a 16 Angstrom periodic box
            where the atoms are always in the same order: water, methanol, water,
            methanol, water, etc.

        This information completely specifies our system to prepare for computing
        the RDF. We initialize our object with this information.

        >>> from mbgdml.analysis.rdf import RDF
        >>> rdf = RDF(Z, entity_ids, comp_ids, cell_vectors)

        From here we need to specify what RDF to compute with ``rdf.run()``.
        We assume the Cartesian coordinates for ``R`` are already loaded as
        a 3D array or memory-map.

        The last two pieces of information are ``comp_id_pair`` and
        ``entity_idxs``. ``comp_id_pair`` specifies what components or species
        we want to compute our RDF with respect to. In this example, we want
        :math:`O_{w}`-:math:`O_{m}`. ``entity_idxs`` specifies which atom in
        each entity to use. Oxygen is the first atom in both water and
        methanol (i.e., index of ``0``).

        >>> comp_id_pair = ('h2o', 'meoh')
        >>> entity_idxs = (0, 0)

        We can then compute our RDF!

        >>> bins, gr = rdf.run(R, comp_id_pair, entity_idxs)

        Notes
        -----
        ``inter_only`` only comes into play when there is a chance of also
        computing intramolecular distances during the RDF calculation. Take the
        hydroxyl OH RDF in a pure methanol simulation for instance. Our
        ``comp_id_pair`` and ``entity_idxs`` would be ``('meoh', 'meoh')`` and
        ``(0, 1)``. The O-H intramolecular bond distance would be a perfectly
        valid atom pair. Usually we are interested in intermolecular distances.
        ``inter_only`` controls whether intramolecular distances are included
        (``inter_only = False``) or not (``inter_only = True``).

        ``entity_idxs`` can specify one or more atoms for each component. For
        example, if you wanted to compute the OH RDF of pure water you could
        use ``entity_idxs = (0, (1, 2))``.

        TODO: Support different cell sizes for each structure.
        """
        self._setup(comp_id_pair, entity_idxs)

        # Computing histogram.
        # Serial operation.
        if not self.use_ray:
            for i in range(0, len(R), step):
                count, volume_contrib, n_R = _bin_distances(
                    R, i, self._atom_pairs, self._hist_settings, self.cell_vectors
                )
                self._count += count
                self._cuml_volume += volume_contrib
                self._n_analyzed += n_R
        # Parallel operation with ray.
        else:
            _bin_distances_remote = ray.remote(_bin_distances)
            chunk_size = min(
                self._max_chunk_size,
                int(len(tuple(range(0, len(R), step))) / self.n_workers),
            )
            chunker = chunk_iterable(range(0, len(R), step), chunk_size)

            R = ray.put(R)
            atom_pairs = ray.put(self._atom_pairs)
            hist_settings = ray.put(self._hist_settings)
            cell_vectors = ray.put(self.cell_vectors)

            # Initialize ray workers
            workers = []

            def add_worker(workers, chunker):
                try:
                    chunk = list(next(chunker))
                except StopIteration:
                    return
                workers.append(
                    _bin_distances_remote.remote(
                        R, chunk, atom_pairs, hist_settings, cell_vectors
                    )
                )
                return

            for _ in range(self.n_workers):
                add_worker(workers, chunker)

            while len(workers) != 0:
                done_id, workers = ray.wait(workers)

                count, volume_contrib, n_R = ray.get(done_id)[0]
                self._count += count
                self._cuml_volume += volume_contrib
                self._n_analyzed += n_R

                add_worker(workers, chunker)

        # Normalize the RDF
        norm = self._n_analyzed  # Number of analyzed frames
        # Volume in each radial shell
        vols = np.power(self.edges, 3)
        norm *= 4 / 3 * np.pi * np.diff(vols)  # Array of shape self.edges
        # Average number density
        N = self._n_pairs
        avg_volume = self._cuml_volume / self._n_analyzed
        norm *= N / avg_volume

        self.results = self._count / norm

        return self.bins, self.results
