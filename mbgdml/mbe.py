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

"""Many-body utilities."""

import itertools
import math
import numpy as np
import ray
from .utils import chunk_array, chunk_iterable, gen_combs
from .stress import to_voigt, virial_finite_diff
from .logger import GDMLLogger

log = GDMLLogger(__name__)


def gen_r_entity_combs(r_prov_spec, entity_ids_lower):
    r"""Generate ``entity_id`` combinations of a specific size for a single
    structure.

    ``r_prov_spec[2:]`` provides ``entity_ids`` of a reference structure.
    For example, consider ``r_prov_spec = [0, 0, 34, 5, 2]``.
    This tells us that the structure contains three entities, and the
    ``entity_ids`` of each are ``34``, ``5``, and ``2`` in the structure
    provenance.

    ``gen_r_entity_combs`` generates all combinations (without replacement)
    based on the number of entities in ``entity_ids_lower``.
    If ``entity_ids_lower`` is ``[0, 0, 0, 1, 1, 1]`` (i.e., number of
    lower-order entities is 2), then ``gen_r_entity_combs`` produces all 3
    choose 2 combinations.

    Parameters
    ----------
    r_prov_spec : :obj:`numpy.ndarray`, ndim: ``1``
        Structure provenance specifications of a single structure.
    entity_ids_lower : :obj:`numpy.ndarray`, ndim: ``1``
        A uniquely identifying integer specifying what atoms belong to
        which entities for lower-order structures. Entities can be a related
        set of atoms, molecules, or functional group. For example, a water and
        methanol molecule could be ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.

    Yield
    -----
    :obj:`tuple`
        Entity combinations of a single structure.
    """
    n_entities_lower = len(set(entity_ids_lower))
    entity_combs = itertools.combinations(r_prov_spec[2:], n_entities_lower)
    for comb in entity_combs:
        yield comb


def mbe_worker(
    r_shape,
    entity_ids,
    r_prov_specs,
    r_idxs,
    E_lower,
    Deriv_lower,
    entity_ids_lower,
    r_prov_specs_lower,
):
    r"""Worker for computing many-body contributions.

    This does not take into account if many-body contributions are being
    ``'added'`` or ``'removed'``. This just sums up all possible contributions.

    Parameters
    ----------
    r_shape : :obj:`tuple`, ndim: ``1``
        Shape of a single structure. For example, ``(n, 3)`` where ``n`` is the
        number of atoms.
    entity_ids : :obj:`numpy.ndarray`, ndim: ``1``
        A uniquely identifying integer specifying what atoms belong to
        which entities for lower-order structures. Entities can be a related
        set of atoms, molecules, or functional group. For example, a water and
        methanol molecule could be ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.
    r_prov_specs : :obj:`numpy.ndarray`, ndim: ``2``
        Structure provenance IDs. This specifies the ``r_prov_id``, structure
        index from the ``r_prov_id`` source, and ``entity_ids`` making up
        the structure.
    r_idxs : :obj:`list`, ndim: ``1``
        Structure indices of the original ``E`` and ``Deriv`` arrays for this
        worker.
    E_lower : :obj:`numpy.ndarray`, ndim: ``1``
        Energies of lower-order structures.
    Deriv_lower : :obj:`numpy.ndarray`, ndim: ``3``
        Potential energy surface derivatives (i.e., gradients or forces
        depending on the sign) of lower-order structures.
    entity_ids_lower : :obj:`numpy.ndarray`, ndim: ``1``
        A uniquely identifying integer specifying what atoms belong to
        which entities for lower-order structures. Entities can be a related
        set of atoms, molecules, or functional group. For example, a water and
        methanol molecule could be ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.
    r_prov_specs_lower : :obj:`numpy.ndarray`, ndim: ``2``
        Structure provenance IDs. This specifies the ``r_prov_id``, structure
        index from the ``r_prov_id`` source, and ``entity_ids`` making up
        lower-order structures.

    Returns
    -------
    :obj:`list`
        Structure indices of the original ``E`` and ``Deriv`` arrays for this
        worker.
    :obj:`numpy.ndarray`
        Many-body energy contributions for the worker structure indices.
    :obj:`numpy.ndarray`
        Many-body derivative (i.e., gradients or forces depending on the sign)
        contributions for the worker structure indices.
    """
    log.debug("MBE contributions")
    log.debug("Parent size %i", np.unique(entity_ids).shape[0])
    E = np.zeros(len(r_idxs))
    Deriv = np.zeros((len(r_idxs), *r_shape))

    # Loop through every structure in the reference data set.
    for i, i_r in enumerate(r_idxs):
        # We have to match the molecule information for each reference
        # structure to the molecules in this lower-order structure to remove
        # the right contribution.
        r_prov_spec = r_prov_specs[i_r]  # r_prov_specs of this structure.
        log.debug("Supersystem structure r_prov_spec: %r", r_prov_spec.tolist())

        # Loop through every molecular combination.
        for entity_comb in gen_r_entity_combs(r_prov_spec, entity_ids_lower):
            # r_prov_spec of the lower-order structure.
            # Combines [r_prov_id, r_idx] with the entity combination.
            r_prov_spec_lower_comb = np.block([r_prov_spec[:2], np.array(entity_comb)])
            log.debug("Fragment r_prov_spec: %r", r_prov_spec_lower_comb.tolist())

            # Index of the structure in the lower data set.
            try:
                log.debug("Searching for fragment's r_prov_spec")
                i_r_lower = np.where(
                    np.all(r_prov_specs_lower == r_prov_spec_lower_comb, axis=1)
                )[0][0]
                log.debug("Found fragment index: %d", i_r_lower)
            except IndexError as e:
                log.debug("Could not find fragment. This can be expected sometimes.")
                if "for axis 0 with size 0" in str(e):
                    continue
                raise

            deriv_lower = Deriv_lower[i_r_lower]

            # Adding or removing energy contributions.
            E[i] += E_lower[i_r_lower]
            log.debug("Fragment energy: %r", E_lower[i_r_lower])

            # Adding or removing derivative contributions.
            # We have to determine the corresponding atom indices of both the
            # reference and lower-order structure.
            # This is done by matching the entity_id_prov between the two r_prov_specs.
            # The position of entity_id_prov in r_prov_specs (with r_prov_id and
            # r_idx removed) specifies the entity_id of the

            # entity_id_prov: the original entity_id of the structure provenance.
            # entity_id: the entity_id in the reference structure.
            # entity_id_lower: the entity_id in the lower-order structure.
            # i_z: atom indices of the reference structure to add or remove
            # lower-order structure.
            # i_z_lower: atom indices of lower-order contribution.
            for entity_id_prov in r_prov_spec_lower_comb[2:]:
                entity_id = np.where(r_prov_spec[2:] == entity_id_prov)[0][0]
                i_z = np.where(entity_ids == entity_id)[0]
                log.debug("atom slice : %r", i_z.tolist())

                entity_id_lower = np.where(
                    r_prov_spec_lower_comb[2:] == entity_id_prov
                )[0][0]
                i_z_lower = np.where(entity_ids_lower == entity_id_lower)[0]

                Deriv[i][i_z] += deriv_lower[i_z_lower]

            log.debug("Fragment successfully accounted for")

    return r_idxs, E, Deriv


def gen_r_idxs_worker(r_prov_specs, r_prov_ids_lower, n_workers):
    r"""Generates the assigned structures for each worker.

    Parameters
    ----------
    r_prov_specs : :obj:`numpy.ndarray`, ndim: ``2``
        Structure provenance IDs. This specifies the ``r_prov_id``, structure
        index from the ``r_prov_id`` source, and ``entity_ids`` making up
        the structure.
    r_prov_ids_lower : :obj:`dict` {:obj:`int`: :obj:`str`}
        Species an ID (:obj:`int`) to uniquely identifying labels for each
        lower-order structure if it originated from another source. Labels
        should always be ``md5_structures``. For example,
        ``{0: '6038e101da7fc0085978741832ebc7ad',
        1: 'eeaf93dec698de3ecb55e9292bd9dfcb'}``.
    n_workers : :obj:`int`
        Number of parallel workers. Can range from ``1`` to the number of CPUs
        available.

    Yields
    ------
    :obj:`numpy.ndarray`
        Structure indices for the worker.
    """
    # Select all structures that we can remove contributions based on
    # r_prov_ids in the lower contributions.
    r_idxs = []
    for r_prov_id_lower in r_prov_ids_lower.keys():
        r_idxs.extend(np.where(r_prov_specs[:, 0] == r_prov_id_lower)[0].tolist())

    task_size = math.ceil(len(r_idxs) / n_workers)
    for i in range(0, len(r_idxs), task_size):
        r_idxs_worker = r_idxs[i : i + task_size]
        yield r_idxs_worker


# pylint: disable=too-many-branches, too-many-statements
def mbe_contrib(
    E,
    Deriv,
    entity_ids,
    r_prov_ids,
    r_prov_specs,
    E_lower,
    Deriv_lower,
    entity_ids_lower,
    r_prov_ids_lower,
    r_prov_specs_lower,
    operation="remove",
    use_ray=False,
    ray_address="auto",
    n_workers=1,
):
    r"""Adds or removes energy and derivative (i.e., gradients or forces)
    contributions from a reference.

    We use the term "lower" to refer to the lower-order (i.e., smaller) systems.
    These are the energy and derivative contributions to remove or add to a
    reference.

    Making :math:`n`-body predictions (i.e., ``operation = "add"``) will often
    not have ``r_prov_ids`` or ``r_prov_specs`` as all lower contributions are
    derived exclusively from these structures. Use :obj:`None` for both of these
    and this function will assume that all ``_lower`` properties apply.

    Parameters
    ----------
    E : :obj:`numpy.ndarray`, ndim: ``1``
        Energies to add or remove contributions from (i.e., reference).
    Deriv : :obj:`numpy.ndarray`, ndim: ``3``
        Potential energy surface derivatives (i.e., gradients or forces
        depending on the sign) of reference structures.
    entity_ids : :obj:`numpy.ndarray`, ndim: ``1``
        A uniquely identifying integer specifying what atoms belong to
        which entities for reference structures. Entities can be a related
        set of atoms, molecules, or functional group. For example, a water and
        methanol molecule could be ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.
    r_prov_ids : :obj:`dict` {:obj:`int`: :obj:`str`} or :obj:`None`
        Species an ID (:obj:`int`) to uniquely identifying labels for each
        structure if it originated from another source. Labels should
        always be ``md5_structures``. For example,
        ``{0: '6038e101da7fc0085978741832ebc7ad', 1:
        'eeaf93dec698de3ecb55e9292bd9dfcb'}``.
    r_prov_specs : :obj:`numpy.ndarray`, ndim: ``2`` or :obj:`None`
        Structure provenance IDs. This specifies the ``r_prov_id``, structure
        index from the ``r_prov_id`` source, and ``entity_ids`` making up
        the structure.
    E_lower : :obj:`numpy.ndarray`, ndim: ``1``
        Lower-order energies.
    Deriv_lower : :obj:`numpy.ndarray`, ndim: ``3``
        Potential energy surface derivatives (i.e., gradients or forces
        depending on the sign) of lower-order structures.
    entity_ids_lower : :obj:`numpy.ndarray`, ndim: ``1``
        A uniquely identifying integer specifying what atoms belong to
        which entities for lower-order structures. Entities can be a related
        set of atoms, molecules, or functional group. For example, a water and
        methanol molecule could be ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.
    r_prov_ids_lower : :obj:`dict` {:obj:`int`: :obj:`str`}
        Species an ID (:obj:`int`) to uniquely identifying labels for each
        lower-order structure if it originated from another source. Labels
        should always be ``md5_structures``. For example,
        ``{0: '6038e101da7fc0085978741832ebc7ad',
        1: 'eeaf93dec698de3ecb55e9292bd9dfcb'}``.
    r_prov_specs_lower : :obj:`numpy.ndarray`, ndim: ``2``
        Structure provenance IDs. This specifies the ``r_prov_id``, structure
        index from the ``r_prov_id`` source, and ``entity_ids`` making up
        lower-order structures.
    operation : :obj:`str`, default: ``'remove'``
        ``'add'`` or ``'remove'`` the contributions.
    use_ray : :obj:`bool`, default: ``False``
        Use `ray <https://docs.ray.io/en/latest/>`__ to parallelize
        computations.
    n_workers : :obj:`int`, default: ``1``
        Total number of workers available for ray. This is ignored if ``use_ray``
        is ``False``.
    ray_address : :obj:`str`, default: ``"auto"``
        Ray cluster address to connect to.

    Returns
    -------
    :obj:`numpy.ndarray`
        Energies with lower-order contributions added or removed.
    :obj:`numpy.ndarray`
        Derivatives with lower-order contributions added or removed.
    """
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
    else:
        log.debug("Not using ray")

    log.debug("Computing many-body expansion contributions")
    if operation not in ("add", "remove"):
        raise ValueError(f'{operation} is not "add" or "remove"')
    log.debug("MBE operation : %s", operation)

    # Checks that the r_prov_md5 hashes match the same r_prov_id
    if (r_prov_ids is not None) and (r_prov_ids != {}):
        log.debug("Checking if r_prov_ids match")
        for r_prov_id_lower, r_prov_md5_lower in r_prov_ids_lower.items():
            assert r_prov_md5_lower == r_prov_ids[r_prov_id_lower]
        log.debug("All r_prov_ids match")
    # Assume that all lower models apply.
    else:
        log.debug("No r_prov_ids exist")
        assert r_prov_specs is None or r_prov_specs.shape == (1, 0)
        assert len(set(r_prov_specs_lower[:, 0])) == 1
        n_r = len(E)
        n_entities = len(set(entity_ids))
        r_prov_specs = np.empty((n_r, 2 + n_entities), dtype=int)
        r_prov_specs[:, 0] = r_prov_specs_lower[0][0]
        r_prov_specs[:, 1] = np.array(list(range(n_r)))
        for entity_id in range(n_entities):
            r_prov_specs[:, entity_id + 2] = entity_id

    r_shape = Deriv.shape[1:]
    log.debug("Shape of example structure: %r", r_shape)

    if not use_ray:
        log.debug("use_ray is False (using serial operation)")
        r_idxs = next(gen_r_idxs_worker(r_prov_specs, r_prov_ids_lower, 1))
        r_idxs_worker, E_worker, Deriv_worker = mbe_worker(
            r_shape,
            entity_ids,
            r_prov_specs,
            r_idxs,
            E_lower,
            Deriv_lower,
            entity_ids_lower,
            r_prov_specs_lower,
        )
        if operation == "remove":
            E[r_idxs_worker] -= E_worker
            Deriv[r_idxs_worker] -= Deriv_worker
        elif operation == "add":
            E[r_idxs_worker] += E_worker
            Deriv[r_idxs_worker] += Deriv_worker
    else:
        log.debug("use_ray is True")
        if not ray.is_initialized():
            log.debug("ray is not initialized")
            # Try to connect to already running ray service (from ray cli).
            try:
                log.debug("Trying to connect to ray at address %r", ray_address)
                ray.init(address=ray_address)
            except ConnectionError:
                ray.init(num_cpus=n_workers)

        # Generate all workers.
        worker = ray.remote(mbe_worker)
        worker_list = []
        for r_idxs in gen_r_idxs_worker(r_prov_specs, r_prov_ids_lower, n_workers):
            worker_list.append(
                worker.options(num_cpus=1).remote(
                    r_shape,
                    entity_ids,
                    r_prov_specs,
                    r_idxs,
                    E_lower,
                    Deriv_lower,
                    entity_ids_lower,
                    r_prov_specs_lower,
                )
            )

        # Run all workers.
        while len(worker_list) != 0:
            done_id, worker_list = ray.wait(worker_list)

            r_idxs_worker, E_worker, Deriv_worker = ray.get(done_id)[0]
            log.debug("Worker %r has finished", done_id)
            if operation == "remove":
                E[r_idxs_worker] -= E_worker
                Deriv[r_idxs_worker] -= Deriv_worker
            elif operation == "add":
                E[r_idxs_worker] += E_worker
                Deriv[r_idxs_worker] += Deriv_worker

    return E, Deriv


def decomp_to_total(
    E_nbody,
    F_nbody,
    entity_ids,
    entity_combs,
    use_ray=False,
    n_workers=1,
    ray_address="auto",
):
    r"""Sum decomposed :math:`n`-body energies and forces for total
    :math:`n`-body contribution.

    This is a wrapper around :func:`mbgdml.mbe.mbe_contrib`.

    Parameters
    ----------
    E_nbody : :obj:`numpy.ndarray`, ndim: ``1``
        All :math:`n`-body energies. :obj:`numpy.nan` values are allowed for
        structures that were beyond the descriptor cutoff.
    F_nbody : :obj:`numpy.ndarray`, ndim: ``3``
        All :math:`n`-body energies. :obj:`numpy.nan` values are allowed for
        structures that were beyond the descriptor cutoff.
    entity_ids : :obj:`numpy.ndarray`, ndim: ``1``
        Integers specifying which atoms belong to which entities for the
        supersystem (not the :math:`n`-body structure).
    entity_combs : :obj:`numpy.ndarray`, ndim: ``2``
        Structure indices and Entity IDs of all structures in increasing
        order of :math:`n`. Column 0 is the structure index and
        the remaining columns are entity IDs.
    use_ray : :obj:`bool`, default: ``False``
        Use `ray <https://docs.ray.io/en/latest/>`__ to parallelize
        computations.
    n_workers : :obj:`int`, default: ``1``
        Total number of workers available for ray. This is ignored if ``use_ray``
        is ``False``.
    ray_address : :obj:`str`, default: ``"auto"``
        Ray cluster address to connect to.

    Returns
    -------
    :obj:`numpy.ndarray`, ndim: ``1``
        Total :math:`n`-body energies.
    """
    # Allocating arrays
    n_atoms = len(entity_ids)
    n_r = len(np.unique(entity_combs[:, 0]))
    E = np.zeros((n_r,))
    F = np.zeros((n_r, n_atoms, 3))

    entity_ids_lower = []
    for i in range(len(entity_combs[0][1:])):
        entity_ids_lower.extend([i for j in range(np.count_nonzero(entity_ids == i))])
    entity_ids_lower = np.array(entity_ids_lower)

    r_prov_specs_lower = np.zeros((len(entity_combs), 1))
    r_prov_specs_lower = np.hstack((r_prov_specs_lower, entity_combs))

    E_nbody = np.nan_to_num(E_nbody, copy=False, nan=0.0)
    F_nbody = np.nan_to_num(F_nbody, copy=False, nan=0.0)

    E, F = mbe_contrib(
        E,
        F,
        entity_ids,
        None,
        None,
        E_nbody,
        F_nbody,
        entity_ids_lower,
        {0: ""},
        r_prov_specs_lower,
        operation="add",
        use_ray=use_ray,
        n_workers=n_workers,
        ray_address=ray_address,
    )
    return E, F


class mbePredict:
    r"""Predict energies and forces of structures using machine learning
    many-body models.

    This can be parallelized with ray but needs to be initialized with
    the ray cli or a script using this class. Note that initializing ray tasks
    comes with some overhead and can make smaller computations much slower.
    Only GDML models can run in parallel.
    """

    def __init__(
        self,
        models,
        predict_model,
        use_ray=False,
        n_workers=1,
        ray_address="auto",
        wkr_chunk_size=None,
        alchemy_scalers=None,
        periodic_cell=None,
        compute_stress=False,
    ):
        r"""
        Parameters
        ----------
        models : :obj:`list` of :obj:`mbgdml.models.Model`
            Machine learning model objects that contain all information to make
            predictions using ``predict_model``.
        predict_model : ``callable``
            A function that takes ``Z``, ``R``, ``entity_ids``, ``nbody_gen``, ``model``
            and computes energies and forces. This will be turned into a ray remote
            function if ``use_ray`` is ``True``. This can return total properties
            or all individual :math:`n`-body energies and forces.
        use_ray : :obj:`bool`, default: ``False``
            Use `ray <https://docs.ray.io/en/latest/>`__ to parallelize
            computations.
        n_workers : :obj:`int`, default: ``1``
            Total number of workers available for ray. This is ignored if ``use_ray``
            is ``False``.
        ray_address : :obj:`str`, default: ``"auto"``
            Ray cluster address to connect to.
        wkr_chunk_size : :obj:`int`, default: :obj:`None`
            Number of :math:`n`-body structures to assign to each spawned
            worker with ray. If :obj:`None`, it will divide up the number of
            predictions by ``n_workers``.
        alchemical_scaling : :obj:`list` of :class:`~mbgdml.alchemy.mbeAlchemyScale`, \
        default: :obj:`None`
            Alchemical scaling of :math:`n`-body interactions of entities.

            .. warning::

                This has not been thoroughly tested.
        periodic_cell : :obj:`mbgdml.periodic.Cell`, default: :obj:`None`
            Use periodic boundary conditions defined by this object. If this
            is not :obj:`None` only :meth:`~mbgdml.mbe.mbePredict.predict` can be used.
        compute_stress : :obj:`bool`, default: ``False``

            .. danger::

                This implementation is experimental and has not been verified. We do
                not recommend using this.

            Compute the internal virial contribution,
            :math:`\mathbf{W} \left( \mathbf{r}^N \right) / V`, of :math:`N`
            particles at positions :math:`\mathbf{r}` to the pressure stress tensor of
            a periodic box with volume :math:`V`. The kinetic contribution is not
            computed here.
        """
        self.models = models
        self.predict_model = predict_model

        if models[0].type == "gap":
            assert not use_ray
        elif models[0].type == "schnet":
            assert not use_ray

        self.use_ray = use_ray  # Do early because some setters are dependent on this.
        self.n_workers = n_workers
        self.wkr_chunk_size = wkr_chunk_size
        if alchemy_scalers is None:
            alchemy_scalers = []
        self.alchemy_scalers = alchemy_scalers
        self.periodic_cell = periodic_cell
        self.compute_stress = compute_stress

        self.virial_form = "group"
        r"""The form to use from
        `10.1063/1.3245303 <https://doi.org/10.1063/1.3245303>`__ to compute the
        internal virial contribution to the pressure stress tensor.
        :math:`\mathbf{W} \left( \mathbf{r}^N \right)` is the internal virial of
        :math:`N` atoms and :math:`\otimes` is the outer tensor product (i.e.,
        ``np.multiply.outer``).

        ``group`` (**default**)

            This computes the virial by considering contributions based on groups
            of atoms. The number of groups, number of atoms in each group, and
            number of groups is completely arbitrary. Thus, we can straightforwardly
            use :math:`n`-body combinations as groups and get more insight into
            the virial contributions origin.

            .. math::

                \mathbf{W} \left( \mathbf{r}^N \right) =
                \sum_{k \in \mathbf{0}} \sum_{w = 1}^{N_k} \mathbf{r}_w^k
                \otimes \mathbf{F}_w^k.

            Here, :math:`k` is a group of atoms that must be in the local cell
            (i.e., :math:`k \in \mathbf{0}`). :math:`w` is the atom index within
            group :math:`k` (:math:`N_k` is the total number of atoms in the group).

            .. important::

                This has to be specifically implemented in the relevant predictor
                functions.

        ``finite_diff``

            Uses finite differences by perturbing the cell vectors.

        :type: :obj:`str`
        """
        self.finite_diff_dh = 1e-4
        r"""Forward and backward displacement of the cell vectors for finite
        differences.

        Default: ``1e-4``

        :type: :obj:`float`
        """
        self.use_voigt = False
        r"""Convert the stress tensor to the 1D Voigt notation (xx, yy, zz, yz, xz, xy).

        Default: ``False``

        :type: :obj:`bool`
        """
        self.only_normal_stress = False
        r"""Only compute normal (xx, yy, and zz) stress. All other elements will be
        zero. This is recommended for MD simulations to avoid altering box angular
        momentum due to the antisymmetric contributions (yz, xz, and xy).

        Default: ``False``

        :type: :obj:`bool`
        """
        self.box_scaling_type = "anisotropic"
        r"""Treatment of box scaling when
        :attr:`mbgdml.mbe.mbePredict.only_normal_stress` is ``True``.

        ``anisotropic`` (**default**)

            No modifications are made to normal stresses which allows box vectors
            to scale independently.

        ``isotropic``

            Average the normal stresses so box vectors will scale identically.

        :type: :obj:`str`
        """

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

            self.models = [ray.put(model) for model in models]
            # if alchemy_scalers is not None:
            #    self.alchemy_scalers = [
            #        ray.put(scaler) for scaler in alchemy_scalers
            #    ]
            self.predict_model = ray.remote(predict_model)

    @property
    def periodic_cell(self):
        r"""Periodic cell for MBE predictions.

        :type: :obj:`mbgdml.periodic.Cell`
        """
        if hasattr(self, "_periodic_cell"):
            return self._periodic_cell

        return None

    @periodic_cell.setter
    def periodic_cell(self, var):
        if var is not None:
            if self.use_ray:
                var = ray.put(var)
        self._periodic_cell = var

    def get_avail_entities(self, comp_ids_r, comp_ids_model):
        r"""Determines available ``entity_ids`` for each ``comp_id`` in a
        structure.

        Parameters
        ----------
        comp_ids_r : :obj:`numpy.ndarray`, ndim: ``1``
            Component IDs of the structure to predict.
        comp_ids_model : :obj:`numpy.ndarray`, ndim: ``1``
            Component IDs of the model.

        Returns
        -------
        :obj:`list`
            (length: ``len(comp_ids_r)``)
        """
        # Models have a specific entity order that needs to be conserved in the
        # predictions. Here, we create a ``avail_entity_ids`` list where each item
        # is a list of all entities in ``r`` that match the model entity.
        avail_entity_ids = []
        for comp_id in comp_ids_model:
            matching_entity_ids = np.where(comp_id == comp_ids_r)[0]
            avail_entity_ids.append(matching_entity_ids)
        return avail_entity_ids

    # pylint: disable=too-many-branches, too-many-statements
    def compute_nbody(self, Z, R, entity_ids, comp_ids, model):
        r"""Compute all :math:`n`-body contributions of a single structure
        using a :obj:`mbgdml.models.Model` object.

        When ``use_ray = True``, this acts as a driver that spawns ray tasks of
        the ``predict_model`` function.

        Parameters
        ----------
        Z : :obj:`numpy.ndarray`, ndim: ``1``
            Atomic numbers of all atoms in the system with respect to ``r``.
        R : :obj:`numpy.ndarray`, shape: ``(len(Z), 3)``
            Cartesian coordinates of a single structure.
        entity_ids : :obj:`numpy.ndarray`, shape: ``(len(Z),)``
            Integers specifying which atoms belong to which entities.
        comp_ids : :obj:`numpy.ndarray`, shape: ``(len(entity_ids),)``
            Relates each ``entity_id`` to a fragment label. Each item's index
            is the label's ``entity_id``.
        model : :obj:`mbgdml.models.Model`
            Model that contains all information needed by ``model_predict``.

        Returns
        -------
        :obj:`float`
            Total :math:`n`-body energy of ``r``.
        :obj:`numpy.ndarray`
            (shape: ``(len(Z), 3)``) - Total :math:`n`-body atomic forces of ``r``.
        :obj:`numpy.ndarray`
            (optional, shape: ``(len(Z), 3)``) - The internal virial
            contribution to the pressure stress tensor in units of energy.
        """
        # Unify r shape.
        if R.ndim == 3:
            log.debug("R has three dimensions (instead of two)")
            if R.shape[0] == 1:
                log.debug("R[0] was selected to proceed")
                R = R[0]
            else:
                raise ValueError("R.ndim is not 2 (only one structure is allowed)")

        # Creates a generator for all possible n-body combinations regardless
        # of cutoffs.
        if self.use_ray:
            model_comp_ids = ray.get(model).comp_ids
        else:
            model_comp_ids = model.comp_ids
        avail_entity_ids = self.get_avail_entities(comp_ids, model_comp_ids)
        log.debug("Available entity IDs: %r", avail_entity_ids)
        nbody_gen = gen_combs(avail_entity_ids)

        kwargs_pred = {}
        if self.alchemy_scalers is not None:
            nbody_order = len(model_comp_ids)
            kwargs_pred["alchemy_scalers"] = [
                alchemy_scaler
                for alchemy_scaler in self.alchemy_scalers
                if alchemy_scaler.order == nbody_order
            ]

        periodic_cell = self.periodic_cell
        compute_stress = (
            self.compute_stress and bool(periodic_cell) and self.virial_form == "group"
        )
        if compute_stress:
            virial = np.zeros((3, 3), dtype=np.float64)
            kwargs_pred["compute_virial"] = True

        # Runs the predict_model function to calculate all n-body energies
        # with this model.
        if not self.use_ray:
            E, F, *virial_model = self.predict_model(
                Z, R, entity_ids, nbody_gen, model, periodic_cell, **kwargs_pred
            )
            if compute_stress:
                virial += virial_model[0]
        else:
            E = 0.0
            F = np.zeros(R.shape, dtype=np.float64)

            # Put all common data into the ray object store.
            Z = ray.put(Z)
            R = ray.put(R)
            entity_ids = ray.put(entity_ids)

            nbody_gen = tuple(nbody_gen)
            if self.wkr_chunk_size is None:
                wkr_chunk_size = math.ceil(len(nbody_gen) / self.n_workers)
            else:
                wkr_chunk_size = self.wkr_chunk_size
            nbody_chunker = chunk_iterable(nbody_gen, wkr_chunk_size)
            workers = []

            # Initialize workers
            predict_model = self.predict_model

            def add_worker(workers, chunk):
                workers.append(
                    predict_model.remote(
                        Z,
                        R,
                        entity_ids,
                        chunk,
                        model,
                        periodic_cell,
                        **kwargs_pred,
                    )
                )

            for _ in range(self.n_workers):
                try:
                    chunk = next(nbody_chunker)
                    add_worker(workers, chunk)
                except StopIteration:
                    break

            # Start workers and collect results.
            while len(workers) != 0:
                done_id, workers = ray.wait(workers)
                E_worker, F_worker, *virial_model = ray.get(done_id)[0]

                E += E_worker
                F += F_worker
                if compute_stress:
                    virial += virial_model[0]

                try:
                    chunk = next(nbody_chunker)
                    add_worker(workers, chunk)
                except StopIteration:
                    pass

            # Cleanup object store
            del Z, entity_ids

        if compute_stress:
            return E, F, virial
        return E, F

    # pylint: disable=too-many-branches, too-many-statements
    def compute_nbody_decomp(self, Z, R, entity_ids, comp_ids, model):
        r"""Compute all :math:`n`-body contributions of a single structure
        using a :obj:`mbgdml.models.Model` object.

        Stores all individual entity ID combinations, energies and forces.
        This is more memory intensive. Structures that fall outside the
        descriptor cutoff will have :obj:`numpy.nan` as their energy and forces.

        When ``use_ray = True``, this acts as a driver that spawns ray tasks of
        the ``predict_model`` function.

        Parameters
        ----------
        Z : :obj:`numpy.ndarray`, ndim: ``1``
            Atomic numbers of all atoms in the system with respect to ``r``.
        r : :obj:`numpy.ndarray`, shape: ``(len(Z), 3)``
            Cartesian coordinates of a single structure.
        entity_ids : :obj:`numpy.ndarray`, shape: ``(len(Z),)``
            Integers specifying which atoms belong to which entities.
        comp_ids : :obj:`numpy.ndarray`, shape: ``(len(entity_ids),)``
            Relates each ``entity_id`` to a fragment label. Each item's index
            is the label's ``entity_id``.
        model : :obj:`mbgdml.models.Model`
            Model that contains all information needed by ``model_predict``.

        Returns
        -------
        :obj:`numpy.ndarray`
            (ndim: ``1``) -
            Energies of all possible :math:`n`-body structures. Some elements
            can be :obj:`numpy.nan` if they are beyond the descriptor cutoff.
        :obj:`numpy.ndarray`
            (ndim: ``3``) -
            Atomic forces of all possible :math:`n`-body structure. Some
            elements can be :obj:`numpy.nan` if they are beyond the descriptor
            cutoff.
        :obj:`numpy.ndarray`
            (ndim: ``2``) - All possible entity IDs.
        """
        # Unify r shape.
        if R.ndim == 3:
            log.debug("R has three dimensions (instead of two)")
            if R.shape[0] == 1:
                log.debug("R[0] was selected to proceed")
                R = R[0]
            else:
                raise ValueError("R.ndim is not 2 (only one structure is allowed)")

        # Creates a generator for all possible n-body combinations regardless
        # of cutoffs.
        if self.use_ray:
            model_comp_ids = ray.get(model).comp_ids
        else:
            model_comp_ids = model.comp_ids
        avail_entity_ids = self.get_avail_entities(comp_ids, model_comp_ids)
        nbody_gen = gen_combs(avail_entity_ids)

        # Explicitly evaluate n-body generator.
        comb0 = next(nbody_gen)
        n_entities = len(comb0)
        entity_combs = np.fromiter(itertools.chain.from_iterable(nbody_gen), np.int64)
        # Reshape and add initial combination back.
        if len(comb0) == 1:
            entity_combs.shape = len(entity_combs), 1
        else:
            entity_combs.shape = int(len(entity_combs) / n_entities), n_entities
        entity_combs = np.vstack((np.array(comb0), entity_combs))

        # Runs the predict_model function to calculate all n-body energies
        # with this model.
        if not self.use_ray:
            E, F, entity_combs = self.predict_model(
                Z, R, entity_ids, entity_combs, model
            )
        else:
            if entity_combs.ndim == 1:
                n_atoms = np.count_nonzero(entity_ids == entity_combs[0])
            else:
                n_atoms = 0
                for i in entity_combs[0]:
                    n_atoms += np.count_nonzero(entity_ids == i)

            # Put all common data into the ray object store.
            Z = ray.put(Z)
            R = ray.put(R)
            entity_ids = ray.put(entity_ids)

            # Allocate memory for energies and forces.
            E = np.empty(len(entity_combs), dtype=np.float64)
            F = np.empty((len(entity_combs), n_atoms, 3), dtype=np.float64)
            E[:] = np.nan
            F[:] = np.nan

            if self.wkr_chunk_size is None:
                wkr_chunk_size = math.ceil(len(entity_combs) / self.n_workers)
            else:
                wkr_chunk_size = self.wkr_chunk_size
            nbody_chunker = chunk_array(entity_combs, wkr_chunk_size)
            workers = []

            # Initialize workers
            predict_model = self.predict_model

            def add_worker(workers, chunk):
                workers.append(predict_model.remote(Z, R, entity_ids, chunk, model))

            for _ in range(self.n_workers):
                try:
                    chunk = next(nbody_chunker)
                    add_worker(workers, chunk)
                except StopIteration:
                    break

            # Start workers and collect results.
            while len(workers) != 0:
                done_id, workers = ray.wait(workers)
                E_worker, F_worker, entity_combs_wkr = ray.get(done_id)[0]

                i_start = np.where((entity_combs_wkr[0] == entity_combs).all(1))[0][0]
                i_end = i_start + len(entity_combs_wkr)
                E[i_start:i_end] = E_worker
                F[i_start:i_end] = F_worker

                try:
                    chunk = next(nbody_chunker)
                    add_worker(workers, chunk)
                except StopIteration:
                    pass

        return E, F, entity_combs

    def predict(self, Z, R, entity_ids, comp_ids):
        r"""Predict the energies and forces of one or multiple structures.

        Parameters
        ----------
        Z : :obj:`numpy.ndarray`, ndim: ``1``
            Atomic numbers of all atoms in the system with respect to ``R``.
        R : :obj:`numpy.ndarray`, shape: ``(N, len(Z), 3)``
            Cartesian coordinates of ``N`` structures to predict.
        entity_ids : :obj:`numpy.ndarray`, shape: ``(N,)``
            Integers specifying which atoms belong to which entities.
        comp_ids : :obj:`numpy.ndarray`, shape: ``(N,)``
            Relates each ``entity_id`` to a fragment label. Each item's index
            is the label's ``entity_id``.

        Returns
        -------
        :obj:`numpy.ndarray`
            (shape: ``(N,)``) - Predicted many-body energy
        :obj:`numpy.ndarray`
            (shape: ``(N, len(Z), 3)``) - Predicted atomic forces.
        :obj:`numpy.ndarray`
            (optional, shape: ``(N, len(Z), 3)``) - The pressure stress tensor in units
            of energy/distance\ :sup:`3`.
        """
        t_predict = log.t_start()
        if R.ndim == 2:
            R = np.array([R])
        n_R = R.shape[0]
        compute_stress = self.compute_stress and bool(self.periodic_cell)

        # Preallocate memory for energies and forces.
        E = np.zeros((n_R,), dtype=np.float64)
        F = np.zeros(R.shape, dtype=np.float64)
        if compute_stress:
            assert self.virial_form in ("group", "finite_diff")
            stress = np.zeros((n_R, 3, 3), dtype=np.float64)

        # Compute all energies and forces with every model.
        for i, r in enumerate(R):
            for model in self.models:
                # Extra returns are present for stress. The *stress_nbody captures it.
                # If it is not requested, it will just be an empty list.
                E_nbody, F_nbody, *virial_nbody = self.compute_nbody(
                    Z, r, entity_ids, comp_ids, model
                )
                E[i] += E_nbody
                F[i] += F_nbody
                if compute_stress and self.virial_form == "group":
                    stress[i] += virial_nbody[0]

            if compute_stress and self.virial_form == "finite_diff":
                periodic_cell = self.periodic_cell
                if self.use_ray:
                    periodic_cell = ray.get(periodic_cell)
                cell_v = periodic_cell.cell_v
                stress[i] = virial_finite_diff(
                    Z,
                    r,
                    entity_ids,
                    comp_ids,
                    cell_v,
                    self,
                    only_normal_stress=self.only_normal_stress,
                )

        if compute_stress:  # pylint: disable=too-many-nested-blocks
            periodic_cell = self.periodic_cell
            if self.use_ray:
                periodic_cell = ray.get(periodic_cell)
            stress /= periodic_cell.volume

            if self.only_normal_stress:
                for n in range(stress.shape[0]):
                    for i in range(0, 3):
                        for j in range(0, 3):
                            if i != j:
                                stress[n][i][j] = 0.0

                    if self.box_scaling_type == "isotropic":
                        stress_average = np.trace(stress[n]) / 3
                        for i in range(0, 3):
                            stress[n][i][i] = stress_average

            if self.use_voigt:
                stress = np.array([to_voigt(r_stress) for r_stress in stress])

            log.t_stop(
                t_predict, message="Predictions took {time} s", precision=3, level=10
            )
            return E, F, stress

        log.t_stop(
            t_predict, message="Predictions took {time} s", precision=3, level=10
        )
        return E, F

    def predict_decomp(self, Z, R, entity_ids, comp_ids):
        r"""Predict the energies and forces of one or multiple structures.

        Parameters
        ----------
        Z : :obj:`numpy.ndarray`, ndim: ``1``
            Atomic numbers of all atoms in the system with respect to ``R``.
        R : :obj:`numpy.ndarray`, shape: ``(N, len(Z), 3)``
            Cartesian coordinates of ``N`` structures to predict.
        entity_ids : :obj:`numpy.ndarray`, shape: ``(N,)``
            Integers specifying which atoms belong to which entities.
        comp_ids : :obj:`numpy.ndarray`, shape: ``(N,)``
            Relates each ``entity_id`` to a fragment label. Each item's index
            is the label's ``entity_id``.

        Returns
        -------
        :obj:`list`
            :math:`n`-body energies of all structures in increasing order of
            :math:`n`.
        :obj:`list`
            :math:`n`-body forces of all structures in increasing order of
            :math:`n`.
        :obj:`list`
            Structure indices and Entity IDs of all structures in increasing
            order of :math:`n`. Column 0 is the structure index in ``R``, and
            the remaining columns are entity IDs.
        :obj:`list`
            :math:`n`-body orders of each returned item.
        """
        t_predict = log.t_start()
        if R.ndim == 2:
            R = np.array([R])

        E_data, F_data, entity_comb_data = [], [], []

        # We want to perform all same order n-body contributions.
        if self.use_ray:
            models_ = [ray.get(model) for model in self.models]
        else:
            models_ = self.models

        model_nbody_orders = [model.nbody_order for model in models_]
        nbody_orders = sorted(set(model_nbody_orders))
        for nbody_order in nbody_orders:
            E_nbody, F_nbody, entity_comb_nbody = [], [], []
            nbody_models = []
            for i in range(len(model_nbody_orders)):
                if models_[i].nbody_order == nbody_order:
                    nbody_models.append(self.models[i])

            for model in nbody_models:
                for i, r in enumerate(R):
                    E, F, entity_combs = self.compute_nbody_decomp(
                        Z, r, entity_ids, comp_ids, model
                    )
                    entity_combs = np.hstack(
                        (
                            np.array([[i] for _ in range(len(entity_combs))]),
                            entity_combs,
                        )
                    )
                    E_nbody.extend(E)
                    F_nbody.extend(F)
                    entity_comb_nbody.extend(entity_combs)

            E_data.append(np.array(E_nbody))
            F_data.append(np.array(F_nbody))
            entity_comb_data.append(np.array(entity_comb_nbody))
        log.t_stop(
            t_predict,
            message="Decomposed predictions took {time} s",
            precision=3,
            level=10,
        )
        return E_data, F_data, entity_comb_data, nbody_orders
