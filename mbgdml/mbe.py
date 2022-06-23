# MIT License
# 
# Copyright (c) 2022, Alex M. Maldonado
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

def gen_r_entity_combs(r_prov_spec, entity_ids_lower):
    """Generate ``entity_id`` combinations of a specific size for a single
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
    entity_combs = itertools.combinations(
        r_prov_spec[2:], n_entities_lower
    )
    for comb in entity_combs:
        yield comb

def mbe_worker(
    r_shape, entity_ids, r_prov_specs, r_idxs, E_lower, Deriv_lower,
    entity_ids_lower, r_prov_specs_lower
):
    """Work for computing many-body contributions.

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
    E = np.zeros(len(r_idxs))
    Deriv = np.zeros((len(r_idxs), *r_shape))
    
    # Loop through every structure in the reference data set.
    for i in range(len(r_idxs)):
        i_r = r_idxs[i]
        # We have to match the molecule information for each reference
        # structure to the molecules in this lower-order structure to remove
        # the right contribution.
        r_prov_spec = r_prov_specs[i_r]  # r_prov_specs of this structure.

        # Loop through every molecular combination.
        for entity_comb in gen_r_entity_combs(r_prov_spec, entity_ids_lower):
            # r_prov_spec of the lower-order structure.
            # Combines [r_prov_id, r_idx] with the entity combination.
            r_prov_spec_lower_comb = np.block(
                [r_prov_spec[:2], np.array(entity_comb)]
            )
            
            # Index of the structure in the lower data set.
            i_r_lower = np.where(
                np.all(r_prov_specs_lower == r_prov_spec_lower_comb, axis=1)
            )[0][0]

            deriv_lower = Deriv_lower[i_r_lower]

            # Adding or removing energy contributions.
            E[i] += E_lower[i_r_lower]
            
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

                entity_id_lower = np.where(
                    r_prov_spec_lower_comb[2:] == entity_id_prov
                )[0][0]
                i_z_lower = np.where(
                    entity_ids_lower == entity_id_lower
                )[0]
                
                Deriv[i][i_z] += deriv_lower[i_z_lower]
                
    return r_idxs, E, Deriv

def gen_r_idxs_worker(r_prov_specs, r_prov_ids_lower, n_workers):
    """Generates the assigned structures for each worker.

    Parameters
    ----------
    r_prov_specs : :obj:`numpy.ndarray`, ndim: ``2``
        Structure provenance IDs. This specifies the ``r_prov_id``, structure
        index from the ``r_prov_id`` source, and ``entity_ids`` making up
        the structure.
    r_prov_ids_lower : :obj:`dict` {:obj:`int`: :obj:`str`}
        Species an ID (:obj:`int`) to uniquely identifying labels for each
        lower-order structure if it originated from another reptar file. Labels
        should always be ``md5_structures``. For example,
        ``{0: '6038e101da7fc0085978741832ebc7ad', 1: 'eeaf93dec698de3ecb55e9292bd9dfcb'}``.
    n_workers : :obj:`int`
        Number of parallel workers. Can range from ``1`` to the umber of CPUs
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
        r_idxs.extend(np.where(r_prov_specs[:,0] == r_prov_id_lower)[0].tolist())
    
    task_size = math.ceil(len(r_idxs)/n_workers)
    for i in range(0, len(r_idxs), task_size):
        r_idxs_worker = r_idxs[i:i + task_size]
        yield r_idxs_worker

def mbe_contrib(
    E, Deriv, entity_ids, r_prov_ids, r_prov_specs, E_lower, Deriv_lower,
    entity_ids_lower, r_prov_ids_lower, r_prov_specs_lower, operation='remove',
    n_workers=1
):
    """Adds or removes energy and derivative (i.e., gradients or forces)
    contributions from a reference.

    We use the term "lower" to refer to the lower-order (i.e., smaller) systems.
    These are the energy and derivative contributions to remove or add to a
    reference.

    Making :math:`n`-body predictions (i.e., ``operation = 'add'``) will often
    not have ``r_prov_ids`` or ``r_prov_specs``as all lower contributions are
    derived exclusively from these structures. Use ``None`` for both of these
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
    r_prov_ids : :obj:`dict` {:obj:`int`: :obj:`str`} or ``None``
        Species an ID (:obj:`int`) to uniquely identifying labels for each
        structure if it originated from another reptar file. Labels should
        always be ``md5_structures``. For example,
        ``{0: '6038e101da7fc0085978741832ebc7ad', 1: 'eeaf93dec698de3ecb55e9292bd9dfcb'}``.
    r_prov_specs : :obj:`numpy.ndarray`, ndim: ``2`` or ``None``
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
        lower-order structure if it originated from another reptar file. Labels
        should always be ``md5_structures``. For example,
        ``{0: '6038e101da7fc0085978741832ebc7ad', 1: 'eeaf93dec698de3ecb55e9292bd9dfcb'}``.
    r_prov_specs_lower : :obj:`numpy.ndarray`, ndim: ``2``
        Structure provenance IDs. This specifies the ``r_prov_id``, structure
        index from the ``r_prov_id`` source, and ``entity_ids`` making up
        lower-order structures.
    operation : :obj:`str`, default: ``'remove'``
        ``'add'`` or ``'remove'`` the contributions.
    n_workers : :obj:`int`, default: ``1``
        Number of workers. Can range from ``1`` to the total number of CPUs
        available. If larger than ``1``, ray tasks are spawned.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        Energies with lower-order contributions added or removed.
    :obj:`numpy.ndarray`
        Derivatives with lower-order contributions added or removed.
    """
    if operation != 'add' and operation != 'remove':
        raise ValueError(f'{operation} is not "add" or "remove"')
    
    # Checks that the r_prov_md5 hashes match the same r_prov_id
    if (r_prov_ids is not None) and (r_prov_ids != {}):
        for r_prov_id_lower, r_prov_md5_lower in r_prov_ids_lower.items():
            assert r_prov_md5_lower == r_prov_ids[r_prov_id_lower]
    # Assume that all lower models apply.
    else:
        assert r_prov_specs is None or r_prov_specs.shape == (1, 0)
        assert len(set(r_prov_specs_lower[:,0])) == 1
        n_r = len(E)
        n_entities = len(set(entity_ids))
        r_prov_specs = np.empty((n_r, 2 + n_entities), dtype=int)
        r_prov_specs[:,0] = r_prov_specs_lower[0][0]
        r_prov_specs[:,1] = np.array([i for i in range(n_r)])
        for entity_id in range(n_entities):
            r_prov_specs[:,entity_id+2] = entity_id

    r_shape = Deriv.shape[1:]

    if n_workers == 1:
        r_idxs = next(
            gen_r_idxs_worker(r_prov_specs, r_prov_ids_lower, n_workers)
        )
        r_idxs_worker, E_worker, Deriv_worker = mbe_worker(
            r_shape, entity_ids, r_prov_specs, r_idxs, E_lower, Deriv_lower,
            entity_ids_lower, r_prov_specs_lower
        )
        if operation == 'remove':
            E[r_idxs_worker] -= E_worker
            Deriv[r_idxs_worker] -= Deriv_worker
        elif operation == 'add':
            E[r_idxs_worker] += E_worker
            Deriv[r_idxs_worker] += Deriv_worker
    else:
        # Check ray.
        import os
        import ray
        num_cpus = os.cpu_count()
        if not ray.is_initialized():
            # Try to connect to already running ray service (from ray cli).
            try:
                ray.init(address='auto')
            except ConnectionError:
                ray.init(num_cpus=num_cpus)
        
        # Generate all workers.
        num_cpus_worker = math.floor(num_cpus/n_workers)
        worker = ray.remote(mbe_worker)
        worker_list = []
        for r_idxs in gen_r_idxs_worker(r_prov_specs, r_prov_ids_lower, n_workers):
            worker_list.append(
                worker.options(num_cpus=num_cpus_worker).remote(
                    r_shape, entity_ids, r_prov_specs, r_idxs, E_lower,
                    Deriv_lower, entity_ids_lower, r_prov_specs_lower
                )
            )

        # Run all workers.
        while len(worker_list) != 0:
            done_id, worker_list = ray.wait(worker_list)
            
            r_idxs_worker, E_worker, Deriv_worker = ray.get(done_id)[0]
            if operation == 'remove':
                E[r_idxs_worker] -= E_worker
                Deriv[r_idxs_worker] -= Deriv_worker
            elif operation == 'add':
                E[r_idxs_worker] += E_worker
                Deriv[r_idxs_worker] += Deriv_worker
    
    return E, Deriv

