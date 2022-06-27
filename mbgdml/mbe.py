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
import logging
import math
import numpy as np
import os

log = logging.getLogger(__name__)

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

class mbePredict(object):
    """Predict energies and forces of structures using machine learning
    many-body models.
    """

    def __init__(
        self, models, predict_model, use_ray=False, n_cores=None,
        wkr_chunk_size=100
    ):
        """
        Parameters
        ----------
        models : :obj:`list` of :obj:`mbgdml.predict.mlWorker`
            Machine learning model objects that contain all information to make
            predictions using ``predict_model``.
        predict_model : ``callable``
            A function that takes ``z, r, entity_ids, nbody_gen, model`` and
            computes energies and forces. This will be turned into a ray remote
            function if ``use_ray = True``.
        use_ray : :obj:`bool`, default: ``False``
            Parallelize predictions using ray. Note that initializing ray tasks
            comes with some overhead and can make smaller computations much
            slower. Thus, this is only recommended with more than 15 or so
            entities.
        n_cores : :obj:`int`, default: ``None``
            Total number of cores available for predictions when using ray. If
            ``None``, then this is determined by ``os.cpu_count()``.
        wkr_chunk_size : :obj:`int`, default: ``100``
            Number of :math:`n`-body structures to assign to each spawned
            worker with ray.
        """
        self.models = models
        self.predict_model = predict_model

        self.use_ray = use_ray
        if n_cores is None:
            n_cores = os.cpu_count()
        self.n_cores = n_cores
        self.wkr_chunk_size = wkr_chunk_size
        if use_ray:
            global ray
            import ray
            assert ray.is_initialized()
            self.models = [ray.put(model) for model in models]
            self.predict_model = ray.remote(predict_model)

    def get_avail_entities(self, comp_ids_r, comp_ids_model):
        """Determines available ``entity_ids`` for each ``comp_id`` in a
        structure.

        Parameters
        ----------
        comp_ids_r : :obj:`numpy.ndarray`, ndim: ``1``
            Component IDs of the structure to predict.
        comp_ids_model : :obj:`numpy.ndarray`, ndim: ``1``
            Component IDs of the model.
        
        Returns
        -------
        :obj:`list`, length: ``len(comp_ids_r)``

        """
        # Models have a specific entity order that needs to be conserved in the
        # predictions. Here, we create a ``avail_entity_ids`` list where each item
        # is a list of all entities in ``r`` that match the model entity.
        avail_entity_ids = []
        for comp_id in comp_ids_model:
            matching_entity_ids = np.where(comp_id == comp_ids_r)[0]
            avail_entity_ids.append(matching_entity_ids)
        return avail_entity_ids
    
    def gen_entity_combinations(self, avail_entity_ids):
        """Generator for entity combinations where each entity comes from a
        specified list (i.e., from ``get_avail_entities``).

        Parameters
        ----------
        avail_entity_ids : :obj:`list` of :obj:`numpy.ndarray`
            A list of ``entity_ids`` that match the ``comp_id`` of each
            model ``entity_id``. Note that the index of the
            :obj:`numpy.ndarray` is equal to the model ``entity_id`` and the
            values are ``entity_ids`` that match the ``comp_id``.
        
        Yields
        ------
        :obj:`tuple`
            Entity IDs to retrieve cartesian coordinates for ML prediction.
        """
        nbody_combinations = itertools.product(*avail_entity_ids)
        # Excludes combinations that have repeats (e.g., (0, 0) and (1, 1. 2)).
        nbody_combinations = itertools.filterfalse(
            lambda x: len(set(x)) <  len(x), nbody_combinations
        )
        # At this point, there are still duplicates in this iterator.
        # For example, (0, 1) and (1, 0) are still included.
        for combination in nbody_combinations:
            # Sorts entity is to avoid duplicate structures.
            # For example, if combination is (1, 0) the sorted version is not
            # equal and will not be included.
            if sorted(combination) == list(combination):
                yield combination
    
    def chunk(self, iterable, n):
        """Chunk an iterable into ``n`` objects.

        Parameters
        ----------
        iterable : ``iterable``
        n : :obj:`int`
            Size of each chunk.
        
        Yields
        ------
        :obj:`tuple`
            ``n`` objects.
        """
        iterator = iter(iterable)
        for first in iterator:
            yield tuple(
                itertools.chain([first], itertools.islice(iterator, n - 1))
            )
    
    def compute_nbody(self, z, r, entity_ids, comp_ids, model):
        """Compute all :math:`n`-body contributions of a single structure
        using a :obj:`mbgdml.predict.mlModel`` object.

        When ``use_ray = True``, this acts as a driver that spawns ray tasks of
        the ``predict_model`` function.

        Parameters
        ----------
        z : :obj:`numpy.ndarray`, ndim: ``1``
            Atomic numbers of all atoms in the system with respect to ``r``.
        r : :obj:`numpy.ndarray`, shape: ``(len(z), 3)``
            Cartesian coordinates of a single structure.
        entity_ids : :obj:`numpy.ndarray`, shape: ``(N,)``
            Integers specifying which atoms belong to which entities.
        comp_ids : :obj:`numpy.ndarray`, shape: ``(N,)``
            Relates each ``entity_id`` to a fragment label. Each item's index
            is the label's ``entity_id``.
        model : ``callable``

        Returns
        -------
        """
        # Unify r shape.
        if r.ndim == 3:
            log.debug('r has three dimensions (instead of two)')
            if r.shape[0] == 1:
                log.debug('r[0] was selected to proceed')
                r = r[0]
            else:
                raise ValueError(
                    'r.ndim is not 2 (only one structure is allowed)'
                )
        
        E = 0.
        F = np.zeros(r.shape, dtype=np.double)

        # Creates a generator for all possible n-body combinations regardless
        # of cutoffs.
        if self.use_ray:
            model_comp_ids = ray.get(model).comp_ids
        else:
            model_comp_ids = model.comp_ids
        avail_entity_ids = self.get_avail_entities(comp_ids, model_comp_ids)
        nbody_gen = self.gen_entity_combinations(avail_entity_ids)
        
        # Runs the predict_model function to calculate all n-body energies
        # with this model.
        if not self.use_ray:
            E, F = self.predict_model(
                z, r, entity_ids, nbody_gen, model
            )
        else:
            # Put all common data into the ray object store.
            z = ray.put(z)
            r = ray.put(r)
            entity_ids = ray.put(entity_ids)

            nbody_gen = tuple(nbody_gen)
            nbody_chunker = self.chunk(nbody_gen, self.wkr_chunk_size)
            workers = []

            # Initialize workers 
            predict_model = self.predict_model
            def add_worker(workers, chunk):
                workers.append(
                    predict_model.remote(
                        z, r, entity_ids, chunk, model
                    )
                )
            for _ in range(self.n_cores):
                try:
                    chunk = next(nbody_chunker)
                    add_worker(workers, chunk)
                except StopIteration:
                    break
            
            # Start workers and collect results.
            while len(workers) != 0:
                done_id, workers = ray.wait(workers)
                E_wkr, F_wkr = ray.get(done_id)[0]
                E += E_wkr
                F += F_wkr
                try:
                    chunk = next(nbody_chunker)
                    add_worker(workers, chunk)
                except StopIteration:
                    pass
        
        return E, F
    
    def predict(self, z, R, entity_ids, comp_ids):
        """Predict the energies and forces of one or multiple structures.

        Parameters
        ----------
        z : :obj:`numpy.ndarray`, ndim: ``1``
            Atomic numbers of all atoms in the system with respect to ``R``.
        R : :obj:`numpy.ndarray`, shape: ``(N, len(z), 3)``
            Cartesian coordinates of ``N`` structures to predict.
        entity_ids : :obj:`numpy.ndarray`, shape: ``(N,)``
            Integers specifying which atoms belong to which entities.
        comp_ids : :obj:`numpy.ndarray`, shape: ``(N,)``
            Relates each ``entity_id`` to a fragment label. Each item's index
            is the label's ``entity_id``.
        
        Returns
        -------
        :obj:`numpy.ndarray`, shape: ``(N,)``
            Predicted many-body energy
        :obj:`numpy.ndarray`, shape: ``(N, len(z), 3)``
            Predicted atomic forces.
        """
        if R.ndim == 2:
            R = np.array([R])
        
        # Preallocate memory for energies and forces.
        E = np.zeros((R.shape[0],), dtype=np.double)
        F = np.zeros(R.shape, dtype=np.double)

        # Compute all energies and forces with every model.
        for i in range(len(E)):
            for model in self.models:
                E_nbody, F_nbody = self.compute_nbody(
                    z, R[i], entity_ids, comp_ids, model
                )
                E[i] += E_nbody
                F[i] += F_nbody
            
        return E, F
