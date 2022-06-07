# MIT License
# 
# Copyright (c) 2018-2020, Stefan Chmiela
# Copyright (c) 2020-2022, Alex M. Maldonado
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

from __future__ import print_function

import multiprocessing as mp

Pool = mp.get_context('fork').Pool

import sys
from functools import partial

import numpy as np
import scipy.optimize
import scipy.spatial.distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .desc import Desc

import logging
log = logging.getLogger(__name__)

glob = {}


def share_array(arr_np, typecode):
    arr = mp.RawArray(typecode, arr_np.ravel())
    return arr, arr_np.shape


def _bipartite_match_wkr(i, n_train, same_z_cost):

    global glob

    adj_set = np.frombuffer(glob['adj_set']).reshape(glob['adj_set_shape'])
    v_set = np.frombuffer(glob['v_set']).reshape(glob['v_set_shape'])
    match_cost = np.frombuffer(glob['match_cost']).reshape(glob['match_cost_shape'])

    adj_i = scipy.spatial.distance.squareform(adj_set[i, :])
    v_i = v_set[i, :, :]

    match_perms = {}
    for j in range(i + 1, n_train):

        adj_j = scipy.spatial.distance.squareform(adj_set[j, :])
        v_j = v_set[j, :, :]

        cost = -np.fabs(v_i).dot(np.fabs(v_j).T)
        cost += same_z_cost * np.max(np.abs(cost))

        _, perm = scipy.optimize.linear_sum_assignment(cost)

        adj_i_perm = adj_i[:, perm]
        adj_i_perm = adj_i_perm[perm, :]

        score_before = np.linalg.norm(adj_i - adj_j)
        score = np.linalg.norm(adj_i_perm - adj_j)

        match_cost[i, j] = score
        if score >= score_before:
            match_cost[i, j] = score_before
        elif not np.isclose(score_before, score):  # otherwise perm is identity
            match_perms[i, j] = perm
        
    return match_perms

def bipartite_match(R, z, lat_and_inv=None, max_processes=None):
    global glob
    log.info('Performing Bipartite matching ...')

    n_train, n_atoms, _ = R.shape

    # penalty matrix for mixing atom species
    log.debug('Atom mixing penalties')
    same_z_cost = np.repeat(z[:, None], len(z), axis=1) - z
    same_z_cost[same_z_cost != 0] = 1
    log.log_array(same_z_cost, level=10)

    match_cost = np.zeros((n_train, n_train))

    desc = Desc(n_atoms, max_processes=max_processes)

    adj_set = np.empty((n_train, desc.dim))
    v_set = np.empty((n_train, n_atoms, n_atoms))
    for i in range(n_train):
        r = np.squeeze(R[i, :, :])

        if lat_and_inv is None:
            adj = scipy.spatial.distance.pdist(r, 'euclidean')

        else:
            adj = scipy.spatial.distance.pdist(
                r, lambda u, v: np.linalg.norm(desc.pbc_diff(u - v, lat_and_inv))
            )

        w, v = np.linalg.eig(scipy.spatial.distance.squareform(adj))
        v = v[:, w.argsort()[::-1]]

        adj_set[i, :] = adj
        v_set[i, :, :] = v

    glob['adj_set'], glob['adj_set_shape'] = share_array(adj_set, 'd')
    glob['v_set'], glob['v_set_shape'] = share_array(v_set, 'd')
    glob['match_cost'], glob['match_cost_shape'] = share_array(match_cost, 'd')

    pool = Pool(max_processes)

    match_perms_all = {}
    for i, match_perms in enumerate(
        pool.imap_unordered(
            partial(_bipartite_match_wkr, n_train=n_train, same_z_cost=same_z_cost),
            list(range(n_train)),
        )
    ):
        match_perms_all.update(match_perms)

    pool.close()
    pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).

    match_cost = np.frombuffer(glob['match_cost']).reshape(glob['match_cost_shape'])
    match_cost = match_cost + match_cost.T
    match_cost[np.diag_indices_from(match_cost)] = np.inf
    match_cost = csr_matrix(match_cost)

    return match_perms_all, match_cost

def sync_perm_mat(match_perms_all, match_cost, n_atoms):

    tree = minimum_spanning_tree(match_cost, overwrite=True)

    perms = np.arange(n_atoms, dtype=int)[None, :]
    rows, cols = tree.nonzero()
    for com in zip(rows, cols):
        perm = match_perms_all.get(com)
        if perm is not None:
            perms = np.vstack((perms, perm))
    perms = np.unique(perms, axis=0)

    return perms

# convert permutation to dijoined cycles
def to_cycles(perm):
    pi = {i: perm[i] for i in range(len(perm))}
    cycles = []

    while pi:
        elem0 = next(iter(pi)) # arbitrary starting element
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break

        cycles.append(cycle)

    return cycles

# find permutation group with larges cardinality
# note: this is used if transitive closure fails (to salvage at least some permutations)
def salvage_subgroup(perms):

    n_perms, n_atoms = perms.shape
    lcms = []
    for i in range(n_perms):
        cy_lens = [len(cy) for cy in to_cycles(list(perms[i, :]))]
        lcm = np.lcm.reduce(cy_lens)
        lcms.append(lcm)
    keep_idx = np.argmax(lcms)
    perms = np.vstack((np.arange(n_atoms), perms[keep_idx,:]))

    return perms


def complete_sym_group(perms, n_perms_max=None):

    perm_added = True
    while perm_added:
        perm_added = False
        n_perms = perms.shape[0]
        for i in range(n_perms):
            for j in range(n_perms):

                new_perm = perms[i, perms[j, :]]
                if not (new_perm == perms).all(axis=1).any():
                    perm_added = True
                    perms = np.vstack((perms, new_perm))

                    # Transitive closure is not converging! Give up and return identity permutation.
                    if n_perms_max is not None and perms.shape[0] == n_perms_max:
                        log.warning('Transitive closure has failed')
                        return None

    return perms


def find_perms(R, z, lat_and_inv=None, max_processes=None):
    log.info('\n#   Finding symmetries   #')

    m, n_atoms = R.shape[:2]

    # Find matching for all pairs.
    match_perms_all, match_cost = bipartite_match(
        R, z, lat_and_inv, max_processes
    )

    # Remove inconsistencies.
    match_perms = sync_perm_mat(match_perms_all, match_cost, n_atoms)

    # Complete symmetric group.
    # Give up, if transitive closure yields more than 100 unique permutations.
    sym_group_perms = complete_sym_group(match_perms, n_perms_max=100)

    # Limit closure to largest cardinality permutation in the set to get at least some symmetries.
    if sym_group_perms is None:
        match_perms_subset = salvage_subgroup(match_perms)
        sym_group_perms = complete_sym_group(match_perms_subset, n_perms_max=100)

    log.info(f'Found {sym_group_perms.shape[0]} symmetries')
    return sym_group_perms

def inv_perm(perm):

    inv_perm = np.empty(perm.size, perm.dtype)
    inv_perm[perm] = np.arange(perm.size)

    return inv_perm
