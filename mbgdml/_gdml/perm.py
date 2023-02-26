# MIT License
#
# Copyright (c) 2018-2021, Stefan Chmiela
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

from __future__ import print_function

import multiprocessing as mp
from functools import partial
import numpy as np
import scipy.optimize
import scipy.spatial.distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from ase import Atoms
from ase.geometry.analysis import Analysis
from .desc import Desc, _pdist, _squareform
from ..logger import GDMLLogger

Pool = mp.get_context("fork").Pool

log = GDMLLogger(__name__)

glob = {}


def share_array(arr_np, typecode):
    arr = mp.RawArray(typecode, arr_np.ravel())
    return arr, arr_np.shape


def _bipartite_match_wkr(i, n_train, same_z_cost):

    global glob  # pylint: disable=global-variable-not-assigned

    adj_set = np.frombuffer(glob["adj_set"]).reshape(glob["adj_set_shape"])
    v_set = np.frombuffer(glob["v_set"]).reshape(glob["v_set_shape"])
    match_cost = np.frombuffer(glob["match_cost"]).reshape(glob["match_cost_shape"])

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
    global glob  # pylint: disable=global-variable-not-assigned

    log.info("Performing Bipartite matching ...")

    n_train, n_atoms, _ = R.shape

    # penalty matrix for mixing atom species
    log.debug("Atom mixing penalties")
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
            adj = scipy.spatial.distance.pdist(r, "euclidean")

        else:

            adj_tri = _pdist(r, lat_and_inv)
            adj = _squareform(adj_tri)  # our vectorized format to full matrix
            adj = scipy.spatial.distance.squareform(
                adj
            )  # full matrix to numpy vectorized format

        w, v = np.linalg.eig(scipy.spatial.distance.squareform(adj))
        v = v[:, w.argsort()[::-1]]

        adj_set[i, :] = adj
        v_set[i, :, :] = v

    glob["adj_set"], glob["adj_set_shape"] = share_array(adj_set, "d")
    glob["v_set"], glob["v_set_shape"] = share_array(v_set, "d")
    glob["match_cost"], glob["match_cost_shape"] = share_array(match_cost, "d")

    pool = None
    map_func = map
    if max_processes != 1 and mp.cpu_count() > 1:
        pool = Pool((max_processes or mp.cpu_count()) - 1)  # exclude main process
        map_func = pool.imap_unordered

    match_perms_all = {}
    for i, match_perms in enumerate(
        map_func(
            partial(_bipartite_match_wkr, n_train=n_train, same_z_cost=same_z_cost),
            list(range(n_train)),
        )
    ):
        match_perms_all.update(match_perms)

    if pool is not None:
        pool.close()
        # Wait for the worker processes to terminate (to measure total runtime
        # correctly).
        pool.join()
        pool = None

    match_cost = np.frombuffer(glob["match_cost"]).reshape(glob["match_cost_shape"])
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
    pi = {i: perm[i] for i in range(len(perm))}  # pylint: disable=invalid-name
    cycles = []

    while pi:
        elem0 = next(iter(pi))  # arbitrary starting element
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

    n_perms, _ = perms.shape

    all_long_cycles = []
    for i in range(n_perms):
        long_cycles = [cy for cy in to_cycles(list(perms[i, :])) if len(cy) > 1]
        all_long_cycles += long_cycles

    # pylint: disable-next=invalid-name
    def _cycle_intersects_with_larger_one(cy):

        # pylint: disable-next=invalid-name
        for ac in all_long_cycles:
            if len(cy) < len(ac):
                if not set(cy).isdisjoint(ac):
                    return True

        return False

    keep_idx_many = []
    for i in range(n_perms):

        # is this permutation valid?
        # remove permutations that contain cycles that share elements with larger
        # cycles in other perms
        long_cycles = [cy for cy in to_cycles(list(perms[i, :])) if len(cy) > 1]

        ignore_perm = any(list(map(_cycle_intersects_with_larger_one, long_cycles)))

        if not ignore_perm:
            keep_idx_many.append(i)

    perms = perms[keep_idx_many, :]

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

                    # Transitive closure is not converging! Give up and return identity
                    # permutation.
                    if n_perms_max is not None and perms.shape[0] == n_perms_max:
                        return None

    return perms


def find_perms(R, z, lat_and_inv=None, max_processes=None):
    log.info("\n#   Finding symmetries   #")

    _, n_atoms = R.shape[:2]

    # Find matching for all pairs.
    match_perms_all, match_cost = bipartite_match(R, z, lat_and_inv, max_processes)

    # Remove inconsistencies.
    match_perms = sync_perm_mat(match_perms_all, match_cost, n_atoms)

    # Complete symmetric group.
    # Give up, if transitive closure yields more than 100 unique permutations.
    sym_group_perms = complete_sym_group(match_perms, n_perms_max=100)

    # Limit closure to largest cardinality permutation in the set to get at least some
    # symmetries.
    if sym_group_perms is None:
        match_perms_subset = salvage_subgroup(match_perms)
        sym_group_perms = complete_sym_group(
            match_perms_subset,
            n_perms_max=100,
        )

    log.info("Found %d symmetries", sym_group_perms.shape[0])
    return sym_group_perms


def find_extra_perms(R, z, lat_and_inv=None):

    _, n_atoms = R.shape[:2]

    # nanotube
    R = R.copy()
    frags = find_frags(R[0], z, lat_and_inv=lat_and_inv)
    print(frags)

    perms = np.arange(n_atoms)[None, :]

    plane_3idxs = [280, 281, 273]  # half outer
    add_perms = find_perms_via_reflection(R[0], frags[1], plane_3idxs)
    perms = np.vstack((perms, add_perms))

    perms = np.unique(perms, axis=0)
    sym_group_perms = complete_sym_group(perms)
    print(sym_group_perms.shape)

    return sym_group_perms


def find_frags(r, z, lat_and_inv=None):

    print("Finding permutable non-bonded fragments... (assumes Ang!)")

    lat = None
    if lat_and_inv:
        lat = lat_and_inv[0]

    n_atoms = r.shape[0]
    # only use first molecule in dataset to find connected components (fix me later,
    # maybe) # *0.529177249
    atoms = Atoms(z, positions=r, cell=lat, pbc=lat is not None)

    adj = Analysis(atoms).adjacency_matrix[0]
    _, labels = connected_components(csgraph=adj, directed=False, return_labels=True)

    frags = [np.where(labels == label)[0] for label in np.unique(labels)]
    n_frags = len(frags)

    if n_frags == n_atoms:
        print(
            "Skipping fragment symmetry search (something went wrong, "
            "e.g. length unit not in Angstroms, etc.)"
        )
        return None

    print("| Found " + str(n_frags) + " disconnected fragments.")

    return frags


def find_frag_perms(R, z, lat_and_inv=None, max_processes=None):

    _, n_atoms = R.shape[:2]
    lat, _ = lat_and_inv

    # only use first molecule in dataset to find connected components (fix me later,
    # maybe) # *0.529177249
    atoms = Atoms(z, positions=R[0], cell=lat, pbc=lat is not None)

    adj = Analysis(atoms).adjacency_matrix[0]
    _, labels = connected_components(csgraph=adj, directed=False, return_labels=True)

    frags = [np.where(labels == label)[0] for label in np.unique(labels)]
    n_frags = len(frags)

    if n_frags == n_atoms:
        print(
            "Skipping fragment symmetry search (something went wrong, "
            "e.g. length unit not in Angstroms, etc.)"
        )
        return [range(n_atoms)]

    print("| Found " + str(n_frags) + " disconnected fragments.")

    # match fragments to find identical ones (allows permutations of fragments)
    swap_perms = [np.arange(n_atoms)]
    for f1 in range(n_frags):  # pylint: disable=invalid-name
        for f2 in range(f1 + 1, n_frags):  # pylint: disable=invalid-name

            sort_idx_f1 = np.argsort(z[frags[f1]])
            sort_idx_f2 = np.argsort(z[frags[f2]])
            inv_sort_idx_f2 = inv_perm(sort_idx_f2)

            z1 = z[frags[f1]][sort_idx_f1]  # pylint: disable=invalid-name
            z2 = z[frags[f2]][sort_idx_f2]  # pylint: disable=invalid-name

            if np.array_equal(z1, z2):  # fragment have the same composition

                # pylint: disable-next=invalid-name
                for ri in range(
                    min(10, R.shape[0])
                ):  # only use first molecule in dataset for matching (fix me later)

                    R_match1 = R[ri, frags[f1], :]  # pylint: disable=invalid-name
                    R_match2 = R[ri, frags[f2], :]  # pylint: disable=invalid-name

                    # if np.array_equal(z1, z2):

                    # pylint: disable-next=invalid-name
                    R_pair = np.concatenate(
                        (R_match1[None, sort_idx_f1, :], R_match2[None, sort_idx_f2, :])
                    )

                    perms = find_perms(
                        R_pair, z1, lat_and_inv=lat_and_inv, max_processes=max_processes
                    )

                    # embed local permutation into global context
                    # pylint: disable-next=invalid-name
                    for p in perms:

                        match_perm = sort_idx_f1[p][inv_sort_idx_f2]

                        swap_perm = np.arange(n_atoms)
                        swap_perm[frags[f1]] = frags[f2][match_perm]
                        swap_perm[frags[f2][match_perm]] = frags[f1]
                        swap_perms.append(swap_perm)

    swap_perms = np.unique(np.array(swap_perms), axis=0)

    # complete symmetric group
    sym_group_perms = complete_sym_group(swap_perms)
    print(
        "| Found "
        + str(sym_group_perms.shape[0])
        + " fragment permutations after closure."
    )

    # match fragments with themselves (to find symmetries in each fragment)

    def _frag_perm_to_perm(n_atoms, frag_idxs, frag_perms):

        # frag_idxs - indices of the fragment (one fragment!)
        # frag_perms - N fragment permutations (Nxn_atoms)

        perms = np.arange(n_atoms)[None, :]
        for fp in frag_perms:  # pylint: disable=invalid-name

            p = np.arange(n_atoms)  # pylint: disable=invalid-name
            p[frag_idxs] = frag_idxs[fp]
            perms = np.vstack((p[None, :], perms))

        return perms

    if n_frags > 1:
        print("| Finding symmetries in individual fragments.")
        for f in range(n_frags):

            R_frag = R[:, frags[f], :]  # pylint: disable=invalid-name
            z_frag = z[frags[f]]

            frag_perms = find_perms(
                R_frag, z_frag, lat_and_inv=lat_and_inv, max_processes=max_processes
            )

            perms = _frag_perm_to_perm(n_atoms, frags[f], frag_perms)
            sym_group_perms = np.vstack((perms, sym_group_perms))

            print(f"{perms.shape[0]} perms")

        sym_group_perms = np.unique(sym_group_perms, axis=0)
    sym_group_perms = complete_sym_group(sym_group_perms)

    return sym_group_perms


def _frag_perm_to_perm(n_atoms, frag_idxs, frag_perms):

    # frag_idxs - indices of the fragment (one fragment!)
    # frag_perms - N fragment permutations (Nxn_atoms)

    perms = np.arange(n_atoms)[None, :]
    for fp in frag_perms:  # pylint: disable=invalid-name

        p = np.arange(n_atoms)  # pylint: disable=invalid-name
        p[frag_idxs] = frag_idxs[fp]
        perms = np.vstack((p[None, :], perms))

    return perms


def find_perms_in_frag(R, z, frag_idxs, lat_and_inv=None, max_processes=None):

    n_atoms = R.shape[1]

    R_frag = R[:, frag_idxs, :]  # pylint: disable=invalid-name
    z_frag = z[frag_idxs]

    frag_perms = find_perms(
        R_frag, z_frag, lat_and_inv=lat_and_inv, max_processes=max_processes
    )

    perms = _frag_perm_to_perm(n_atoms, frag_idxs, frag_perms)

    return perms


def find_perms_via_alignment(
    pts_full,
    frag_idxs,
    align_a_idxs,
    align_b_idxs,
):
    # pylint: disable=invalid-name

    # alignment indices are included in fragment
    assert np.isin(align_a_idxs, frag_idxs).all()
    assert np.isin(align_b_idxs, frag_idxs).all()

    assert len(align_a_idxs) == len(align_b_idxs)

    pts = pts_full[frag_idxs, :]

    align_a_pts = pts_full[align_a_idxs, :]
    align_b_pts = pts_full[align_b_idxs, :]

    ctr = np.mean(pts, axis=0)
    align_a_pts -= ctr
    align_b_pts -= ctr

    ab_cov = align_a_pts.T.dot(align_b_pts)
    u, _, vh = np.linalg.svd(ab_cov)
    R = u.dot(vh)

    if np.linalg.det(R) < 0:
        vh[2, :] *= -1  # multiply 3rd column of V by -1
        R = u.dot(vh)

    pts -= ctr
    pts_R = pts.copy()

    pts_R = R.dot(pts_R.T).T

    pts += ctr
    pts_R += ctr

    pts_full_R = pts_full.copy()
    pts_full_R[frag_idxs, :] = pts_R

    R_pair = np.vstack((pts_full[None, :, :], pts_full_R[None, :, :]))

    adj = scipy.spatial.distance.cdist(R_pair[0], R_pair[1], "euclidean")
    _, perm = scipy.optimize.linear_sum_assignment(adj)

    return perm


def find_perms_via_reflection(r, frag_idxs, plane_3idxs):
    """compute normal of plane defined by atoms in 'plane_idxs'"""
    # pylint: disable=invalid-name

    is_plane_defined_by_bond_centers = isinstance(plane_3idxs[0], tuple)
    if is_plane_defined_by_bond_centers:
        a = (r[plane_3idxs[0][0], :] + r[plane_3idxs[0][1], :]) / 2
        b = (r[plane_3idxs[1][0], :] + r[plane_3idxs[1][1], :]) / 2
        c = (r[plane_3idxs[2][0], :] + r[plane_3idxs[2][1], :]) / 2
    else:
        a = r[plane_3idxs[0], :]
        b = r[plane_3idxs[1], :]
        c = r[plane_3idxs[2], :]

    ab = b - a
    ab /= np.linalg.norm(ab)

    ac = c - a
    ac /= np.linalg.norm(ac)

    normal = np.cross(ab, ac)[:, None]

    # compute reflection matrix
    reflection = np.eye(3) - 2 * normal.dot(normal.T)

    r_R = r.copy()
    r_R[frag_idxs, :] = reflection.dot(r[frag_idxs, :].T).T

    adj = scipy.spatial.distance.cdist(r, r_R, "euclidean")
    _, perm = scipy.optimize.linear_sum_assignment(adj)

    print_perm_colors(perm, r, plane_3idxs)

    return perm


def print_perm_colors(perm, pts, plane_3idxs=None):
    # pylint: disable=invalid-name
    idx_done = []
    c = -1
    for i in range(perm.shape[0]):
        if i not in idx_done and perm[i] not in idx_done:
            c += 1
            idx_done += [i]
            idx_done += [perm[i]]

    from matplotlib import cm  # pylint: disable=import-outside-toplevel

    viridis = cm.get_cmap("prism")
    colors = viridis(np.linspace(0, 1, c + 1))

    print("---")
    print("select all; color [255,255,255]")

    if plane_3idxs is not None:

        def pts_str(x):
            return "{" + str(x[0]) + ", " + str(x[1]) + ", " + str(x[2]) + "}"

        is_plane_defined_by_bond_centers = isinstance(plane_3idxs[0], tuple)
        if is_plane_defined_by_bond_centers:
            a = (pts[plane_3idxs[0][0], :] + pts[plane_3idxs[0][1], :]) / 2
            b = (pts[plane_3idxs[1][0], :] + pts[plane_3idxs[1][1], :]) / 2
            c = (pts[plane_3idxs[2][0], :] + pts[plane_3idxs[2][1], :]) / 2
        else:
            a = pts[plane_3idxs[0], :]
            b = pts[plane_3idxs[1], :]
            c = pts[plane_3idxs[2], :]

        print(
            "draw plane1 300 PLANE "
            + pts_str(a)
            + " "
            + pts_str(b)
            + " "
            + pts_str(c)
            + ";color $plane1 green"
        )

    idx_done = []
    c = -1
    for i in range(perm.shape[0]):
        if i not in idx_done and perm[i] not in idx_done:

            c += 1
            color_str = (
                "["
                + str(int(colors[c, 0] * 255))
                + ","
                + str(int(colors[c, 1] * 255))
                + ","
                + str(int(colors[c, 2] * 255))
                + "]"
            )

            if i != perm[i]:
                print("select atomno=" + str(i + 1) + "; color " + color_str)
                print("select atomno=" + str(perm[i] + 1) + "; color " + color_str)
            idx_done += [i]
            idx_done += [perm[i]]

    print("---")


def inv_perm(perm):

    inv_perm_array = np.empty(perm.size, perm.dtype)
    inv_perm_array[perm] = np.arange(perm.T.size)

    return inv_perm_array
