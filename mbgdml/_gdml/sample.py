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

import numpy as np

def draw_strat_sample(T, n, excl_idxs=None):
    """
    Draw sample from dataset that preserves its original distribution.

    The distribution is estimated from a histogram were the bin size is
    determined using the Freedman-Diaconis rule. This rule is designed to
    minimize the difference between the area under the empirical
    probability distribution and the area under the theoretical
    probability distribution. A reduced histogram is then constructed by
    sampling uniformly in each bin. It is intended to populate all bins
    with at least one sample in the reduced histogram, even for small
    training sizes.

    Parameters
    ----------
    T : :obj:`numpy.ndarray`
        Dataset to sample from.
    n : int
        Number of examples.
    excl_idxs : :obj:`numpy.ndarray`, optional
        Array of indices to exclude from sample.

    Returns
    -------
    :obj:`numpy.ndarray`
        Array of indices that form the sample.
    """
    if excl_idxs is None or len(excl_idxs) == 0:
        excl_idxs = None

    if n == 0:
        return np.array([], dtype=np.uint)

    if T.size == n:  # TODO: this only works if excl_idxs=None
        assert excl_idxs is None
        return np.arange(n)

    if n == 1:
        idxs_all_non_excl = np.setdiff1d(
            np.arange(T.size), excl_idxs, assume_unique=True
        )
        return np.array([np.random.choice(idxs_all_non_excl)])

    # Freedman-Diaconis rule
    h = 2 * np.subtract(*np.percentile(T, [75, 25])) / np.cbrt(n)
    n_bins = int(np.ceil((np.max(T) - np.min(T)) / h)) if h > 0 else 1
    n_bins = min(
        n_bins, int(n / 2)
    )  # Limit number of bins to half of requested subset size.

    bins = np.linspace(np.min(T), np.max(T), n_bins, endpoint=False)
    idxs = np.digitize(T, bins)

    # Exclude restricted indices.
    if excl_idxs is not None and excl_idxs.size > 0:
        idxs[excl_idxs] = n_bins + 1  # Impossible bin.

    uniq_all, cnts_all = np.unique(idxs, return_counts=True)

    # Remove restricted bin.
    if excl_idxs is not None and excl_idxs.size > 0:
        excl_bin_idx = np.where(uniq_all == n_bins + 1)
        cnts_all = np.delete(cnts_all, excl_bin_idx)
        uniq_all = np.delete(uniq_all, excl_bin_idx)

    # Compute reduced bin counts.
    reduced_cnts = np.ceil(cnts_all / np.sum(cnts_all, dtype=float) * n).astype(int)
    reduced_cnts = np.minimum(
        reduced_cnts, cnts_all
    )  # limit reduced_cnts to what is available in cnts_all

    # Reduce/increase bin counts to desired total number of points.
    reduced_cnts_delta = n - np.sum(reduced_cnts)

    while np.abs(reduced_cnts_delta) > 0:

        # How many members can we remove from an arbitrary bucket, without any bucket with more than one member going to zero?
        max_bin_reduction = np.min(reduced_cnts[np.where(reduced_cnts > 1)]) - 1

        # Generate additional bin members to fill up/drain bucket counts of subset. This array contains (repeated) bucket IDs.
        outstanding = np.random.choice(
            uniq_all,
            min(max_bin_reduction, np.abs(reduced_cnts_delta)),
            p=(reduced_cnts - 1) / np.sum(reduced_cnts - 1, dtype=float),
            replace=True,
        )
        uniq_outstanding, cnts_outstanding = np.unique(
            outstanding, return_counts=True
        )  # Aggregate bucket IDs.

        outstanding_bucket_idx = np.where(
            np.in1d(uniq_all, uniq_outstanding, assume_unique=True)
        )[
            0
        ]  # Bucket IDs to Idxs.
        reduced_cnts[outstanding_bucket_idx] += (
            np.sign(reduced_cnts_delta) * cnts_outstanding
        )
        reduced_cnts_delta = n - np.sum(reduced_cnts)

    # Draw examples for each bin.
    idxs_train = np.empty((0,), dtype=int)
    for uniq_idx, bin_cnt in zip(uniq_all, reduced_cnts):
        idx_in_bin_all = np.where(idxs.ravel() == uniq_idx)[0]
        idxs_train = np.append(
            idxs_train, np.random.choice(idx_in_bin_all, bin_cnt, replace=False)
        )

    return idxs_train