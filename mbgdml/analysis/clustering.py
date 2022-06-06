# MIT License
# 
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

import logging
import numpy as np

log = logging.getLogger(__name__)

def get_clustered_data(cl_idxs, data):
    """Cluster data according to cluster indices.

    Parameters
    ----------
    cl_idxs : :obj:`list` of :obj:`numpy.ndarray`
        Structure indices stored in clusters.
    data : :obj:`numpy.ndarray`
        Iterative data with respect to structure indices.
    
    Returns
    -------
    :obj:`list` of :obj:`numpy.ndarray`
        ``data`` clustered with respect to ``cl_idxs``.
    """
    log.info(f'Assembling data into {len(cl_idxs)} clusters')
    log.debug(f'Example data:')
    log.log_array(np.array(data[0]), level=10)
    data_cl = []
    for idxs in cl_idxs:
        d = np.array(data[idxs])
        data_cl.append(d)
    return data_cl

def agglomerative(data, kwargs={'n_clusters': 10}):
    """Cluster data using ``sklearn.cluster.AgglomerativeClustering``.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Feature or precomputed distance matrix used to cluster similar
        structures.
    kwargs : :obj:`dict`, default: ``{'n_clusters': 10}``
        Keyword arguments to pass to ``sklearn.cluster.AgglomerativeClustering``
        with the exception of ``n_clusters``.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        Cluster labels of structures in ``R_desc``.
    """
    from sklearn.cluster import AgglomerativeClustering
    log.info('Agglomerative clustering')
    log.debug(kwargs)
    log.debug(f'Data example: {data[0]}')
    t_cluster = log.t_start()
    cluster_labels = AgglomerativeClustering(**kwargs).fit_predict(data)
    log.t_stop(t_cluster)
    return cluster_labels

def kmeans(data, kwargs={'n_clusters': 5, 'init': 'k-means++'}):
    """Cluster data using ``sklearn.cluster.KMeans``.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Feature or precomputed distance matrix used to cluster similar
        structures. For example, atomic pairwise distances.
    kwargs : :obj:`dict`, default: ``{'n_clusters': 5, 'init': 'k-means++'}``
        Keyword arguments to pass to ``sklearn.cluster.KMeans``
        with the exception of ``n_clusters``.
    """
    from sklearn.cluster import KMeans
    log.info('K-means clustering')
    log.debug(kwargs)
    log.debug(f'Data example: {data[0]}')
    t_cluster = log.t_start()
    cluster_labels = KMeans(**kwargs).fit_predict(data)
    log.t_stop(t_cluster)
    return cluster_labels

def cluster_structures(cl_data, cl_algs, cl_kwargs):
    """Performs :math:`n`-stage clustering of structures based on features.

    Parameters
    ----------
    cl_data : :obj:`tuple` of :obj:`numpy.ndarray`
        Data to cluster with ``cl_algs``
    cl_algs : :obj:`tuple` of ``callable``
        Clustering algorithms to use.
    cluster_kwargs : :obj:`tuple` of :obj:`dict`
        Keyword arguments for each clustering algorithm.

    Returns
    -------
    :obj:`list`
        :obj:`numpy.ndarray` of structure indices in each cluster.
    """
    n_structures = len(cl_data[0])
    cl_idxs = [np.array(range(n_structures))]
    log.info(f'Clustering {n_structures} structures')
    
    t_clustering = log.t_start()
    # Loop through each clustering routine
    for i_cluster in range(len(cl_algs)):
        data = np.array(cl_data[i_cluster])
        cl_alg = cl_algs[i_cluster]
        kwargs = cl_kwargs[i_cluster]

        cl_idxs_new = []
        # Loop though every current cluster group.
        for idxs in cl_idxs:
            cl_labels = cl_alg(data[idxs], kwargs)
            
            for label in set(cl_labels):
                ind = np.concatenate(np.argwhere(cl_labels == label))
                cl_idxs_new.append(idxs[ind])

        cl_idxs = cl_idxs_new
    
    log.t_stop(t_clustering, message='Total clustering time: {time} s')
    return cl_idxs

def get_cluster_losses(loss_func, loss_kwargs):
    """Computes the loss of each group using a loss function with energy
    and force errors as inputs.

    Parameters
    ----------
    loss_func : ``callable``
        Computes the loss of a group with ``E_errors`` and ``F_errors`` as 
        kwargs.
    loss_kwargs : :obj:`dict`
        Data to pass to ``loss_func``.

    Returns
    -------
    :obj:`numpy.ndarray`
    """
    log.info('Computing cluster losses')
    loss_kws = list(loss_kwargs.keys())
    n_clusters = len(loss_kwargs[loss_kws[0]])

    losses = []
    for i in range(n_clusters):
        cluster_loss_kwargs = {
            kwarg: loss_kwargs[kwarg][i] for kwarg in loss_kws
        }
        losses.append(loss_func(**cluster_loss_kwargs))
    
    return np.array(losses)

def idx_loss_f_mse(**kwargs):
    r"""Compute force MSE of individual structures.

    Parameters
    ----------
    **kwargs : :obj:`dict`
        Keyword arguments for the loss function.

        ``F_errors`` : (:obj:`numpy.ndarray`) -
        Force predictions errors of all structures in a cluster.
    
    Returns
    -------
    :obj:`numpy.ndarray`
    """
    F_errors = kwargs['F_errors']
    F_shape = F_errors.shape
    F_errors = F_errors.reshape((F_shape[0], F_shape[1]*F_shape[2]))
    se = F_errors**2
    mse = np.mean(se, axis=1)
    return mse

def idx_loss_f_rmse(**kwargs):
    """Compute force RMSE of individual structures.

    Parameters
    ----------
    **kwargs : :obj:`dict`
        Keyword arguments for the loss function.

        ``F_errors`` : (:obj:`numpy.ndarray`) -
        Force predictions errors of all structures in a cluster.
    
    Returns
    -------
    :obj:`numpy.ndarray`
    """
    mse = idx_loss_f_mse(**kwargs)
    return np.sqrt(mse)

def cluster_loss_F_mse(**kwargs):
    """Compute force MSE from force errors of several structures.

    Parameters
    ----------
    **kwargs : :obj:`dict`
        Keyword arguments for the loss function.

        ``F_errors`` : (:obj:`numpy.ndarray`) -
        Force predictions errors of all structures in a cluster.

    Returns
    -------
    :obj:`float`
    """
    return np.mean((kwargs['F_errors'])**2)

def cluster_loss_F_rmse(**kwargs):
    """Compute force RMSE from force errors of several structures.

    Parameters
    ----------
    **kwargs : :obj:`dict`
        Keyword arguments for the loss function.

        ``F_errors`` : (:obj:`numpy.ndarray`) -
        Force predictions errors of all structures in a cluster.

    Returns
    -------
    :obj:`float`
    """
    mse = cluster_loss_F_mse(**kwargs)
    return np.sqrt(mse)
