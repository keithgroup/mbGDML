# MIT License
# 
# Copyright (c) 2020 monopsony
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

"""Identifies problematic (high error) structures for model to train on next.
Some code within this module is modified from https://github.com/fonsecag/MLFF.
"""

import logging
import numpy as np
import os
from scipy.spatial.distance import pdist

from . import clustering
from ..predict import mbPredict
from ..utils import save_json

log = logging.getLogger(__name__)

class prob_structures:
    """Find problematic structures for models in datasets.
    
    Clusters all structures in a dataset using agglomerative and k-means
    algorithms using a structural descriptor and energies.
    """

    def __init__(self, models):
        """
        Parameters
        ----------
        models : :obj:`list` of :obj:`str` or :obj:`dict`
            Contains paths or dictionaries of many-body GDML models.
        """
        self.predict = mbPredict(models)
        self.model_dicts = self.predict.models
    
    def get_pd(self, R):
        """Computes pairwise distances from atomic positions.
        
        Parameters
        ----------
        R : :obj:`numpy.ndarray`, shape: ``(n_samples, n_atoms, 3)``
            Atomic positions.
        
        Returns
        -------
        :obj:`numpy.ndarray`, shape: ``(n_samples, n_atoms*(n_atoms-1)/2)``
            Pairwise distances of atoms in each structure.
        """
        assert R.ndim == 3
        n_samples, n_atoms, _ = R.shape
        n_pd = int(n_atoms*((n_atoms-1)/2))
        R_pd = np.zeros((n_samples, n_pd))

        for i in range(len(R)):
            R_pd[i] = pdist(R[i])

        return R_pd

    def prob_cl_indices(self, cl_idxs, cl_losses):
        """Identify problematic dataset indices.

        Parameters
        ----------
        cl_idxs : :obj:`list` of :obj:`numpy.ndarray`
            Clustered dataset indices.
        cl_losses : :obj:`numpy.ndarray`
            Losses for each cluster in ``cl_idxs``.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            Dataset indices from clusters with higher-than-average losses.
        """
        log.info('Finding problematic structures')
        mean_loss = np.mean(cl_losses)
        cl_idxs_prob = np.concatenate(np.argwhere(cl_losses > mean_loss))
        clusters = np.array(cl_idxs, dtype=object)[cl_idxs_prob]
        idxs = np.concatenate(clusters)
        return idxs
    
    def n_cl_samples(self, n_sample, cl_weights, cl_pop, cl_losses):
        """Number of dataset indices to sample from each cluster.

        Parameters
        ----------
        n_sample : :obj:`int`
            Total number of dataset indices to sample from all clusters.
        cl_weights : :obj:`numpy.ndarray`
            Normalized cluster weights.
        cl_pop : :obj:`numpy.ndarray`
            Cluster populations.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            Number of dataset indices to sample from each cluster.
        """
        samples = np.array(cl_weights * n_sample)
        samples = np.floor(samples)

        # Check that selections do not sample more than the population
        for i in range(len(samples)):
            if samples[i] > cl_pop[i]:
                samples[i] = cl_pop[i]

        # Try to have at least one sample from each cluster
        # (in order of max loss)
        arg_max = (-cl_losses).argsort()
        for i in arg_max:
            if np.sum(samples) == n_sample:
                return samples
            if samples[i] == 0:
                samples[i] = 1
        
        # If there are still not enough samples, we start adding additional
        # samples in order of highest cluster losses.
        for i in arg_max:
            if np.sum(samples) == n_sample:
                return samples
            if samples[i] < cl_pop[i]:
                samples[i] += 1

        return samples.astype(int)
    
    def select_prob_indices(self, n_select, cl_idxs, idx_loss_cl):
        """Select ``n`` problematic dataset indices based on weighted cluster
        losses and distribution.

        Parameters
        ----------
        n_select : :obj:`int`
            Number of problematic dataset indices to select.
        cl_idxs : :obj:`list` of :obj:`numpy.ndarray`
            Clustered dataset indices.
        idx_loss_cl : :obj:`list` of :obj:`numpy.ndarray`
            Clustered individual structure losses. Same shape as ``cl_idxs``.
        
        Returns
        -------
        :obj:`numpy.ndarray``, shape: ``(n_select,)``
            Problematic dataset indices.
        """
        log.info('\nSelecting problematic structures')
        cl_losses = np.array([np.mean(losses) for losses in idx_loss_cl])
        cl_pop = np.array([len(_) for _ in cl_idxs])  # Cluster population

        log.info('Computing cluster loss weights')
        cl_weights = (cl_losses / np.sum(cl_losses)) * (cl_pop / np.sum(cl_pop))
        cl_weights_norm = np.array(cl_weights) / np.sum(cl_weights)

        Ns = self.n_cl_samples(n_select, cl_weights_norm, cl_pop, cl_losses)

        log.info('Sampling structures')
        n_cl = len(cl_losses)
        prob_idxs = []
        for i in range(n_cl):
            losses = idx_loss_cl[i]
            idxs = cl_idxs[i]
            ni = int(Ns[i])

            argmax = np.argsort(-losses)[:ni]
            prob_idxs.extend(idxs[argmax])

        prob_idxs = np.array(prob_idxs)
        log.debug('Selected dataset indices:')
        log.log_array(prob_idxs, level=10)
        return prob_idxs
    
    def find(
        self, dset, n_find, dset_is_train=True, write_json=True, save_dir='.'
    ):
        """Find problematic structures in a dataset.

        Uses agglomerative and k-means clustering on a dataset. First, the
        dataset is split into ``10`` clusters based on atomic pairwise
        distances. Then each cluster is further split into ``5`` clusters based
        on energies.

        Energies and forces are predicted, and then problematic structures are
        taken from clusters with higher-than-average losses. Here, the force MSE
        is used as the loss function.

        Finally, ``n_find`` structures are sampled from the 100 clusters based
        on a weighted cluster error distribution.

        Parameters
        ----------
        dset : :obj:`mbgdml.data.dataset.dataSet`
            Dataset to cluster and analyze errors.
        n_find : :obj:`int`
            Number of dataset indices to find.
        dset_is_train : :obj:`bool`, default: ``True``
            If ``dset`` is the training dataset. Training indices will be
            dropped from the analyses.
        write_json : :obj:`bool`, default: ``True``
            Write JSON file detailing clustering and prediction errors.
        save_dir : :obj:`str`, default: ``'.'``
            Directory to save any files.
        """
        log.info(
            '---------------------------\n'
            '|   Finding Problematic   |\n'
            '|       Structures        |\n'
            '---------------------------\n'
        )
        if write_json:
            self.json_dict = {}

        log.info('Loading dataset\n')
        Z, R, E, F = dset.z, dset.R, dset.E, dset.F
        entity_ids, comp_ids = dset.entity_ids, dset.comp_ids

        # Perform clustering based on pairwise distances and energies
        R_pd = self.get_pd(R)
        cl_data = (R_pd, E.reshape(-1, 1))
        cl_algos = (clustering.agglomerative, clustering.kmeans)
        cl_kwargs = ({'n_clusters': 10}, {'n_clusters': 5})
        cl_idxs = clustering.cluster_structures(
            cl_data, cl_algos, cl_kwargs
        )
        if write_json:
            self.json_dict['clustering'] = {}
            self.json_dict['clustering']['indices'] = cl_idxs
            cl_pop = [len(i) for i in cl_idxs]
            self.json_dict['clustering']['population'] = cl_pop

        log.info('\nPredicting structures')
        E_pred, F_pred = self.predict.predict(Z, R, entity_ids, comp_ids)
        log.info('Computing prediction errors')
        E_errors = E_pred - E
        F_errors = F_pred - F
        log.debug('Energy errors')
        log.log_array(E_errors, level=10)
        log.debug('Force errors')
        log.log_array(F_errors, level=10)

        log.info('\nAggregating errors')
        E_errors_cl = clustering.get_clustered_data(cl_idxs, E_errors)
        F_errors_cl = clustering.get_clustered_data(cl_idxs, F_errors)

        # Computing cluster losses
        loss_kwargs = {'F_errors': F_errors_cl}
        cl_losses = clustering.get_cluster_losses(
            clustering.cluster_loss_F_mse, loss_kwargs
        )
        if write_json:
            self.json_dict['clustering']['loss_function'] = 'force_mse'
            self.json_dict['clustering']['losses'] = cl_losses
            

        prob_indices = self.prob_cl_indices(cl_idxs, cl_losses)

        if dset_is_train:
            log.info('Dropping indices already in training set')
            if len(self.model_dicts) != 1:
                log.warning(
                    'Cannot drop training indices if there are multiple models'
                )
                log.warning('Not dropping any indices')

            assert len(self.model_dicts) == 1
            train_idxs = self.model_dicts[0]['idxs_train']
            log.debug('Training indices')
            log.log_array(train_idxs, level=10)
            n_prob_idxs0 = len(prob_indices)
            prob_indices = np.setdiff1d(prob_indices, train_idxs)
            n_prob_idxs1 = len(prob_indices)
            log.debug(f'Removed {n_prob_idxs0-n_prob_idxs1} structures')

        # Problematic clustering
        log.info('Refine clustering of problematic structures')
        # Switching to local idxs (not dataset indices) for clustering.
        R_pd_prob = R_pd[prob_indices]
        cl_data_prob = (R_pd_prob,)
        cl_algos_prob = (clustering.agglomerative,)
        cl_kwargs_prob = ({'n_clusters': 100},)
        cl_idxs_prob = clustering.cluster_structures(
            cl_data_prob, cl_algos_prob, cl_kwargs_prob
        )
        # switching back to dataset idxs
        cl_idxs_prob = [
            np.array(prob_indices[idxs]) for idxs in cl_idxs_prob
        ]

        if write_json:
            self.json_dict['problematic_clustering'] = {}
            self.json_dict['problematic_clustering']['indices'] = cl_idxs_prob
            cl_pop_prob = [len(i) for i in cl_idxs_prob]
            self.json_dict['problematic_clustering']['population'] = cl_pop_prob

        log.info('Aggregating errors for problematic structures')
        E_errors_cluster_prob = clustering.get_clustered_data(
            cl_idxs_prob, E_errors
        )
        F_errors_cluster_prob = clustering.get_clustered_data(
            cl_idxs_prob, F_errors
        )
        idx_loss_kwargs = {'F_errors': F_errors}
        structure_loss = clustering.idx_loss_f_mse(
            **idx_loss_kwargs
        )

        structure_loss_cl = clustering.get_clustered_data(
            cl_idxs_prob, structure_loss
        )
        if write_json:
            self.json_dict['problematic_clustering']['loss_function'] = 'force_mse'
            self.json_dict['problematic_clustering']['losses'] = structure_loss_cl

        next_idxs = self.select_prob_indices(
            n_find, cl_idxs_prob, structure_loss_cl
        )
        if write_json:
            self.json_dict['add_training_indices'] = next_idxs
            save_json(
                os.path.join(save_dir, 'find_problematic_indices.json'),
                self.json_dict
            )

        return next_idxs
