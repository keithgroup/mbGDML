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
from ..mbe import mbePredict
from ..utils import save_json

log = logging.getLogger(__name__)

class prob_structures:
    """Find problematic structures for models in datasets.
    
    Clusters all structures in a dataset using agglomerative and k-means
    algorithms using a structural descriptor and energies.
    """

    def __init__(
        self, models, predict_model, use_ray=False, n_cores=None,
        wkr_chunk_size=100
    ):
        """
        Parameters
        ----------
        models : :obj:`list` of :obj:`mbgdml.predict.model`
            Machine learning model objects that contain all information to make
            predictions using ``predict_model``.
        predict_model : ``callable``
            A function that takes ``z, r, entity_ids, nbody_gen, model`` and
            computes energies and forces. This will be turned into a ray remote
            function if ``use_ray = True``. This can return total properties
            or all individual :math:`n`-body energies and forces.
        use_ray : :obj:`bool`, default: ``False``
            Parallelize predictions using ray. Note that initializing ray tasks
            comes with some overhead and can make smaller computations much
            slower. Thus, this is only recommended with more than 10 or so
            entities.
        n_cores : :obj:`int`, default: ``None``
            Total number of cores available for predictions when using ray. If
            ``None``, then this is determined by ``os.cpu_count()``.
        wkr_chunk_size : :obj:`int`, default: ``100``
            Number of :math:`n`-body structures to assign to each spawned
            worker with ray.
        """
        self.models = models
        self.mbe_pred = mbePredict(
            models, predict_model, use_ray, n_cores, wkr_chunk_size
        )
    
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
        # TODO: Fix circumstance when cl_idxs_prob is less than n_find
        log.info('Finding problematic structures')
        mean_loss = np.mean(cl_losses)
        cl_idxs_prob = np.concatenate(np.argwhere(cl_losses >= mean_loss))
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
        self, dset, n_find, dset_is_train=True, train_idxs=None,
        write_json=True, save_cl_plot=True,
        kwargs_subplot={'figsize': (5.5, 3), 'constrained_layout': True},
        kwargs_plot={'lolli_color': '#223F4B', 'annotate_cl_idx': False},
        image_format='png', image_dpi=600, save_dir='.'
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
        dset : :obj:`mbgdml.data.dataSet`
            Dataset to cluster and analyze errors.
        n_find : :obj:`int`
            Number of dataset indices to find.
        dset_is_train : :obj:`bool`, default: ``True``
            If ``dset`` is the training dataset. Training indices will be
            dropped from the analyses.
        train_idxs : :obj:`numpy.ndarray`, ndim: ``1``, default: ``None``
            Training indices that will be dropped if ``dset_is_train`` is
            ``True``. These do not need to be provided for GDML models (as they
            are already stored in the model).
        write_json : :obj:`bool`, default: ``True``
            Write JSON file detailing clustering and prediction errors.
        save_cl_plot : :obj:`bool`, default: ``True``
            Plot cluster losses and histogram.
        kwargs_subplot : :obj:`dict`, optional
            Subplot keyword arguments.
        kwargs_plot : :obj:`dict`, optional
            ``self.plot_cl_losses`` keyword arguments.
        image_format : :obj:`str`, default: ``png``
            Format to save the image in.
        image_dpi : :obj:`int`, default: ``600``
            Dots per inch to save the image.
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
        
        # Removing training indices.
        R_idxs_orig = np.array(list(range(len(R))))
        if dset_is_train:
            log.info('Dropping indices already in training set')
            if len(self.models) != 1:
                log.warning(
                    'Cannot drop training indices if there are multiple models'
                )
                log.warning('Not dropping any indices')
            assert len(self.models) == 1

            if train_idxs is None:
                try:
                    train_idxs = self.models[0]._model_dict['idxs_train']
                except Exception:
                    raise AttributeError('Training indices were not provided')
            else:
                assert isinstance(train_idxs, np.ndarray)
            log.debug('Training indices')
            log.log_array(train_idxs, level=10)

            n_Ri = len(R_idxs_orig)
            log.info(f'There are a total of {n_Ri} structures')
            R_idxs = np.setdiff1d(R_idxs_orig, train_idxs)
            n_Rf = len(R_idxs)
            log.info(f'Removed {n_Ri-n_Rf} structures')
        else: 
            R_idxs = [np.array(range(n_structures))]
        # Note: Indices from this point on do not directly map to their index
        # in the dataset. We have to convert back to their original indices
        # when necessary. We refer to R_idxs as no-training indices.

        # Perform clustering based on pairwise distances and energies
        R, E, F = R[R_idxs], E[R_idxs], F[R_idxs]
        R_pd = self.get_pd(R)
        cl_data = (R_pd, E.reshape(-1, 1))
        cl_algos = (clustering.agglomerative, clustering.kmeans)
        cl_kwargs = ({'n_clusters': 10}, {'n_clusters': 5})
        cl_idxs = clustering.cluster_structures(
            cl_data, cl_algos, cl_kwargs
        )

        cl_pop = [len(i) for i in cl_idxs]
        if write_json:
            # Convert back to dataset indices just to write.
            # The no-train indices is still needed to compute errors and
            # problematic clustering.
            cl_idxs_write = [np.array(R_idxs[idxs]) for idxs in cl_idxs]
            self.json_dict['clustering'] = {}
            self.json_dict['clustering']['indices'] = cl_idxs_write
            self.json_dict['clustering']['population'] = cl_pop

        log.info('\nPredicting structures')
        E_pred, F_pred = self.mbe_pred.predict(Z, R, entity_ids, comp_ids)
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

        # Problematic clustering
        log.info('Refine clustering of problematic structures')
        # Switching to problematic idxs for clustering.
        R_pd_prob = R_pd[prob_indices]
        cl_data_prob = (R_pd_prob,)
        cl_algos_prob = (clustering.agglomerative,)
        cl_kwargs_prob = ({'n_clusters': 100},)
        cl_idxs_prob = clustering.cluster_structures(
            cl_data_prob, cl_algos_prob, cl_kwargs_prob
        )
        # switching back to no-training idxs
        cl_idxs_prob = [
            np.array(prob_indices[idxs]) for idxs in cl_idxs_prob
        ]

        cl_pop_prob = [len(i) for i in cl_idxs_prob]
        if write_json:
            # Convert back to dataset indices just to write.
            cl_idxs_prob_write = [np.array(R_idxs[idxs]) for idxs in cl_idxs_prob]
            self.json_dict['problematic_clustering'] = {}
            self.json_dict['problematic_clustering']['indices'] = cl_idxs_prob_write
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
        # Convert back to dataset indices.
        next_idxs = R_idxs[next_idxs]
        if write_json:
            self.json_dict['add_training_indices'] = next_idxs
            save_json(
                os.path.join(save_dir, 'find_problematic_indices.json'),
                self.json_dict
            )
        
        if save_cl_plot:
            kwargs_subplot = {'figsize': (5.5, 3), 'constrained_layout': True}
            fig = self.plot_cl_losses(
                cl_pop, cl_losses, 'Force MSE', kwargs_subplot=kwargs_subplot,
                **kwargs_plot
            )
            fig.savefig(
                os.path.join(save_dir, f'cl_losses.{image_format}'),
                dpi=image_dpi
            )

        return next_idxs
    
    def plot_cl_losses(
        self, cl_pop, cl_losses, loss_label, kwargs_subplot={},
        lolli_color='#223F4B', annotate_cl_idx=False
    ):
        """Plot cluster losses and population histogram using matplotlib.

        Parameters
        ----------
        cl_pop : :obj:`numpy.ndarray`
            Cluster populations (unsorted).
        cl_losses : :obj:`numpy.ndarray`
            Cluster losses (unsorted).
        kwargs_subplot : :obj:`dict`
            Subplot keyword arguments.
        lolli_color : :obj:`str`, default: ``#223F4B``
            Lollipop color.
        annotate_cl_idx : :obj:`bool`, default: ``False``
            Add the cluster index above the cluster loss value.
        
        Returns
        -------
        ``object``
            A matplotlib figure object.
        """
        import matplotlib.pyplot as plt
        cl_width = 1

        cl_losses = np.array(cl_losses)
        cl_pop = np.array(cl_pop)

        loss_sort = np.argsort(cl_losses)
        cl_pop = cl_pop[loss_sort]
        cl_losses = cl_losses[loss_sort]

        n_cl = len(cl_pop)
        cl_plot_x = np.array(range(n_cl))*cl_width

        fig, ax_pop = plt.subplots(nrows=1, ncols=1, **kwargs_subplot)
        ax_loss = ax_pop.twinx()

        ax_loss.yaxis.set_ticks_position('left')
        ax_loss.yaxis.set_label_position('left')
        ax_pop.yaxis.set_ticks_position('right')
        ax_pop.yaxis.set_label_position('right')

        # Cluster losses
        ax_loss.set_ylabel(loss_label)
        ax_loss.vlines(
            x=cl_plot_x, ymin=0, ymax=cl_losses, linewidth=0.8, color=lolli_color
        )
        ax_loss.scatter(cl_plot_x, cl_losses, s=2, color=lolli_color)

        # Losses mean
        ax_loss.axhline(
            np.mean(cl_losses), color=lolli_color, alpha=1,
            linewidth=1.0, linestyle=':'
        )
        ax_loss.text(
            0.5, np.mean(cl_losses), 'Mean', fontsize=8
        )

        # population histogram (bar chart)
        ax_pop.set_xlabel('Cluster')
        ax_pop.set_ylabel('Size')
        edge_shift = cl_width/2
        edges = [i-edge_shift for i in cl_plot_x] + [cl_plot_x[-1] + edge_shift]
        ax_pop.stairs(
            values=cl_pop, edges=edges, fill=False, baseline=0.0, zorder=-1.0,
            edgecolor='lightgrey', alpha=1.0
        )

        # Annotate with cluster index
        if annotate_cl_idx:
            for i in range(len(loss_sort)):
                cl_idx = loss_sort[i]
                cl_x = cl_plot_x[i]
                if cl_idx < 10:
                    x_disp = -1.5
                else:
                    x_disp = -2.7
                ax_loss.annotate(
                    str(cl_idx), (cl_x, cl_losses[i]), xytext=(x_disp, 3), xycoords='data',
                    fontsize=4, fontweight='bold', textcoords='offset points',
                    color=lolli_color
                )

        # Handle axes label
        ax_pop.set_xticks([])

        ax_loss.set_xlim(left=edges[0], right=edges[-1])

        ax_loss.set_ylim(bottom=0)
        ax_pop.set_ylim(bottom=0)

        return fig
