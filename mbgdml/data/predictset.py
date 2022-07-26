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

# pylint: disable=E1101


import numpy as np
from .. import __version__ as mbgdml_version
from .basedata import mbGDMLData
from ..mbe import mbePredict, decomp_to_total


class predictSet(mbGDMLData):
    """A predict set is a data set with mbGDML predicted energy and forces
    instead of training data.

    When analyzing many structures using mbGDML it is easier to
    predict all many-body contributions once and then analyze the stored data.

    Attributes
    ----------
    name : :obj:`str`
        File name of the predict set.
    theory : :obj:`str`
        Specifies the level of theory used for GDML training.
    entity_ids : :obj:`numpy.ndarray`
        A uniquely identifying integer specifying what atoms belong to
        which entities. Entities can be a related set of atoms, molecules,
        or functional group. For example, a water and methanol molecule
        could be ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.
    comp_ids : :obj:`numpy.ndarray`
        Relates ``entity_id`` to a fragment label for chemical components
        or species. Labels could be ``WAT`` or ``h2o`` for water, ``MeOH``
        for methanol, ``bz`` for benzene, etc. There are no standardized
        labels for species. The index of the label is the respective
        ``entity_id``. For example, a water and methanol molecule could
        be ``['h2o', 'meoh']``.
    """

    def __init__(self, pset=None):
        """
        Parameters
        ----------
        pset : :obj:`str` or :obj:`dict`, optional
            Predict set path or dictionary to initialize with.
        """
        self.name = 'predictset'
        self.type = 'p'
        self.mbgdml_version = mbgdml_version
        self._loaded = False
        self._predicted = False
        if pset is not None:
            self.load(pset)  
    
    def prepare(self):
        """Prepares a predict set by calculated the decomposed energy and
        force contributions.

        Note
        -----
        You must load a dataset first to specify ``z``, ``R``, ``entity_ids``,
        and ``comp_ids``.
        """
        E_nbody, F_nbody, entity_combs, nbody_orders = self.mbePredict.predict_decomp(
            self.z, self.R, self.entity_ids, self.comp_ids
        )
        self.nbody_orders = nbody_orders
        for i in range(len(nbody_orders)):
            order = nbody_orders[i]
            setattr(self, f'E_{order}', E_nbody[i])
            setattr(self, f'F_{order}', F_nbody[i])
            setattr(self, f'entity_combs_{order}', entity_combs[i])
        self._predicted = True

    def asdict(self):
        """Converts object into a custom :obj:`dict`.

        Returns
        -------
        :obj:`dict`
        """     
        pset = {
            'type': np.array('p'),
            'version': np.array(self.mbgdml_version),
            'name': np.array(self.name),
            'theory': np.array(self.theory),
            'z': self.z,
            'R': self.R,
            'r_unit': np.array(self.r_unit),
            'E_true': self.E_true,
            'e_unit': np.array(self.e_unit),
            'F_true': self.F_true,
            'entity_ids': self.entity_ids,
            'comp_ids': self.comp_ids,
            'nbody_orders': self.nbody_orders,
            'models_md5': self.models_md5
        }
        for nbody_order in self.nbody_orders:
            pset[f'E_{nbody_order}'] = getattr(self, f'E_{nbody_order}')
            pset[f'F_{nbody_order}'] = getattr(self, f'F_{nbody_order}')
            pset[f'entity_combs_{nbody_order}'] = getattr(
                self, f'entity_combs_{nbody_order}'
            )

        if self._loaded == False and self._predicted == False:
            if not hasattr(self, 'dataset') or not hasattr(self, 'mbgdml'):
                raise AttributeError(
                    'No data can be predicted or is not loaded.'
                )
            else: 
                self.prepare(self.z, self.R, self.entity_ids, self.comp_ids)
        
        return pset
    
    @property
    def e_unit(self):
        """Units of energy. Options are ``eV``, ``hartree``, ``kcal/mol``, and
        ``kJ/mol``.

        :type: :obj:`str`
        """
        return self._e_unit

    @e_unit.setter
    def e_unit(self, var):
        self._e_unit = var

    @property
    def E_true(self):
        """True energies from data set.

        :type: :obj:`numpy.ndarray`
        """
        return self._E_true
    
    @E_true.setter
    def E_true(self, var):
        self._E_true = var

    @property
    def F_true(self):
        """True forces from data set.

        :type: :obj:`numpy.ndarray`
        """
        return self._F_true
    
    @F_true.setter
    def F_true(self, var):
        self._F_true = var
 
    def load(self, pset):
        """Reads predict data set and loads data.
        
        Parameters
        ----------
        pset : :obj:`str` or :obj:`dict`
            Path to predict set ``npz`` file or a dictionary.
        """
        if isinstance(pset, str):
            pset = dict(np.load(pset, allow_pickle=True))
        elif not isinstance(pset, dict):
            raise TypeError(f'{type(pset)} is not supported.')

        self.name = str(pset['name'][()])
        self.theory = str(pset['theory'][()])
        self.version = str(pset['version'][()])
        self.z = pset['z']
        self.R = pset['R']

        self.r_unit = str(pset['r_unit'][()])
        self.e_unit = str(pset['e_unit'][()])
        self.E_true = pset['E_true']
        self.F_true = pset['F_true']
        self.entity_ids = pset['entity_ids']
        self.comp_ids = pset['comp_ids']
        self.nbody_orders = pset['nbody_orders']
        self.models_md5 = pset['models_md5']

        for i in pset['nbody_orders']:
            setattr(self, f'E_{i}', pset[f'E_{i}'])
            setattr(self, f'F_{i}', pset[f'F_{i}'])
            setattr(self, f'entity_combs_{i}', pset[f'entity_combs_{i}'])

        self._loaded = True

    def nbody_predictions(self, nbody_orders, n_workers=1):
        """Energies and forces of all structures including ``nbody_order``
        contributions.

        Predict sets have data that is broken down into many-body contributions.
        This function sums the many-body contributions up to the specified
        level; for example, ``3`` returns the energy and force predictions when
        including one, two, and three body contributions/corrections.

        Parameters
        ----------
        nbody_orders : :obj:`list` of :obj:`int`
            :math:`n`-body orders to include.
        
        Returns
        -------
        :obj:`numpy.ndarray`
            Energy of structures up to and including n-order corrections.
        :obj:`numpy.ndarray`
            Forces of structures up to an including n-order corrections.
        """
        if n_workers != 1:
            global ray
            import ray
            ray.is_initialized()

        E = np.zeros(self.E_true.shape)
        F = np.zeros(self.F_true.shape)
        for nbody_order in nbody_orders:
            E_decomp = getattr(self, f'E_{nbody_order}')
            F_decomp = getattr(self, f'F_{nbody_order}')
            entity_combs = getattr(self, f'entity_combs_{nbody_order}')
            if len(nbody_orders) == 1:
                return E_decomp, F_decomp
            E_nbody, F_nbody = decomp_to_total(
                E_decomp, F_decomp, self.entity_ids, entity_combs
            )
            E += E_nbody
            F += F_nbody
        return E, F

    def load_dataset(self, dset):
        """Loads data set in preparation to create a predict set.

        Parameters
        ----------
        dset : :obj:`str` or :obj:`dict`
            Path to data set or :obj:`dict` with at least the following data:

            ``z`` (:obj:`numpy.ndarray`, ndim: ``1``) - 
                Atomic numbers.

            ``R`` (:obj:`numpy.ndarray`, ndim: ``3``) - 
                Cartesian coordinates.

            ``E`` (:obj:`numpy.ndarray`, ndim: ``1``) - 
                Reference, or true, energies of the structures we will predict.

            ``F`` (:obj:`numpy.ndarray`, ndim: ``3``) - 
                Reference, or true, forces of the structures we will predict.

            ``entity_ids`` (:obj:`numpy.ndarray`, ndim: ``1``) - 
                An array specifying which atoms belong to which entities.

            ``comp_ids`` (:obj:`numpy.ndarray`, ndim: ``1``) - 
                An array relating ``entity_id`` to a fragment label for chemical
                components or species.

            ``theory`` (:obj:`str`) - 
                The level of theory used to compute energy and forces.

            ``r_unit`` (:obj:`str`) - 
                Units of distance.
                
            ``e_unit`` (:obj:`str`) - 
                Units of energy.
        """
        if isinstance(dset, str):
            dset = dict(np.load(dset, allow_pickle=True))
        elif not isinstance(dset, dict):
            raise TypeError(f'{type(dset)} is not supported')

        self.z = dset['z']
        self.R = dset['R']
        E = dset['E']
        if not isinstance(dset['E'], np.ndarray):
            E = np.array(E)
        if E.ndim == 0:
            E = np.array([E])
        self.E_true = E
        self.F_true = dset['F']
        self.entity_ids = dset['entity_ids']
        self.comp_ids = dset['comp_ids']

        if isinstance(dset['theory'], np.ndarray):
            self.theory = str(dset['theory'].item())
        else:
            self.theory = dset['theory']
        if isinstance(dset['r_unit'], np.ndarray):
            self.r_unit = str(dset['r_unit'].item())
        else:
            self.r_unit = dset['r_unit']
        if isinstance(dset['e_unit'], np.ndarray):
            self.e_unit = str(dset['e_unit'].item())
        else:
            self.e_unit = dset['e_unit']
    
    def load_models(
        self, models, predict_model, use_ray=False, n_cores=None,
        wkr_chunk_size=100
    ):
        """Loads model(s) in preparation to create a predict set.

        Parameters
        ----------
        models : :obj:`list` of :obj:`mbgdml.predict.mlWorker`
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
        if use_ray:
            global ray
            import ray
            assert ray.is_initialized()
        self.mbePredict = mbePredict(
            models, predict_model, use_ray, n_cores, wkr_chunk_size
        )
        
        models_md5, nbody_orders = [], []
        for model in self.mbePredict.models:
            if use_ray:
                model = ray.get(model)
            nbody_orders.append(model.nbody_order)
            try:
                models_md5.append(model.md5)
            except AttributeError:
                pass
        self.nbody_orders = np.array(nbody_orders)
        self.models_md5 = np.array(models_md5)
