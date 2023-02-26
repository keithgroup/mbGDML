# MIT License
#
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

import ray
import numpy as np
from .. import __version__ as mbgdml_version
from .basedata import mbGDMLData
from ..mbe import mbePredict, decomp_to_total
from ..logger import GDMLLogger

log = GDMLLogger(__name__)


class PredictSet(mbGDMLData):
    r"""A predict set is a data set with mbGDML predicted energy and forces
    instead of training data.

    When analyzing many structures using mbGDML it is easier to predict all many-body
    contributions once and then analyze the stored data.
    """

    def __init__(self, pset=None, Z_key="Z", R_key="R", E_key="E", F_key="F"):
        r"""
        Parameters
        ----------
        pset : :obj:`str` or :obj:`dict`, default: ``None``
            Predict set path or dictionary to initialize with.
        Z_key : :obj:`str`, default: ``Z``
            :obj:`dict` key in ``pset`` for atomic numbers.
        R_key : :obj:`str`, default: ``R``
            :obj:`dict` key in ``pset`` for Cartesian coordinates.
        E_key : :obj:`str`, default: ``E``
            :obj:`dict` key in ``pset`` for energies.
        F_key : :obj:`str`, default: ``F``
            :obj:`dict` key in ``pset`` for atomic forces.
        """
        self.name = "predictset"
        r"""File name of the predict set.

        Default: ``"predictset"``

        :type: :obj:`str`
        """
        self.type = "p"
        self.mbgdml_version = mbgdml_version
        self._loaded = False
        self._predicted = False

        # Set keys for atomic properties.
        self.Z_key = Z_key
        self.R_key = R_key
        self.E_key = E_key
        self.F_key = F_key

        if pset is not None:
            self.load(pset)

    def prepare(self):
        r"""Prepares a predict set by calculated the decomposed energy and
        force contributions.

        Note
        -----
        You must load a dataset first to specify ``Z``, ``R``, ``entity_ids``,
        and ``comp_ids``.
        """
        E_nbody, F_nbody, entity_combs, nbody_orders = self.mbePredict.predict_decomp(
            self.Z, self.R, self.entity_ids, self.comp_ids
        )
        self.nbody_orders = nbody_orders
        for i, order in enumerate(nbody_orders):
            setattr(self, f"E_{order}", E_nbody[i])
            setattr(self, f"F_{order}", F_nbody[i])
            setattr(self, f"entity_combs_{order}", entity_combs[i])
        self._predicted = True

    def asdict(self):
        r"""Converts object into a custom :obj:`dict`.

        Returns
        -------
        :obj:`dict`
        """
        pset = {
            "type": np.array("p"),
            "version": np.array(self.mbgdml_version),
            "name": np.array(self.name),
            "theory": np.array(self.theory),
            self.Z_key: self.Z,
            self.R_key: self.R,
            "r_unit": np.array(self.r_unit),
            "E_true": self.E_true,
            "e_unit": np.array(self.e_unit),
            "F_true": self.F_true,
            "entity_ids": self.entity_ids,
            "comp_ids": self.comp_ids,
            "nbody_orders": self.nbody_orders,
            "models_md5": self.models_md5,
        }
        for nbody_order in self.nbody_orders:
            pset[f"E_{nbody_order}"] = getattr(self, f"E_{nbody_order}")
            pset[f"F_{nbody_order}"] = getattr(self, f"F_{nbody_order}")
            pset[f"entity_combs_{nbody_order}"] = getattr(
                self, f"entity_combs_{nbody_order}"
            )

        if self._loaded is False and self._predicted is False:
            if not hasattr(self, "dataset") or not hasattr(self, "mbgdml"):
                raise AttributeError("No data can be predicted or is not loaded.")
            self.prepare()

        return pset

    @property
    def e_unit(self):
        r"""Units of energy. Options are ``eV``, ``hartree``, ``kcal/mol``, and
        ``kJ/mol``.

        :type: :obj:`str`
        """
        return self._e_unit

    @e_unit.setter
    def e_unit(self, var):
        self._e_unit = var

    @property
    def E_true(self):
        r"""True energies from data set.

        :type: :obj:`numpy.ndarray`
        """
        return self._E_true

    @E_true.setter
    def E_true(self, var):
        self._E_true = var  # pylint: disable=invalid-name

    @property
    def F_true(self):
        r"""True forces from data set.

        :type: :obj:`numpy.ndarray`
        """
        return self._F_true

    @F_true.setter
    def F_true(self, var):
        self._F_true = var  # pylint: disable=invalid-name

    @property
    def entity_ids(self):
        r"""1D array specifying which atoms belong to which entities.

        An entity represents a related set of atoms such as a single molecule,
        several molecules, or a functional group. For mbGDML, an entity usually
        corresponds to a model trained to predict energies and forces of those
        atoms. Each ``entity_id`` is an :obj:`int` starting from ``0``.

        It is conceptually similar to PDBx/mmCIF ``_atom_site.label_entity_ids``
        data item.

        Examples
        --------
        A single water molecule would be ``[0, 0, 0]``. A water (three atoms)
        and methanol (six atoms) molecule in the same structure would be
        ``[0, 0, 0, 1, 1, 1, 1, 1, 1]``.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, "_entity_ids"):
            return self._entity_ids

        return np.array([])

    @entity_ids.setter
    def entity_ids(self, var):
        self._entity_ids = np.array(var)

    @property
    def comp_ids(self):
        r"""A 1D array relating ``entity_id`` to a fragment label for chemical
        components or species. Labels could be ``WAT`` or ``h2o`` for water,
        ``MeOH`` for methanol, ``bz`` for benzene, etc. There are no
        standardized labels for species. The index of the label is the
        respective ``entity_id``. For example, a water and methanol molecule
        could be ``['h2o', 'meoh']``.

        Examples
        --------
        Suppose we have a structure containing a water and methanol molecule.
        We can use the labels of ``h2o`` and ``meoh`` (which could be
        anything): ``['h2o', 'meoh']``. Note that the
        ``entity_id`` is a :obj:`str`.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, "_comp_ids"):
            return self._comp_ids

        return np.array([])

    @property
    def theory(self):
        r"""The level of theory used to compute energy and gradients of the data
        set.

        :type: :obj:`str`
        """
        if hasattr(self, "_theory"):
            return self._theory

        return "n/a"

    @theory.setter
    def theory(self, var):
        self._theory = var

    def load(self, pset):
        r"""Reads predict data set and loads data.

        Parameters
        ----------
        pset : :obj:`str` or :obj:`dict`
            Path to predict set ``npz`` file or a dictionary.
        """
        if isinstance(pset, str):
            pset = dict(np.load(pset, allow_pickle=True))
        elif not isinstance(pset, dict):
            raise TypeError(f"{type(pset)} is not supported.")

        self.name = str(pset["name"][()])
        self.theory = str(pset["theory"][()])
        self.version = str(pset["version"][()])
        self.Z = pset[self.Z_key]
        self.R = pset[self.R_key]

        self.r_unit = str(pset["r_unit"][()])
        self.e_unit = str(pset["e_unit"][()])
        self.E_true = pset[f"{self.E_key}_true"]
        self.F_true = pset[f"{self.F_key}_true"]
        self.entity_ids = pset["entity_ids"]
        self.comp_ids = pset["comp_ids"]
        self.nbody_orders = pset["nbody_orders"]
        self.models_md5 = pset["models_md5"]

        for i in pset["nbody_orders"]:
            setattr(self, f"{self.E_key}_{i}", pset[f"{self.E_key}_{i}"])
            setattr(self, f"{self.F_key}_{i}", pset[f"{self.F_key}_{i}"])
            setattr(self, f"entity_combs_{i}", pset[f"entity_combs_{i}"])

        self._loaded = True

    @comp_ids.setter
    def comp_ids(self, var):
        self._comp_ids = np.array(var)

    def nbody_predictions(
        self, nbody_orders, use_ray=False, n_workers=1, ray_address="auto"
    ):
        r"""Energies and forces of all structures including ``nbody_order``
        contributions.

        Predict sets have data that is broken down into many-body contributions.
        This function sums the many-body contributions up to the specified
        level; for example, ``3`` returns the energy and force predictions when
        including one, two, and three body contributions/corrections.

        Parameters
        ----------
        nbody_orders : :obj:`list` of :obj:`int`
            :math:`n`-body orders to include.
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
            Energy of structures up to and including n-order corrections.
        :obj:`numpy.ndarray`
            Forces of structures up to an including n-order corrections.
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

        E = np.zeros(self.E_true.shape)
        F = np.zeros(self.F_true.shape)
        for nbody_order in nbody_orders:
            E_decomp = getattr(self, f"{self.E_key}_{nbody_order}")
            F_decomp = getattr(self, f"{self.F_key}_{nbody_order}")
            entity_combs = getattr(self, f"entity_combs_{nbody_order}")
            if len(nbody_orders) == 1:
                return E_decomp, F_decomp
            E_nbody, F_nbody = decomp_to_total(
                E_decomp, F_decomp, self.entity_ids, entity_combs
            )
            E += E_nbody
            F += F_nbody
        return E, F

    def load_dataset(self, dset, Z_key="Z", R_key="R", E_key="E", F_key="F"):
        r"""Loads data set in preparation to create a predict set.

        Parameters
        ----------
        dset : :obj:`str` or :obj:`dict`
            Path to data set or :obj:`dict` with at least the following data:
        Z_key : :obj:`str`, default: ``Z``
            :obj:`dict` key in ``dset_path`` for atomic numbers.
        R_key : :obj:`str`, default: ``R``
            :obj:`dict` key in ``dset_path`` for atomic Cartesian coordinates.
        E_key : :obj:`str`, default: ``E``
            :obj:`dict` key in ``dset_path`` for energies.
        F_key : :obj:`str`, default: ``F``
            :obj:`dict` key in ``dset_path`` for atomic forces.
        """
        if isinstance(dset, str):
            dset = dict(np.load(dset, allow_pickle=True))
        elif not isinstance(dset, dict):
            raise TypeError(f"{type(dset)} is not supported")

        self.Z = dset[Z_key]
        self.R = dset[R_key]
        E = dset[E_key]
        if not isinstance(dset[E_key], np.ndarray):
            E = np.array(E)
        if E.ndim == 0:
            E = np.array([E])
        self.E_true = E
        self.F_true = dset[F_key]
        self.entity_ids = dset["entity_ids"]
        self.comp_ids = dset["comp_ids"]

        if isinstance(dset["theory"], np.ndarray):
            self.theory = str(dset["theory"].item())
        else:
            self.theory = dset["theory"]
        if isinstance(dset["r_unit"], np.ndarray):
            self.r_unit = str(dset["r_unit"].item())
        else:
            self.r_unit = dset["r_unit"]
        if isinstance(dset["e_unit"], np.ndarray):
            self.e_unit = str(dset["e_unit"].item())
        else:
            self.e_unit = dset["e_unit"]

    def load_models(
        self,
        models,
        predict_model,
        use_ray=False,
        n_workers=1,
        ray_address="auto",
        wkr_chunk_size=100,
    ):
        r"""Loads model(s) in preparation to create a predict set.

        Parameters
        ----------
        models : :obj:`list` of :obj:`mbgdml.models.Model`
            Machine learning model objects that contain all information to make
            predictions using ``predict_model``.
        predict_model : ``callable``
            A function that takes ``Z, R, entity_ids, nbody_gen, model`` and
            computes energies and forces. This will be turned into a ray remote
            function if ``use_ray = True``. This can return total properties
            or all individual :math:`n`-body energies and forces.
        use_ray : :obj:`bool`, default: ``False``
            Use `ray <https://docs.ray.io/en/latest/>`__ to parallelize
            computations.
        n_workers : :obj:`int`, default: ``1``
            Total number of workers available for ray. This is ignored if ``use_ray``
            is ``False``.
        ray_address : :obj:`str`, default: ``"auto"``
            Ray cluster address to connect to.
        wkr_chunk_size : :obj:`int`, default: ``100``
            Number of :math:`n`-body structures to assign to each spawned
            worker with ray.
        """
        if use_ray:
            assert ray.is_initialized()
        self.mbePredict = mbePredict(
            models, predict_model, use_ray, n_workers, ray_address, wkr_chunk_size
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
