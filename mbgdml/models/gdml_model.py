# MIT License
#
# Copyright (c) 2018-2022 Stefan Chmiela, Gregory Fonseca
# Copyright (c) 2022-2023, Alex M. Maldonado
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
from .base import Model
from ..utils import md5_data
from .._gdml.desc import _from_r
from .. import __version__ as mbgdml_version
from ..logger import GDMLLogger

log = GDMLLogger(__name__)


# pylint: disable-next=invalid-name
class gdmlModel(Model):
    def __init__(self, model, comp_ids=None, criteria=None, for_predict=True):
        """
        Parameters
        ----------
        model : :obj:`str` or :obj:`dict`
            Path to GDML npz model or :obj:`dict`.
        comp_ids : ``iterable``, default: :obj:`None`
            Model component IDs that relate entity IDs of a structure to a
            fragment label. This overrides any model ``comp_ids`` if they are
            present.
        criteria : :obj:`mbgdml.descriptors.Criteria`, default: :obj:`None`
            Initialized descriptor criteria for accepting a structure based on
            a descriptor and cutoff.
        for_predict : :obj:`bool`, default: ``True``
            Unpack the model for use in predictors.
        """
        super().__init__(criteria)

        self.type = "gdml"

        self._load(model, comp_ids)
        if for_predict:
            self._unpack()

    def _load(self, model, comp_ids):
        if isinstance(model, str):
            log.debug("Loading model from %r", model)
            model = dict(np.load(model, allow_pickle=True))
        elif isinstance(model, dict):
            log.debug("Loading model from dictionary")
        else:
            raise TypeError(f"{type(model)} is not string or dict")

        self.model_dict = model

        if "z" in model:
            self.Z = model["z"]
        elif "Z" in model:
            self.Z = model["Z"]
        else:
            raise KeyError("z or Z not found in model")
        self.r_unit = model["r_unit"].item()

        if "entity_ids" in model.keys():
            self.entity_ids = model["entity_ids"]
        else:
            self.entity_ids = None

        if comp_ids is not None:
            self.comp_ids = comp_ids
        else:
            self.comp_ids = model["comp_ids"]
        self.nbody_order = len(self.comp_ids)

    def _unpack(self):
        model = self.model_dict

        self.sig = model["sig"]
        self.n_atoms = self.Z.shape[0]
        self.desc_func = _from_r

        self.lat_and_inv = (
            (model["lattice"], np.linalg.inv(model["lattice"]))
            if "lattice" in model
            else None
        )
        self.n_train = model["R_desc"].shape[1]
        self.stdev = model["std"] if "std" in model else 1.0
        self.integ_c = model["c"]
        self.n_perms = model["perms"].shape[0]
        self.tril_perms_lin = model["tril_perms_lin"]

        self.use_torch = False
        self.R_desc_perms = (
            np.tile(model["R_desc"].T, self.n_perms)[:, self.tril_perms_lin]
            .reshape(self.n_train, self.n_perms, -1, order="F")
            .reshape(self.n_train * self.n_perms, -1)
        )
        self.R_d_desc_alpha_perms = (
            np.tile(model["R_d_desc_alpha"], self.n_perms)[:, self.tril_perms_lin]
            .reshape(self.n_train, self.n_perms, -1, order="F")
            .reshape(self.n_train * self.n_perms, -1)
        )

        if "alphas_E" in model.keys():
            self.alphas_E_lin = np.tile(
                model["alphas_E"][:, None], (1, self.n_perms)
            ).ravel()
        else:
            self.alphas_E_lin = None

    @property
    def code_version(self):
        r"""mbGDML version used to train the model.

        Raises
        ------
        AttributeError
            If accessing information when no model is loaded.
        """

        if hasattr(self, "model_dict"):
            return self.model_dict["code_version"].item()

        raise AttributeError("No model is loaded.")

    @property
    def md5(self):
        r"""Unique MD5 hash of model.

        :type: :obj:`str`
        """
        md5_properties_all = [
            "md5_train",
            "Z",
            "R_desc",
            "R_d_desc_alpha",
            "sig",
            "alphas_F",
        ]
        md5_keys = []
        md5_dict = {}
        for key in md5_properties_all:
            if hasattr(self, key):
                md5_keys.append(key)
                md5_dict[key] = getattr(self, key)
        md5_string = md5_data(md5_dict, md5_keys)
        return md5_string

    def add_modifications(self, dset):
        r"""mbGDML-specific modifications of models.

        Transfers information from data set to model.

        Parameters
        ----------
        dset : :obj:`dict`
            Dictionary of a mbGDML data set.
        """
        dset_keys = ["entity_ids", "comp_ids"]
        for key in dset_keys:
            self.model_dict[key] = dset[key]
        self.model_dict["code_version"] = np.array(mbgdml_version)
        self.model_dict["md5"] = np.array(self.md5)

    def save(self, path):
        r"""Save model dict as npz file.

        Changes in model attributions are not considered.

        Parameters
        ----------
        path : :obj:`str`
            Path to save the npz file.
        """
        np.savez_compressed(path, **self.model_dict)
