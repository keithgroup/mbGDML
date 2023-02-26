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

import os
import numpy as np
from cclib.parser.utils import convertor
from .basedata import mbGDMLData
from .. import utils
from .. import _version
from ..logger import GDMLLogger

mbgdml_version = _version.get_versions()["version"]

log = GDMLLogger(__name__)


class DataSet(mbGDMLData):
    r"""For creating, loading, manipulating, and using data sets."""

    def __init__(self, dset_path=None, Z_key="Z", R_key="R", E_key="E", F_key="F"):
        """
        Parameters
        ----------
        dset_path : :obj:`str`, optional
            Path to a `npz` file.
        Z_key : :obj:`str`, default: ``Z``
            :obj:`dict` key in ``dset_path`` for atomic numbers.
        R_key : :obj:`str`, default: ``R``
            :obj:`dict` key in ``dset_path`` for Cartesian coordinates.
        E_key : :obj:`str`, default: ``E``
            :obj:`dict` key in ``dset_path`` for energies.
        F_key : :obj:`str`, default: ``F``
            :obj:`dict` key in ``dset_path`` for atomic forces.
        """
        self.type = "d"
        self.name = "dataset"

        # Set keys for atomic properties.
        self.Z_key = Z_key
        self.R_key = R_key
        self.E_key = E_key
        self.F_key = F_key

        if dset_path is not None:
            self.load(dset_path)

    @property
    def name(self):
        r"""Human-readable label for the data set.

        :type: :obj:`str`
        """
        if hasattr(self, "_name"):
            return self._name

        return None

    @name.setter
    def name(self, var):
        self._name = str(var)

    @property
    def r_prov_ids(self):
        r"""Specifies structure sets IDs/labels and corresponding MD5 hashes.

        Keys are the Rset IDs (:obj:`int`) and values are MD5 hashes
        (:obj:`str`) for the particular structure set.

        This is used as a breadcrumb trail that specifies where each structure
        in the data set originates from.

        Examples
        --------
        >>> dset.r_prov_ids
        {0: '2339670ad87a606cb11a72191dfd9f58'}

        :type: :obj:`dict`
        """
        if hasattr(self, "_r_prov_ids"):
            return self._r_prov_ids

        return {}

    @r_prov_ids.setter
    def r_prov_ids(self, var):
        self._r_prov_ids = var

    @property
    def r_prov_specs(self):
        r"""An array specifying where each structure in ``R`` originates from.

        A ``(n_R, 1 + n_entity)`` array where each row contains the Rset ID
        from ``r_prov_ids`` (e.g., 0, 1, 2, etc.) then the structure index and
        entity_ids from the original full structure in the structure set.

        If there has been no previous sampling, an array of shape (1, 0)
        is returned.

        :type: :obj:`numpy.ndarray`

        Examples
        --------
        >>> dset.r_prov_specs  # [r_prov_id, r_index, entity_1, entity_2, entity_3]
        array([[0, 985, 46, 59, 106],
               [0, 174, 51, 81, 128]])
        """
        if hasattr(self, "_r_prov_specs"):
            return self._r_prov_specs

        return np.array([[]], dtype="int_")

    @r_prov_specs.setter
    def r_prov_specs(self, var):
        self._r_prov_specs = var

    @property
    def F(self):
        r"""Atomic forces of atoms in structure(s).

        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of structures and ``n`` is the number of atoms with three
        Cartesian components.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, "_F"):
            return self._F

        return np.array([[[]]])

    @F.setter
    def F(self, var):
        self._F = var  # pylint: disable=invalid-name

    @property
    def E(self):
        r"""The energies of structure(s).

        A :obj:`numpy.ndarray` with shape of ``(n,)`` where ``n`` is the number
        of atoms.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, "_E"):
            return self._E

        return np.array([])

    @E.setter
    def E(self, var):
        self._E = var  # pylint: disable=invalid-name

    @property
    def e_unit(self):
        r"""Units of energy. Options are ``'eV'``, ``'hartree'``,
        ``'kcal/mol'``, and ``'kJ/mol'``.

        :type: :obj:`str`
        """
        if hasattr(self, "_e_unit"):
            return self._e_unit

        return "n/a"

    @e_unit.setter
    def e_unit(self, var):
        self._e_unit = var

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

    @property
    def E_min(self):  # pylint: disable=invalid-name
        r"""Minimum energy of all structures.

        :type: :obj:`float`
        """
        return float(np.min(self.E.ravel()))

    @property
    def E_max(self):  # pylint: disable=invalid-name
        r"""Maximum energy of all structures.

        :type: :obj:`float`
        """
        return float(np.max(self.E.ravel()))

    @property
    def E_var(self):  # pylint: disable=invalid-name
        r"""Energy variance.

        :type: :obj:`float`
        """
        return float(np.var(self.E.ravel()))

    @property
    def E_mean(self):  # pylint: disable=invalid-name
        r"""Mean of all energies.

        :type: :obj:`float`
        """
        return float(np.mean(self.E.ravel()))

    @property
    def F_min(self):  # pylint: disable=invalid-name
        r"""Minimum atomic force in all structures.

        :type: :obj:`float`
        """
        return float(np.min(self.F.ravel()))

    @property
    def F_max(self):  # pylint: disable=invalid-name
        r"""Maximum atomic force in all structures.

        :type: :obj:`float`
        """
        return float(np.max(self.F.ravel()))

    @property
    def F_var(self):  # pylint: disable=invalid-name
        r"""Force variance.

        :type: :obj:`float`
        """
        return float(np.var(self.F.ravel()))

    @property
    def F_mean(self):  # pylint: disable=invalid-name
        r"""Mean of all forces.

        :type: :obj:`float`
        """
        return float(np.mean(self.F.ravel()))

    @property
    def md5(self):
        r"""Unique MD5 hash of data set.

        Notes
        -----
        ``Z`` and ``R`` are always used to generate the MD5 hash. If available,
        :obj:`mbgdml.data.DataSet.E` and :obj:`mbgdml.data.DataSet.F` are used.

        :type: :obj:`str`
        """
        try:
            return self.asdict()["md5"].item().decode()
        except BaseException:
            print("Not enough information in dset for MD5")
            raise

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

    @comp_ids.setter
    def comp_ids(self, var):
        self._comp_ids = np.array(var)

    @property
    def mb(self):
        r"""Many-body expansion order of this data set. This is :obj:`None` if the
        data set does not contain many-body energies and forces.

        :type: :obj:`int`
        """
        if hasattr(self, "_mb"):
            return self._mb

        return None

    @mb.setter
    def mb(self, var):
        self._mb = int(var)

    @property
    def mb_dsets_md5(self):
        r"""All MD5 hash of data sets used to remove n-body contributions from
        data sets.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, "_mb_dsets_md5"):
            return self._mb_dsets_md5

        return np.array([])

    @mb_dsets_md5.setter
    def mb_dsets_md5(self, var):
        self._mb_dsets_md5 = var.astype(str)

    @property
    def mb_models_md5(self):
        r"""All MD5 hash of models used to remove n-body contributions from
        models.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, "_mb_models_md5"):
            return self._mb_models_md5

        return np.array([])

    @mb_models_md5.setter
    def mb_models_md5(self, var):
        self._mb_models_md5 = var.astype(str)

    # pylint: disable-next=invalid-name
    def convertE(self, E_units):
        r"""Convert energies and updates ``e_unit``.

        Parameters
        ----------
        E_units : :obj:`str`
            Desired units of energy. Options are ``'eV'``, ``'hartree'``,
            ``'kcal/mol'``, and ``'kJ/mol'``.
        """
        self._E = convertor(self.E, self.e_unit, E_units)
        self.e_unit = E_units

    # pylint: disable-next=invalid-name
    def convertR(self, R_units):
        r"""Convert coordinates and updates ``r_unit``.

        Parameters
        ----------
        R_units : :obj:`str`
            Desired units of coordinates. Options are ``'Angstrom'`` or
            ``'bohr'``.
        """
        self._R = convertor(self.R, self.r_unit, R_units)
        self.r_unit = R_units

    # pylint: disable-next=invalid-name
    def convertF(self, force_e_units, force_r_units, e_units, r_units):
        r"""Convert forces.

        Does not change ``e_unit`` or ``r_unit``.

        Parameters
        ----------
        force_e_units : :obj:`str`
            Specifies package-specific energy units used in calculation.
            Available units are ``'eV'``, ``'hartree'``, ``'kcal/mol'``, and
            ``'kJ/mol'``.
        force_r_units : :obj:`str`
            Specifies package-specific distance units used in calculation.
            Available units are ``'Angstrom'`` and ``'bohr'``.
        e_units : :obj:`str`
            Desired units of energy. Available units are ``'eV'``,
            ``'hartree'``, ``'kcal/mol'``, and ``'kJ/mol'``.
        r_units : :obj:`str`
            Desired units of distance. Available units are ``'Angstrom'`` and
            ``'bohr'``.
        """
        self._F = utils.convert_forces(
            self.F, force_e_units, force_r_units, e_units, r_units
        )

    def _update(self, dataset):
        r"""Updates object attributes.

        Parameters
        ----------
        dataset : :obj:`dict`
            Contains all information and arrays stored in data set.
        """
        self.name = dataset["name"].item()
        self._Z = dataset[self.Z_key]
        self._R = dataset[self.R_key]
        self._E = dataset[self.E_key]
        self._F = dataset[self.F_key]
        self._r_unit = dataset["r_unit"].item()
        try:
            self._e_unit = dataset["e_unit"].item()
        except KeyError:
            self._e_unit = "n/a"
        try:
            self.mbgdml_version = dataset["mbgdml_version"].item()
        except KeyError:
            # Some old data sets do not have this information.
            # This is unessential, so we will just ignore this.
            pass
        try:
            self.theory = dataset["theory"].item()
        except KeyError:
            self.theory = "n/a"
        # mbGDML added data set information.
        if "mb" in dataset.keys():
            self.mb = dataset["mb"].item()
        if "mb_models_md5" in dataset.keys():
            self.mb_models_md5 = dataset["mb_models_md5"]
        if "mb_dsets_md5" in dataset.keys():
            self.mb_dsets_md5 = dataset["mb_dsets_md5"]

        try:
            self.r_prov_specs = dataset["r_prov_specs"][()]
            self.r_prov_ids = dataset["r_prov_ids"][()]
            self.entity_ids = dataset["entity_ids"]
            self.comp_ids = dataset["comp_ids"]
        except KeyError:
            pass

        if "centered" in dataset.keys():
            self.centered = dataset["centered"].item()

    def load(self, dataset_path):
        r"""Read data set.

        Parameters
        ----------
        dataset_path : :obj:`str`
            Path to NumPy ``npz`` file.
        """
        dataset_npz = np.load(dataset_path, allow_pickle=True)
        npz_type = dataset_npz.f.type.item()

        if npz_type != "d":
            raise ValueError(f"{npz_type} is not a data set.")

        self._update(dict(dataset_npz))

    # pylint: disable-next=too-many-branches
    def asdict(self, gdml_keys=True):
        r"""Converts object into a custom :obj:`dict`.

        Parameters
        ----------
        gdml_keys : :obj:`bool`, default: ``True``
            Data sets can use any keys to specify atomic data. However, mbGDML uses
            the standard of ``Z`` for atomic numbers, ``R`` for structure coordinates,
            ``E`` for energies, and ``F`` for forces. Using this option changes the
            data set keys to GDML keys.

        Returns
        -------
        :obj:`dict`
        """
        # Data always available for data sets.
        dataset = {
            "type": np.array("d"),
            "mbgdml_version": np.array(mbgdml_version),
            "name": np.array(self.name),
            "r_prov_ids": np.array(self.r_prov_ids),
            "r_prov_specs": np.array(self.r_prov_specs),
            self.Z_key: np.array(self.Z),
            self.R_key: np.array(self.R),
            "r_unit": np.array(self.r_unit),
            "entity_ids": self.entity_ids,
            "comp_ids": self.comp_ids,
        }
        if gdml_keys:
            md5_properties = ["Z", "R"]
        else:
            md5_properties = [self.Z_key, self.R_key]

        # When starting a new data set from a structure set, there will not be
        # any energy or force data. Thus, we try to add the data if available,
        # but will not error out if the data is not available.
        # Theory.
        try:
            dataset["theory"] = np.array(self.theory)
        except BaseException:
            pass

        # Energies.
        try:
            dataset[self.E_key] = np.array(self.E)
            dataset["e_unit"] = np.array(self.e_unit)
            dataset["E_min"] = np.array(self.E_min)
            dataset["E_max"] = np.array(self.E_max)
            dataset["E_mean"] = np.array(self.E_mean)
            dataset["E_var"] = np.array(self.E_var)
            if gdml_keys:
                md5_properties.append("E")
            else:
                md5_properties.append(self.E_key)
        except BaseException:
            pass

        # Forces.
        try:
            dataset[self.F_key] = np.array(self.F)
            dataset["F_min"] = np.array(self.F_min)
            dataset["F_max"] = np.array(self.F_max)
            dataset["F_mean"] = np.array(self.F_mean)
            dataset["F_var"] = np.array(self.F_var)
            if gdml_keys:
                md5_properties.append("F")
            else:
                md5_properties.append(self.F_key)
        except BaseException:
            pass

        # mbGDML information.
        if hasattr(self, "mb") and self.mb is not None:
            dataset["mb"] = np.array(self.mb)
        if len(self.mb_models_md5) > 0:
            dataset["mb_models_md5"] = np.array(self.mb_models_md5)
        if len(self.mb_dsets_md5) > 0:
            dataset["mb_dsets_md5"] = np.array(self.mb_dsets_md5)

        if hasattr(self, "centered"):
            dataset["centered"] = np.array(self.centered)

        if gdml_keys:
            dataset["Z"] = dataset.pop(self.Z_key)
            dataset["R"] = dataset.pop(self.R_key)
            dataset["E"] = dataset.pop(self.E_key)
            dataset["F"] = dataset.pop(self.F_key)

        dataset["md5"] = np.array(utils.md5_data(dataset, md5_properties))
        return dataset

    def print(self):
        r"""Prints all structure coordinates, energies, and forces of a data set."""
        num_config = self.R.shape[0]
        for config in range(num_config):
            r_prov_spec = self.r_prov_specs[config]
            print(f"-----Configuration {config}-----")
            print(
                f"r_prov_id: {int(r_prov_spec[0])}     "
                f"Structure index: {int(r_prov_spec[1])}"
            )
            print(f"Molecule indices: {r_prov_spec[2:]}")
            print(f"Coordinates:\n{self.R[config]}")
            print(f"Energy: {self.E[config]}")
            print(f"Forces:\n{self.F[config]}\n")

    def write_xyz(self, save_dir):
        r"""Saves xyz file of all structures in data set.

        Parameters
        ----------
        save_dir : :obj:`str`
        """
        xyz_path = os.path.join(save_dir, self.name)
        utils.write_xyz(xyz_path, self.Z, self.R)
