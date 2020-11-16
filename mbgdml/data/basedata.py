# MIT License
# 
# Copyright (c) 2020, Alex M. Maldonado
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
from sgdml.utils import io as sgdml_io
from cclib.parser.utils import convertor
from mbgdml import utils
import mbgdml.solvents as solvents


class mbGDMLData():
    """
    Parent class for mbGDML data and predict sets.

    Attributes
    ----------
    system_info : dict
        Information describing the system.
    """

    def __init__(self):
        pass
    
    @property
    def z(self):
        """Atomic numbers of all atoms in data set structures.
        
        A ``(n,)`` shape array of type :obj:`numpy.int32` containing atomic
        numbers of atoms in the structures in order as they appear.

        :type: :obj:`numpy.ndarray`
        """
        return self._z
    

    @z.setter
    def z(self, var):
        self._z = var

    
    @property
    def system_info(self):
        """
        """
        if self.z.ndim == 1:
            z_symbols = utils.atoms_by_element(self.z.tolist())
            return solvents.system_info(z_symbols)
        else:
            return {'system': 'unknown'}
    

    @property
    def R(self):
        """Atomic coordinates of structure(s).
        
        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of structures and ``n`` is the number of atoms with three 
        Cartesian components.

        :type: :obj:`numpy.ndarray`
        """
        return self._R
    

    @R.setter
    def R(self, var):
        self._R = var
    

    @property
    def r_unit(self):
        """Units of distance. Options are ``'Angstrom'`` or ``'bohr'``.

        :type: :obj:`str`
        """
        return self._r_unit
    

    @r_unit.setter
    def r_unit(self, var):
        self._r_unit = var
    

    @property
    def F(self):
        """Atomic forces of atoms in structure(s).
        
        A :obj:`numpy.ndarray` with shape of ``(m, n, 3)`` where ``m`` is the
        number of structures and ``n`` is the number of atoms with three 
        Cartesian components.

        :type: :obj:`numpy.ndarray`
        """
        return self._F
    

    @F.setter
    def F(self, var):
        self._F = var
    

    @property
    def E(self):
        """The energies of structure(s).
        
        A :obj:`numpy.ndarray` with shape of ``(n,)`` where ``n`` is the number
        of atoms.

        :type: :obj:`numpy.ndarray`
        """
        if hasattr(self, '_E'):
            return self._E
        else:
            raise AttributeError('No energies were provided in data set.')
    

    @E.setter
    def E(self, var):
        self._E = var
    

    @property
    def e_unit(self):
        """Units of energy. Options are ``'eV'``, ``'hartree'``,
        ``'kcal/mol'``, and ``'kJ/mol'``.

        :type: :obj:`str`
        """
        return self._e_unit
    

    @e_unit.setter
    def e_unit(self, var):
        self._e_unit = var
    

    @property
    def E_min(self):
        """Minimum energy of all structures.

        :type: :obj:`float`
        """
        return float(np.min(self.E.ravel()))
    

    @property
    def E_max(self):
        """Maximum energy of all structures.

        :type: :obj:`float`
        """
        return float(np.max(self.E.ravel()))
    

    @property
    def E_var(self):
        """Energy variance.

        :type: :obj:`float`
        """
        return float(np.var(self.E.ravel()))
        
    
    @property
    def E_mean(self):
        """Mean of all energies.

        :type: :obj:`float`
        """
        return float(np.mean(self.E.ravel()))
    

    @property
    def F_min(self):
        """Minimum atomic force in all structures.

        :type: :obj:`float`
        """
        return float(np.min(self.F.ravel()))
    

    @property
    def F_max(self):
        """Maximum atomic force in all structures.

        :type: :obj:`float`
        """
        return float(np.max(self.F.ravel()))
    

    @property
    def F_var(self):
        """Force variance.

        :type: :obj:`float`
        """
        return float(np.var(self.F.ravel()))
    

    @property
    def F_mean(self):
        """Mean of all forces.

        :type: :obj:`float`
        """
        return float(np.mean(self.F.ravel()))
    

    def convertE(self, E_units):
        """Convert energies and updates :attr:`e_unit`.

        Parameters
        ----------
        E_units : :obj:`str`
            Desired units of energy. Options are ``'eV'``, ``'hartree'``,
            ``'kcal/mol'``, and ``'kJ/mol'``.
        """
        self._E = convertor(self.E, self.e_unit, E_units)
        self.e_unit = E_units
    
    
    def convertR(self, R_units):
        """Convert coordinates and updates :attr:`r_unit`.

        Parameters
        ----------
        R_units : :obj:`str`
            Desired units of coordinates. Options are ``'Angstrom'`` or
            ``'bohr'``.
        """
        self._R = convertor(self.R, self.r_unit, R_units)
        self.r_unit = R_units


    def convertF(self, force_e_units, force_r_units, e_units, r_units):
        """Convert forces.

        Does not change :attr:`e_unit` or :attr:`r_unit`.

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
    

    def add_system_info(self, dict_data):
        """Adds information about the system to the model.
        
        Parameters
        ----------
        dataset : :obj:`dict`
            Contains all data as :obj:`numpy.ndarray` objects.
        
        Returns
        -------
        :obj:`dict`
            Contains all data as :obj:`numpy.ndarray` objects along with added
            system information.
        
        Notes
        -----
        If the system is a solvent, the 'solvent' name and 'cluster_size'
        is included.
        """
        z_symbols = utils.atoms_by_element(dict_data['z'].tolist())
        system_info = solvents.system_info(z_symbols)
        dict_data['system'] = np.array(system_info['system'])
        if dict_data['system'] == 'solvent':
            dict_data['solvent'] = np.array(
                system_info['solvent_name']
            )
            dict_data['cluster_size'] = np.array(
                system_info['cluster_size']
            )
        return dict_data


    def save(self, name, data, save_dir):
        """General save function for GDML data sets and models.
        
        Parameters
        ----------
        name : :obj:`str`
            Name of the file to be saved not including the ``npz`` extension.
        data : :obj:`dict`
            Data to be saved to ``npz`` file.
        save_dir : :obj:`str`
            Directory to save the file (with or without the ``'/'`` suffix).
        """
        save_dir = utils.norm_path(save_dir)
        save_path = save_dir + name + '.npz'
        np.savez_compressed(save_path, **data)

