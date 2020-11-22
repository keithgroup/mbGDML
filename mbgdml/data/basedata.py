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
    def n_z(self):
        """Number of atoms.

        :type: :obj:`int`
        """
        if self.z.ndim == 2:
            return int(self.z.shape[0])
        elif self.z.ndim == 3:
            return int(self.z.shape[1])

    
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
    def n_R(self):
        """Number of structures.

        :type: :obj:`int`
        """
        if self.R.ndim == 2:
            return 1
        elif self.R.ndim == 3:
            return int(self.R.shape[0])
    

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
    def e_unit(self):
        """Units of energy. Options are ``'eV'``, ``'hartree'``,
        ``'kcal/mol'``, and ``'kJ/mol'``.

        :type: :obj:`str`
        """
        return self._e_unit
    

    @e_unit.setter
    def e_unit(self, var):
        self._e_unit = var
    

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

