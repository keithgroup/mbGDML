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
from mbgdml import utils
import mbgdml.solvents as solvents


class mbGDMLData():
    """Parent class for mbGDML data and predict sets.
    """

    def __init__(self):
        pass


    def get_system_info(self, atoms):
        """Describes the dataset system.
        
        Parameters
        ----------
        atoms : list
            Atomic numbers of all atoms in the system. The atoms
            are repeated; for example, water is ['H', 'H', 'O'].
        """

        self.system_info = solvents.system_info(atoms)
    

    def add_system_info(self, dict_data):
        """Adds information about the system to the model.
        
        Parameters
        ----------
        dataset : dict
            Custom data structure that contains all information for a
            GDML dataset.
        
        Returns
        -------
        dict
            An updated GDML dataset with additional information regarding
            the system.
        
        Note
        ----
            If the system is a solvent, the 'solvent' name and 'cluster_size'
            is included.
        """

        if not hasattr(self, 'system_info'):
            self.get_system_info(dict_data['z'].tolist())
        
        dict_data['system'] = self.system_info['system']
        if dict_data['system'] == 'solvent':
            dict_data['solvent'] = self.system_info['solvent_name']
            dict_data['cluster_size'] = self.system_info['cluster_size']
        
        return dict_data


    def save(self, name, data, save_dir, is_dataset):
        """General save function for GDML data sets and models.
        
        Parameters
        ----------
        name : str
            Name of the file to be saved not including the
            extension.
        data : dict
            Base variables for dataset or model.
        save_dir : str
            Directory to save the file.
        is_dataset : bool
            Is the file a dataset? Controls whether the md5 of
            the file is saved.
        """

        save_dir = utils.norm_path(save_dir)
        if is_dataset:
            data['md5'] = sgdml_io.dataset_md5(data)
        save_path = save_dir + name + '.npz'
        np.savez_compressed(save_path, **data)
