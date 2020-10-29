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

from periodictable import elements

class solvent():
    """Identifies and describes the solvent for MB-GDML.
    
    Represents the solvent that makes up the cluster MB-GDML is being trained
    from.
    
    Attributes
    ----------
    system : str
        Designates what kind of system this is. Currently only 'solvent' 
        systems.
    solvent_name : str
        Name of the solvent.
    solvent_label : str
        A label for the solvent for filenaming purposes.
    solvent_molec_size : int
        Total number of atoms in one solvent molecule.
    cluster_size : int
        Total number of solvent molecules in cluster.
    """


    def __init__(self, atom_list):

        # Gets available solvents
        self.all_solvents = solvent_data
        self.identify_solvent(atom_list, self.all_solvents)


    def atom_numbers(self, chem_formula):
        """Provides a dictionary of atoms and their quantity from chemical
        formula.
        
        Parameters
        ----------
        chem_formula : str
            The chemical formula of a single solvent  molecule. For example, 
            'CH3OH' for methanol, 'H2O' for water, and 'C2H3N' for acetonitrile.
        
        Returns
        -------
        dict
            Contains the atoms as their elemental symbol for keys and their 
            quantity as values.
        
        Examples
        -------
        >>> atom_numbers('CH3OH')
        """

        string_list = list(chem_formula)
        atom_dict = {}
        str_index = 0
        while str_index < len(string_list):
            next_index = str_index + 1
            if string_list[str_index].isalpha():
                # Checks to see if there is more than one of this atom type.
                try:
                    if string_list[next_index].isalpha():
                        number = 1
                        index_change  = 1
                    elif string_list[next_index].isdigit():
                        number = int(string_list[next_index])
                        index_change  = 2
                except IndexError:
                    number = 1
                    index_change  = 1

            # Adds element numbers to atom_dict
            if string_list[str_index] in atom_dict:
                atom_dict[string_list[str_index]] += number
            else:
                atom_dict[string_list[str_index]] = number
            
            str_index += index_change

        return atom_dict


    def identify_solvent(self, atom_list, all_solvents):
        """Identifies the solvent from a repeated list of elements.
        
        Parameters
        ----------
        atom_list : list
            Elements as strings. Elements should be repeated. For example, 
            ['H', 'H', 'O', 'O', 'H', 'H']. Note that the order does not 
            matter; only the quantity.
        all_solvents : dict
            Contains all solvents described in solvents.json. Keys are the name 
            of the solvent, and the values are dicts 'label' and 'formula' that 
            provide a solvent label and chemical formula, respectively.
        """

        # Converts atoms identified by their atomic number into element symbols
        # for human redability.
        if str(atom_list[0]).isdigit():
            atom_list_elements = []
            for atom in atom_list:
                atom_list_elements.append(str(elements[atom]))
            atom_list = atom_list_elements

        # Determines quantity of each element in atom list.
        # Example: {'H': 4, 'O': 2}
        atom_num = {}
        for atom in atom_list:
            if atom in atom_num.keys():
                atom_num[atom] += 1
            else:
                atom_num[atom] = 1
        
        # Identifies solvent by comparing multiples of element numbers for each
        # solvent in the json file. Note, this does not differentiate
        # between isomers or compounds with the same chemical formula.
        # Loops through available solvents and solvent_atoms.
        for solvent in all_solvents:
            
            solvent_atoms = self.atom_numbers(all_solvents[solvent]['formula'])
            # Number of elements in atom_list should equal that of the solvent.
            if len(atom_num) == len(solvent_atoms):

                # Tests all criteria to identify solvent. If this fails, it moves
                # onto the next solvent. If all of them fail, it raises an error.
                try:
                    # Checks that the number of atoms is a multiple of the solvent.
                    # Also checks that the multiples of all the atoms are the same.
                    solvent_numbers = []

                    # Checks that atoms are multiples of a solvent.
                    for atom in solvent_atoms:
                        multiple = atom_num[atom] / solvent_atoms[atom]
                        if multiple.is_integer():
                            solvent_numbers.append(multiple)
                        else:
                            raise ValueError

                    # Checks that all multiples are the same.
                    test_multiple = solvent_numbers[0]
                    for multiple in solvent_numbers:
                        if multiple != test_multiple:
                            raise ValueError
                    
                    
                    self.solvent_name = str(solvent)
                    self.solvent_label = str(all_solvents[solvent]['label'])
                    self.solvent_molec_size = int(sum(solvent_atoms.values()))
                    self.cluster_size = int(atom_num[atom] \
                                            / solvent_atoms[atom])
                    self.system = 'solvent'  # For future
                    # developments that could involve solute-solvent systems.
                    
                except:
                    pass

        if not hasattr(self, 'solvent_name'):
            print('The solvent could not be identified.')


def system_info(atoms):
    """Determines information about the system key to mbGDML.
    
    Parameters
    ----------
    atoms : list
        Atomic numbers of all atoms in the system.
    
    Returns
    -------
    dict
        System information useful for mbGDML. Information includes 'system' 
        which categorizes the system and provides specific information.
    
    Notes
    -----
    For a 'solvent' system the additional information returned is the
    'solvent_name', 'solvent_label', 'solvent_molec_size', and 'cluster_size'.
    """
    
    system_info = solvent(atoms)
    system_info_dict = {'system': system_info.system}
    if system_info_dict['system'] is 'solvent':
        system_info_dict['solvent_name'] = system_info.solvent_name
        system_info_dict['solvent_label'] = system_info.solvent_label
        system_info_dict['solvent_molec_size'] = system_info.solvent_molec_size
        system_info_dict['cluster_size'] = system_info.cluster_size
    
    return system_info_dict

solvent_data = {
    "water": {
        "label": "H2O",
        "formula": "H2O"
    },
    "acetonitrile": {
        "label": "acn",
        "formula": "C2H3N"
    },
    "methanol": {
        "label": "MeOH",
        "formula": "CH3OH"
    }
}