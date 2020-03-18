import os
import json
from periodictable import elements
# TODO Create solvent class
# TODO Create solvent file that contains solvent information to initialize objects

class solvent():
    """Identifies and describes the solvent for MB-GDML.
    
    Represents the solvent that makes up the cluster MB-GDML is being trained
    from.
    
    Attributes:
        all_solvents (dict): Contains all solvents described in solvents.json.
            Keys are the name of the solvent, and the values are dicts 'label'
            and 'formula' that provide a solvent label and chemical formula,
            respectively.
        solvent_name (str): Name of the solvent.
        solvent_label (str): A label for the solvent for filenaming purposes.
        solvent_molec_size (int): Total number of atoms in one solvent molecule.
        cluster_size (int): Total number of solvent molecules in cluster.
    """

    def __init__(self, atom_list):

        # Gets available solvents from json file.
        solvent_json = os.path.dirname(os.path.realpath(__file__)) \
                       + '/solvents.json'
        with open(solvent_json, 'r') as solvent_file:
            solvent_data=solvent_file.read()
        all_solvents = json.loads(solvent_data)
        
        self.identify_solvent(atom_list, all_solvents)

        

    def atom_numbers(self, chem_formula):
        """Provides a dictionary of atoms and their quantity from chemical
        formula.
        
        Args:
            chem_formula (str): the chemical formula of a single solvent
                molecule. For example, 'CH3OH' for methanol, 'H2O' for water,
                and 'C2H3N' for acetonitrile.
        
        Returns:
            dict: contains the atoms as their elemental symbol for keys and
                their quantity as values.
        
        Example:
            atom_numbers('CH3OH')
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
        
        Args:
            atom_list (lst): List of elements as strings. Elements should be
                repeated. For example, ['H', 'H', 'O', 'O', 'H', 'H']. Note that
                the order does not matter; only the quantity.
            all_solvents (dict): Contains all solvents described in
            solvents.json. Keys are the name of the solvent, and the values are
            dicts 'label' and 'formula' that provide a solvent label and
            chemical formula, respectively.
        
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
                    
                    
                except:
                    pass

        if not hasattr(self, 'solvent_name'):
            print('The solvent could not be identified.')


solvent_atoms_old = {
    'water': {'H': 2, 'O': 1},
    'methanol': {'C': 1, 'H': 4, 'O': 1},
    'acetonitrile': {'C': 2, 'H': 3, 'N': 1}
}

solvent_size_old ={
    'water': 3,
    'methanol': 6,
    'acetonitrile': 6
}

solvent_labels_old ={
    'water': 'H2O',
    'methanol': 'MeOH',
    'acetonitrile': 'acn'
}

def identify_solvent(atom_list):
    """Identifies the solvent from a repeated list of elements from an xyz file.
    
    Args:
        atom_list (lst): List of elements as strings. Elements should be
            repeated. For example, ['H', 'H', 'O', 'O', 'H', 'H']. Note that
            the order does not matter; only the quantity.
    
    Returns:
        dict: Contains 'solvent', 'label', 'solvent_size_old', and
                'cluster_size'.
    """

    # Converts atoms identified by their atomic number into element symbols
    # for human redability.
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
    # solvent in solvent_atoms_old dictionary. Note, this does not differentiate
    # between isomers or compounds with the same chemical formula.
    # Loops through available solvents and solvent_atoms_old.
    for solvent in solvent_atoms_old:

        # Number of elements in atom_list should equal that of the solvent.
        if len(atom_num) == len(solvent_atoms_old[solvent]):

            # Tests all criteria to identify solvent. If this fails, it moves
            # onto the next solvent. If all of them fail, it raises an error.
            try:
                # Checks that the number of atoms is a multiple of the solvent.
                # Also checks that the multiples of all the atoms are the same.
                solvent_numbers = []

                # Checks that atoms are multiples of a solvent.
                for atom in solvent_atoms_old[solvent]:
                    multiple = atom_num[atom] / solvent_atoms_old[solvent][atom]
                    if multiple.is_integer():
                        solvent_numbers.append(multiple)
                    else:
                        raise ValueError

                # Checks that all multiples are the same.
                test_multiple = solvent_numbers[0]
                for multiple in solvent_numbers:
                    if multiple != test_multiple:
                        raise ValueError
                
                return {
                    'solvent': solvent,
                    'solvent_label': solvent_labels_old[solvent],
                    'solvent_size': solvent_size_old[solvent],
                    'cluster_size': int(atom_num[atom] \
                                         / solvent_atoms_old[solvent][atom])
                }
            except:
                pass
    
    print('The solvent could not be identified.')
    raise ValueError
