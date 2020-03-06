from periodictable import elements

solvent_atoms = {
    'water': {'H': 2, 'O': 1},
    'methanol': {'C': 1, 'H': 4, 'O': 1},
    'acetonitrile': {'C': 2, 'H': 3, 'N': 1}
}

solvent_size ={
    'water': 3,
    'methanol': 6,
    'acetonitrile': 6
}

solvent_labels ={
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
        dict: Contains 'solvent' and 'size' information.
    """

    # Converts atoms identified by their atomic number into element symbols
    # for human redability.
    atom_list_elements = []
    for atom in atom_list:
        atom_list_elements.append(str(elements[atom]))
    atom_list = atom_list_elements

    # Determines quantity of each element in atom list.
    # Example: {'H': 4, 'O': 2}
    atom_numbers = {}
    for atom in atom_list:
        if atom in atom_numbers.keys():
            atom_numbers[atom] += 1
        else:
            atom_numbers[atom] = 1
    
    # Identifies solvent by comparing multiples of element numbers for each
    # solvent in solvent_atoms dictionary. Note, this does not differentiate
    # between isomers or compounds with the same chemical formula.
    # Loops through available solvents and solvent_atoms.
    for solvent in solvent_atoms:

        # Number of elements in atom_list should equal that of the solvent.
        if len(atom_numbers) == len(solvent_atoms[solvent]):

            # Tests all criteria to identify solvent. If this fails, it moves
            # onto the next solvent. If all of them fail, it raises an error.
            try:
                # Checks that the number of atoms is a multiple of the solvent.
                # Also checks that the multiples of all the atoms are the same.
                solvent_numbers = []

                # Checks that atoms are multiples of a solvent.
                for atom in solvent_atoms[solvent]:
                    multiple = atom_numbers[atom] / solvent_atoms[solvent][atom]
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
                    'label': solvent_labels[solvent],
                    'solvent_size': solvent_size[solvent],
                    'molecule_size': int(atom_numbers[atom] \
                                         / solvent_atoms[solvent][atom])
                }
            except:
                pass
    
    print('The solvent could not be identified.')
    raise ValueError
