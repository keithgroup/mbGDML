import numpy as np
from cclib.io import ccread
from cclib.parser.utils import convertor
from periodictable import elements
from mbsgdml import utils

def prepare_training(
    segment_calc_folder, training_folder, solvent, temperature, iteration
):
    
    if training_folder[-1] != '/':
        training_folder = training_folder + '/'

    all_out_files = utils.get_files(segment_calc_folder, 'out')

    # Organizes segment type into dictionary.
    out_files_dict = {}
    while len(all_out_files) > 0:
        # Start with last file.
        ref_file = all_out_files[0]

        # Grabs molecule label.
        label_ref = ref_file.split('/')[-1].split('-')[1]
        # Adding all output files matching reference label into list.
        collected_out_files = [file for file in all_out_files if \
                               ('-' + label_ref + '-') in file]
        out_files_dict[label_ref] = collected_out_files


        # Removes all occurrences.
        all_out_files = [file for file in all_out_files if \
                         ('-' + label_ref + '-') not in file]
    
    # Specifies number of atoms in each molecule and solvent label.
    if solvent[0] == 'water':
        solvent_label = 'H2O'
        num_atoms = 3
    
    # Determines max number of solvent number for labeling.
    segment_list = list(out_files_dict.keys())
    largest_segment = max(segment_list, key=len)
    max_cluster_data = ccread(out_files_dict[str(largest_segment)][0])
    max_num_molecules = int(len(max_cluster_data.atomnos) / num_atoms)

    for segment in out_files_dict:

        # Naming segment file
        gdml_file_name = training_folder \
                                 + str(segment) + '-' \
                                 + str(max_num_molecules) + solvent_label \
                                 + '-' + str(temperature) + 'K-' \
                                 + str(iteration) + '-gdml.txt'

        # Loops through each MD step for segment and adds to gdml file.
        step_index = 0
        while step_index < len(out_files_dict[segment]):
            segment_step = [file for file in out_files_dict[segment] if \
                           ('-step' + str(step_index) + '-') in file]
            
            if len(segment_step) == 1:
                segment_data = ccread(segment_step[0])
                atoms = list(segment_data.atomnos)
                num_atoms = len(atoms)
                coords = segment_data.atomcoords[-1]
                energy = convertor(segment_data.mpenergies[0][-1], 'eV', 'kcal/mol')
                forces = utils.convert_gradients(segment_data.grads[0], num_atoms)
                

                # Combining atoms, coordinates, and forces.
                gdml_data = []
                atom_index = 0
                while atom_index < len(atoms):
                    atom_string = '  ' + str(elements[atoms[atom_index]])
                    coord_string = np.array2string(
                                       coords[atom_index],
                                       suppress_small=True, separator='   ',
                                       formatter={'float_kind':'{:0.6f}'.format}
                                   )[1:-1].replace(' -', '-') + '    '
                    force_string = np.array2string(
                                       forces[atom_index],
                                       suppress_small=True, separator='   ',
                                       formatter={'float_kind':'{:0.8f}'.format}
                                   )[1:-1].replace(' -', '-') + '\n'
                    
                    # Adds proper spacing between atom and first coordinate.
                    element_space = len(atom_string) - 1
                    if coords[atom_index][0] < 0:
                        neg_space = 1
                    else:
                        neg_space = 0
                    num_spaces = 10 - element_space - neg_space
                    spaces = '{:<' + str(num_spaces) + '}'
                    atom_string = spaces.format(atom_string)

                    # Adds proper spacing between coordinates and forces.
                    # Negative numbers.
                    if forces[atom_index][0] > 0:
                        coord_string = coord_string + ' '
                    # Greater than 10.
                    if abs(forces[atom_index][0]) >= 10:
                        coord_string = coord_string[:-1]
                    

                    gdml_data.append(atom_string + coord_string + force_string)
                    
                    atom_index += 1
                            
                lines = [str(num_atoms) + '\n', '# ' + str(energy) + '\n'] + gdml_data

                # Writing or appending gdml file (will overwrite).
                if step_index == 0:
                    file_mode = 'w'
                else:
                    file_mode = 'a'
                
                with open(gdml_file_name, file_mode) as gdml_file:
                    gdml_file.writelines(lines)
                

            else:
                print('There is not a unique output file for MD step '\
                       + str(step_index) + ' for segment ' + str(segment))
                return None
            
            step_index += 1
        
    
    return None