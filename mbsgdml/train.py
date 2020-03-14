import os
import numpy as np
from cclib.io import ccread
from cclib.parser.utils import convertor
from periodictable import elements
from mbsgdml import utils
from mbsgdml.solvents import solvent

class PartitionCalcOutput():
    """Quantum chemistry output file for all MD steps of a single partition.

    Output file that contains electronic energies and gradients of the same
    partition from a single MD trajectory. For a single dimer partition of a
    n step MD trajectory would have n coordinates, single point energies,
    and gradients.

    Attributes:
        output_file (str): Path to quantum chemistry output file for a
            partition.
        output_name (str): The name of the quantum chemistry output file
            (includes extension).
        cluster (str): The label identifying the partition of the MD trajectory.
        temp (str): Set point temperautre for the MD thermostat.
        iter (int): Identifies the iteration of the MD iteration.
    """

    def __init__(self, output_path):
        self.output_file = output_path
        self.get_label_info()
        self.cclib_data = ccread(self.output_file)
        self.get_gdml_data()
        self.get_solvent_info()
    
    def get_label_info(self):
        """Gets info from output file name.

        Output file should be labeled in the following manner:
        'out-NumSolv-temperature-iteration-partition.out'.
            'NumSolv' tells us original sized solvent cluster (e.g., 4MeOH);
            'temperature' is the set point for the thermostat (e.g., 300K);
            'iteration' identifies the MD simulation (e.g., 2);
            'partition' identifies what solvent molecules are in the partition
                (e.g., CD).
        A complete example would be 'out-4MeOH-300K-2-CD.out'.
        """
        self.output_name = self.output_file.split('/')[-1]
        split_label = self.output_name.split('-')
        self.cluster = str(split_label[1])
        self.temp = str(split_label[2])
        self.iter = int(split_label[3])
        self.partition = str(split_label[4].split('.')[0])
        self.partition_size = int(len(self.partition))
    

    def get_solvent_info(self):
        self.solvent = solvent(self.atoms.tolist())
    
    def get_gdml_data(self):
        """Parses GDML-relevant data from partition output file.
        """
        try:
            self.atoms = self.cclib_data.atomnos
            self.coords = self.cclib_data.atomcoords
            self.grads = self.cclib_data.grads
            if hasattr(self.cclib_data, 'mpenergies'):
                self.energies = self.cclib_data.mpenergies
            elif hasattr(self.cclib_data, 'scfenergies'):
                self.energies = self.cclib_data.scfenergies
            else:
                raise KeyError
        except:
            print('Something happened while parsing output file.')
            print('Please check ' + str(self.output_name) + ' output file.')
            
    def write_gdml_data(self, gdml_data_dir):
        """Writes and categorizes GDML file in a common GDML data directory.
        
        This should be the last function called after all necessary data is
        collected from the output file.

        GDML file is categorized according to its solvent, partition size,
        temperature, and MD iteration in that order.

        Args:
            gdml_data_dir (str): Path to common GDML data directory.
        """

        gdml_data_dir = utils.norm_path(gdml_data_dir)

        # Preparing directories.
        # /path/to/gdml-files/solventlabel/partitionsize/temp/iteration
        gdml_solvent = self.solvent.solvent_label
        gdml_partition_size = ['monomer', 'dimer', 'trimer', 'tetramer', 'pentamer']
        gdml_solvent_dir = utils.norm_path(
            gdml_data_dir + gdml_solvent
        )
        gdml_partition_size_dir = utils.norm_path(
            gdml_solvent_dir + gdml_partition_size[self.partition_size - 1]
        )
        gdml_temp_dir = utils.norm_path(
            gdml_partition_size_dir + str(self.temp)
        )
        gdml_iter_dir = utils.norm_path(
            gdml_temp_dir + str(self.iter)
        )
        all_dir = [gdml_solvent_dir, gdml_partition_size_dir, gdml_temp_dir,
                   gdml_iter_dir]
        for directory in all_dir:
            try:
                os.chdir(directory)
            except:
                os.mkdir(directory)
                os.chdir(directory)
        
        # Writing GDML file.
        self.gdml_file = gdml_iter_dir + self.partition \
                         + '-' + self.cluster \
                         + '-' + self.temp \
                         + '-' + str(self.iter) \
                         + '-gdml.xyz'
        with open(self.gdml_file, 'w') as gdml_file:
            step_index = 0
            atom_list = utils.atoms_by_element(self.atoms)
            atom_num = self.solvent.solvent_molec_size \
                       * self.solvent.cluster_size
            while step_index < len(self.coords):

                # Number of atoms.
                gdml_file.write(str(atom_num) + '\n')

                # Comment with energy.
                step_energy = convertor(
                    self.energies[step_index][0], 'eV', 'kcal/mol'
                )
                gdml_file.write('# ' + str(step_energy) + '\n')

                # Atomic positions and gradients.
                atom_index = 0
                while atom_index < len(self.atoms):
                    atom_string = atom_list[atom_index] + '    '
                    coord_string = np.array2string(
                                       self.coords[step_index][atom_index],
                                       suppress_small=True, separator='   ',
                                       formatter={'float_kind':'{:0.8f}'.format}
                                   )[1:-1] + '    '
                    grad_string = np.array2string(
                                       self.grads[step_index][atom_index],
                                       suppress_small=True, separator='   ',
                                       formatter={'float_kind':'{:0.8f}'.format}
                                   )[1:-1] + '    '

                    atom_line = ''.join([atom_string, coord_string, grad_string])
                    atom_line = atom_line.replace(' -', '-') + '\n'
                    gdml_file.write(atom_line)

                    atom_index += 1
            
                step_index += 1

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
