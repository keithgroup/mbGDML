import os
import re
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
        partition (str): Identifies what solvent molecules are in the partition.
        partition_size (int): The number of solvent molecules in the partition.
        cclib_data (obj): Contains all data parsed from output file.
        atoms (np.array): A (n) array containing n atomic numbers.
        coords (np.array): A (m, n, 3) array containing the atomic coordinates
            of n atoms of m MD steps.
        grads (np.array): A (m, n, 3) array containing the atomic gradients
            of n atoms of m MD steps.
        energies (np.array): A (m, n) array containing the energies of n atoms
            of m MD steps.
        solvent (obj): From Solvent class and contains solvent information.
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
                    atom_string = atom_list[atom_index] + '     '
                    coord_string = np.array2string(
                        self.coords[step_index][atom_index],
                        suppress_small=True, separator='     ',
                        formatter={'float_kind':'{:0.8f}'.format}
                    )[1:-1] + '     '
                    # Converts from Eh/bohr to kcal/(angstrom mol)
                    converted_grads = self.grads[step_index][atom_index] \
                                      * ( 627.50947414 / 0.5291772109)
                    grad_string = np.array2string(
                        converted_grads,
                        suppress_small=True, separator='     ',
                        formatter={'float_kind':'{:0.8f}'.format}
                    )[1:-1] + '     '

                    atom_line = ''.join([atom_string, coord_string, grad_string])

                    # Cleaning string for alignments with negative numbers and
                    # double-digit numbers.
                    atom_line = atom_line.replace(' -', '-') + '\n'
                    neg_double = re.findall(' -[0-9][0-9].[0-9]', atom_line.rstrip())
                    pos_double = re.findall(' [0-9][0-9].[0-9]', atom_line.rstrip())
                    for value in neg_double:
                        atom_line = atom_line.replace(value, value[1:])
                    for value in pos_double:
                        atom_line = atom_line.replace(value, value[1:])

                    gdml_file.write(atom_line)

                    atom_index += 1
            
                step_index += 1

def prepare_gdml_files(partition_calc_dir, gdml_data_dir):
    
    partition_calc_dir = utils.norm_path(partition_calc_dir)

    all_out_files = utils.get_files(partition_calc_dir, 'out')

    for out_file in all_out_files:
        print('Writing the GDML file for ' + out_file.split('/')[-1] + ' ...')
        calc = PartitionCalcOutput(out_file)
        calc.write_gdml_data(gdml_data_dir)

def prepare_partition_dataset(gdml_partition_dir, write_dir):

    gdml_partition_dir = utils.norm_path(gdml_partition_dir)
    write_dir = utils.norm_path(write_dir)

    # Gets all GDML files within a partition.
    all_gdml_files = utils.natsort_list(
        utils.get_files(gdml_partition_dir, 'gdml.xyz')
    )

    # Gets naming information from one of the GDML files.
    gdml_file_example = all_gdml_files[0].split('/')[-1][:-4].split('-')
    gdml_partitions = ['monomer', 'dimer', 'trimer', 'tetramer', 'pentamer']
    gdml_partition_size = gdml_partitions[len(gdml_file_example[0]) - 1]
    gdml_cluster = gdml_file_example[1]
    gdml_partition_file = write_dir \
                          + '-'.join([gdml_cluster, gdml_partition_size]) \
                          + '-gdml-dataset.xyz'
    
    # TODO write data in npz instead of extended xyz file.
    # Writes all partitions to a single extended-xyz GDML file.
    open(gdml_partition_file, 'w').close()
    for partition in all_gdml_files:
        print('Writing the ' + partition.split('/')[-1] + ' file.')
        with open(partition, 'r') as partition_file:
            partition_data = partition_file.read()
        with open(gdml_partition_file, 'a+') as gdml_file:
            gdml_file.write(partition_data)

def prepare_gdml_dataset(gdml_solvent_dir, write_dir):

    # Gets all directories which should contain all GDML files of a single
    # partition size.
    gdml_solvent_dir = utils.norm_path(gdml_solvent_dir)
    partition_dirs = os.listdir(gdml_solvent_dir)
    file_index = 0
    while file_index < len(partition_dirs):
        partition_dirs[file_index] = gdml_solvent_dir \
                                     + partition_dirs[file_index]
        file_index += 1
    partition_dirs = [item for item in partition_dirs if os.path.isdir(item)]
    
    for size in partition_dirs:
        prepare_partition_dataset(size, write_dir)
