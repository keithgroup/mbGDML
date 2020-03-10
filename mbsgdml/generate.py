import os
from mbsgdml import utils, partition, calculate

def data_sets(trajfolder):
    """Driver for generating all positions and gradients from MD trajectories.

    The provided folder contains the MD trajectories used for partitioning a
    solvent into n-body contributions. Typically three iterations of
    three different temperatures (100, 300, and 500 Kelvin) are used to provide
    trajectory files. These trajectory files are then used to make partitions
    containing n solvent molecules.

    The trajectories must be labled with the following scheme:
    'solvent info-temperature-iteration-*traj*.xyz'. For example,
    '4MeOH-300K-1-md-trajectory.xyz'.
    
    Args:
        trajfolder (path): folder containing all MD trajectories to be included
            in solvent data sets. Must contain 'traj' in filename and have an
            'xyz' file extension.
    
    Example:
        data_sets('/path/to/dir')
    """

    trajfolder = utils.norm_path(trajfolder)
    trajs = utils.natsort_list(utils.get_files(trajfolder, 'traj'))
    
    print('Making partition calculation directory.')
    parentdir = utils.norm_path(trajfolder + 'partition_calculations')
    try:
        os.mkdir(parentdir)
    except:
        print('There is already a calculation directory.')
        print('Please make sure you are not duplicating any calculations.')
        print('Terminating.')
        return None
    
    for traj in trajs:
        os.chdir(parentdir)

        traj_filename = traj.split('/')[-1][:-4]
        traj_info = traj_filename.split('-')
        temp = traj_info[1]
        iteration = traj_info[2]

        tempdir = utils.norm_path(parentdir + str(temp))
        iterdir = utils.norm_path(tempdir + str(iteration))
        try:
            os.chdir(tempdir)
        except:
            os.mkdir(tempdir)
        try:
            os.chdir(iterdir)
        except:
            os.mkdir(iterdir)

        print('Working on ' + traj_filename + ' ...')
        traj_partitions = partition.partition_trajectory(traj)
        for traj_partition in traj_partitions:
            calculate.partition_engrad(
                'orca', iterdir,
                traj_partitions[traj_partition], temp, iteration
            )
    
    return None
