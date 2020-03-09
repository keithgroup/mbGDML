from mbsgdml import utils, partition

def data_sets(trajfolder):
    """Driver for generating all positions and gradients from MD trajectories.

    The provided folder contains the MD trajectories used for partitioning a
    solvent into n-body contributions. Typically three iterations of
    three different temperatures (100, 300, and 500 Kelvin) are used to provide
    trajectory files. These trajectory files are then used to make partitions
    containing n solvent molecules.

    The trajectories must be labled with the following scheme:
    'solvent info'-'temperature'-'iteration'-*traj*.xyz. For example,
    '4MeOH-300K-1-md-trajectory.xyz'.
    
    Args:
        trajfolder (path): folder containing all MD trajectories to be included
            in solvent data sets. Must contain 'traj' in filename and have an
            'xyz' file extension.
    """
    
    trajs  = utils.natsort_list(utils.get_files(trajfolder, 'traj'))
    
    for traj in trajs:
        traj_filename = traj.split('/')[-1][:-4]
        traj_info = traj_filename.split('-')
        temp = traj_info[1]
        iteration = traj_info[2]
        print(traj_filename)

        traj_partition = partition.partition_trajectory(traj)

#test_partition = partition.partition_trajectory('/home/alex/repos/MB-sGDML/tests/4MeOH-300K-1-md-trajectory.xyz')
#partition_engrad(
#    'orca', '/home/alex/repos/MB-sGDML/tests', test_partition['CD'], 300, 1
#)
    return None

data_sets('/home/alex/Dropbox/keith/projects/gdml/data/md/4MeOH-md')