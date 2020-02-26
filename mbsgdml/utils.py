import os
import cclib
from natsort import natsorted, ns

def norm_path(path):
    """Normalizes directory paths to be consistent.
    
    Args:
        path (string): Path to a directory.
    
    Returns:
        normd_path: Normalized path.
    """

    normd_path = path  # Initializes path variable.

    # Makes sure path ends with forward slash.
    if normd_path[-1] != '/':
        normd_path = path + '/'
    
    return normd_path

def get_files(path, expression):
    """Returns paths to all files in a given directory that matches a provided
    expression in the file name. Commonly used to find all files of a certain
    type, e.g. output or xyz files.
    
    Args:
        path (string): Specifies the folder to search.
        expression (string): Expression to be tested against all file names in
        'path'.
    
    Returns:
        list: all absolute paths to files matching the provided expression.
    """
    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        
        index = 0
        while index < len(filenames):
            filenames[index] = dirpath + '/' + filenames[index]
            index += 1


        all_files.extend(filenames)
    
    
    files = []
    for file in all_files:
        if expression in file:
            files.append(file)

    return files

def make_folder(folder):
    """Creates folder at specified path.
    If the current folder exists, it creates another folder
    with an added number.
    
    Args:
        pathFolder (str): Path to desired folder.
    
    Returns:
        str: Final path of new folder; ends in '/'.
    """
    
    # First tries to create the desired directory.
    try:

        os.mkdir(folder)
        return folder + '/'

    # If there is already a directory with the same name,
    # append a positive integer until there is no previously existing directory.
    except FileExistsError:

        indexDir = 1
        dirExists = True
        while dirExists:
            try:
                pathFolderIteration = folder + '-' + str(indexDir)
                os.mkdir(pathFolderIteration)

                dirExists = False

                return pathFolderIteration + '/'

            # Increments number by 1 until it finds the lowest number.
            except FileExistsError:
                indexDir += 1


def natsort_list(unsorted_list):
    """Basic function that organizes a list based on human (or natural) sorting
    methodology.
    
    Args:
        unsorted_list (list): List of strings.
    
    Returns:
        list: Sorted list of string.
    """
    sorted_list = natsorted(unsorted_list, alg=ns.IGNORECASE)

    return sorted_list