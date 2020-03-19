# Quick file to test parts of MB-GDML as they are developed.
import mbgdml

'''
mbgdml.train.create_gdml_xyz(
    '/home/alex/Dropbox/keith/projects/gdml/data/partitions/calculations/4H2O/',
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-datasets/'
)



mbgdml.train.gdml_xyz_datasets(
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-datasets/H2O'
)
'''
# script to convert xyz to npz
# sgdml_dataset_from_extxyz.py


test = mbgdml.train.MBGDMLTrain(
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-datasets/H2O',
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-models/H2O'
)
test.train_GDML(
    test.dataset_paths[0], 500, 500, 10, 1e-15
)

