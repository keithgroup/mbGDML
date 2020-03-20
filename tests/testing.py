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

train_num = 500
validate_num = 200
test_num = 200
sigma_range = '2:10:300'

for dataset in test.dataset_paths:
    
    if 'monomer' in dataset:
        test.train_GDML(
            dataset, train_num, validate_num, test_num, sigma_range
        )

    
    if 'dimer' in dataset:
        test.train_GDML(
            dataset, train_num, validate_num, test_num, sigma_range
        )


    if 'trimer' in dataset:
        test.train_GDML(
            dataset, train_num, validate_num, test_num, sigma_range
        )


    if 'tetramer' in dataset:
        test.train_GDML(
            dataset, train_num, validate_num, test_num, sigma_range
        )
    


