# Quick file to test parts of mbGDML as they are developed.
import mbgdml

'''
mbgdml.train.create_gdml_xyz(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/partitions/calculations/4H2O/',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/gdml-datasets/'
)



mbgdml.train.gdml_xyz_datasets(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/gdml-datasets/H2O'
)
'''
# script to convert xyz to npz
# sgdml_dataset_from_extxyz.py


test = mbgdml.train.MBGDMLTrain(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/quick-methanol/datasets',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/quick-methanol/models'
)

train_num = 100
validate_num = 100
test_num = 100
sigma_range = '2:10:100'  # Original is 2:10:100

for dataset in test.dataset_paths:
    
    if '3body' in dataset:
        test.train_GDML(
            dataset, train_num, validate_num, test_num, sigma_range
        )
      



