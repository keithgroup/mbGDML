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
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-datasets/MeOH',
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-models/MeOH'
)

train_num = 500
validate_num = 300
sigma_para = 10
lambda_para = 1e-15
info_string = 'train' + str(train_num) \
              + '-valid' + str(validate_num) \
              + '-sig' + str(sigma_para) \
              + '-lam' + str(lambda_para)

for dataset in test.dataset_paths:
    
    if 'trimer' in dataset:
        test.load_dataset(dataset)
        test.train_GDML(
            test.dataset,
            train_num, validate_num, sigma_para, lambda_para,
            info_string
        )

