# Quick file to test parts of mbGDML as they are developed.
import mbgdml

'''
mbgdml.data.create_datasets(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/partitions/calculations/4H2O/',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/',
    'bohr', 'hartree', theory='MP2/def2-TZVP'
)
'''

'''
mbgdml.data.combine_datasets(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/MeOH/4mer',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/MeOH'
)
'''


'''
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
'''     



