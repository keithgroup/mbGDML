# Quick file to test parts of mbGDML as they are developed.
import mbgdml


mbgdml.data.create_datasets(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/partitions/calculations/5H2O/',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/',
    'bohr', 'hartree', theory='MP2.def2-TZVP'
)


'''
mbgdml.data.combine_datasets(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/MeOH/4mer',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/MeOH'
)
'''


'''
test = mbgdml.train.MBGDMLTrain(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/H2O/4H2O-1mer-dataset.npz',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/models/H2O'
)

train_num = 300
validate_num = 300
test_num = 300
sigma_range = '2:10:100'  # Original is 2:10:100
test.train_GDML(
    test.dataset, train_num, validate_num, test_num, sigma_range
)
 '''



