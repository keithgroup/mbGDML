# Quick file to test parts of mbGDML as they are developed.
import mbgdml

'''
mbgdml.data.create_datasets(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/partitions/calculations/4MeOH/',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/',
    'bohr', 'hartree', theory='MP2.def2-TZVP'
)
'''

'''
mbgdml.data.combine_datasets(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/5H2O/5mer',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/5H2O'
)
'''

'''
train_num = 300
validate_num = 300
test_num = 300
sigma_range = '2:10:100'  # Original is 2:10:100
test = mbgdml.train.MBGDMLTrain()
test.load_dataset('/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/4MeOH/4MeOH-1mer-dataset.npz')
test.train_GDML(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/models/',
    train_num, validate_num, test_num, sigma_range
)
'''


# Info from here on out.
solvent_label = '4H2O'  # ***
nbody = 4  # ***

dataset_dir = f'/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/{solvent_label}/'
dataset_path = f'{dataset_dir}{solvent_label}-{str(nbody)}mer-dataset.npz'
models_dir = '/home/alex/Dropbox/keith/projects/mbgdml/data/models/'
solvent_model_dir = f'{models_dir}{solvent_label}/'

'''
# Creates a new mbGDML dataset.
dataset = mbgdml.data.mbGDMLDataset()
dataset.load(dataset_path)
dataset.mb_dataset(nbody, solvent_model_dir)
dataset.save(dataset.base_vars['name'][()], dataset.base_vars, dataset_dir, True)
'''

# Training the new mbGDML model.
mb_dataset_path = f'{dataset_dir}{solvent_label}-{nbody}body-dataset.npz'
train_num = 300
validate_num = 300
test_num = 300
sigma_range = '80:10:200'  # Original is 2:10:100
test = mbgdml.train.MBGDMLTrain()
test.load_dataset(mb_dataset_path)
test.train_GDML(models_dir, train_num, validate_num, test_num, sigma_range)
