# Quick file to test parts of mbGDML as they are developed.
import mbgdml

'''
mbgdml.data.create_datasets(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/partitions/calculations/4H2O/',
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
test.load_dataset('/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/4H2O/4H2O-1mer-dataset.npz')
test.train_GDML(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/models/',
    train_num, validate_num, test_num, sigma_range
)
'''

'''
# Info from here on out.
solvent_label = '4H2O'  # ***
nbody = 4  # ***

dataset_dir = f'/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/{solvent_label}/'
dataset_path = f'{dataset_dir}{solvent_label}-{str(nbody)}mer-dataset.npz'
models_dir = '/home/alex/Dropbox/keith/projects/mbgdml/data/models/'
solvent_model_dir = f'{models_dir}{solvent_label}/'


# Creates a new mbGDML dataset.
dataset = mbgdml.data.mbGDMLDataset()
dataset.load(dataset_path)
dataset.mb_dataset(nbody, solvent_model_dir)
dataset.save(dataset.base_vars['name'][()], dataset.base_vars, dataset_dir, True)


# Training the new mbGDML model.
mb_dataset_path = f'{dataset_dir}{solvent_label}-{nbody}body-dataset.npz'
train_num = 300
validate_num = 300
test_num = 300
sigma_range = '80:10:180'  # Original is 2:10:100
test = mbgdml.train.MBGDMLTrain()
test.load_dataset(mb_dataset_path)
test.train_GDML(models_dir, train_num, validate_num, test_num, sigma_range)
'''
'''
# Testing MD
model_paths = [
    '/home/alex/Dropbox/keith/projects/mbgdml/data/models/4H2O/4H2O-1mer-model-MP2.def2-TZVP-train300-sym6.npz',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/models/4H2O/4H2O-2body-model-MP2.def2-TZVP-train300-sym72.npz',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/models/4H2O/4H2O-3body-model-MP2.def2-TZVP-train300-sym12.npz'
]

md = mbgdml.calculate.mbGDMLMD(
    'methanol',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/md/mbgdml/4H2O/structures/4H2O-minimum.xyz'
)

md.load_calculator(model_paths)
md.relax()
md.run(100, 0.5, 300)
'''

# Testing prediction set
model_paths = [
    '/home/alex/Dropbox/keith/projects/mbgdml/data/models/4H2O/4H2O-1mer-model-MP2.def2-TZVP-train300-sym2.npz',
    '/home/alex/Dropbox/keith/projects/mbgdml/data/models/4H2O/4H2O-2body-model-MP2.def2-TZVP-train300-sym8.npz'
]
dataset_path = '/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/4H2O/4H2O-2mer-dataset.npz'
test = mbgdml.data.mbGDMLPredictset(dataset_path, model_paths)
test.create_predictset()

print('debug')

test.save(test.base_vars['name'], test.base_vars, '/home/alex/Dropbox/keith/projects/mbgdml/data/analysis/4H2O/', False)