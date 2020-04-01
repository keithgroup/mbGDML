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
test_dataset = mbgdml.data.mbGDMLDataset()
test_dataset.load('/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/4H2O/4H2O-4mer-dataset.npz') #####

test_model1 = mbgdml.data.mbGDMLModel()
test_model1.load('/home/alex/Dropbox/keith/projects/mbgdml/data/models/4H2O/4H2O-1mer-model-MP2.def2-TZVP-train300-sym2.npz') #####
test_dataset.mb_dataset(test_model1.model)

test_model2 = mbgdml.data.mbGDMLModel()
test_model2.load('/home/alex/Dropbox/keith/projects/mbgdml/data/models/4H2O/4H2O-2body-model-MP2.def2-TZVP-train300-sym8.npz') #####
test_dataset.mb_dataset(test_model2.model)

test_model3 = mbgdml.data.mbGDMLModel()
test_model3.load('/home/alex/Dropbox/keith/projects/mbgdml/data/models/4H2O/4H2O-3mer-model-MP2.def2-TZVP-train300-sym48.npz') #####
test_dataset.mb_dataset(test_model3.model)

#test_model4 = mbgdml.data.mbGDMLModel()
#test_model4.load('/home/alex/Dropbox/keith/projects/mbgdml/data/models/4H2O/4H2O-2body-model-MP2.def2-TZVP-train300-sym8.npz') #####
#test_dataset.mb_dataset(test_model4.model)
'''
'''
test_dataset.save(test_dataset.base_vars['name'][()], test_dataset.base_vars,
                  '/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/4H2O', True) #####
'''
'''
train_num = 300
validate_num = 300
test_num = 300
sigma_range = '2:10:100'  # Original is 2:10:100
test = mbgdml.train.MBGDMLTrain()
test.load_dataset('/home/alex/Dropbox/keith/projects/mbgdml/data/datasets/4H2O/4H2O-4body-dataset.npz') #####
test.train_GDML(
    '/home/alex/Dropbox/keith/projects/mbgdml/data/models/',
    train_num, validate_num, test_num, sigma_range
)
'''