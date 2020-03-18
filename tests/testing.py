# Quick file to test parts of MB-GDML as they are developed.
import mbgdml

'''
mbgdml.train.prepare_gdml_files(
    '/home/alex/Dropbox/keith/projects/gdml/data/partitions/calculations/4acn/',
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-datasets/'
)
'''

'''
mbgdml.train.prepare_gdml_dataset(
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-datasets/5H2O',
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-datasets/5H2O'
)
'''

test = mbgdml.train.MBGDMLTrain(
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-datasets/H2O',
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-models/H2O'
)
print(test.dataset_dir)
print(test.dataset_paths)