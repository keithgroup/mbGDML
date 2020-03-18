# Quick file to test parts of MB-sGDML as they are developed.
import mbsgdml

'''
mbsgdml.train.prepare_gdml_files(
    '/home/alex/Dropbox/keith/projects/gdml/data/partitions/calculations/4acn/',
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-datasets/'
)
'''

'''
mbsgdml.train.prepare_gdml_dataset(
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-datasets/5H2O',
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-datasets/5H2O'
)
'''

test = mbsgdml.train.MBGDMLTrain(
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-datasets/H2O',
    '/home/alex/Dropbox/keith/projects/gdml/data/gdml-models/H2O'
)
print(test.dataset_dir)
print(test.dataset_paths)