# Tests an already trained model for classifying a given input paper into the corresponding Sustainable Development goals

from preprocess import get_validation_files, get_training_files

validFilesDict = get_validation_files(preprocess=False)
trainFiles = get_training_files()

