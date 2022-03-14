# Tests an already trained model for classifying a given input paper into the corresponding Sustainable Development goals

from preprocess_validation_files import get_validation_files

validFilesDict = get_validation_files(preprocess=False)
values = list(validFilesDict.values())

fp = open(values[0][0], 'r')
print(fp.read())
fp.close()
