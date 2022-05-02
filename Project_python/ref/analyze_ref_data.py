# script for getting statistic and analyze those texts used in the training or validation phases
import sys
sys.path.insert(1, '../Project_python/')
import conf
import data

paths = conf.get_paths()
raw_orgFiles, sdgs_orgFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
raw_orgFilesCompact, sdgs_orgFilesCompact = data.get_sdgs_org_files(paths["SDGs_inf"], compact_per_sdg=True)
raw_natureShort, sdgs_nature, index_natureAbstracts = data.get_nature_abstracts()
raw_natureExt, sdgs_natureAll, index_natureFull = data.get_nature_files(abstract=True, kw=True, intro=True, body=True, concl=True)

# Check the health of all the texts so that they are valid
print('# Checking org Files...')
indexes = []
for file, sdgs in zip(raw_orgFiles, sdgs_orgFiles):
    print('File: {}, SDGs:{}, nWords: {}'.format(raw_orgFiles.index(file), sdgs, len(file.split(' '))))
    print(' - Text: ', file)
    valid = input('Is valid?: ')
    if len(valid) > 0:
        indexes.append(raw_orgFiles.index(file))
print('Invalid texts: ', indexes)

# Count number of files per sdg

# Count the number of words per file... see if they are too large or too short

