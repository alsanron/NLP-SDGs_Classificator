# script for getting statistic and analyze those texts used in the training or validation phases
import sys
sys.path.insert(1, '../Project_python/')
import conf
import data
import numpy as np
import matplotlib.pyplot as plt
import warnings

paths = conf.get_paths()
raw_orgFiles, sdgs_orgFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
# raw_natureShort, sdgs_nature, index_natureAbstracts = data.get_nature_abstracts()
# raw_natureExt, sdgs_natureAll, index_natureFull = data.get_nature_files(abstract=True, kw=True, intro=True, body=True, concl=True)
# raw_pathFinder, sdgs_pathFinder = data.get_sdgs_pathfinder(paths["ref"], min_words=200)
raw_extraFiles, sdgs_extra = data.get_extra_manual_files(paths["ref"])


# Check the health of all the texts so that they are valid
def show_data(files_data, sdgs_data, label):
    print('# Checking org Files...')
    indexes = []; sdgs = []; words = []
    for file, sdg in zip(files_data, sdgs_data):
        # print('File: {}, SDGs:{}, nWords: {}'.format(files_data.index(file), sdgs, len(file.split(' '))))
        # print(' - Text: ', file)
        # valid = input('Is valid?: ')
        # if len(valid) > 0:
        #     indexes.append(files_data.index(file))
        sdgs.append(sdg)
        nWords = len(file.split(' '))
        if nWords > 500:
            warnings.warn('File: {} has {} words'.format(files_data.index(file), nWords))
        words.append(nWords)
    print('- nFiles: ', len(words))
    print('- Words: ', [min(words), np.mean(words), max(words)]) 

    ranges = ['-100', '100-200', '200-300', '300-400', '400+']
    wordsNpy = np.array(words)
    counts = np.zeros(5)
    for nWord in words:
        if nWord < 100:
            counts[0] += 1
        elif nWord < 200:
            counts[1] += 1
        elif nWord < 300:
            counts[2] += 1
        elif nWord < 400:
            counts[3] += 1
        else:
            counts[4] += 1
    if not(len(words) == sum(counts)): raise ValueError('Check algorithm')
    plt.figure()
    plt.bar(ranges, counts)
    plt.xlabel('Number of words')
    plt.ylabel("Number of {} files".format(label))
    plt.show()

    counts = np.zeros(17)
    for group in sdgs_data:
        for sdg in group:
            counts[sdg - 1] += 1
    plt.figure()
    plt.bar([str(ii) for ii in range(1,18)], counts)
    plt.xlabel('SDG')
    plt.ylabel("Number of {} files".format(label))
    plt.show()

def analize_corpus(corpus, sdgs):
    countPerSdg = np.zeros(17)
    for sdgG in sdgs:
        for sdg in sdgG:
            countPerSdg[sdg - 1] += 1
    countPerSdgStr = ["SDG{}:{}".format(sdg, int(count)) for sdg, count in zip(range(1,18), countPerSdg)]
    countPerSdgStr = " | ".join(countPerSdgStr)
    print(countPerSdgStr)

# show_data(raw_orgFiles, sdgs_orgFiles, "training")
#show_data(raw_orgFiles, sdgs_orgFiles, "SDGs-UN information")
print("## ORG Files: "); analize_corpus(corpus=raw_orgFiles, sdgs=sdgs_orgFiles)
# print("## Manual Files: ");  analize_corpus(corpus=raw_extraFiles, sdgs=sdgs_extra)
