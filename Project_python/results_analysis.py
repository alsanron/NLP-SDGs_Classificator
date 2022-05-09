# Analysis of the results obtained separately with the NMF, LDA and Top2Vec models
# from cProfile import label
# from curses import raw
import pandas as pd
import numpy as np
import conf
import matplotlib.pyplot as plt
import time

paths = conf.get_paths()
pathFolder = paths["out"] + "Global/"

training_set = 0
model = 4

print('######### Loading texts...')
def parse_ascii_sdgs_association(sdgs_ascii):
    sdgs = sdgs_ascii.split('|')
    scores = []
    for sdg in sdgs:
        score = float(sdg.split(':')[1])
        scores.append(score)
    return np.array(scores)

if training_set:
    top2vec = pd.read_excel(paths["out"] + "Top2vec/" + "test_top2vec_training_files0.xlsx")
    nmf = pd.read_excel(paths["out"] + "NMF/" + "test_nmf_training_files.xlsx")
    lda = pd.read_excel(paths["out"] + "LDA/" + "test_lda_training_files0.xlsx")
else:
    nmf = pd.read_excel(paths["out"] + "NMF/" + "test_nmf_abstracts.xlsx")
    lda = pd.read_excel(paths["out"] + "LDA/" + "test_lda_abstracts0.xlsx")
    top2vec = pd.read_excel(paths["out"] + "Top2vec/" + "test_top2vec_abstracts0.xlsx")
    
texts = list(top2vec["text"])
labeledSDGs = list(top2vec["real"])

sdgs_ascii_nmf = list(nmf["sdgs_association"])
sdgs_nmf = [parse_ascii_sdgs_association(sdgs_ascii) for sdgs_ascii in sdgs_ascii_nmf]

sdgs_ascii_lda = list(lda["sdgs_association"])
sdgs_lda = [parse_ascii_sdgs_association(sdgs_ascii) for sdgs_ascii in sdgs_ascii_lda]

sdgs_ascii_top2vec = list(top2vec["sdgs_association"])
sdgs_top2vec = [parse_ascii_sdgs_association(sdgs_ascii) for sdgs_ascii in sdgs_ascii_top2vec]

def filter_nmf(raw_sdgs, min_valid):
    # threshold = np.mean(raw_sdgs) 
    threshold = np.median(sorted(raw_sdgs))
    validSDGs = (raw_sdgs - threshold) > 0.0
    
    potential_sdgs = np.zeros(17)
    for sdg, sdgIndex in zip(raw_sdgs, range(17)):
        if sdg >= min_valid or (validSDGs[sdgIndex] and sdg >= min_valid):
            potential_sdgs[sdgIndex] = sdg
    discarded_sdgs = raw_sdgs - potential_sdgs

    return (potential_sdgs, discarded_sdgs)

def filter_top2vec(raw_sdgs, min_valid):
    # threshold = np.mean(raw_sdgs) 
    threshold = np.median(sorted(raw_sdgs))
    validSDGs = (raw_sdgs - threshold) > 0.0
    
    potential_sdgs = np.zeros(17)
    for sdg, sdgIndex in zip(raw_sdgs, range(17)):
        if sdg >= min_valid or (validSDGs[sdgIndex] and sdg >= min_valid):
            potential_sdgs[sdgIndex] = sdg
    discarded_sdgs = raw_sdgs - potential_sdgs

    return (potential_sdgs, discarded_sdgs)

def identify_sdgs(sdgs_nmf, sdgs_lda, sdgs_top2vec):
    sdgs_new = np.zeros(17)
    for index, nmf, lda, top2vec in zip(range(len(sdgs_nmf)), sdgs_nmf, sdgs_lda, sdgs_top2vec):
        sdgs = np.array([nmf, lda, top2vec])
        if (sdgs >= 0.3).any():
            tmp = [ii for ii in sdgs if ii >= 0.3]
            sdgs_new[index] = np.mean(tmp)
        elif index == 2 and top2vec >=0.2:
            tmp = [ii for ii in sdgs if ii >= 0.2]
            sdgs_new[index] = np.mean(tmp)
        else:
            count1 = [ii for ii in sdgs if ii >= 0.2]
            count2 = [ii for ii in sdgs if ii >= 0.1]
            if len(count1) >= 1 and len(count2) >= 2:
                tmp = [ii for ii in sdgs if ii >= 0.2]
                sdgs_new[index] = np.mean(tmp)
    return sdgs_new
        

xlabel_sdgs = range(1,18)
xlabel_np = np.array(xlabel_sdgs)
valid_any = 0; valid_all = 0; count_any = 0; count_all = 0
for textIndex in range(len(sdgs_ascii_top2vec)):
    print('Text {} of {}'.format(textIndex + 1, len(sdgs_ascii_top2vec)))
    plt.figure(figsize=(8, 8))
    
    if model == 1:
        potential_sdgs, discarded_sdgs = filter_nmf(sdgs_nmf[textIndex], min_valid=0.10)
        plt.bar(xlabel_np - 0.1, potential_sdgs, width=0.15, label='valid-nmf')
        plt.bar(xlabel_np + 0.1, discarded_sdgs, width=0.15, label='disc-nmf')
        path_save = pathFolder + "Images/NMF/"
    elif model == 2:
        plt.bar(xlabel_np, sdgs_lda[textIndex], width=0.15, label='lda')
        path_save = pathFolder + "Images/LDA/"
    elif model == 3:
        potential_sdgs, discarded_sdgs = filter_top2vec(sdgs_top2vec[textIndex], min_valid=0.15)
        plt.bar(xlabel_np - 0.1, potential_sdgs, width=0.15, label='valid-top2vec')
        plt.bar(xlabel_np + 0.1, discarded_sdgs, width=0.15, label='disc-top2vec')
        path_save = pathFolder + "Images/TOP2VEC/"
    elif model == -1:
        plt.bar(xlabel_np - 0.15, sdgs_nmf[textIndex], width=0.15, label='nmf')
        plt.bar(xlabel_np + 0.0, sdgs_lda[textIndex], width=0.15, label='lda')
        plt.bar(xlabel_np + 0.15, sdgs_top2vec[textIndex], width=0.15, label='top2vec')
        path_save = pathFolder + "Images/ALL/"
    else:
        potential_sdgs = identify_sdgs(sdgs_nmf[textIndex], sdgs_lda[textIndex], sdgs_top2vec[textIndex])
        plt.bar(xlabel_np, potential_sdgs, width=0.3, label='all')
        path_save = pathFolder + "Images/ALL_FILTER/"
    plt.xticks(xlabel_sdgs)
    plt.xlabel('SDGS')
    plt.ylabel("Score")
    plt.ylim(top=0.5)
    plt.title('SDGs to identify: {}'.format(labeledSDGs[textIndex]))
    plt.legend()
    # plt.savefig(path_save + "text{}.png".format(textIndex))
    # print('################# \r')
    # print(texts[textIndex])
    # plt.show()
    plt.close()
     
    tmp = 0
    count_all += 1
    ass_sdgs = labeledSDGs[textIndex][1:-1].split(',')
    ass_sdgs = [int(sdg) for sdg in ass_sdgs]
    predict_sdgs = []
    for sdg in potential_sdgs:
        if sdg > 0:
            predict_sdgs.append(list(potential_sdgs).index(sdg) + 1)
    for sdg in ass_sdgs:
        count_any += 1
        if sdg in predict_sdgs:
            tmp += 1
            valid_any += 1
    if tmp == len(ass_sdgs):
        valid_all += 1
    
perc_any = valid_any / count_any * 100
perc_all = valid_all / count_all * 100
print('Summary -> any: {:.2f} %, all: {:.2f} %'.format(perc_any, perc_all))