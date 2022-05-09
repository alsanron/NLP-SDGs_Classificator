# Analysis of the results obtained separately with the NMF, LDA and Top2Vec models
from cProfile import label
import pandas as pd
import numpy as np
import conf
import matplotlib.pyplot as plt

paths = conf.get_paths()
pathFolder = paths["out"] + "Send/"

nmf = pd.read_excel(pathFolder + "test_abstract_nmf.xlsx")
lda = pd.read_excel(pathFolder + "test_abstract_lda.xlsx")
top2vec = pd.read_excel(pathFolder + "test_abstract_top2vec.xlsx")

texts = list(top2vec["text"])
labeledSDGs = list(top2vec["real"])

def parse_ascii_sdgs_association(sdgs_ascii):
    sdgs = sdgs_ascii.split('|')
    scores = []
    for sdg in sdgs:
        score = float(sdg.split(':')[1])
        scores.append(score)
    return scores

sdgs_ascii_nmf = list(nmf["sdgs_association"])
sdgs_nmf = [parse_ascii_sdgs_association(sdgs_ascii) for sdgs_ascii in sdgs_ascii_nmf]

sdgs_ascii_lda = list(lda["sdgs_association"])
sdgs_lda = [parse_ascii_sdgs_association(sdgs_ascii) for sdgs_ascii in sdgs_ascii_lda]

sdgs_ascii_top2vec = list(top2vec["sdgs_association"])
sdgs_top2vec = [parse_ascii_sdgs_association(sdgs_ascii) for sdgs_ascii in sdgs_ascii_top2vec]

xlabel_sdgs = range(1,18)
xlabel_np = np.array(xlabel_sdgs)
for textIndex in range(len(sdgs_ascii_nmf)):
    plt.figure(figsize=(10, 8))
    plt.bar(xlabel_np - 0.2, sdgs_nmf[textIndex], width=0.15, label='nmf')
    plt.bar(xlabel_np, sdgs_lda[textIndex], width=0.15, label='lda')
    plt.bar(xlabel_np + 0.2, sdgs_top2vec[textIndex], width=0.15, label='top2vec')
    plt.xticks(xlabel_sdgs)
    plt.xlabel('SDGS')
    plt.ylabel("Score")
    plt.title('SDGs to identify: {}'.format(labeledSDGs[textIndex]))
    plt.legend()
    plt.savefig(paths["out"] + "Send/Images/" + "text{}.png".format(textIndex))
    plt.close()