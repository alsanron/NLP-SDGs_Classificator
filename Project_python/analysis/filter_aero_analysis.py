# Configures the project paths: they can be launched from any code
from cProfile import label
from cmath import isnan
from pkgutil import iter_importers
import sys, os
sys.path.append(os.path.realpath('.'))
import conf
conf.import_paths()
paths = conf.get_paths()


filter_threshold = 0.3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools

df = pd.read_excel(paths["out"] + "All/df_test_aero.xlsx")

list_sdgs = []; list_scores = []
for ids in list(df["id_sdgs"]):
    if not isinstance(ids, str): continue
    
    sdgs_l = []; scores = []
    for sdgs in ids.split(','):
        score = float(sdgs.split(':')[0]); sdg = int(sdgs.split(':')[1])
        if score >= filter_threshold:
            sdgs_l.append(sdg); scores.append(score)
    list_sdgs.append(sdgs_l); list_scores.append(scores)
    
tools.plot_SDGsidentified(list_sdgs, list_scores, with_score=True, fontsize=14, path_out=paths["out"] + "All/total_weight_sdgs_filtered.png") 


def compare_lower_higher_citations(fontsize=14):
    xlabel = [ii for ii in range(1, 18)]
    median_index = len(list_scores) // 2
    countsLow, countstr = tools.count_texts_per_score(list_sdgs[:median_index], list_scores[:median_index])
    countsHigh, countstr = tools.count_texts_per_score(list_sdgs[median_index:], list_scores[median_index:])
    
    plt.figure(figsize=(8, 8))
    plt.bar(np.array(xlabel[0]) - 0.1, countsLow[0], width=0.2, alpha=1.0, color='blue', label="Lower than the median")
    plt.bar(np.array(xlabel) - 0.1, countsLow, width=0.2, alpha=1.0, color='blue')
    
    plt.bar(np.array(xlabel[1]) + 0.1, countsHigh[0], width=0.2, alpha=1.0, color='red', label="Higher than the median")
    plt.bar(np.array(xlabel) + 0.1, countsHigh, width=0.2, alpha=1.0, color='red')
    plt.xticks(xlabel)
    plt.xlabel('SDG', fontsize=fontsize)
    plt.ylabel("Total weight (sum of individual score)", fontsize=fontsize)
    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
    path_out = paths["out"] + "All/lower_higher_citations_filtered.png"
    plt.legend()
    if os.path.exists(path_out): 
        os.remove(path_out) # otherwise, old figures are not overwritten
    plt.savefig(path_out)
    
compare_lower_higher_citations(fontsize=14)