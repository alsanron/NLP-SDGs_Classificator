# script for getting statistic and analyze those texts used in the training or validation phases
import string
import sys
sys.path.insert(1, '../Project_python/')
import data
import tools
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

pathOut = "analysis/out/"

def get_nfiles_per_wordscount(texts:list[str], label:str="", show:bool=False):
    indexes = []; sdgs = []; 
    words_per_text = [tools.count_words(text) for text in texts]

    ranges = ['-100', '100-200', '200-300', '300-400', '400+']
    count_per_range = np.zeros(5)
    for nWord in words_per_text:
        if nWord < 100:
            count_per_range[0] += 1
        elif nWord < 200:
            count_per_range[1] += 1
        elif nWord < 300:
            count_per_range[2] += 1
        elif nWord < 400:
            count_per_range[3] += 1
        else:
            count_per_range[4] += 1
            
    if not(len(words_per_text) == sum(count_per_range)): raise ValueError('Check algorithm')
    
    if len(label) > 0:
        print('# Proceeding to plot')
        plt.figure(figsize=(12, 8))
        plt.bar(ranges, count_per_range)
        plt.xlabel('Number of words')
        plt.ylabel("Number of texts: {}".format(label))
        figPath = pathOut + "nfiles_wordrange_" + label + ".png"
        if os.path.exists(figPath): 
            os.remove(figPath) # otherwise, old figures are not overwritten
        plt.savefig(figPath)
        
        if show: plt.show()
        
    return count_per_range


def analyze_texts(texts:list[str], sdgs:list[list[int]], label:str="", show:bool=False):
    get_nfiles_per_wordscount(texts, label, show) # the plot of files per words range
    
    countPerSdg, countPerSdgStr = tools.count_texts_per_sdg(sdgs)
    meanWords, meanWordsStr = tools.count_meanwords_per_sdg(texts, sdgs)
    
    if len(label) > 0:
        plt.figure(figsize=(12, 8))
        sdgs = ["{}".format(ii) for ii in range(1,18)]
        plt.bar(sdgs, countPerSdg)
        plt.xlabel('SDG')
        plt.ylabel("Number of texts: {}".format(label))
        figPath = pathOut + "ntexts_per_sdg_" + label + ".png"
        if os.path.exists(figPath): 
            os.remove(figPath) # otherwise, old figures are not overwritten
        plt.savefig(figPath)
        
        if show: plt.show()
        
        plt.figure(figsize=(12, 8))
        plt.bar(sdgs, meanWords)
        plt.xlabel('SDG')
        plt.ylabel("Mean number of words: {}".format(label))
        figPath = pathOut + "meanwords_per_sdg_" + label + ".png"
        if os.path.exists(figPath): 
            os.remove(figPath) # otherwise, old figures are not overwritten
        plt.savefig(figPath)
        
        if show: plt.show()
    

# CHECK FOR REPEATED TEXTS
# dataset = data.get_dataset()
# tools.search_for_repeated_texts(dataset["standard"], ratio=0.8) 

# labels = ["org", "manual_extra", "nature_abstract", "nature_all"]
# labels = ["org", "manual_extra"]
# for label in labels:
#     dataset = data.get_dataset(filter=[label])
#     analyze_texts(texts=dataset["standard"], sdgs=dataset["sdgs"], label=label, show=False)
    
labels = ["org", "manual_extra"]
dataset = data.get_dataset(filter=labels, requires_update=False)
analyze_texts(texts=dataset["standard"], sdgs=dataset["sdgs"], label="training", show=False)


labels = ["nature_abstract"]
dataset = data.get_dataset(filter=labels, requires_update=False)
analyze_texts(texts=dataset["standard"], sdgs=dataset["sdgs"], label="validation", show=False)
