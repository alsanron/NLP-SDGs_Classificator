# script for getting statistic and analyze those texts used in the training or validation phases
import string
import sys
sys.path.insert(1, '../Project_python/')
import data
import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

pathOut = "analysis/out/"


def get_nfiles_per_wordscount(texts:list[str], label:str="", show:bool=False):
    indexes = []; sdgs = []; 
    words_per_text = [tools.count_words(text) for text in texts]

    def_fontSize = 24 # default font size used for the plots
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
        plt.xlabel('Number of words', fontsize=def_fontSize)
        plt.ylabel("Number of texts: {}".format(label), fontsize=def_fontSize)
        plt.tick_params(axis='x', labelsize=def_fontSize)
        plt.tick_params(axis='y', labelsize=def_fontSize)
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
    def_fontSize = 30 # default font size used for the plots
    if len(label) > 0:
        plt.figure(figsize=(12, 8))
        sdgs = ["{}".format(ii) for ii in range(1,18)]
        plt.bar(sdgs, countPerSdg)
        plt.xlabel('SDG', fontsize=def_fontSize)
        plt.ylabel("Number of texts: {}".format(label), fontsize=def_fontSize)
        plt.tick_params(axis='x', labelsize=def_fontSize)
        plt.tick_params(axis='y', labelsize=def_fontSize)
        figPath = pathOut + "ntexts_per_sdg_" + label + ".png"
        if os.path.exists(figPath): 
            os.remove(figPath) # otherwise, old figures are not overwritten
        plt.savefig(figPath)
        
        if show: plt.show()
        
        plt.figure(figsize=(12, 8))
        plt.bar(sdgs, meanWords)
        plt.xlabel('SDG', fontsize=def_fontSize)
        plt.ylabel("Mean number of words: {}".format(label), fontsize=def_fontSize)
        plt.tick_params(axis='x', labelsize=def_fontSize)
        plt.tick_params(axis='y', labelsize=def_fontSize)
        figPath = pathOut + "meanwords_per_sdg_" + label + ".png"
        if os.path.exists(figPath): 
            os.remove(figPath) # otherwise, old figures are not overwritten
        plt.savefig(figPath)
        
        if show: plt.show()


def analyze_aero_texts(dataset:pd.DataFrame):
    abstracts = list(dataset["standard"]); years = list(dataset["years"]); 
    citations = list(dataset["citations"]); countries = list(dataset["countries"])

    def get_nfiles_per_years(yearsList:list[float]):
        def_fontSize = 24
        plt.figure(figsize=(12, 8))
        years = []; countPerYear = []
        for year in yearsList:
            if year not in years: 
                # a new year was found. it is added to the list and count initalize to 1
                years.append(year)
                countPerYear.append(1) 
            else:
                countPerYear[years.index(year)] += 1

        plt.bar(years, countPerYear)
        plt.xlabel('Year', fontsize=def_fontSize)
        plt.ylabel("Number of papers", fontsize=def_fontSize)
        plt.tick_params(axis='x', labelsize=def_fontSize)
        plt.tick_params(axis='y', labelsize=def_fontSize)
        figPath = pathOut + "papers_per_year" + ".png"
        if os.path.exists(figPath): 
            os.remove(figPath) # otherwise, old figures are not overwritten
        plt.savefig(figPath)

    def get_ncitations_per_years(yearsList:list[float], nCitationsList:list[int]):
        def_fontSize = 24
        plt.figure(figsize=(12, 8))
        years = []; countPerYear = []
        for year, nCitations in zip(yearsList, nCitationsList):
            if year not in years: 
                years.append(year)
                countPerYear.append(nCitations) 
            else:
                countPerYear[years.index(year)] += nCitations

        plt.bar(years, countPerYear)
        plt.xlabel('Year', fontsize=def_fontSize)
        plt.ylabel("Number of citations", fontsize=def_fontSize)
        plt.tick_params(axis='x', labelsize=def_fontSize)
        plt.tick_params(axis='y', labelsize=def_fontSize)
        figPath = pathOut + "citations_per_year" + ".png"
        if os.path.exists(figPath): 
            os.remove(figPath) # otherwise, old figures are not overwritten
        plt.savefig(figPath)

    def print_countries(countriesList:list[str]):
        countries = []
        for country in countriesList:
            if country not in countries: countries.append(country)
        print(','.join(countries))

    get_nfiles_per_wordscount(abstracts, "aero", show=False) # the plot of files per words range
    get_nfiles_per_years(years)
    get_ncitations_per_years(years, citations)
    print_countries(countries)


def plot_npublications_peryear():
    def_fontSize = 18
    nPublications = [10, 12, 14, 15, 20, 24, 27, 29, 33, 33, 34, 35, 36, 35, 41, 46, 55, 66, 91, 124]
    years = range(2000, 2020)
    plt.figure(figsize=(12, 8))
    plt.bar(years, nPublications)
    plt.xlabel('Year', fontsize=def_fontSize)
    plt.ylabel("Number of publications (in thousands)", fontsize=def_fontSize)
    plt.xticks([2000, 2005, 2010, 2015, 2019], ['2000', '2005', '2010', '2015', '2019'])
    plt.tick_params(axis='x', labelsize=def_fontSize)
    plt.tick_params(axis='y', labelsize=def_fontSize)
    figPath = pathOut + "npublications_per_year" + ".png"
    if os.path.exists(figPath): 
        os.remove(figPath) # otherwise, old figures are not overwritten
    plt.savefig(figPath)
    
labels = ["org", "manual_extra"]
dataset = data.get_dataset(filter=labels, requires_update=False)
analyze_texts(texts=dataset["standard"], sdgs=dataset["sdgs"], label="training", show=False)


labels = ["nature_abstract"]
dataset = data.get_dataset(filter=labels, requires_update=False)
analyze_texts(texts=dataset["standard"], sdgs=dataset["sdgs"], label="validation", show=False)

labels = ["aero"]
dataset = data.get_dataset(filter=labels, requires_update=False)
analyze_aero_texts(dataset)

plot_npublications_peryear()
