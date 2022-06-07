# Script for the evaluation of data in the igem competition
from logging import error
import data
import conf
import pandas as pd
import model_global
import numpy as np
import tools
import matplotlib.pyplot as plt
import warnings

paths = conf.get_paths()
path_out = paths["out"] + "iGEM/"

raw_ig_abstract, ig_inf = data.get_iGEM_files(ref_path=paths["ref"], verbose=True)

def prepare_texts(corpus):
    newCorpus = []
    for text in corpus:
        newCorpus.append(" ".join(tools.tokenize_text(text, lemmatize=True, stem=False ,extended_stopwords=True)))
    return newCorpus
        
# trainFiles = prepare_texts(raw_trainFiles)
ig_abstract = prepare_texts(raw_ig_abstract)

flag_new_identification = 0 # 0: plot results, 1: new identification

if flag_new_identification:
    # LOADING SECTION - ALL MODELS SHOULD HAVE BEEN TRAINED AND SAVED BEFORE THE CALL TO THIS SCRIPT
    print('######## LOADING MODELS...')
    model = model_global.Global_Classifier(paths=paths, verbose=True)
    model.load_models()

    predic, scores = model.test_model(raw_ig_abstract, ig_abstract, associated_SDGs=[], 
                    path_to_plot="", 
                    path_to_excel=path_out + "abstracts.xlsx", 
                    only_bad=False, only_positive=False, filter_low=True)
else:
    def plot_global_count_sdgs(sdgs):
        countPerSdg = np.zeros(17)
        for sdgList in sdgs:
            for sdg in sdgList:
                countPerSdg[sdg - 1] += 1
                
        label_ticks = ["{}".format(ii) for ii in range(1,18)]
        plt.figure()
        plt.bar(label_ticks, countPerSdg)
        plt.xlabel('SDG')
        plt.ylabel("Number of times identified")
        plt.savefig(path_out + "global_count_sdgs.png")
        plt.show()
        
    def plot_global_count_sdgs_per_region(ig_inf, sdgs):
        regions = ["North America", "Europe", "Asia", "Latin America", "Africa"]; 
        colors = ['black', 'red', 'green', 'blue', 'cyan']
        width = 0.14
        spacing = np.array([-2, -1, 0, 1, 2]) * width
        nRegions = len(regions); assert(nRegions == 5)
        countPerSdg = np.zeros((nRegions, 17))
        for sdgList, index in zip(sdgs, range(len(sdgs))):
            region = ig_inf[index]["Region"]
            regionIndex = regions.index(region)
            for sdg in sdgList:
                countPerSdg[regionIndex, sdg - 1] += 1
                
        label_ticks_int = range(1,18)
        label_ticks_float = np.array(label_ticks_int)
        label_ticks_str = ["{}".format(ii) for ii in label_ticks_int]
        
        plt.figure(figsize=(8, 8))
        for ii in range(nRegions):     
            plt.bar(label_ticks_float + spacing[ii], countPerSdg[ii, :], width=width, label=regions[ii], color=colors[ii])
        
        plt.xticks(label_ticks_int)
        plt.xlabel('SDG')
        plt.ylabel("Number of times identified")
        plt.legend()
        plt.savefig(path_out + "global_count_sdgs_per_region.png")
        plt.show()
        
    def plot_timeline_sdg_per_region(ig_inf, sdgs, sdg_query):
        regions = ["North America", "Europe", "Asia", "Latin America", "Africa"]; 
        years = []
        for case in ig_inf:
            year = case["Year"]
            if year not in years: years.append(year)
        nYears = len(years)
        colors = ['black', 'red', 'green', 'blue', 'cyan']
        width = 0.14
        spacing = np.array([-2, -1, 0, 1, 2]) * width
        nRegions = len(regions); assert(nRegions == 5)
        countPerSdg = np.zeros((nRegions, nYears))
        for sdgList, index in zip(sdgs, range(len(sdgs))):
            region = ig_inf[index]["Region"]; year = ig_inf[index]["Year"]
            regionIndex = regions.index(region)
            yearIndex = years.index(year)
            for sdg in sdgList:
                if sdg == sdg_query: countPerSdg[regionIndex, yearIndex] += 1
                
        label_ticks_int = range(nYears)
        label_ticks_float = np.array(label_ticks_int)
        xlabel_ticks = [int(year) for year in years]
        
        plt.figure(figsize=(8, 8))
        for ii in range(nRegions):     
            plt.bar(label_ticks_float + spacing[ii], countPerSdg[ii, :], width=width, label=regions[ii], color=colors[ii])
        
        plt.xticks(label_ticks_int)
        plt.xlabel('SDG')
        plt.ylabel("Number of times identified")
        plt.legend()
        plt.savefig(path_out + "timeline_sdg{}_per_region.png".format(sdg_query))
        plt.show()
        
        
    df = pd.read_excel(path_out + "abstracts.xlsx")
    tmp = list(df["predict"])
    predic_sdgs = tools.parse_sdgs_ascii_list(tmp)
    # plot_global_count_sdgs(predic_sdgs)
    # plot_global_count_sdgs_per_region(ig_inf, predic_sdgs)
    plot_timeline_sdg_per_region(ig_inf, predic_sdgs, 3)
    plot_timeline_sdg_per_region(ig_inf, predic_sdgs, 7)
            
    
    
