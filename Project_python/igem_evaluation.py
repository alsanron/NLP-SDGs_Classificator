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
import os


paths = conf.get_paths()
path_out = paths["out"] + "iGEM/"

raw_ig_abstract, ig_inf = data.get_iGEM_files(ref_path=paths["ref"], verbose=True)

def prepare_texts(corpus):
    newCorpus = []
    for text in corpus:
        newCorpus.append(" ".join(tools.tokenize_text(text, lemmatize=True, stem=False ,extended_stopwords=True)))
    return newCorpus
        
ig_abstract = prepare_texts(raw_ig_abstract)

flag_new_identification = 0 # 0: plot results, 1: new identification
show_plots = 1

if flag_new_identification:
    # LOADING SECTION - ALL MODELS SHOULD HAVE BEEN TRAINED AND SAVED BEFORE THE CALL TO THIS SCRIPT
    print('######## LOADING MODELS...')
    model = model_global.Global_Classifier(paths=paths, verbose=True)
    model.load_models()
    model.test_model(raw_corpus=raw_ig_abstract, corpus=ig_abstract, associated_SDGs=[], 
                 path_to_plot="", 
                path_to_excel=path_out + "abstracts.xlsx", 
                 only_bad=False, only_positive=True, filter_low=True)

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
        if show_plots: plt.show()
        
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
        if show_plots: plt.show()
        
    def plot_global_perc_sdgs_per_region(ig_inf, sdgs):
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
            plt.bar(label_ticks_float + spacing[ii], countPerSdg[ii, :] / sum(countPerSdg[ii, :]) * 100, width=width, label=regions[ii], color=colors[ii])
        
        plt.xticks(label_ticks_int)
        plt.xlabel('SDG')
        plt.ylabel("Contribution to each SDG [%]")
        plt.legend()
        plt.savefig(path_out + "global_percentage_sdgs_per_region.png")
        if show_plots: plt.show()
        
    def plot_timeline_sdg_per_region(ig_inf, sdgs, sdg_query):
        regions = ["North America", "Europe", "Asia", "Latin America", "Africa"]; 
        years = []
        for case in ig_inf:
            year = int(case["Year"])
            if year not in years: years.append(year)
        nYears = len(years)
        colors = ['black', 'red', 'green', 'blue', 'cyan']
        width = 0.14
        spacing = np.array([-2, -1, 0, 1, 2]) * width
        nRegions = len(regions); assert(nRegions == 5)
        countPerSdg = np.zeros((nRegions, nYears))
        for sdgList, index in zip(sdgs, range(len(sdgs))):
            region = ig_inf[index]["Region"]; year = int(ig_inf[index]["Year"])
            regionIndex = regions.index(region)
            yearIndex = years.index(year)
            for sdg in sdgList:
                if sdg == sdg_query: countPerSdg[regionIndex, yearIndex] += 1
                
        label_ticks_float = np.array(years)
        
        plt.figure(figsize=(12, 8))
        for ii in range(nRegions):     
            plt.bar(label_ticks_float + spacing[ii], countPerSdg[ii, :], width=width, label=regions[ii], color=colors[ii])
        
        plt.xticks(years)
        plt.xlabel('Year')
        plt.ylabel("Number of times identified")
        plt.legend()
        plt.title('SDG{}'.format(sdg_query))
        figPath = path_out + "timeline_sdg{}_per_region.png".format(sdg_query)
        if os.path.exists(figPath): 
            os.remove(figPath) # otherwise, old figures are not overwritten
        plt.savefig(figPath)
        if show_plots: plt.show()
        
    def plot_timeline_sdg_identification_per_region(ig_inf, sdgs):
        regions = ["North America", "Europe", "Asia", "Latin America", "Africa"]; 
        years = []
        for case in ig_inf:
            year = int(case["Year"])
            if year not in years: years.append(year)
        nYears = len(years)
        colors = ['black', 'red', 'green', 'blue', 'cyan']
        width = 0.14
        spacing = np.array([-2, -1, 0, 1, 2]) * width
        nRegions = len(regions); assert(nRegions == 5)
        nSDGsIdentified = np.zeros((nRegions, nYears)); nProjects = np.zeros((nRegions, nYears))
        for sdgList, index in zip(sdgs, range(len(sdgs))):
            region = ig_inf[index]["Region"]; year = int(ig_inf[index]["Year"])
            regionIndex = regions.index(region)
            yearIndex = years.index(year)
            
            nProjects[regionIndex, yearIndex] += 1
            for sdg in sdgList:
                nSDGsIdentified[regionIndex, yearIndex] += 1
                
        label_ticks_float = np.array(years)
        
        plt.figure(figsize=(12, 8))
        for ii in range(nRegions):     
            plt.bar(label_ticks_float + spacing[ii], nSDGsIdentified[ii, :] / nProjects[ii, :], width=width, label=regions[ii], color=colors[ii])
        
        plt.xticks(years)
        plt.xlabel('Year')
        plt.ylabel("Number of SDGs identified / Number of projects presented")
        plt.legend()
        figPath = path_out + "timeline_percsdgs_per_year_per_region.png"
        if os.path.exists(figPath): 
            os.remove(figPath) # otherwise, old figures are not overwritten
        plt.savefig(figPath)
        if show_plots: plt.show()
        
    def plot_contribution_per_track(ig_inf, sdgs):
        def filter_track(track:str) -> str:
            return track.replace("(P)", "").replace("/", " & ")

        tracks = [["Food & Energy", "Food & Nutrition", "Energy"], 
                ["Measurement", "Diagnostics"], 
                ["Entrepreneurship", "Policy & Practices"], 
                ["Health & Medicine", "Therapeutics"],
                ["Information Processing"],
                ["Software"],
                ["Hardware"],
                ["Environment"],
                ["Foundational Advance"],
                ["Manufacturing"],
                ["Open"],
                ["Community Labs"],
                ["Art & Design"],
                ["Microfluidics"],
                ["High School"],
                ["New Application"]
                ]
        nTracks = len(tracks)
        texts = [[] for ii in range(nTracks)]
        def get_track_index(track:str) -> int:
            track = filter_track(track)
            for ii in range(nTracks):
                for case in tracks[ii]:
                    if case == track: return ii
            raise ValueError('Track: {} not found'.format(track))
                    
        tracks_sdgs = np.zeros((nTracks, 17))            
        width = 0.3
        for sdgList, index in zip(sdgs, range(len(sdgs))):
            inf = ig_inf[index]
            if int(inf["Year"]) < 2014 or len(inf["Track"].split(" ")) < 1: continue
            track_index = get_track_index(inf["Track"])
            texts[track_index].append(inf["Abstract"])
            for sdg in sdgList:
                tracks_sdgs[track_index, sdg - 1] += 1
                
        label_ticks_int = range(1,18)
        for ii in range(nTracks):
            plt.figure(figsize=(8, 8))
            plt.bar(label_ticks_int, tracks_sdgs[ii, :] / sum(tracks_sdgs[ii, :]) * 100, width=width)
            plt.xticks(label_ticks_int)
            plt.xlabel('SDG')
            plt.ylabel("Contribution per sdg [%]")
            plt.legend()
            plt.title(" - ".join(tracks[ii]) + ". Number of files: {}".format(len(texts[ii])))
            plt.savefig(path_out + "Tracks/track_{}.png".format(ii))
            if show_plots: plt.show()
            
        df = pd.DataFrame()
        textJoin = []; trackJoin = []; wordsCounter = []
        for listText, index in zip(texts, range(len(texts))):
            for text in listText:
                textJoin.append(text); trackJoin.append(tracks[index])
                wordsCounter.append(len(str(text).split(' ')))
        print('Average number of words: {:.2f}'.format(np.mean(wordsCounter)))
        df ["texts"] = textJoin
        df["track"] = trackJoin
        df.to_csv(path_out + "Tracks/texts_per_track.csv")
            
    df = pd.read_excel(path_out + "abstracts.xlsx")
    tmp = list(df["predict"])
    predic_sdgs = tools.parse_sdgs_ascii_list(tmp)
    # plot_global_count_sdgs(predic_sdgs)
    # plot_global_count_sdgs_per_region(ig_inf, predic_sdgs)
    # plot_global_perc_sdgs_per_region(ig_inf, predic_sdgs)
    # for ii in range(1,18):
    #     plot_timeline_sdg_per_region(ig_inf, predic_sdgs, ii)
    plot_timeline_sdg_identification_per_region(ig_inf, predic_sdgs)
    plot_contribution_per_track(ig_inf, predic_sdgs)
            
    
    
