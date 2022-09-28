# Analysis of the results obtained with the Aero database

# Configures the project paths: they can be launched from any code
from cProfile import label
from pkgutil import iter_importers
import sys, os
sys.path.append(os.path.realpath('.'))
import conf
conf.import_paths()

# Configuration flags
identify_sdgs = True # true: all the texts are identified, false: it used previous stored data

# Imports required to work properly
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

print('# Loading aero dataset...')
paths = conf.get_paths()
ds_aero = data.get_dataset(requires_update=False, filter=["aero"])
raw_files = ds_aero["standard"]; files = ds_aero["lem"]

print('# Loading models...')
model = model_global.Global_Classifier(paths=paths, verbose=True)
model.load_models()

if identify_sdgs:
    print('# Identifying SDGs in texts...')
    predic, scores, predicStr = model.test_model(raw_corpus=raw_files, corpus=files, associated_SDGs=[], 
                 path_to_plot="", path_to_excel=paths["out"] + "All/test_aero.xlsx", 
                 only_bad=False, score_threshold=-1,  only_positive=True, filter_low=True, only_main_topic=False)
    ds_aero["id_sdgs"] = predicStr
    pd.DataFrame(ds_aero).to_excel(paths["out"] + "All/df_test_aero.xlsx")
    print('# Results were updated')
    
def get_sdgs_scores(row_sdgs:str):
    try: 
        elems = row_sdgs.split(',') 
        scores = []; sdgs=[]
        for elem in elems:
            scores.append(float(elem.split(':')[0]))
            sdgs.append(float(elem.split(':')[1]))
    except:
        print('# Input: ')
        scores = []; sdgs=[]
    return scores, sdgs

def parse_list(sdgs_list):
    scores = []; sdgs = []
    for sdg in sdgs_list:
        sc, sd = get_sdgs_scores(sdg)
        scores.append(sc); sdgs.append(sd)
    return scores, sdgs      
    
ds = pd.read_excel(paths["out"] + "All/df_test_aero.xlsx") 
list_scores, list_sdgs = parse_list(list(ds["id_sdgs"]))
 
print('# Obtaining total number of SDGs identified')
tools.plot_SDGsidentified(list_sdgs, list_scores, with_score=True, fontsize=14, path_out=paths["out"] + "All/total_weight_sdgs.png") 
tools.plot_SDGsidentified(list_sdgs, list_scores, with_score=False, fontsize=14, path_out=paths["out"] + "All/sdgs_identified.png") 

print('# Obtaining the evolution of the SDGs with the years')
def plot_evolution_with_years(fontsize:int=14):
    plt.figure(figsize=(8, 8))
    positions = [-0.15, -0.05, 0.05, 0.15]
    labels = ["SDG7", "SDG9", "SDG11", "SDG13"]
    colors = ["yellow", "blue", "red", "green"]
    years = range(2017, 2022)
    sdgsPerYear = [[], [], [], []]

    for year in years:
        df = ds.loc[ds['years'] == year]
        list_scores, list_sdgs = parse_list(list(df["id_sdgs"]))
        counts, countstr = tools.count_texts_per_score(list_sdgs, list_scores)
        
        sdgsPerYear[0].append(counts[6]); sdgsPerYear[1].append(counts[8])
        sdgsPerYear[2].append(counts[10]); sdgsPerYear[3].append(counts[12])
    for ii in range(4):
        xx = 2017
        plt.bar(xx + positions[ii], sdgsPerYear[ii][0], width=0.1, alpha=1, color=colors[ii], label=labels[ii])
        
    for ii in range(4):
        for jj in range(2018, 2022):
            plt.bar(jj + positions[ii], sdgsPerYear[ii][years.index(jj)], width=0.1, alpha=1, color=colors[ii])

    plt.xticks(years)
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel("Total weight (sum of individual score)", fontsize=fontsize)
    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
    plt.legend()
    plt.savefig(paths["out"] + "All/evolution_sdgs_year.png")

plot_evolution_with_years(fontsize=14)

print('# Plotting the contribution of the papers < median and above')
citations = list(ds["citations"])
dfOrdered = ds.sort_values(by=['citations'])
median_index = len(list_scores) // 2
print('# The median is: {}'.format(list(dfOrdered["citations"])[median_index]))
list_scores, list_sdgs = parse_list(list(dfOrdered["id_sdgs"]))

print([len(list_scores), median_index])


def compare_lower_higher_citations(fontsize=14):
    xlabel = [ii for ii in range(1, 18)]
    countsLow, countstr = tools.count_texts_per_score(list_sdgs[:median_index], list_scores[:median_index])
    countsHigh, countstr = tools.count_texts_per_score(list_sdgs[median_index:], list_scores[median_index:])
    
    plt.figure(figsize=(8, 8))
    plt.bar(np.array(xlabel[0]) - 0.1, countsLow[0], width=0.2, alpha=1.0, color='green', label="Lower than the median")
    plt.bar(np.array(xlabel) - 0.1, countsLow, width=0.2, alpha=1.0, color='green')
    
    plt.bar(np.array(xlabel[1]) + 0.1, countsHigh[0], width=0.2, alpha=1.0, color='red', label="Higher than the median")
    plt.bar(np.array(xlabel) + 0.1, countsHigh, width=0.2, alpha=1.0, color='red')
    plt.xticks(xlabel)
    plt.xlabel('SDG', fontsize=fontsize)
    plt.ylabel("Total weight (sum of individual score)", fontsize=fontsize)
    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
    path_out = paths["out"] + "All/lower_higher_citations.png"
    plt.legend()
    if os.path.exists(path_out): 
        os.remove(path_out) # otherwise, old figures are not overwritten
    plt.savefig(path_out)
    
compare_lower_higher_citations(fontsize=14)


print('# Plotting pie charts')
countries = list(ds["countries"])
list_countries = []; count_countries = []
for country in countries:
    if not country in list_countries: 
        list_countries.append(country)
        count_countries.append(1)
    else:
        count_countries[list_countries.index(country)] += 1
listC = [[country, count] for country, count in zip(list_countries, count_countries)]

def order_second(elem):
    return elem[1]
listC.sort(key=order_second, reverse=True)
a=2
# import matplotlib.pyplot as plt

# # Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
# sizes = [15, 30, 45, 10]
# explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# plt.show()