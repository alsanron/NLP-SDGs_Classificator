# Analysis of the results obtained with the Aero database

# Configures the project paths: they can be launched from any code
from pkgutil import iter_importers
import sys, os
sys.path.append(os.path.realpath('.'))
import conf
conf.import_paths()

# Configuration flags
identify_sdgs = False # true: all the texts are identified, false: it used previous stored data

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
                 only_bad=False, score_threshold=-1,  only_positive=True, filter_low=True)
    ds_aero["id_sdgs"] = predicStr
    pd.DataFrame(ds_aero).to_excel(paths["out"] + "All/df_test_aero.xlsx")
    print('# Results were updated')
    
def get_sdgs_scores(row_sdgs:str):
    elems = row_sdgs.split(',') 
    scores = []; sdgs=[]
    for elem in elems:
        scores.append(float(elem.split(':')[0]))
        sdgs.append(float(elem.split(':')[0]))
    return scores, sdgs

def parse_list(sdgs_list):
    scores = []; sdgs = []
    for sdg in sdgs_list:
        sc, sd = get_sdgs_scores(sdgs)
        scores.append(sc); sd.append(sd)
    return scores, sdgs      
    
ds = pd.read_excel(paths["out"] + "All/df_test_aero.xlsx")  
 
print('# Obtaining total number of SDGs identified')
tools.plot_SDGsidentified(sdgs_identified:list[list[int]], path_out:str="", show:bool=False)   
