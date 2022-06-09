# TESTS THE COMBINATION OF ALL THE MODELS
from logging import error
import data
import conf
import pandas as pd
import model_global
import numpy as np
import tools

#%% Data loading
paths = conf.get_paths()

# PREPROCESS THE INPUT TEXTS
print('######## LOADING TEXTS...')
raw_orgFiles, sdgs_orgFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
raw_natureShort, sdgs_nature, index_abstracts = data.get_nature_abstracts()
raw_natureExt, sdgs_natureAll, index_full = data.get_nature_files(abstract=True, kw=True, intro=True, body=True, concl=True)

def prepare_texts(corpus):
    newCorpus = []
    for text in corpus:
        newCorpus.append(" ".join(tools.tokenize_text(text, lemmatize=True, stem=False ,extended_stopwords=True)))
    return newCorpus
        
# trainFiles = prepare_texts(raw_trainFiles)
orgFiles = prepare_texts(raw_orgFiles)
natureShort = prepare_texts(raw_natureShort)

# LOADING SECTION - ALL MODELS SHOULD HAVE BEEN TRAINED AND SAVED BEFORE THE CALL TO THIS SCRIPT
print('######## LOADING MODELS...')
model = model_global.Global_Classifier(paths=paths, verbose=True)
model.load_models()
model.test_model(raw_corpus=raw_natureShort, corpus=natureShort, associated_SDGs=sdgs_nature, 
                 path_to_plot="", 
                 path_to_excel=paths["out"] + "All/all_model.xlsx", 
                 only_bad=False, score_threshold=3.0,  only_positive=True, filter_low=True)
