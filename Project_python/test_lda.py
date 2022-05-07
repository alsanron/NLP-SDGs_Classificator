# %%
# script for testing the bertopic functionality and classes
from logging import error
import data
import conf
import pandas as pd
import tools
import json
import os
import model_lda
import tomotopy as tp

paths = conf.get_paths()
raw_orgFiles, sdgs_orgFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
raw_natureShort, sdgs_nature, index_abstracts = data.get_nature_abstracts()
raw_natureExt, sdgs_natureAll, index_full = data.get_nature_files(abstract=True, kw=True, intro=True, body=True, concl=True)
# raw_pathFinder, sdgs_pathFinder = data.get_sdgs_pathfinder(paths["ref"], min_words=200)
# raw_extraFiles, sdgs_extra = data.get_extra_manual_files(paths["ref"])
# raw_healthcare, sdgs_healthcare = data.get_health_care_files(paths["ref"], n_files=100)

def prepare_texts(corpus):
    newCorpus = []
    for text in corpus:
        newCorpus.append(" ".join(tools.tokenize_text(text, lemmatize=False, stem=False ,extended_stopwords=True)))
    return newCorpus
        
# trainFiles = prepare_texts(raw_trainFiles)
orgFiles = prepare_texts(raw_orgFiles)
# extraFiles = prepare_texts(raw_extraFiles)
# healthcareFiles = prepare_texts(raw_healthcare)
natureShort = prepare_texts(raw_natureShort)
natureExt = prepare_texts(raw_natureExt)

# %%
lda = model_lda.LDA_classifier(k=16, min_cf=0, min_df=3, seed=1)
lda.set_conf(paths)

trainData = [orgFiles, sdgs_orgFiles]
lda.train_model(trainData, iterations=100)

pathOut = paths["out"] + "LDA/" + "topics_{}.csv".format(lda.k)
lda.print_summary(top_words=30, 
                  path_csv=pathOut
                  )
lda.map_model_topics_to_sdgs(path_csv="", normalize=True)