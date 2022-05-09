from logging import error
import data
import conf
import pandas as pd
import tools
import model_top2vec
import json
import os

#%% Data loading
paths = conf.get_paths()

# PREPROCESS THE INPUT TEXTS
print('######## LOADING TEXTS...')
raw_orgFiles, sdgs_orgFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
raw_natureShort, sdgs_nature, index_abstracts = data.get_nature_abstracts()
raw_natureExt, sdgs_natureAll, index_full = data.get_nature_files(abstract=True, kw=True, intro=True, body=True, concl=True)
# raw_pathFinder, sdgs_pathFinder = data.get_sdgs_pathfinder(paths["ref"], min_words=200)
raw_extraFiles, sdgs_extra = data.get_extra_manual_files(paths["ref"])
raw_healthcare, sdgs_healthcare = data.get_health_care_files(paths["ref"], n_files=100)

train = 0

# TRAINING SECTION
print('######## TRAINING MODELS...')
top2vec = model_top2vec.Top2Vec_classifier(paths, verbose=True)
trainData = [raw_orgFiles + raw_extraFiles + raw_healthcare, sdgs_orgFiles + sdgs_extra + sdgs_healthcare]

if train:
    top2vec.train(train_data=trainData, embedding_model="all-MiniLM-L6-v2", method="learn", ngram=True, min_count=1, workers=8, embedding_batch_size=10, tokenizer=False, split=False, nSplit=25) #"all-MiniLM-L6-v2", universal-sentence-encoder
    top2vec.save()
else:
    top2vec.load(trainData)
    
top2vec.map_model_topics_to_sdgs(normalize=True,
                                 path_csv=(paths["out"] + "Top2vec/" + "topics.csv")
                                 )
# TESTING SECTION
print('######## TESTING MODELS...')
top2vec.test_model(corpus=raw_natureShort, associated_SDGs=sdgs_nature,
                   filter_low=True, score_threshold=0.2, only_positive=True,
                     path_to_excel=(paths["out"] + "Top2vec/" + "test_abstracts.xlsx"), 
                     only_bad=False, 
                     )
top2vec.test_model(corpus=raw_natureExt, associated_SDGs=sdgs_natureAll,
                   filter_low=True, score_threshold=0.2, only_positive=True,
                     path_to_excel=(paths["out"] + "Top2vec/" + "test_full.xlsx"), 
                     only_bad=False, 
                     )