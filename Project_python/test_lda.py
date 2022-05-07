# %%
# script for testing the bertopic functionality and classes
from logging import error
import data
import conf
import pandas as pd
import tools
import json
import os
import numpy as np
import model_lda
from gensim.models import Phrases
from gensim.corpora.dictionary import Dictionary

paths = conf.get_paths()
raw_orgFiles, sdgs_orgFiles = data.get_sdgs_org_files(paths["SDGs_inf"], compact=True)
raw_natureShort, sdgs_nature, index_abstracts = data.get_nature_abstracts()
raw_natureExt, sdgs_natureAll, index_full = data.get_nature_files(abstract=True, kw=True, intro=True, body=True, concl=True)
# raw_pathFinder, sdgs_pathFinder = data.get_sdgs_pathfinder(paths["ref"], min_words=200)
# raw_extraFiles, sdgs_extra = data.get_extra_manual_files(paths["ref"])
# raw_healthcare, sdgs_healthcare = data.get_health_care_files(paths["ref"], n_files=100)

def prepare_texts(corpus):
    newCorpus = []
    for text in corpus:
        newCorpus.append(tools.tokenize_text(text, lemmatize=True, stem=False ,extended_stopwords=True))
    return newCorpus
        
# trainFiles = prepare_texts(raw_trainFiles)
orgFiles = prepare_texts(raw_orgFiles)
# extraFiles = prepare_texts(raw_extraFiles)
# healthcareFiles = prepare_texts(raw_healthcare)
natureShort = prepare_texts(raw_natureShort)
natureExt = prepare_texts(raw_natureExt)

# %%
# lda = model_lda.LDA_classifier(k=16, min_cf=0, min_df=3, seed=1)
# lda.set_conf(paths)

trainData = [orgFiles, sdgs_orgFiles]
if 1:
    bigram = Phrases(trainData[0], min_count=10)
    for idx in range(len(trainData[0])):
        for token in bigram[trainData[0][idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                trainData[0][idx].append(token)
# for text in trainData[0]:
#     print(' | '.join(text))
#     a=input()
# PREPARE TRAINING DATA
dict = Dictionary(trainData[0])
dict.filter_extremes(no_below=1, no_above=0.7)
dict[0] # just to load the dict
id2word = dict.id2token
corpus = [dict.doc2bow(text) for text in trainData[0]]

# num_topics = 16
# chunksize = 30
# passes = 200
# iterations = 500

eval_every = None  # Don't evaluate model perplexity, takes too much time.

modelPath = paths["model"] + "lda_model"

if 1:   
    if 1:
        optimData = pd.read_excel(paths["ref"] + "optimization_lda.xlsx")
        out_sum_per_topic = []; out_stats = []
        for ii in range(len(optimData)):
            print('# Case: {} of {}'.format(ii, len(optimData)))
            num_topics = optimData["num_topics"][ii]
            chunksize = optimData["chunksize"][ii]
            passes = optimData["passes"][ii]
            iterations = optimData["iterations"][ii]
            update_every = optimData["update_every"][ii]
            lda = model_lda.LDA_classifier(corpus=corpus, id2word=id2word, 
                                            chunksize=chunksize,
                                            # alpha='auto',
                                            # eta='auto',
                                            iterations=iterations,
                                            num_topics=num_topics,
                                            passes=passes,
                                            minimum_probability=0.0001,
                                            update_every=update_every,
                                            # eval_every=eval_every,
                                            random_state=1
                                            )
            lda.set_conf(paths, dict)
            lda.print_summary(top_words=30, 
    #                   #path_csv=pathOut
                    )
            sumPerTopic, listAscii = lda.map_model_topics_to_sdgs(trainData, path_csv="", normalize=True, verbose=True)
            out_sum_per_topic.append(listAscii)
            out_stats.append("mean: {:.2f}, std: {:.2f}".format(np.mean(sumPerTopic), np.std(sumPerTopic)))
        optimData["sum_per_topic"] = out_sum_per_topic
        optimData["stats"] = out_stats
        optimData.to_excel(paths["out"] + "LDA/" + "out_optimization1text_final.xlsx")
    else:
        num_topics = 17
        chunksize = 2000
        passes = 100
        iterations = 1000
        update_every = 1
        lda = model_lda.LDA_classifier(corpus=corpus, id2word=id2word, 
                                        chunksize=chunksize,
                                        # alpha='auto',
                                        # eta='auto',
                                        iterations=iterations,
                                        num_topics=num_topics,
                                        passes=passes,
                                        minimum_probability=0.0001,
                                        update_every=update_every,
                                        # eval_every=eval_every,
                                        random_state=1
                                        )
        lda.set_conf(paths, dict)
        lda.print_summary(top_words=30, 
#                   #path_csv=pathOut
                )
        sumPerTopic, listAscii = lda.map_model_topics_to_sdgs(trainData, path_csv="", normalize=True, verbose=True)
else:
    lda = model_lda.LDA_classifier.load(modelPath)
    

pathOut = paths["out"] + "LDA/" + "topics_{}.csv".format(num_topics)
# lda.print_summary(top_words=30, 
#                   #path_csv=pathOut
#                   )
# sumPerTopic, listAscii = lda.map_model_topics_to_sdgs(trainData, path_csv="", normalize=True, verbose=False)
# print(listAscii)