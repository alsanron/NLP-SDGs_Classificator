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

# To adjust list:
# TODO EXPAND THE STOPWORDS LIST
# TODO FILTRAR SCORES < 0.05?

paths = conf.get_paths()
raw_orgFiles, sdgs_orgFiles = data.get_sdgs_org_files(paths["SDGs_inf"], compact=True)
raw_natureShort, sdgs_nature, index_abstracts = data.get_nature_abstracts()
raw_natureExt, sdgs_natureAll, index_full = data.get_nature_files(abstract=True, kw=True, intro=True, body=True, concl=True)
# raw_pathFinder, sdgs_pathFinder = data.get_sdgs_pathfinder(paths["ref"], min_words=200)
# raw_extraFiles, sdgs_extra = data.get_extra_manual_files(paths["ref"])
# raw_healthcare, sdgs_healthcare = data.get_health_care_files(paths["ref"], n_files=100)

######## GLOBAL CONFIGURATION
optim_excel = "optimization_lda.xlsx"; out_optim = "out_optimization.xlsx"
optimize = True

lemmatize = False
bigrams = True; min_count_bigram = 10
trigrams = True; min_count_trigram = 5
min_words_count = 1 # minimum number of times a word must appear in the corpus. It should be small since the training set is small
max_words_frequency = 0.7 # max frequency of a word appearing in corpus
        
######## CODE 
def prepare_texts(corpus):
    newCorpus = []
    for text in corpus:
        newCorpus.append(tools.tokenize_text(text, lemmatize=lemmatize, stem=False ,extended_stopwords=True))
    return newCorpus

print('- Preparing texts...')        
# trainFiles = prepare_texts(raw_trainFiles)
orgFiles = prepare_texts(raw_orgFiles)
# extraFiles = prepare_texts(raw_extraFiles)
# healthcareFiles = prepare_texts(raw_healthcare)
natureShort = prepare_texts(raw_natureShort)
natureExt = prepare_texts(raw_natureExt)

trainData = [orgFiles, sdgs_orgFiles]

if bigrams:
        print('Creating bigrams vocabulary...')
        bigram = Phrases(trainData[0], min_count=min_count_bigram)
        for idx in range(len(trainData[0])):
            for token in bigram[trainData[0][idx]]:
                if token.count("_") == 1:
                    trainData[0][idx].append(token)
        if trigrams:
            print('Creating trigrams vocabulary...')
            trigram = Phrases(trainData[0], min_count=min_count_trigram)
            for idx in range(len(trainData[0])):
                for token in trigram[trainData[0][idx]]:
                    if token.count("_") == 2:
                        trainData[0][idx].append(token)

dict = Dictionary(trainData[0])
dict.filter_extremes(no_below=min_words_count, no_above=max_words_frequency)
dict[0] # just to load the dict
id2word = dict.id2token
corpus = [dict.doc2bow(text) for text in trainData[0]]

print('Training model...')
optimData = pd.read_excel(paths["ref"] + optim_excel)
out_perc_global = []; out_perc_any = []
for ii in range(len(optimData)):
    print('# Case: {} of {}'.format(ii + 1, len(optimData)))
    num_topics = optimData["num_topics"][ii]
    chunksize = optimData["chunksize"][ii]
    passes = optimData["passes"][ii]
    iterations = optimData["iterations"][ii]
    update_every = optimData["update_every"][ii]
    topics_csv = optimData["topics_csv"][ii]
    out_test_excel = optimData["out_test_excel"][ii]
    score_threshold = optimData["score_threshold"][ii]
    only_bad = optimData["only_bad"][ii]
    only_positive = optimData["only_positive"][ii]
    
    lda = model_lda.LDA_classifier(corpus=corpus, id2word=id2word,
                                    chunksize=chunksize,
                                    iterations=iterations,
                                    num_topics=num_topics,
                                    passes=passes,
                                    minimum_probability=0.0001,
                                    update_every=update_every,
                                    eval_every=None,
                                    random_state=1
                                    )
    print('Model postprocessing...')
    lda.set_conf(paths, dict)
    lda.print_summary(top_words=30, 
                        path_csv=(paths["out"] + "LDA/" + topics_csv)
                    )
    sumPerTopic, listAscii = lda.map_model_topics_to_sdgs(trainData, path_csv="", normalize=True, verbose=True)
    # lda.save_model()
    
    print('Testing model...')
    rawSDG, perc_valid_global, perc_valid_any = lda.test_model(natureShort, sdgs_nature, path_to_plot="", 
                                                               path_to_excel=(paths["out"] +     "LDA/" + out_test_excel), only_bad=only_bad, 
                                                               score_threshold=score_threshold,
                                                               only_positive=only_positive)
    out_perc_global.append(perc_valid_global); out_perc_any.append(perc_valid_any)
optimData["perc_global"] = out_perc_global
optimData["perc_any"] = out_perc_any
optimData.to_excel(paths["out"] + "LDA/" + out_optim)