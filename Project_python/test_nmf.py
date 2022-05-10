from logging import error
import data
import conf
import pandas as pd
import tools
import model_nmf
import json
import os


def test_text_preprocess(texts, n_texts=20):

    if len(texts) < n_texts: error("Hey, check input texts")
    outPath = "test/"
    texts = texts[0:(n_texts - 1)]
    
    print("#### No lemmatize, no stem")
    with open(outPath + "text_preprocess.txt", "w", encoding="utf-8") as f:
        for text in texts:
            f.write("###################################### \r")
            f.write("##### ORIGINAL: \r" + text + "\r\r")
            f.write("##### NO_LEM_NO_STEM: \r" + " ".join(tools.tokenize_text(text, lemmatize=False, stem=False)) + "\r\r")
            f.write("##### LEM_NO_STEM: \r" + " ".join(tools.tokenize_text(text, lemmatize=True, stem=False)) + "\r\r")
            f.write("##### NO_LEM_STEM: \r" + " ".join(tools.tokenize_text(text, lemmatize=False, stem=True)) + "\r\r")
            f.write("##### LEM_STEM: \r" + " ".join(tools.tokenize_text(text, lemmatize=True, stem=True)) + "\r\r")
        f.close()
    
#%% Data loading
paths = conf.get_paths()

# PREPROCESS THE INPUT TEXTS
print('######## LOADING TEXTS...')
raw_orgFiles, sdgs_orgFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
raw_natureShort, sdgs_nature, index_abstracts = data.get_nature_abstracts()
raw_natureExt, sdgs_natureAll, index_full = data.get_nature_files(abstract=True, kw=True, intro=True, body=True, concl=True)
# raw_pathFinder, sdgs_pathFinder = data.get_sdgs_pathfinder(paths["ref"], min_words=200)

topWords = 30
def prepare_texts(corpus):
    newCorpus = []
    for text in corpus:
        newCorpus.append(" ".join(tools.tokenize_text(text, lemmatize=True, stem=False ,extended_stopwords=True)))
    return newCorpus
        
# trainFiles = prepare_texts(raw_trainFiles)
orgFiles = prepare_texts(raw_orgFiles)
natureShort = prepare_texts(raw_natureShort)
natureExt = prepare_texts(raw_natureExt)

trainData = [orgFiles, sdgs_orgFiles]

# TRAINING SECTION
print('######## TRAINING MODELS...')
nmf = model_nmf.NMF_classifier(paths,verbose=False)

# nmf.train(train_data=trainData, n_topics=16, ngram=(1,2), min_df=1)
# nmf.save()
nmf.load(n_topics=16)
nmf.train_data = trainData
nmf.map_model_topics_to_sdgs(n_top_words=topWords, normalize=True, path_csv="out/topics_nmf_global_bigram3.csv")

# TESTING SECTION
print('######## TESTING MODELS...')
nmf.test_model(corpus=natureShort, associated_SDGs=sdgs_nature, score_threshold=0.2,
               segmentize=-1, filter_low=True,
               path_to_excel=(paths["out"] + "NMF/" + "test_nmf_abstracts.xlsx")
               )

nmf.test_model(corpus=natureExt, associated_SDGs=sdgs_natureAll, score_threshold=0.2,
               segmentize=-1, filter_low=True,
               path_to_excel=(paths["out"] + "NMF/" + "test_nmf_full.xlsx")
               )

nmf.test_model(corpus=trainData[0], associated_SDGs=trainData[1], score_threshold=0.2,
               segmentize=-1, filter_low=True,
               path_to_excel=(paths["out"] + "NMF/" + "test_nmf_training_files.xlsx")
               )
