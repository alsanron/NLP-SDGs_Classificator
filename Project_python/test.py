from logging import error
import data
import conf
import pandas as pd
import tools


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
# natureAbstracts = data.get_nature_abstracts(paths["Nature"])
# natureFiles = data.get_nature_files(paths["Nature"])
# abstracts = data.get_previous_classified_abstracts(paths["Abstracts"])
sdgsFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
textsPathfinder = data.get_sdgs_pathfinder(paths["ref"])

texts = [elem[0] for elem in textsPathfinder]
test_text_preprocess(texts)