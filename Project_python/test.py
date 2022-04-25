from logging import error
import data
import conf
import pandas as pd
import tools
import model
import json


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
    
def test_individual_models_nmf():
    paths = conf.get_paths()
    models_nmf = model.get_individual_models_per_sdg(flag_train=True)
    sdgsFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
    
    
#%% Data loading
paths = conf.get_paths()
# natureAbstracts = data.get_nature_abstracts(paths["Nature"])
# natureFiles = data.get_nature_files(paths["Nature"])
# abstracts = data.get_previous_classified_abstracts(paths["Abstracts"])
# sdgsFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
# textsPathfinder = data.get_sdgs_pathfinder(paths["ref"])

# texts = [elem[0] for elem in textsPathfinder]
# test_text_preprocess(texts)
trainFiles = [file[0] for file in data.get_sdgs_org_files(paths["SDGs_inf"])]
topWords = 25

lda = model.LDA_classifier(paths)
lda.train_individual_model_per_sdg(trainFiles)
# lda.export_individual_model_topics_to_csv("out/topics_lda_individual_models_monogram.csv", n_top_words=topWords)
lda.train_global_model(trainFiles, n_topics=18)
lda.export_global_model_topics_to_csv("out/topics_lda_global_unordered.csv", n_top_words=topWords)
# lda.map_model_topics_to_sdgs(n_top_words=topWords, path_csv="out/topics_lda_global_monogram.csv")

nmf = model.NMF_classifier(paths)
nmf.train_individual_model_per_sdg(multigrams=(1,1))
# nmf.load_individual_model_per_sdg()
nmf.export_individual_model_topics_to_csv("out/topics_nmf_individual_models_monogram.csv", n_top_words=topWords)

nmf.train_global_model(trainFiles, n_topics=17, multigrams=(1,1))
# nmf.load_global_model(n_topics=17)
nmf.map_model_topics_to_sdgs(n_top_words=topWords, path_csv="out/topics_nmf_global_monogram.csv")
# nmf.export_global_model_topics_to_csv("out/topics_global.csv", n_top_words=25)
# nmf.map_model_topics_to_sdgs(n_top_words=25, path_csv="out/topics_global.csv")

with open(paths["ref"] + "ext_database.json", "r") as f:
    json_dump = f.read()
    f.close()
validationDB = json.loads(json_dump)
# print('{} validation files'.format{len(validationDB)})
nmf.test_model(database=validationDB, path_excel="out/matrix_classification_abstract.xlsx", abstract=True)
nmf.test_model(database=validationDB, path_excel="out/matrix_classification_abstract_kw.xlsx", abstract=True, kw=True)
nmf.test_model(database=validationDB, path_excel="out/matrix_classification_abstract_kw_intro.xlsx", abstract=True, kw=True, intro=True)
nmf.test_model(database=validationDB, path_excel="out/matrix_classification_abstract_kw_intro_body.xlsx", abstract=True, kw=True, intro=True, body=True)
nmf.test_model(database=validationDB, path_excel="out/matrix_classification_all.xlsx", abstract=True,kw=True, intro=True, body=True, concl=True)
nmf.test_model(database=validationDB, path_excel="out/matrix_classification_abstract_conclus.xlsx", abstract=True, concl=True)