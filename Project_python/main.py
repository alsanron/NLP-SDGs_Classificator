# Tests an already trained model for classifying a given input paper into the corresponding Sustainable Development goals

from preprocess import get_validation_files, get_training_files
import tools
import pandas as pd
import train
import json
import validate
import optimize


#%% Configuration
paths = dict()
paths["ref"] = "ref/"
paths["training"] = "ref/Training/"
paths["validation"] = "ref/Validation/"
paths["out"] = "out/"


#%% Training and validation data load
# validFilesDict = get_validation_files(preprocess=False, refPath=paths["validation"])
# trainFiles = get_training_files(refPath=paths["training"])
f = open(paths["ref"] + "SDG_titles.json")
sdgs_title = json.load(f)
f.close()


#%%  17 models are trained for classifying each SDG
multigrams = (1,1)
nTopics = 1
nTopWords = 25
res_singleSDG = []
df = pd.DataFrame()
colNames = []
for ii in range(1, 18):
    trainFiles = get_training_files(refPath=paths["training"], sdg=ii, abstracts=False)
    nmf_res = train.train_nmf(trainFiles, n_topics=nTopics, ngram=multigrams)
    res_singleSDG.append(nmf_res)
    topics = train.get_topics(model=res_singleSDG[ii - 1][0], vectorizer=res_singleSDG[ii - 1][1], n_top_words=nTopWords,       n_topics=nTopics)
    df = pd.concat([df, topics], ignore_index=True, axis=1)
    colNames.append("Topic #{:02d} - {}".format(ii, sdgs_title["SDG{}".format(ii)]))

df.columns = colNames
df.to_csv("out/single_topics_n{}.csv".format(17))

#%% 1 model with nTopics is trained with the same documents. This model is used for text classification. Each topic gets associated with the real SDG based on its similarity with the topics obtained in the 17 models.

#%% BRUT FORCE OPTIMIZATION OF THE NUMBER_TOPICS AND MULTIGRAMS
# range_topics = range(15,24,2)
# range_multigrams = range(1,4)
# optimize.nmf_brut_force(paths, single_sdgs_models=res_singleSDG, 
#                         n_top_words=nTopWords, 
#                         range_topics=range_topics, 
#                         range_multigrams=range_multigrams)

#%% COMPARISON OF THE RESULTS WHEN USING ORIGINAL TEXTS AND ADDING NEW CORPORA: ERC Papers
[percOk, percents, okPerSDG, countPerSDG, exclude_sdg, returnValidFiles] = optimize.train_validate_model(paths, 
                        single_sdgs_models=res_singleSDG, n_top_words=nTopWords, verbose=True,
                        n_topics=17, n_multigrams=(1,1), 
                        abstracts=True,
                        alpha_w=0.0000)
print("OK: {:.2f}, Excluded: {}".format(percOk, exclude_sdg))


#%% Those files which were classified as valid, are returned as training files and used for the model training in the next iteration
# nIterations = 3
# returnValidFiles = []
# for ii in range(0, nIterations):
#     print("Iteration #{}".format(ii + 1))
#     [percOk, percents, okPerSDG, countPerSDG, exclude_sdg, returnValidFiles] = optimize.train_validate_model(paths, 
#                     single_sdgs_models=res_singleSDG, n_top_words=nTopWords, 
#                     n_topics=17, n_multigrams=(1,1), 
#                     abstracts=True,
#                     alpha_w=0.0000,
#                     new_training=returnValidFiles)
#     print("New training files: {}, OK: {:.2f}, Excluded: {}".format(len(returnValidFiles), percOk, exclude_sdg))