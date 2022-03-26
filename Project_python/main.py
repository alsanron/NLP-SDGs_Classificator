# Tests an already trained model for classifying a given input paper into the corresponding Sustainable Development goals

from preprocess import get_validation_files, get_training_files
import tools
import pandas as pd
import train
import json
import validate


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
nTopWords = 30
res_singleSDG = []
df = pd.DataFrame()
colNames = []
for ii in range(1, 18):
    trainFiles = get_training_files(refPath=paths["training"], sdg=ii)
    nmf_res = train.train_nmf(trainFiles, n_topics=nTopics, ngram=multigrams)
    res_singleSDG.append(nmf_res)
    topics = train.get_topics(model=res_singleSDG[ii - 1][0], vectorizer=res_singleSDG[ii - 1][1], n_top_words=nTopWords,       n_topics=nTopics)
    df = pd.concat([df, topics], ignore_index=True, axis=1)
    colNames.append("Topic #{:02d} - {}".format(ii, sdgs_title["SDG{}".format(ii)]))

df.columns = colNames
df.to_csv("out/single_topics_n{}.csv".format(17))

#%% 1 model with nTopics is trained with the same documents. This model is used for text classification. Each topic gets associated with the real SDG based on its similarity with the topics obtained in the 17 models.
nTopics = 17
trainFiles = get_training_files(refPath=paths["training"])
nmf_res = train.train_nmf(trainFiles, n_topics=nTopics, ngram=multigrams)
topics = train.get_topics(model=nmf_res[0], vectorizer=nmf_res[1], n_top_words=nTopWords, n_topics=nTopics)
[topics_association, sdgs_coh, sdgs_found] = validate.map_model_topics_to_sdgs(res_singleSDG, topics, 
                                                                               pathToCsv=paths["out"]+"association_map.csv", 
                                                                               verbose=True)
validFilesDict = get_validation_files(preprocess=False, refPath=paths["validation"])
[percOk, percents, okPerSDG, countPerSDG] = validate.validate_model(model=nmf_res[0], 
                                                                    vectorizer=nmf_res[1],                         topics_association=topics_association,                        sdgs_mapped=sdgs_found,                         validFilesDict=validFilesDict,
                                                                    pathToCsv=paths["out"]+"results.csv")

#%% Show how good the model fits the validation files, and extract the topics
# topics = train.get_topics(model=model_nmf, vectorizer=vect_nmf, n_top_words=nTopWords, n_topics=nTopics)
# topics.to_csv("out/topics_n{}.csv".format(nTopics))

# Optimize the number of topics for the best fit with the files. Maybe 17 does not imply ok.

# Show how good classifies the model the new test files according to the previous class

# Optimize the number of topics to maximize the classification accuracy

# Introduce as variable the multigrams, and test if the results are improved