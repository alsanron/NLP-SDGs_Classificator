# Tests an already trained model for classifying a given input paper into the corresponding Sustainable Development goals

import preprocess
import tools
import pandas as pd
import train
import json
import validate
import optimize
import conf
import data

paths = conf.get_paths()
sdgs_title = data.get_sdg_titles(paths["ref"])

#%%  17 models are trained for classifying each SDG
def get_individual_models_per_sdg(flag_train):
    # trains the passed number of models with the information of the onu or returns the already trained models
    
    if flag_train:
        # trains all the models
        multigrams = (1,2)
        nTopics = 1
        nTopWords = 30
        n_sdgs = 17
        
        res_singleSDG = []
        df = pd.DataFrame()
        colNames = []
        for ii in range(1, n_sdgs):
            print("## Training model {}".format(ii))
            trainData = data.get_sdgs_org_files(refPath=paths["SDGs_inf"], sdg=ii)
            trainFiles = [file[0] for file in trainData]
            model, vectorizer = train.train_nmf(trainFiles, n_topics=nTopics, ngram=multigrams)
            res_singleSDG.append([model, vectorizer])
            topics = train.get_topics(model=model, vectorizer=vectorizer, n_top_words=nTopWords, n_topics=nTopics)
            df = pd.concat([df, topics], ignore_index=True, axis=1)
            colNames.append("Topic #{:02d} - {}".format(ii, sdgs_title["SDG{}".format(ii)]))
            
            tools.save_obj(model, paths["model"] + "model_1topic_sdg{}.pickle".format(ii))
            tools.save_obj(vectorizer, paths["model"] + "vect_1topic_sdg{}.pickle".format(ii))

        df.columns = colNames
        df.to_csv("out/topics_individual_model_n{}.csv".format(17))
    else:
        # loads the models that have been trained previously
        
    
get_individual_models_per_sdg(flag_train=True)

#%% 1 model with nTopics is trained with the same documents. This model is used for text classification. Each topic gets associated with the real SDG based on its similarity with the topics obtained in the 17 models.

#%% BRUT FORCE OPTIMIZATION OF THE NUMBER_TOPICS AND MULTIGRAMS
# range_topics = range(15,24,2)
# range_multigrams = range(1,4)
# optimize.nmf_brut_force(paths, single_sdgs_models=res_singleSDG, 
#                         n_top_words=nTopWords, 
#                         range_topics=range_topics, 
#                         range_multigrams=range_multigrams)

#%% COMPARISON OF THE RESULTS WHEN USING ORIGINAL TEXTS AND ADDING NEW CORPORA: ERC Papers
aa = 2
[percOk, percents, okPerSDG, countPerSDG, exclude_sdg, returnValidFiles] = optimize.train_validate_model(paths, 
                        single_sdgs_models=res_singleSDG, n_top_words=nTopWords, verbose=False,
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