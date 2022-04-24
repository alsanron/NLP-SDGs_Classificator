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
import model



#%% BRUT FORCE OPTIMIZATION OF THE NUMBER_TOPICS AND MULTIGRAMS
# range_topics = range(15,24,2)
# range_multigrams = range(1,4)
# optimize.nmf_brut_force(paths, single_sdgs_models=res_singleSDG, 
#                         n_top_words=nTopWords, 
#                         range_topics=range_topics, 
#                         range_multigrams=range_multigrams)

#%% COMPARISON OF THE RESULTS WHEN USING ORIGINAL TEXTS AND ADDING NEW CORPORA: ERC Papers
# aa = 2
# [percOk, percents, okPerSDG, countPerSDG, exclude_sdg, returnValidFiles] = optimize.train_validate_model(paths, 
#                         single_sdgs_models=res_singleSDG, n_top_words=nTopWords, verbose=False,
#                         n_topics=17, n_multigrams=(1,1), 
#                         abstracts=True,
#                         alpha_w=0.0000)
# print("OK: {:.2f}, Excluded: {}".format(percOk, exclude_sdg))


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