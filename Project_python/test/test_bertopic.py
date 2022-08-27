# Configures the project paths: they can be launched from any code
from pkgutil import iter_importers
import sys, os
sys.path.append(os.path.realpath('.'))
import conf
conf.import_paths()

# Configuration flags
flag_train = False # If set to True, then the models are trained based on the csv.
n_repetitions = 10
score_threshold = 0.2; filter = True; normalize = False

# Real imports required by the file for work properly
from logging import error
import data
import conf
import pandas as pd
import numpy as np
import model_bertopic
import tools

# Loads all the datasets
print('# Loading datasets...')
paths = conf.get_paths()
ds_train = data.get_dataset(requires_update=False, filter=["org", "manual_extra"])
ds_valid_short = data.get_dataset(requires_update=False, filter=["nature_abstract"])
ds_valid_long = data.get_dataset(requires_update=False, filter=["nature_all"])

raw_orgFiles = ds_train["standard"]
raw_natureShort = ds_valid_short["standard"]
raw_natureExt = ds_valid_long["standard"]
   
orgFiles = ds_train["lem"]; sdgs_org = ds_train["sdgs"]
natureShort = ds_valid_short["lem"]; sdgs_natureShort = ds_valid_short["sdgs"]
natureLong = ds_valid_long["lem"]; sdgs_natureLong = ds_valid_long["sdgs"]

topics_seed_list = data.get_sdgs_seed_list(refPath=paths["ref"])

# Training section
print('# Training model...')
bertopic = model_bertopic.BERTopic_classifier(paths)

optim_param = pd.read_excel(paths["ref"] + "optimization_bertopic.xlsx")
ngram = list(optim_param["ngram"])
embedding_model = list(optim_param["embedding_model"])
top_n_words = list(optim_param["top_n_words"])
min_topic_size = list(optim_param["min_topic_size"])
nr_topics = list(optim_param["nr_topics"])
diversity = list(optim_param["diversity"])
seed_topic_list  = list(optim_param["seed_topic_list"])
parsed = list(optim_param["parsed"])
nOptims = len(optim_param)

optim_param = pd.DataFrame(np.repeat(optim_param.values, n_repetitions, axis=0))

tops_ascii = []; tops_raw = []; stats = []; perc_test = []; perc_train = []
for ii in range(nOptims):
    for jj in range(n_repetitions):
      print('# Training case: {}.{} of {}'.format(ii + 1, jj + 1, nOptims))
      
      if parsed[ii]: trainData = [orgFiles, sdgs_org]
      else: trainData = [raw_orgFiles, sdgs_org]
      
      if seed_topic_list[ii] == True: topic_list = topics_seed_list
      else: topic_list = None
        
      ngram_range = ngram[ii][1:-1].split(",") # from ascii to tuple int
      ngram_range = (int(ngram_range[0]), int(ngram_range[1]))

      if flag_train:
        bertopic.train_model(train_data=trainData, 
                      embedding_model=embedding_model[ii], # Others: all-MiniLM-L6-v2, all-MiniLM-L12-v2, all-mpnet-base-v2
                      n_gram_range=ngram_range, # the default parameter is (1,1)
                      top_n_words=top_n_words[ii], min_topic_size=min_topic_size[ii], # default parameters
                      nr_topics=nr_topics[ii], # reduce the number of topics to this number
                      diversity=diversity[ii], # value can be used between 0, 1
                      calculate_probabilities=True, 
                      seed_topic_list=topic_list,
                      verbose=False
                      )
        bertopic.print_summary(path_csv=paths["out"] + "Bertopic/topics_it_{}case_{}.csv".format(jj, ii))
        tools.save_obj(bertopic, paths["model"] + "bertopic_it{}_case{}.pickle".format(jj, ii))
      else:
        modelPath = paths["model"] + "bertopic.pickle"
        bertopic = tools.load_obj(paths["model"] + "bertopic.pickle")
        np.savetxt(paths["out"] + "Bertopic/association_matrix.csv", bertopic.topics_association, delimiter=",")
        print('# Loading model from: ' + modelPath)
      
      fig = bertopic.model.visualize_topics()
      fig.write_html(paths["out"] + "Bertopic/" + "vis_topics_it_{}case_{}.html".format(jj, ii))
      
      fig =  bertopic.model.visualize_barchart()
      fig.write_html(paths["out"] + "Bertopic/" + "vis_barchart_it_{}case_{}.html".format(jj, ii))
      
      fig = bertopic.model.visualize_heatmap()
      fig.write_html(paths["out"] + "Bertopic/" + "vis_heatmap_it_{}case_{}.html".format(jj, ii))
    
      print('# Testing training files')
      predic, maxSDG, perc_global, perc_single = bertopic.test_model(corpus=trainData[0], associated_SDGs=trainData[1], filter_low=filter,
                                          score_threshold=score_threshold, only_positive=False, 
                                          path_to_excel=(paths["out"] + "Bertopic/" + "test_training_files_it{}case_{}.xlsx".format(jj, ii)), 
                                          only_bad=False, expand_factor=1.0, normalize=normalize)
      perc_train.append("{:.2f} - {:.2f}".format(perc_single, perc_global))
      # tools.analyze_predict_real_sdgs(trainData[1], predic, path_out=paths["out"] + "Bertopic/", case_name="training_files_it{}case_{}".format(jj, ii), show=False)
      tools.plot_ok_vs_nok_SDGsidentified(trainData[1], predic, paths["out"] + "Bertopic/" + "sdgs_training_it{}_{}.png".format(jj, ii))
      
      expandFactor = 1 / maxSDG
      print('# Expand factor: {:.2f}'.format(expandFactor))
      
      if parsed[ii]: testTexts = natureShort
      else: testTexts = raw_natureShort
      
      print('# Testing test files')
      predic, maxSDG, perc_global, perc_single = bertopic.test_model(corpus=testTexts, associated_SDGs=sdgs_natureShort, filter_low=filter, 
                                          score_threshold=0.15, only_positive=True, 
                                          path_to_excel=(paths["out"] + "Bertopic/" + "test_nature_short_it{}case_{}.xlsx".format(jj, ii)),
                                          only_bad=False, expand_factor=1.0, normalize=normalize)
      perc_test.append("{:.2f} - {:.2f}".format(perc_single, perc_global))
      # tools.analyze_predict_real_sdgs(sdgs_natureShort, predic, path_out=paths["out"] + "Bertopic/", case_name="nature_short_it{}case_{}".format(jj, ii), show=False)
      tools.plot_ok_vs_nok_SDGsidentified(sdgs_natureShort, predic, paths["out"] + "Bertopic/" + "sdgs_test_it{}_{}.png".format(jj, ii))
      
      if not flag_train: 
        print('# Closing testing, no training...')
        break
      
    if not flag_train: 
        print('# Closing testing, no training...')
        break
      
if flag_train:    
  # Only stores the results if the model was trained...
  optim_param["perc_test"] = perc_test
  optim_param["perc_train"] = perc_train
  optim_param.to_excel(paths["out"] + "Bertopic/" + "optimization_out.xlsx")