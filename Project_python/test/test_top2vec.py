# Configures the project paths: they can be launched from any code
from pkgutil import iter_importers
import sys, os
sys.path.append(os.path.realpath('.'))
import conf
conf.import_paths()

# Configuration flags
flag_train = False # If set to True, then the models are trained based on the csv.
n_repetitions = 10

# Real imports required by the file for work properly
from logging import error
import data
import conf
import pandas as pd
import model_top2vec
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


# TRAINING SECTION
print('# Training model...')
top2vec = model_top2vec.Top2Vec_classifier(paths, verbose=True)

for jj in range(n_repetitions):
  optim_param = pd.read_excel(paths["ref"] + "optimization_top2vec.xlsx")
  min_count = list(optim_param["min_count"])
  ngram = list(optim_param["ngram"])
  embedding_model = list(optim_param["embedding_model"])
  speed = list(optim_param["speed"])
  use_embedding_model_tokenizer  = list(optim_param["use_embedding_model_tokenizer "])
  parsed = list(optim_param["parsed"])
  filter = list(optim_param["filter"])
  only_positive = list(optim_param["only_positive"])
  
  tops_ascii = []; tops_raw = []; stats = []; perc_test = []; perc_train = []
  for ii in range(len(optim_param)):
    print('# Optimizing case: {} of {}'.format(ii + 1, len(optim_param)))
    
    if parsed[ii]: trainData = [orgFiles, sdgs_org]
    else: trainData = [raw_orgFiles, sdgs_org]
    

    # try: 
    if flag_train:
      top2vec.train(train_data=trainData, embedding_model=embedding_model[ii], method=speed[ii], 
                    ngram=ngram[ii], min_count=min_count[ii], workers=8, tokenizer=use_embedding_model_tokenizer[ii]) # "doc2vec", "all-MiniLM-L6-v2", universal-sentence-encoder

      [sum_per_topic_raw, sum_per_topic_ascii] = top2vec.map_model_topics_to_sdgs(normalize=True,
                                    path_csv=(paths["out"] + "Top2vec/" + "topics_it{}_{}.csv".format(jj, ii)),
                                    version=2
                                )
      tools.save_obj(top2vec, paths["model"] + "top2vec_it{}_{}.pickle".format(jj, ii))
    else: 
      top2vec = tools.load_obj(path=(paths["model"] + "top2vec.pickle"))
      [sum_per_topic_raw, sum_per_topic_ascii] = top2vec.map_model_topics_to_sdgs(normalize=True,
                                    path_csv=(paths["out"] + "Top2vec/" + "topics.csv"),
                                    version=2)
    
    perc_global_train, perc_single_train, probs_per_sdg_train, maxSDG, pred_sdgs = top2vec.test_model(corpus=trainData[0], associated_SDGs=trainData[1],
                  filter_low=filter[ii], score_threshold=0.2, only_positive=only_positive[ii],
                    path_to_excel=(paths["out"] + "Top2vec/" + "test_top2vec_training_files_it{}_{}.xlsx".format(jj, ii)), 
                    only_bad=False, expand_factor=1.0, normalize=False
                    )
    tools.plot_ok_vs_nok_SDGsidentified(trainData[1], pred_sdgs, paths["out"] + "Top2vec/" + "sdgs_train_it{}_{}.png".format(jj, ii))
    expandFactor = 1 / maxSDG
    print('# Expand factor: {:.2f}'.format(expandFactor))
  
    if parsed[ii]: testTexts = natureShort
    else: testTexts = raw_natureShort
    perc_global, perc_single, probs_per_sdg_test, maxSDG, pred_sdgs = top2vec.test_model(corpus=testTexts, associated_SDGs=sdgs_natureShort,
                    filter_low=filter[ii], score_threshold=0.1, only_positive=only_positive[ii],
                      path_to_excel=(paths["out"] + "Top2vec/" + "test_top2vec_abstractsLow_it{}_{}.xlsx".format(jj, ii)), 
                      only_bad=False, expand_factor=expandFactor, version=1, normalize=False, normalize_threshold=0.2
                      )
    tools.plot_ok_vs_nok_SDGsidentified(sdgs_natureShort, pred_sdgs, paths["out"] + "Top2vec/" + "sdgs_test_it{}_{}.png".format(jj, ii))
    
    pred_sdgs = pd.DataFrame(pred_sdgs)
    pred_sdgs.to_csv(paths["out"] + "ALL/Individual/pred_test_top2vec.csv")
    
    if not flag_train: 
      print('# Closing testing, no training...')
      break
    
  if not flag_train: 
      print('# Closing testing, no training...')
      break
    
    # except:
    #   print('# Case {} raised an exception, proceeding to next case'.format(ii))

  # if n_repetitions > 1:
  #   usr = input('Continue? (y/n): ')
  #   usr = usr.lower()
  #   if usr == "n": break
    