from logging import error
import data
import conf
import pandas as pd
import model_bertopic
import numpy as np
import tools

#%% Data loading
paths = conf.get_paths()

# PREPROCESS THE INPUT TEXTS
print('######## LOADING TEXTS...')
raw_orgFiles, sdgs_orgFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
raw_natureShort, sdgs_nature, index_abstracts = data.get_nature_abstracts()
raw_natureExt, sdgs_natureAll, index_full = data.get_nature_files(abstract=True, kw=True, intro=True, body=True, concl=True)
raw_extraFiles, sdgs_extra = data.get_extra_manual_files(paths["ref"], sdg_query=[1,3,10,15])
topics_seed_list = data.get_sdgs_seed_list(refPath=paths["ref"])

def prepare_texts(corpus):
    newCorpus = []
    for text in corpus:
        newCorpus.append(" ".join(tools.tokenize_text(text, lemmatize=True, stem=False ,extended_stopwords=True)))
    return newCorpus
        
orgFiles = prepare_texts(raw_orgFiles)
natureShort = prepare_texts(raw_natureShort)
natureExt = prepare_texts(raw_natureExt)
extraFiles = prepare_texts(raw_extraFiles)

optimize = 1

# TRAINING SECTION
print('######## TRAINING MODELS...')
bertopic = model_bertopic.BERTopic_classifier(paths)

for jj in range(1):
  if optimize:
    optim_param = pd.read_excel(paths["ref"] + "Bertopic/" + "optimization_bertopic.xlsx")
    ngram = list(optim_param["ngram"])
    embedding_model = list(optim_param["embedding_model"])
    top_n_words = list(optim_param["top_n_words"])
    min_topic_size = list(optim_param["min_topic_size"])
    nr_topics = list(optim_param["nr_topics"])
    diversity = list(optim_param["diversity"])
    seed_topic_list  = list(optim_param["seed_topic_list"])
    ext_dataset = list(optim_param["ext_dataset"])
    parsed = list(optim_param["parsed"])
    
    tops_ascii = []; tops_raw = []; stats = []; perc_test = []; perc_train = []
    for ii in range(len(optim_param)):
      print('Optimizing case: {} of {}'.format(ii + 1, len(optim_param)))
      
      if ext_dataset[ii]:
        if parsed[ii]: trainData = [orgFiles + extraFiles, sdgs_orgFiles + sdgs_extra ]
        else: 
          trainData = [raw_orgFiles + raw_extraFiles, sdgs_orgFiles + sdgs_extra]
      else:
        if parsed[ii]: trainData = [orgFiles, sdgs_orgFiles]
        else: trainData = [raw_orgFiles, sdgs_orgFiles]
        
      if seed_topic_list[ii] == True: topic_list = topics_seed_list
      else: topic_list = None
        
      ngram_range = ngram[ii][1:-1].split(",") # from ascii to tuple int
      ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        
      # store the training files in csv
      # df = pd.DataFrame()
      # df["files"] = trainData[0]
      # df.to_csv(paths["out"] + "Bertopic/training_texts.csv")

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
      
      fig = bertopic.model.visualize_topics()
      fig.write_html(paths["out"] + "Bertopic/" + "vis_topics_it_{}case_{}.html".format(jj, ii))
      
      fig =  bertopic.model.visualize_barchart()
      fig.write_html(paths["out"] + "Bertopic/" + "vis_barchart_it_{}case_{}.html".format(jj, ii))
      
      fig = bertopic.model.visualize_heatmap()
      fig.write_html(paths["out"] + "Bertopic/" + "vis_heatmap_it_{}case_{}.html".format(jj, ii))

      tools.save_obj(bertopic, paths["model"] + "bertopic_it{}_case{}.pickle".format(jj, ii))
      
      filter = True; normalize = False
      
      print('## Testing training files')
      perc_global_train, perc_single_train, probs_per_sdg_train, maxSDG = bertopic.test_model(corpus=trainData[0], associated_SDGs=trainData[1], filter_low=filter, score_threshold=0.2, only_positive=True, path_to_excel=(paths["out"] + "Bertopic/" + "test_training_files_it{}case_{}.xlsx".format(jj, ii)), only_bad=False, expand_factor=1.0, normalize=normalize)
      expandFactor = 1 / maxSDG
      print('Expand factor: {:.2f}'.format(expandFactor))
      
      if parsed[ii]: testTexts = natureShort
      else: testTexts = raw_natureShort
      
      print('## Testing test files')
      perc_global_test, perc_single_test, probs_per_sdg_test, maxSDG = bertopic.test_model(corpus=testTexts, associated_SDGs=sdgs_nature, filter_low=filter, score_threshold=0.15, only_positive=True, path_to_excel=(paths["out"] + "Bertopic/" + "test_nature_short_it{}case_{}.xlsx".format(jj, ii)), only_bad=False, expand_factor=1.0, normalize=normalize)
      
      # perc_global, perc_single, probs_per_sdg_test, maxSDG = top2vec.test_model(corpus=raw_natureShortFilt, associated_SDGs=sdgs_natureFilt,
      #             filter_low=filter, score_threshold=0.5, only_positive=True,
      #               path_to_excel=(paths["out"] + "Top2vec/" + "test_top2vec_abstractsHigh_it{}_{}.xlsx".format(jj, ii)), 
      #               only_bad=False, expand_factor=2, version=1, normalize=normalize, normalize_threshold=0.25
      #               )
      
      # perc_global, perc_single, probs_per_sdg_test = top2vec.test_model(corpus=raw_natureExt, associated_SDGs=sdgs_natureAll,
      #                 filter_low=True, score_threshold=0.2, only_positive=True,
      #                   path_to_excel=(paths["out"] + "Top2vec/" + "test_top2vec_full{}.xlsx".format(ii)), 
      #                   only_bad=False, expand_factor=2, version=1
      #                   )
      
      perc_test.append("{:.3f}, {:.3f}".format(perc_global_test, perc_single_test))
      perc_train.append("{:.3f}, {:.3f}".format(perc_global_train, perc_single_train))
      
    optim_param["perc_test"] = perc_test
    optim_param["perc_train"] = perc_train
    optim_param.to_excel(paths["out"] + "Bertopic/" + "optimization_out.xlsx")
    
    # usr = input('Continue? (y/n): ')
    # usr = usr.lower()
    # if usr == "n": break
    
  else:
      top2vec.load(trainData)
    