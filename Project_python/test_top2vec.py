from logging import error
import data
import conf
import pandas as pd
import model_top2vec
import numpy as np
import tools

#%% Data loading
paths = conf.get_paths()

# PREPROCESS THE INPUT TEXTS
print('######## LOADING TEXTS...')
raw_orgFiles, sdgs_orgFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
raw_natureShort, sdgs_nature, index_abstracts = data.get_nature_abstracts()
raw_natureShortFilt, sdgs_natureFilt = data.get_nature_abstracts_filtered()
raw_natureExt, sdgs_natureAll, index_full = data.get_nature_files(abstract=True, kw=True, intro=True, body=True, concl=True)
# raw_pathFinder, sdgs_pathFinder = data.get_sdgs_pathfinder(paths["ref"], min_words=200)
raw_extraFiles, sdgs_extra = data.get_extra_manual_files(paths["ref"])
raw_healthcare, sdgs_healthcare = data.get_health_care_files(paths["ref"])

def prepare_texts(corpus):
    newCorpus = []
    for text in corpus:
        newCorpus.append(" ".join(tools.tokenize_text(text, lemmatize=True, stem=False ,extended_stopwords=True)))
    return newCorpus
        
orgFiles = prepare_texts(raw_orgFiles)
natureShort = prepare_texts(raw_natureShort)
natureShortFilt = prepare_texts(raw_natureShortFilt)
natureExt = prepare_texts(raw_natureExt)
extraFiles = prepare_texts(raw_extraFiles)
healthcare = prepare_texts(raw_healthcare)

optimize = 1

# TRAINING SECTION
print('######## TRAINING MODELS...')
top2vec = model_top2vec.Top2Vec_classifier(paths, verbose=True)

for jj in range(1):
  if optimize:
    optim_param = pd.read_excel(paths["ref"] + "Top2vec/" + "optimization_top2vec.xlsx")
    min_count = list(optim_param["min_count"])
    ngram = list(optim_param["ngram"])
    embedding_model = list(optim_param["embedding_model"])
    speed = list(optim_param["speed"])
    use_embedding_model_tokenizer  = list(optim_param["use_embedding_model_tokenizer "])
    ext_dataset = list(optim_param["ext_dataset"])
    parsed = list(optim_param["parsed"])
    
    tops_ascii = []; tops_raw = []; stats = []; perc_test = []; perc_train = []
    for ii in range(len(optim_param)):
      print('Optimizing case: {} of {}'.format(ii + 1, len(optim_param)))
      
      if ext_dataset[ii]:
        if parsed[ii]: trainData = [orgFiles + healthcare, sdgs_orgFiles + sdgs_healthcare]
        else: 
          trainData = [raw_orgFiles + raw_healthcare + raw_extraFiles, sdgs_orgFiles + sdgs_healthcare + sdgs_extra]
          # trainData = [raw_orgFiles + raw_healthcare, sdgs_orgFiles + sdgs_healthcare]
      else:
        if parsed[ii]: trainData = [orgFiles, sdgs_orgFiles]
        else: trainData = [raw_orgFiles, sdgs_orgFiles]
        
      # store the training files in csv
      df = pd.DataFrame()
      df["files"] = trainData[0]
      df.to_csv(paths["out"] + "Top2vec/training_texts.csv")

      top2vec.train(train_data=trainData, embedding_model=embedding_model[ii], method=speed[ii], 
                    ngram=ngram[ii], min_count=min_count[ii], workers=8, tokenizer=use_embedding_model_tokenizer[ii]) # "doc2vec", "all-MiniLM-L6-v2", universal-sentence-encoder
      top2vec.save()
      # top2vec.load(train_data=trainData)
      [sum_per_topic_raw, sum_per_topic_ascii] = top2vec.map_model_topics_to_sdgs(normalize=True,
                                    path_csv=(paths["out"] + "Top2vec/" + "topics.csv"),
                                    version=2
                                  )
      tools.save_obj(top2vec, paths["model"] + "top2vec.pickle")
      
      filter = True; normalize = False
      
      perc_global_train, perc_single_train, probs_per_sdg_train, maxSDG = top2vec.test_model(corpus=trainData[0], associated_SDGs=trainData[1],
                    filter_low=filter, score_threshold=0.2, only_positive=False,
                      path_to_excel=(paths["out"] + "Top2vec/" + "test_top2vec_training_files_it{}_{}.xlsx".format(jj, ii)), 
                      only_bad=False, expand_factor=1.0, normalize=normalize
                      )
      expandFactor = 1 / maxSDG
      print('Expand factor: {:.2f}'.format(expandFactor))
    
      if parsed[ii]: testTexts = natureShortFilt
      else: testTexts = raw_natureShortFilt
      perc_global, perc_single, probs_per_sdg_test, maxSDG = top2vec.test_model(corpus=testTexts, associated_SDGs=sdgs_natureFilt,
                      filter_low=filter, score_threshold=0.1, only_positive=False,
                        path_to_excel=(paths["out"] + "Top2vec/" + "test_top2vec_abstractsLow_it{}_{}.xlsx".format(jj, ii)), 
                        only_bad=False, expand_factor=expandFactor, version=1, normalize=normalize, normalize_threshold=0.2
                        )
      
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


      
    #   tops_ascii.append(sum_per_topic_ascii)
    #   tops_raw.append(list(sum_per_topic_raw))
    #   stats.append("{:.3f}, {:.3f}".format(np.mean(sum_per_topic_raw), np.std(sum_per_topic_raw)))
    #   perc_test.append("{:.3f}, {:.3f}".format(perc_global, perc_single))
    #   perc_train.append("{:.3f}, {:.3f}".format(perc_global_train, perc_single_train))
      
    # optim_param["ascii"] = tops_ascii
    # optim_param["stats"] = stats
    # optim_param["perc_test"] = perc_test
    # optim_param["perc_train"] = perc_train
    # optim_param.to_excel(paths["out"] + "Top2vec/" + "optimization_out.xlsx")
    
  else:
      top2vec.load(trainData)
    