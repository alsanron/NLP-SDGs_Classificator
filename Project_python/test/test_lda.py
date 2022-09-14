# Configures the project paths: they can be launched from any code
import sys, os
sys.path.append(os.path.realpath('.'))
import conf
conf.import_paths()

# CONFIGURATION FLAGS
flag_optimize = 1
bigrams  = True; min_count_bigram  = 5
trigrams = True; min_count_trigram = 5
min_words_count = 2 # minimum number of times a word must appear in the corpus. It should be small since the training set is small
max_words_frequency = 0.7 # max frequency of a word appearing in corpus
type_texts = "lem"
expandFactor = 1.3; scoreThreshold = 0.2

optim_excel = "optimization_lda.xlsx"

# Real imports required by the file for work properly
from logging import error
import data
import conf
import pandas as pd
import tools
import model_lda
from gensim.models import Phrases
from gensim.corpora.dictionary import Dictionary

# Loads all the datasets
print('# Loading datasets...')
paths = conf.get_paths()
ds_train = data.get_dataset(requires_update=False, filter=["org", "manual_extra"])
ds_valid_short = data.get_dataset(requires_update=False, filter=["nature_abstract"])
ds_valid_long = data.get_dataset(requires_update=False, filter=["nature_all"])
   
orgFiles = ds_train[type_texts]; sdgs_org = ds_train["sdgs"]
natureShort = ds_valid_short[type_texts]; sdgs_natureShort = ds_valid_short["sdgs"]
natureLong = ds_valid_long[type_texts]; sdgs_natureLong = ds_valid_long["sdgs"]
        
trainData = [orgFiles, sdgs_org]
trainData[0] = [text.split(' ') for text in trainData[0]] # requires the words separated
rmWords = ['ha'] # private rm words list based on the topic-words analysis
for word in rmWords:
    for ii in range(len(trainData[0])): 
        if word in trainData[0][ii]: trainData[0][ii].remove(word)

if bigrams:
    print('# Creating bigrams...')
    bigram = Phrases(trainData[0], min_count=min_count_bigram)
    for idx in range(len(trainData[0])):
        for token in bigram[trainData[0][idx]]:
            if token.count("_") == 1:
                trainData[0][idx].append(token)
    if trigrams:
        print('# Creating trigrams...')
        trigram = Phrases(trainData[0], min_count=min_count_trigram)
        for idx in range(len(trainData[0])):
            for token in trigram[trainData[0][idx]]:
                if token.count("_") == 2:
                    trainData[0][idx].append(token)

print('# Creating dictionary...')
dict = Dictionary(trainData[0])
dict.filter_extremes(no_below=1, no_above=0.7)
dict[0] # just to load the dict
id2word = dict.id2token
corpus = [dict.doc2bow(text) for text in trainData[0]]

tmp= [text.split(' ') for text in natureShort] # requires the words separated
corpusPerplexity = [dict.doc2bow(text) for text in tmp]
        
# Optimization/Calculation section
path_out = paths["out"] + "LDA/"

if flag_optimize:
    optimData = pd.read_excel(paths["ref"] + optim_excel)
    nCases = len(optimData)
    out_perc_global = []; out_perc_any = []; perp = []; repSdgs=[]
    
    for ii in range(nCases):
        print('# Case: {} of {}'.format(ii, nCases))
        num_topics = optimData["num_topics"][ii]
        passes = optimData["passes"][ii]
        iterations = optimData["iterations"][ii]
        score_threshold = optimData["score_threshold"][ii]
        only_positive = optimData["only_positive"][ii]
        
        print('# Training model: nTopic:{}, passes:{}, iterations:{}, score:{}'.format(num_topics, passes, iterations, score_threshold))
        lda = model_lda.LDA_classifier(corpus=corpus, # training data
                                       id2word=id2word, # mapping of ids to words
                                       # chunksize=chunksize, default is 2000
                                       iterations=iterations,
                                       num_topics=num_topics,
                                       passes=passes,
                                       minimum_probability=0.0001, # default is 0.01
                                       # update_every=update_every, default is 1 for online learning
                                       eval_every=None, # default is 10
                                       random_state=1
                                        )
        lda.set_conf(paths, dict)
        lda.print_summary(top_words=50)
        sumPerTopic, listAscii = lda.map_model_topics_to_sdgs(trainData, path_csv=(path_out + "topic_words{}.csv".format(ii)), 
                                                            normalize=True, verbose=True)

        # tools.save_obj(lda, paths["model"] + "lda{}.pickle".format(ii))
        
        print('# Testing model...')
        filter = True; normalize = False
        
        # [rawSDG, perc_valid_global, perc_valid_any, maxSDG, pred_sdgs] = lda.test_model(trainData[0], trainData[1], score_threshold=score_threshold, segmentize=-1, filter_low=filter, normalize=normalize,
        # path_to_excel=(path_out + "test_training{}.xlsx".format(ii)), expand_factor=1.0)
        # tools.plot_ok_vs_nok_SDGsidentified(trainData[1], pred_sdgs, path_out + "sdgs_train{}.png".format(ii))
            
        
        # [rawSDG, perc_valid_global, perc_valid_any, maxSDG, pred_sdgs] = lda.test_model(natureShort, sdgs_natureShort, score_threshold=score_threshold, segmentize=-1, filter_low=filter, normalize=normalize,
        # path_to_excel=(path_out + "test_natureS{}.xlsx".format(ii)), expand_factor=expandFactor)
        # tools.plot_ok_vs_nok_SDGsidentified(sdgs_natureShort, pred_sdgs, path_out + "sdgs_test{}.png".format(ii))
    
        perplexity = lda.log_perplexity(corpusPerplexity)
        print('# Model perplexity: {:.2f}'.format(perplexity))
    
        # out_perc_global.append(perc_valid_global); out_perc_any.append(perc_valid_any)
        perp.append(perplexity); repSdgs.append(listAscii)
        
    # optimData["perc_global"] = out_perc_global
    # optimData["perc_any"] = out_perc_any
    optimData["log_perplexity"] = perp
    optimData["representativitySDGs"] = repSdgs
    optimData.to_excel(path_out + "optimization.xlsx")
else: 
    modelPath = paths["model"] + "lda0.pickle"
    print('# Loading model from: ' + modelPath)
    lda = tools.load_obj(modelPath)
    lda.set_conf(paths, dict)
    # lda.print_summary(top_words=50)
    sumPerTopic, listAscii = lda.map_model_topics_to_sdgs(trainData, path_csv=(path_out + "topic_words.csv"), 
                                                        normalize=True, verbose=True)
        
    print('# Testing model...')
    filter = True; normalize = False  
    [rawSDG, perc_valid_global, perc_valid_any, maxSDG, pred_sdgs] = lda.test_model(trainData[0], trainData[1], score_threshold=scoreThreshold, segmentize=-1, filter_low=filter, normalize=normalize,
    path_to_excel=(path_out + "test_training.xlsx"), expand_factor=expandFactor)
    tools.plot_ok_vs_nok_SDGsidentified(trainData[1], pred_sdgs, path_out + "sdgs_train.png")
        
    [rawSDG, perc_valid_global, perc_valid_any, maxSDG, pred_sdgs] = lda.test_model(natureShort, sdgs_natureShort, score_threshold=scoreThreshold, segmentize=-1, filter_low=filter, normalize=normalize,
    path_to_excel=(path_out + "test_natureS.xlsx"), expand_factor=expandFactor)
    tools.plot_ok_vs_nok_SDGsidentified(sdgs_natureShort, pred_sdgs, path_out + "sdgs_test.png")
    
    pred_sdgs = pd.DataFrame(pred_sdgs)
    pred_sdgs.to_csv(paths["out"] + "ALL/Individual/pred_test_lda.csv")
    