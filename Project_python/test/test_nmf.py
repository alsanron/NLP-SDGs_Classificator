# Configures the project paths: they can be launched from any code
import sys, os
sys.path.append(os.path.realpath('.'))
import conf
conf.import_paths()

# CONFIGURATION FLAGS
flag_optimize = 0

# Real imports required by the file for work properly
import model_nmf
from logging import error
import data
import conf
import pandas as pd
import numpy as np
import tools

# Loads all the datasets
paths = conf.get_paths()
ds_train = data.get_dataset(requires_update=False, filter=["org", "manual_extra"])
ds_valid_short = data.get_dataset(requires_update=False, filter=["nature_abstract"])
ds_valid_long = data.get_dataset(requires_update=False, filter=["nature_all"])

path_out = "out/NMF/"

if flag_optimize:
    optimData = pd.read_excel(paths["ref"] + "optimization_nmf.xlsx")
    nTopics = optimData["num_topics"]
    nIterations = optimData["iterations"]
    nature = optimData["nature"]
    stemming = optimData["stemming"]
    score_threshold = optimData["score_threshold"]
    l1 = optimData["l1"]
    alpha_w = optimData["alpha_w"]
    alpha_h = optimData["alpha_h"]
    
    res_any = []; res_all = []
    
    for ii in range(len(nTopics)):
        print("# Optimizing case: {}, nTopic: {}, nIterations: {}, nature: {}, Stemming: {}, L1: {:.2f}, Alphaw: :{:.2f}, AlphaH: :{:.2f}".format(ii, nTopics[ii], nIterations[ii], nature[ii], stemming[ii], l1[ii], alpha_w[ii], alpha_h[ii]))
    
        if stemming[ii]: type_texts = "lem_stem"
        else: type_texts = "lem"
            
        # all text should have been processed in the same way
        orgFiles = ds_train[type_texts]; sdgs_org = ds_train["sdgs"]
        natureShort = ds_valid_short[type_texts]; sdgs_natureShort = ds_valid_short["sdgs"]
        natureLong = ds_valid_long[type_texts]; sdgs_natureLong = ds_valid_long["sdgs"]

        if nature[ii]: trainData = [orgFiles + natureShort, sdgs_org + sdgs_natureShort]
        else: trainData = [orgFiles, sdgs_org]
        
        try: 
            print('# Training model...')
            nmf = model_nmf.NMF_classifier(paths, verbose=True)

            nmf.train(train_data=trainData, n_topics=nTopics[ii], ngram=(1,3), min_df=2, max_iter=nIterations[ii],
                    l1=l1[ii], alpha_w=alpha_w[ii], alpha_h=alpha_h[ii])
            nmf.map_model_topics_to_sdgs(n_top_words=50, normalize=True, path_csv=path_out + "topics_map{}.csv".format(ii))

            tools.save_obj(nmf, paths["model"] + "nmf{}.pickle".format(ii))

            print('# Testing model...')
            filter = True; normalize = False

            [rawSDG, perc_valid_global, perc_valid_any, maxSDG, pred_sdgs] = nmf.test_model(corpus=trainData[0], associated_SDGs=trainData[1], score_threshold=score_threshold[ii], segmentize=-1, filter_low=filter, normalize=normalize,
                        path_to_excel=(path_out + "test_nmf_training{}.xlsx".format(ii)))
            tools.plot_ok_vs_nok_SDGsidentified(trainData[1], pred_sdgs, path_out + "sdgs_train{}.png".format(ii))
            
            expandFactor = 4
            [rawSDG, perc_valid_global, perc_valid_any, maxSDG, pred_sdgs] = nmf.test_model(corpus=natureShort, associated_SDGs=sdgs_natureShort, score_threshold=score_threshold[ii],
                        segmentize=-1, path_to_excel=(path_out + "test_nmf_natureS{}.xlsx".format(ii)),
                        normalize=normalize, filter_low=filter, expand_factor=expandFactor)
            tools.plot_ok_vs_nok_SDGsidentified(sdgs_natureShort, pred_sdgs, path_out + "sdgs_test{}.png".format(ii))
        
        except:
            print('# Aborting execution of iteration{}'.format(ii))
            perc_valid_any = -1; perc_valid_global = -1
        
        res_any.append(perc_valid_any); res_all.append(perc_valid_global)
        
    outData = optimData
    outData["any"] = res_any
    outData["all"] = res_all
    outData.to_excel(path_out + "optimization{}.xlsx".format(ii))
else:
    print('# Using default-user configuration...')
    
    type_texts = "lem"
    nTopics = 20; maxIter = 1000; l1 = 0.0; alpha_w = 0.0; alpha_h = 0.0
    score = 0.1
    
    orgFiles = ds_train[type_texts]; sdgs_org = ds_train["sdgs"]
    natureShort = ds_valid_short[type_texts]; sdgs_natureShort = ds_valid_short["sdgs"]
    natureLong = ds_valid_long[type_texts]; sdgs_natureLong = ds_valid_long["sdgs"]

    trainData = [orgFiles, sdgs_org]
        
    print('# Training model...')
    nmf = model_nmf.NMF_classifier(paths, verbose=True)

    nmf.train(train_data=trainData, n_topics=nTopics, ngram=(1,3), min_df=2, max_iter=maxIter,
            l1=l1, alpha_w=alpha_w, alpha_h=alpha_h)
    nmf.print_stopwords(path_out + "stopwords.csv")
    nmf.map_model_topics_to_sdgs(n_top_words=50, normalize=True, path_csv=path_out + "topics_map.csv")

    tools.save_obj(nmf, paths["model"] + "nmf.pickle")

    print('# Testing model...')
    filter = True; normalize = False; expandFactor = 4.0
    
    [rawSDG, perc_valid_global, perc_valid_any, maxSDG, pred_sdgs] = nmf.test_model(corpus=trainData[0], associated_SDGs=trainData[1], score_threshold=score, segmentize=-1, filter_low=filter, normalize=normalize,
                path_to_excel=(path_out + "test_nmf_training.xlsx"), expand_factor=expandFactor)
    tools.plot_ok_vs_nok_SDGsidentified(trainData[1], pred_sdgs, path_out + "sdgs_train.png", fontsize=18, color='green')
    
    [rawSDG, perc_valid_global, perc_valid_any, maxSDG, pred_sdgs] = nmf.test_model(corpus=natureShort, associated_SDGs=sdgs_natureShort, score_threshold=score,
                segmentize=-1, path_to_excel=(path_out + "test_nmf_natureS.xlsx"),
                normalize=normalize, filter_low=filter, expand_factor=expandFactor)
    tools.plot_ok_vs_nok_SDGsidentified(sdgs_natureShort, pred_sdgs, path_out + "sdgs_test.png",  fontsize=18, color='red')
    
    pred_sdgs = pd.DataFrame(pred_sdgs)
    pred_sdgs.to_csv(paths["out"] + "ALL/Individual/pred_test_nmf.csv")


