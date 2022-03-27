import pandas as pd
import os
import numpy as np
import tools
import warnings
import matplotlib.pyplot as plt

def validate_model(model, vectorizer, topics_association, sdgs_mapped, validFilesDict, 
                   pathToCsv="", verbose=True):
    # Validates a pretrained model. Since some papers are included in more than 1 topic, then the top 3 output from the model should contain the real SDG associated to the paper. The value should be large enough from the others not relevant topics
    
    exclude_sdg = []
    if sdgs_mapped.count(0) > 0:
        # Then the papers associated to that SDG can not be verified
        index = -1
        for ii in range(0,sdgs_mapped.count(0)):
            index = sdgs_mapped.index(0, index + 1)
            exclude_sdg.append(index + 1)
        warning = 'Excluded SDGS: {}'.format(exclude_sdg)
        warnings.warn(warning)
    
    saveCSV = False
    if len(pathToCsv) > 4:
        saveCSV = True
    paperNames = []; paperPaths = []; paperRealSDGs = []; paperPredictSDGs = []; validPredict = []
    returnValidFiles = []
    for key, value in validFilesDict.items():
        fileName = key
        filePath = value[0]
        fileSDGs = value[1]
        if any(outSDG in fileSDGs for outSDG in exclude_sdg):
            # Then that file should not be checked
            continue
        nSDGsAssociated = len(fileSDGs)
        f = open(filePath, 'r',errors='replace')
        text = f.read()
        f.close()
        tokens = " ".join(tools.lemmatize_text(text))
        query_words_vect = vectorizer.transform([tokens])
        topicFeats = model.transform(query_words_vect)[0]
        sortArgs = topicFeats.argsort()
        predictSDGs = []
        scores = ""
        for ii in range(0,nSDGsAssociated):
            index = sortArgs[-(ii + 1)]
            score = topicFeats[index]
            scoreSDG = topics_association[index]
            predictSDGs.append(scoreSDG)
            # scores += "SC - {:0.2f} - SDG {}".format(score, scoreSDG)
            scores += "#SDG {} ".format(scoreSDG)
            
        if sorted(fileSDGs) == sorted(predictSDGs):
            valid = True
            returnValidFiles.append(tokens)
        else:
            valid = False
        if verbose:
            print("- Valid: {} || SDGS {}, Model: {}".format(valid, fileSDGs, scores))

        paperNames.append(fileName)
        paperPaths.append(filePath)
        paperRealSDGs.append(fileSDGs)
        paperPredictSDGs.append(predictSDGs)
        validPredict.append(valid)
    
    [percOk, percents, okPerSDG, countPerSDG] = compute_statistics(realSDGs=paperRealSDGs, 
                                                                   predictedSDGs=paperPredictSDGs, 
                                                                   validPredict=validPredict,
                                                                   excludedSDGs=exclude_sdg, 
                                                                   pathToWrite="out/stats.txt", 
                                                                   verbose=verbose)
    if saveCSV:
        # Then the mapping result is stored in a csv
        df = pd.DataFrame()
        df['paperName'] = paperNames
        df['paperPath'] = paperPaths
        df['realSDG'] = paperRealSDGs
        df['predictSDG'] = paperPredictSDGs
        df['valid'] = validPredict
        df.to_csv(pathToCsv)
        
    return [percOk, percents, okPerSDG, countPerSDG, exclude_sdg, returnValidFiles]
        
def compute_statistics(realSDGs, predictedSDGs, validPredict, excludedSDGs, 
                       pathToWrite="", 
                       verbose=True):
    if len(pathToWrite) > 3:
        file = open(pathToWrite,'w')
    else:
        file = open('stats.txt', 'w')
        
    nFilesTest = len(realSDGs)
    nFilesClassifiedOk = validPredict.count(True)
    percOk = nFilesClassifiedOk / float(nFilesTest) * 100
    sent = '#### {} OK from {}, {:.2f} % \r'.format(nFilesClassifiedOk, nFilesTest, percOk)
    file.write(sent)
    if verbose:
        print(sent)
    
    okPerSDG = np.zeros(17)
    countPerSDG = np.zeros(17)
    for ii in range(0,nFilesTest):
        for sdg in realSDGs[ii]:
            countPerSDG[sdg - 1] += 1
            okPerSDG[sdg - 1] += int(validPredict[ii])
    sdgs = []
    percents = []
    for ii in range(1,18):
        if not(ii in excludedSDGs):
            sdgs.append('I:{} F:{}'.format(ii, countPerSDG[ii - 1]))
            perc = okPerSDG[ii - 1] / float(countPerSDG[ii - 1]) * 100.0
            percents.append(perc)
            sent = '#### SDG #{}: {:.2f} % OK, from {} files \r'.format(ii, perc, countPerSDG[ii - 1])
            file.write(sent)
            if verbose:
                print(sent)
    file.close()
    
    if verbose:
        plt.bar(sdgs, percents)
        plt.xlabel('SDGS')
        plt.ylabel("Percentage Valid")
        plt.title('Categories Bar Plot')
        plt.show()

    return [percOk, percents, okPerSDG, countPerSDG]
        

def map_model_topics_to_sdgs(res_singleSDG, newTopics, pathToCsv="", verbose=True):
    # Maps each new topic of the general NMF model to an specific SDG obtained from training 17 models
    associated_sdg = []
    nTopics = len(newTopics.columns)
    for ii in range(0,nTopics):
        topicWords = list(newTopics.iloc[:, ii])
        [topic, topic_ind] = get_associated_sdg(res_singleSDG, topicWords, verbose=verbose)
        associated_sdg.append([topic, topic_ind])
    sdgs_coh = [sdg[0] for sdg in associated_sdg]
    topics_association = [sdg[1] for sdg in associated_sdg]
    sdgs_found = [topics_association.count(sdg) for sdg in range(1,18)]
    
    if verbose:
        print(topics_association)
        print(sdgs_found)
        
    if len(pathToCsv) > 4:
        # Then the mapping result is stored in a csv
        df = pd.DataFrame()
        
        col_names = []
        col_data = []
        for sdg in range(1,18):
            if sdg in topics_association:
                sdgCount = topics_association.count(sdg)
                index = -1
                for jj in range(0,sdgCount):
                    index = topics_association.index(sdg, index + 1)
                    colName = "Topic #{} - {}".format(sdg, jj)
                    
                    colWords = list(newTopics.iloc[:, index])
                    df[colName] = colWords
            else:
                colName = "Topic #{:2d} - xx".format(sdg)
                df[colName] = 0
        df.to_csv(pathToCsv)
                
    return [topics_association, sdgs_coh, sdgs_found]




def get_associated_sdg(sdgs_models, query_words, verbose=True):
    query_words = ' '.join(query_words)
    max_values = []
    for sdg in sdgs_models:
        model = sdg[0]
        vect = sdg[1]
        query_words_vect = vect.transform([query_words])
        nmf_features = model.transform(query_words_vect)
        max_values.append(nmf_features.max())
    
    max_coh_val = max(max_values)
    max_coh_ind = max_values.index(max_coh_val)  
    topic_ind = max_coh_ind + 1 
    
    if verbose:
        print("Max coherence: {:0.2f}, SDG # {:2d}".format(max_coh_val, topic_ind))
    
    return [max_coh_val, topic_ind]