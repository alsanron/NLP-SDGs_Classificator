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
    # for validData in validFilesDict:
        # fileName = "[NO NAME]"
        # filePath = "[NO Path]"
        fileName = key
        filePath = value[0]
        # text = validData[0]
        fileSDGs = value[1]
        
        
        if any(outSDG in fileSDGs for outSDG in exclude_sdg):
            # Then that file should not be checked
            continue
        nSDGsAssociated = len(fileSDGs)
        f = open(filePath, 'r',errors='replace')
        text = f.read()
        f.close()
        
        
        tokens = " ".join(tools.tokenize_text(text))
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
        

