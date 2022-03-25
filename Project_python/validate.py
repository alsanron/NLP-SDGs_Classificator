import pandas as pd
import os
import tools

def validate_model(model, vectorizer, topics_association, validFilesDict):
    # Validates a pretrained model. Since some papers are included in more than 1 topic, then the top 3 output from the model should contain the real SDG associated to the paper. The value should be large enough from the others not relevant topics
    for file in list(validFilesDict.values()):
        filePath = file[0]
        fileSDGs = file[1]
        nSDGsAssociated = len(fileSDGs)
        f = open(filePath, 'r',errors='replace')
        text = f.read()
        f.close()
        tokens = " ".join(tools.lemmatize_text(text))
        query_words_vect = vectorizer.transform([tokens])
        topicFeats = model.transform(query_words_vect)[0]
        sortArgs = topicFeats.argsort()
        scores = ""
        for ii in range(0,nSDGsAssociated):
            index = sortArgs[-(ii + 1)]
            score = topicFeats[index]
            scoreSDG = topics_association[index]
            # scores += "SC - {:0.2f} - SDG {}".format(score, scoreSDG)
            scores += "#SDG {} ".format(scoreSDG)
        print("SDGS {}, Model: {}".format(fileSDGs, scores))
        
        


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