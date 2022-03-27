
from preprocess import get_training_files, get_validation_files
import train
import validate
import pandas as pd


def nmf_brut_force(paths, single_sdgs_models, n_top_words, range_topics, range_multigrams, abstracts=False):
    df = pd.DataFrame()
    for nTopics in range_topics:
        print('---- Testing with {} topics'.format(nTopics))
        for ii in range_multigrams:
            multigrams = (1,ii)
            
            print('---- Testing with {} multigrams'.format(multigrams))

            # trainFiles = get_training_files(refPath=paths["training"], abstracts=abstracts)
            # nmf_res = train.train_nmf(trainFiles, n_topics=nTopics, ngram=multigrams)
            # topics = train.get_topics(model=nmf_res[0], vectorizer=nmf_res[1], n_top_words=n_top_words, n_topics=nTopics)
            # [topics_association, sdgs_coh, sdgs_found] = validate.map_model_topics_to_sdgs(single_sdgs_models, topics, 
            #                                                                             pathToCsv=paths["out"]+"association_map.csv", 
            #                                                                             verbose=False)
            # validFilesDict = get_validation_files(preprocess=False, refPath=paths["validation"])
            # [percOk, percents, okPerSDG, countPerSDG, exclude_sdg] = validate.validate_model(model=nmf_res[0], 
            #                                                                     vectorizer=nmf_res[1],                         topics_association=topics_association,                        sdgs_mapped=sdgs_found,                         validFilesDict=validFilesDict,
            #                                                                     verbose=False,
            #                                                                     pathToCsv=paths["out"]+"results.csv")
            [percOk, percents, okPerSDG, countPerSDG, exclude_sdg, returnValidFiles] = train_validate_model(paths, single_sdgs_models, n_top_words, nTopics, multigrams, abstracts=abstracts)
            
            df_new = pd.DataFrame()
            df_new['nTopics'] = [nTopics]
            df_new['nTopWords'] = [n_top_words]
            df_new['multigram'] = [multigrams]
            df_new['excludedSDGs'] = [exclude_sdg]
            df_new['OverallScore'] = [percOk]
            df = df.append(df_new, ignore_index=True)
    df.to_csv(paths['out'] + "force_brut_optimization.csv") 
    
def train_validate_model(paths, single_sdgs_models, n_top_words, n_topics, n_multigrams, alpha_w=0.0, abstracts=False, new_training=[]):
    trainFiles = get_training_files(refPath=paths["training"], abstracts=abstracts)
    if len(new_training) > 1:
        for file in new_training:
            trainFiles.append(file)
    nmf_res = train.train_nmf(trainFiles, n_topics=n_topics, ngram=n_multigrams, alpha_w=alpha_w)
    topics = train.get_topics(model=nmf_res[0], vectorizer=nmf_res[1], n_top_words=n_top_words, n_topics=n_topics)
    [topics_association, sdgs_coh, sdgs_found] = validate.map_model_topics_to_sdgs(single_sdgs_models, topics, 
                                                                                pathToCsv=paths["out"]+"association_map.csv", 
                                                                                verbose=False)
    validFilesDict = get_validation_files(preprocess=False, refPath=paths["validation"])
    [percOk, percents, okPerSDG, countPerSDG, exclude_sdg, returnValidFiles] = validate.validate_model(model=nmf_res[0], 
                                                                        vectorizer=nmf_res[1],                         topics_association=topics_association,                        sdgs_mapped=sdgs_found,                         validFilesDict=validFilesDict,
                                                                        verbose=False,
                                                                        pathToCsv=paths["out"]+"results.csv")
    return [percOk, percents, okPerSDG, countPerSDG, exclude_sdg, returnValidFiles]