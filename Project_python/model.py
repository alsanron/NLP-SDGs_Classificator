# functions used for testing different model configurations
from signal import valid_signals
import tools
import pandas as pd
import numpy as np
import conf
import data
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import tools
import warnings
warnings.filterwarnings('ignore')


class NMF_classifier:
    paths=[]
    individual_models=[]
    global_model=[]
    topics_association=[]
    verbose=False
    
    def __init__(self, paths, verbose=False):
        self.paths = paths
        self.verbose = verbose
        
    def get_individual_model_per_sdg(self):
        return self.individual_models
    
    def export_individual_model_topics_to_csv(self, path, n_top_words=20):
        df = pd.DataFrame()
        colNames = []
        sdgs_names = data.get_sdg_titles(self.paths["ref"])
        
        ii = 0
        for nmf, sdg in zip(self.individual_models, sdgs_names):
            model = nmf[0]; vectorizer = nmf[1]; ii += 1
            topics = self.get_topics_from_model(model=model, vectorizer=vectorizer, n_top_words=n_top_words)
            df = pd.concat([df, topics], ignore_index=True, axis=1)
            colNames.append("{} - {}".format(sdg, sdgs_names[sdg]))

        df.columns = colNames
        df.to_csv(path)
  
    def load_individual_model_per_sdg(self):
        # loads the models that have been trained previously
        self.individual_models = []
        n_sdgs = 17
        for ii in range(1, n_sdgs + 1):
            model = tools.load_obj(self.paths["model"] + "model_1topic_sdg{}.pickle".format(ii))
            vectorizer = tools.load_obj(self.paths["model"] + "vect_1topic_sdg{}.pickle".format(ii))
            self.individual_models.append([model, vectorizer])

    def train_individual_model_per_sdg(self, multigrams=(1,1)):
        #  17 models are trained for classifying each SDG
        # trains the passed number of models with the information of the onu or returns the already trained models
        # flag_train = True -> then the models are trained, False = models are loaded from memory
        n_sdgs = 17 # the number of texts
        nTopics = 1

        self.individual_models = []
        for ii in range(1, n_sdgs + 1):
            trainData = data.get_sdgs_org_files(refPath=self.paths["SDGs_inf"], sdg=ii)
            trainFiles = [file[0] for file in trainData]
            model, vectorizer = self.__train_nmf(trainFiles, n_topics=nTopics, ngram=multigrams)
            self.individual_models.append([model, vectorizer])
            tools.save_obj(model, self.paths["model"] + "model_1topic_sdg{}.pickle".format(ii))
            tools.save_obj(vectorizer, self.paths["model"] + "vect_1topic_sdg{}.pickle".format(ii))
    
    def load_global_model(self, n_topics):
        model = tools.load_obj(self.paths["model"] + "model_{}topics.pickle".format(n_topics))
        vectorizer = tools.load_obj(self.paths["model"] + "vect_{}topics.pickle".format(n_topics))
        self.global_model = [model, vectorizer]
            
    def train_global_model(self, train_files, n_topics, multigrams):
        if len(self.individual_models) == 0:
            errors.error('individual models not trained yet')
        self.global_model = self.__train_nmf(train_files, n_topics=n_topics, ngram=multigrams)
        tools.save_obj(self.global_model[0], self.paths["model"] + "model_{}topics.pickle".format(n_topics))
        tools.save_obj(self.global_model[1], self.paths["model"] + "vect_{}topics.pickle".format(n_topics))
        
    def test_model(self, database, path_excel, abstract=True, kw=False, intro=False, body=False, concl=False):
        predictedSDGs = []
        realSDGs = []
        scoresSDGs = []
        valids = []
        validsAny = []
        files = []
        abstracts = []
        countPerSDG = np.zeros(17)
        countWellPredictionsPerSDG = np.zeros(17)
        
        for file in database:
            text = ""
            sdgs = database[file]["SDG"]
            for sdg in sdgs:
                countPerSDG[sdg - 1] += 1 # increments the SDGs counter
            if abstract:
                text += database[file]["abstract"]
            if kw:
                text += database[file]["keywords"]
            if intro:
                text += database[file]["introduction"]
            if body:
                text += database[file]["body"]
            if concl:
                text += database[file]["conclusions"]
                
            predic, score = self.map_text_to_sdgs(text, top_score=len(sdgs))  
            valid = False
            if sorted(sdgs) == sorted(predic):
                valid = True
            validSingle = False
            for sdg in sdgs:
                if sdg in predic:
                    validSingle = True
                    countWellPredictionsPerSDG[sdg - 1] += 1
                    break

            predictedSDGs.append(predic)
            realSDGs.append(sdgs)
            scoresSDGs.append(score)
            valids.append(valid)
            validsAny.append(validSingle)
            files.append(file)
            abstracts.append(abstracts)
            
        df = pd.DataFrame()
        df["file"] = files
        # df["abstract"] = abstracts
        df["prediction"] = predictedSDGs
        df["real"] = realSDGs
        df["scores"] = scoresSDGs
        df["valid"] = valids
        df["valid_single"] = validsAny
        
        oks = [ok for ok in valids if ok == True]
        oksSingle = [ok for ok in validsAny if ok == True]
        configStr = "Abstract {} - Kw - {} Intro - {} Body - {} Concl - {}".format(int(abstract), int(kw), int(intro), int(body), int(concl))
        print("#### Config:" + configStr)
        print("- {:.2f} % valid global, {:.2f} % valid any, of {} files".format(len(oks) / len(valids) * 100, len(oksSingle) / len(valids) * 100, len(valids)))
        df.to_excel(path_excel)
        
        sdgs = []
        percents = []
        for ii in range(1, 18):
            # sdgs.append('SDG{}'.format(ii))
            sdgs.append('{}'.format(ii))
            perc = countWellPredictionsPerSDG[ii - 1] / float(countPerSDG[ii - 1]) * 100.0
            percents.append(perc)
        plt.figure()
        plt.bar(sdgs, percents)
        plt.xlabel('SDGS')
        plt.ylabel("Correctly individual identified [%]")
        plt.savefig('out/percentage_valid_' + configStr.replace('-','').replace(' ', '_').replace('__','_') + ".png")
        
        plt.figure()
        plt.bar(sdgs, countPerSDG)
        plt.xlabel('SDGS')
        plt.ylabel("Number papers associated to each SDG")
        plt.savefig("out/counter_files_per_sdg.png")
        
    def map_text_to_sdgs(self, text, top_score):
        tokens = " ".join(tools.tokenize_text(text, lemmatize=True))
        query_words_vect = self.global_model[1].transform([tokens])
        topicFeats = self.global_model[0].transform(query_words_vect)[0]
        sortArgs = topicFeats.argsort()
        predictSDGs = []
        scores = []
        for ii in range(0, top_score):
            index = sortArgs[-(ii + 1)]
            scoreSDG = self.topics_association[index]
            predictSDGs.append(scoreSDG)
            scores.append(topicFeats[index])
        return [predictSDGs, scores]
        
    def map_model_topics_to_sdgs(self, n_top_words, path_csv=""):
        # Maps each new topic of the general NMF model to an specific SDG obtained from training 17 models
        topics = self.get_topics_from_model(self.global_model[0], self.global_model[1], n_top_words=n_top_words)
        nTopics = self.global_model[0].n_components
        associated_sdg = []
        for ii in range(0,nTopics):
            topicWords = list(topics.iloc[:, ii])
            [topic, topic_ind] = self.get_associated_sdg(topicWords)
            associated_sdg.append([topic, topic_ind])
        sdgs_coh = [sdg[0] for sdg in associated_sdg]
        topics_association = [sdg[1] for sdg in associated_sdg]
        self.topics_association = topics_association
        sdgs_found = [topics_association.count(sdg) for sdg in range(1,18)]
    
        if self.verbose:
            print(topics_association)
            print(sdgs_found)
            
        if len(path_csv) > 4:
            # Then the mapping result is stored in a csv
            df = pd.DataFrame()
            sdgs_names = data.get_sdg_titles(self.paths["ref"])
            col_names = []
            col_data = []
            for sdg in range(1,18):
                sdgName = list(sdgs_names.keys())[sdg - 1]
                sdgTitle = sdgs_names[sdgName]
                if sdg in topics_association:
                    sdgCount = topics_association.count(sdg)
                    index = -1
                    for jj in range(0,sdgCount):
                        index = topics_association.index(sdg, index + 1)
                        colName = "{} : {} - {}".format(sdgName, jj, sdgTitle)
                        colWords = list(topics.iloc[:, index])
                        df[colName] = colWords
                else:
                    colName = "{}:xx - {}".format(sdgName, sdgTitle)
                    df[colName] = 0
            df.to_csv(path_csv)

    def get_associated_sdg(self, query_words):
        query_words = ' '.join(query_words)
        max_values = []
        for res_nmf in self.individual_models:
            model = res_nmf[0]; vectorizer = res_nmf[1]
            query_words_vect = vectorizer.transform([query_words])
            nmf_features = model.transform(query_words_vect)
            max_values.append(nmf_features.max())
        
        max_coh_val = max(max_values)
        max_coh_ind = max_values.index(max_coh_val)  
        topic_ind = max_coh_ind + 1 
        
        if self.verbose:
            print("Max coherence: {:0.2f}, SDG # {:2d}".format(max_coh_val, topic_ind))
        
        return [max_coh_val, topic_ind]
        
    def get_topics_from_model(self, model, vectorizer, n_top_words):
        # Returns the n_top_words for each of the n_topics with which a model has been trained
        feat_names = vectorizer.get_feature_names_out()
        
        word_dict = dict()
        for ii in range(model.n_components):    
            #for each topic, obtain the largest values, and add the words they map to into the dictionary.
            words_ids = model.components_[ii].argsort()[:-n_top_words - 1:-1]
            words = [feat_names[key] for key in words_ids]
            word_dict['Topic # {:02d}'.format(ii + 1)] = words
            
        return pd.DataFrame(word_dict)
      
            
    def __train_nmf(self, trainData, n_topics, ngram=(1,1), alpha_w=0.0):
    # Trains a NMF model
    # @param trainData corpus of texts (array). They must be passed as texts, they are tokenized internally
    # @param n_topics number of topics for the model
    # @param ngram (min, max) multigrams to search in the corpus
    # @return [model, vectorizer]
        tokens = []
        for text in trainData:
            tokens.append(" ".join(tools.tokenize_text(text)))
        
        vectorizer = TfidfVectorizer(min_df=2, # They have to appear in at least x documents
                                    stop_words='english', # Remove all stop words from English
                                    encoding='utf-8',
                                    ngram_range=ngram, # min-max
                                    #token_pattern=u'(?u)\b\w*[a-zA-Z]\w*\b' 
                                    )
        vectorized_data = vectorizer.fit_transform(tokens)
        model_nmf = NMF(n_components=n_topics, random_state=5, verbose=0, alpha_W=alpha_w)
        model_nmf.fit(vectorized_data)
        
        return [model_nmf, vectorizer]

    
        
    
    