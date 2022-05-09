# functions used for testing different model configurations
from signal import valid_signals
import tools
import pandas as pd
import numpy as np
import conf
import data
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import tools
import warnings
import tomotopy as tp
from top2vec import Top2Vec
import hashlib
from bertopic import BERTopic
warnings.filterwarnings('ignore')

           
class LDA_classifier:
    paths=[]
    individual_models=[]
    global_model=[]
    perplexity=0
    coherence=0
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
        for model, sdg in zip(self.individual_models, sdgs_names):
            ii += 1
            topics = self.get_topics_from_model(model=model, n_top_words=n_top_words)
            coefs = [elem[0] for elem in topics[0]]
            words = [elem[1] for elem in topics[0]]
            df = pd.concat([df, pd.Series(coefs), pd.Series(words)], ignore_index=True, axis=1)
            colNames.append("Coefs: {} - {}".format(sdg, sdgs_names[sdg]))
            colNames.append("Words")

        df.columns = colNames
        df.to_csv(path)
        
    def export_global_model_topics_to_csv(self, path, n_top_words=20):
        df = pd.DataFrame()
        colNames = []
        topicsRaw = self.get_topics_from_model(model=self.global_model, n_top_words=n_top_words)
        ii = 0
        for topic in topicsRaw:
            ii += 1
            coefs = [elem[0] for elem in topic]
            words = [elem[1] for elem in topic]
            df = pd.concat([df, pd.Series(coefs), pd.Series(words)], ignore_index=True, axis=1)
            # df = pd.concat([df, pd.Series(words)], ignore_index=True, axis=1)
            colNames.append("Topic{}-C".format(ii))
            colNames.append("W")

        df.columns = colNames
        df.to_csv(path)
  
    def load_individual_model_per_sdg(self):
        # loads the models that have been trained previously
        self.individual_models = []
        n_sdgs = 17
        for ii in range(1, n_sdgs + 1):
            model = tools.load_obj(self.paths["model"] + "lda_model_1topic_sdg{}.pickle".format(ii))
            self.individual_models.append(model)

    def train_individual_model_per_sdg(self):
        #  17 models are trained for classifying each SDG
        # trains the passed number of models with the information of the onu or returns the already trained models
        # flag_train = True -> then the models are trained, False = models are loaded from memory
        n_sdgs = 17 # the number of texts
        nTopics = 1

        self.individual_models = []
        for ii in range(1, n_sdgs + 1):
            trainData = data.get_sdgs_org_files(refPath=self.paths["SDGs_inf"], sdg=ii)
            trainFiles = [file[0] for file in trainData]
            model = self.__train_lda(trainFiles, n_topics=nTopics, verbose=False)
            self.individual_models.append(model)
            tools.save_obj(model, self.paths["model"] + "lda_model_1topic_sdg{}.pickle".format(ii))
    
    def load_global_model(self, n_topics):
        model = tools.load_obj(self.paths["model"] + "model_{}topics.pickle".format(n_topics))
        vectorizer = tools.load_obj(self.paths["model"] + "vect_{}topics.pickle".format(n_topics))
        self.global_model = [model, vectorizer]
            
    def train_global_model(self, train_files, n_topics):
        if len(self.individual_models) == 0:
            # errors.error('individual models not trained yet')
            pass
        self.global_model = self.__train_lda(train_files, n_topics=n_topics, verbose=True)
        tools.save_obj(self.global_model, self.paths["model"] + "model_lda_{}topics.pickle".format(n_topics))
        
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
        topics = self.get_topics_from_model(self.global_model, n_top_words=30)
        nTopics = self.global_model.num_topics
        associated_sdg = []
        for ii in range(0,nTopics):
            topicWords = [elem[1] for elem in topics[ii]]
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
        max_values = []
        for model in self.individual_models:
            topics = model.get_document_topics(model.id2word.doc2bow(query_words.split(' ')))
            query_words_vect = vectorizer.transform([query_words])
            nmf_features = model.transform(query_words_vect)
            max_values.append(nmf_features.max())
        
        max_coh_val = max(max_values)
        max_coh_ind = max_values.index(max_coh_val)  
        topic_ind = max_coh_ind + 1 
        
        if self.verbose:
            print("Max coherence: {:0.2f}, SDG # {:2d}".format(max_coh_val, topic_ind))
        
        return [max_coh_val, topic_ind]
        
    def get_topics_from_model(self, model, n_top_words):
        # Returns the n_top_words for each of the n_topics with which a model has been trained
        word_dict = dict()
        topicsRaw = model.show_topics(num_topics=model.num_topics, num_words=n_top_words)
        topicsParsed = []
        for topic in topicsRaw:
            topicStr = topic[1]
            words = []
            for comb in topicStr.split(' + '):
                coef, word = comb.split('*')
                coef = float(coef)
                word = word.replace('"','')
                words.append([coef, word])
            topicsParsed.append(words)
        return topicsParsed
           
    def __train_lda(self, trainData, n_topics, verbose=False):
        # Trains a LDA model
        # @param trainData corpus of texts (array). They must be passed as texts, they are tokenized internally
        # @param n_topics number of topics for the model
        # @return model
        texts = []
        for text in trainData:
            texts.append(tools.tokenize_text(text))
        id2word = corpora.Dictionary(texts)

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=n_topics, 
                                           random_state=5,
                                           update_every=1,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        if verbose:
            print(' - Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
        return lda_model
    
    
class PAM_classifier(tp.PAModel):
    paths=[]
    individual_models=[]
    topics_association=[]
    verbose=False
    
    def set_conf(self, paths, verbose=False):
        self.paths = paths
        self.verbose = verbose
 
    def load_global_model(self, n_topics):
        pass
        # model = tools.load_obj(self.paths["model"] + "model_{}topics.pickle".format(n_topics))
        # vectorizer = tools.load_obj(self.paths["model"] + "vect_{}topics.pickle".format(n_topics))
        # self.global_model = [model, vectorizer]
        
    # def test_model(self, database, path_excel, abstract=True, kw=False, intro=False, body=False, concl=False):
    #     predictedSDGs = []
    #     realSDGs = []
    #     texts = []
    #     scoresSDGs = []
    #     valids = []
    #     validsAny = []
    #     files = []
    #     abstracts = []
    #     countPerSDG = np.zeros(17)
    #     countWellPredictionsPerSDG = np.zeros(17)
        
    #     for file in database:
    #         text = ""
    #         sdgs = database[file]["SDG"]
    #         for sdg in sdgs:
    #             countPerSDG[sdg - 1] += 1 # increments the SDGs counter
    #         if abstract:
    #             text += database[file]["abstract"]
    #         if kw:
    #             text += database[file]["keywords"]
    #         if intro:
    #             text += database[file]["introduction"]
    #         if body:
    #             text += database[file]["body"]
    #         if concl:
    #             text += database[file]["conclusions"]
                
    #         predic, score = self.map_text_to_sdgs(text, top_score=len(sdgs))  
    #         valid = False
    #         if sorted(sdgs) == sorted(predic):
    #             valid = True
    #         validSingle = False
    #         for sdg in sdgs:
    #             if sdg in predic:
    #                 validSingle = True
    #                 countWellPredictionsPerSDG[sdg - 1] += 1
    #                 break

    #         predictedSDGs.append(predic)
    #         texts.append(text)
    #         realSDGs.append(sdgs)
    #         scoresSDGs.append(score)
    #         valids.append(valid)
    #         validsAny.append(validSingle)
    #         files.append(file)
    #         abstracts.append(abstracts)
            
    #     df = pd.DataFrame()
    #     df["file"] = files
    #     # df["abstract"] = abstracts
    #     df["texts"] = texts
    #     df["prediction"] = predictedSDGs
    #     df["real"] = realSDGs
    #     df["scores"] = scoresSDGs
    #     df["valid"] = valids
    #     df["valid_single"] = validsAny
        
    #     oks = [ok for ok in valids if ok == True]
    #     oksSingle = [ok for ok in validsAny if ok == True]
    #     configStr = "Abstract {} - Kw - {} Intro - {} Body - {} Concl - {}".format(int(abstract), int(kw), int(intro), int(body), int(concl))
    #     print("#### Config:" + configStr)
    #     print("- {:.2f} % valid global, {:.2f} % valid any, of {} files".format(len(oks) / len(valids) * 100, len(oksSingle) / len(valids) * 100, len(valids)))
    #     df.to_excel(path_excel)
        
    #     sdgs = []
    #     percents = []
    #     for ii in range(1, 18):
    #         # sdgs.append('SDG{}'.format(ii))
    #         sdgs.append('{}'.format(ii))
    #         perc = countWellPredictionsPerSDG[ii - 1] / float(countPerSDG[ii - 1]) * 100.0
    #         percents.append(perc)
    #     plt.figure()
    #     plt.bar(sdgs, percents)
    #     plt.xlabel('SDGS')
    #     plt.ylabel("Correctly individual identified [%]")
    #     plt.savefig('out/percentage_valid_' + configStr.replace('-','').replace(' ', '_').replace('__','_') + ".png")
        
    #     plt.figure()
    #     plt.bar(sdgs, countPerSDG)
    #     plt.xlabel('SDGS')
    #     plt.ylabel("Number papers associated to each SDG")
    #     plt.savefig("out/counter_files_per_sdg.png")
        
        
    def get_topics_from_model(self, model, n_top_words):
        # Returns the n_top_words for each of the n_topics with which a model has been trained
        word_dict = dict()
        topicsRaw = model.show_topics(num_topics=model.num_topics, num_words=n_top_words)
        topicsParsed = []
        for topic in topicsRaw:
            topicStr = topic[1]
            words = []
            for comb in topicStr.split(' + '):
                coef, word = comb.split('*')
                coef = float(coef)
                word = word.replace('"','')
                words.append([coef, word])
            topicsParsed.append(words)
        return topicsParsed
           
    def train_model(self, train_data):
        # Trains a PAM model
        # @param trainData corpus of texts (array). They must be passed as texts, they are tokenized internally
        # @param n_topics number of topics for the model
        # @return model
        corpus = train_data[0]; associated_sdgs = train_data[1]
        for text in corpus:
            self.add_doc(text)
            
        for i in range(0, 100, 10):
            self.train(10)
            print('Iteration: {}\tLog-likelihood: {}'.format(i, self.ll_per_word))
        
        # self.global_model.train(iter=100)
        
        # for k in range(mdl.k):
        #     print('Top 10 words of topic #{}'.format(k))
        #     print(mdl.get_topic_words(k, top_n=10))
        # doc = self.global_model.make_doc(trainData[0])
        
        # self.global_model.summary()
        
    def print_summary(self):
        pass
        
        
class Top2Vec_classifier:
    paths=[]
    global_model=[]
    topics_association=[]
    verbose=False
    
    def __init__(self, paths, verbose=False):
        self.paths = paths
        self.verbose = verbose
 
    def load_global_model(self):
        self.global_model = Top2Vec.load(self.paths["model"] + "model_top2vec")
            
    def train_global_model(self, train_data, embedding_model="doc2vec", ngram=True, method="learn", workers=8, min_count=2, embedding_batch_size=17, tokenizer=False, split=False, nSplit=25):
        # trains the model based on the training files
        # @param train_files corpus of documents as a list of strings
        # @param method "fast-learn", "learn" or "deep-learn"
        # @param workes number of parallel workers
        # @param min_count minimum number of documents where a word must be to be valid
        corpus = train_data[0]; associated_sdgs = train_data[1]
        
        # class Preprocess:
        #     def __call__(self, text):
        #         tokens = tools.tokenize_text(text, min_word_length=3, lemmatize=False, stem=False, extended_stopwords=True)
        #         return tokens     

        self.global_model = Top2Vec(documents=corpus, embedding_model=embedding_model, min_count=min_count, ngram_vocab=ngram, speed=method, workers=workers, embedding_batch_size=embedding_batch_size, document_chunker="sequential", split_documents=split, chunk_length=nSplit, use_embedding_model_tokenizer=tokenizer)
        
        self.print_model_summary()
        self.global_model.save(self.paths["model"] + "model_top2vec")
        # self.map_model_topics_to_sdgs(associated_sdgs=associated_sdgs)
        
    def reduce_topics(self, num_topics):
        nTopics = self.global_model.get_num_topics()
        if nTopics <= num_topics:
            raise ValueError('Ntopics: {}, new nTopics: {}'.format(nTopics, num_topics))
        newTopics = self.global_model.hierarchical_topic_reduction(num_topics)
        
        
    def test_model(self, corpus, associated_SDGs,  stat_topics=-1, path_to_plot="", path_to_excel="", only_bad=False, score_threshold=3.0, only_positive=False):
        rawSDG = []; 
        predictedSDGs = []
        realSDGs = []
        scoresSDGs = []
        valids = []
        validsAny = []
        texts = []
        statsGlobal = []
        countPerSDG = np.zeros(17)
        countWellPredictionsPerSDG = np.zeros(17)
        
        numTopics = self.global_model.get_num_topics()
        if stat_topics < 0 or stat_topics > numTopics:
            print(' - n_stat topics too large... now -> ', numTopics)
            stat_topics = numTopics
        
        for text, sdgs in zip(corpus, associated_SDGs):
            raw_sdgs, predic, score, raw_topicsScores = self.map_text_to_sdgs(text, n_query=stat_topics, score_threshold=score_threshold, only_positive=only_positive)  
            
            validSingle = False; ii = 0
            for sdg in sdgs:
                countPerSDG[sdg - 1] += 1
                if sdg in predic:
                    validSingle = True
                    ii += 1
                    countWellPredictionsPerSDG[sdg - 1] += 1
            valid = False
            if ii == len(sdgs):
                valid = True
                
            if (only_bad and not(valid)) or not(only_bad):
                raw_sdgsAscii = ["x{}: {:.2f}".format(xx, topic) for topic, xx in zip(raw_sdgs, range(1,18))]
                raw_sdgsAscii = "|".join(raw_sdgsAscii)
                rawSDG.append(raw_sdgsAscii)
                
                stats = [min(raw_sdgs), np.mean(raw_sdgs), max(raw_sdgs)]
                statsAscii = "[{:.2f}, {:.2f}, {:.2f}]".format(stats[0], stats[1], stats[2])
                statsGlobal.append(statsAscii)
                predictedSDGs.append(predic)
                realSDGs.append(sdgs)
                scoresSDGs.append(score)
                texts.append(text)
            valids.append(valid)
            validsAny.append(validSingle)
            
        oks = [ok for ok in valids if ok == True]
        oksSingle = [ok for ok in validsAny if ok == True]
        print("- {:.2f} % valid global, {:.2f} % valid any, of {} files".format(len(oks) / len(valids) * 100, len(oksSingle) / len(valids) * 100, len(valids)))
        
        if len(path_to_excel) > 0:
            df = pd.DataFrame()
            df["text"] = texts
            df["real"] = realSDGs
            df["topics_association"] = rawSDG
            df["stats"] = statsGlobal
            df["prediction"] = predictedSDGs
            df["scores"] = scoresSDGs
            df["all_valid"] = valids
            df["any_valid"] = validsAny
            df.to_excel(path_to_excel)
        
    def map_model_topics_to_sdgs(self, associated_sdgs, num_docs=15, path_csv="", normalize=False):
        # maps each internal topic with the SDGs. A complete text associated to each specific SDG is fetched. Then each topic is compared with each text and the text-associated sdg with the maximum score is selected as the SDG.
        nTopics = self.global_model.get_num_topics()
        topic_sizes, topics_num = self.global_model.get_topic_sizes()
        self.topics_association = np.zeros((nTopics, 17))
        for ii in range(nTopics):   
            if num_docs < 0:
                numDocs = topic_sizes[ii]
            else:
                numDocs = num_docs
            documents, document_scores, document_ids = self.global_model.search_documents_by_topic(topic_num=ii, num_docs=numDocs)
            if normalize: document_scores = document_scores / sum(document_scores)
            sdgs = np.zeros(17)
            for id, score in zip(document_ids, document_scores):
                realSDG = associated_sdgs[id]
                for sdg in realSDG:
                    sdgs[sdg - 1] += score * 1
            self.topics_association[ii] = sdgs
            # if normalize:
            #     self.topics_association[ii] = sdgs / sum(sdgs)
            listAscii = ["x{}: {:.2f}".format(xx, topic) for topic, xx in zip(self.topics_association[ii], range(1,18))]
            print('Topic{:2d}: '.format(ii), ' | '.join(listAscii))
            
        if len(path_csv) > 4:
            # Then the mapping result is stored in a csv
            df = pd.DataFrame()
            col_names = []
            col_data = []
            sdgTitles = data.get_sdg_titles(self.paths["ref"])
            topic_words, word_scores, topic_nums = self.global_model.get_topics()
            # for sdg in sdgTitles:
            #     sdgTitle = sdgTitles[sdg]
            #     colName = "{} - {}".format(sdg, sdgTitle)
            #     colWords = []
            #     sdgInt = list(sdgTitles.keys()).index(sdg) + 1
            #     for ii, index in zip(self.topics_association, range(nTopics)):
            #         if ii == sdgInt:
            #             words = list(topic_words[index])
            #             colWords.append(words[0:30])
            #     df[colName] = colWords[0]
            for words, index in zip(topic_words, topic_nums):
                df["topic{}".format(index)] = list(words)
            df.to_csv(path_csv)
            
    def map_text_to_sdgs(self, text, n_query, score_threshold, only_positive=False):
        topics_words, word_scores, topic_scores, topic_nums = self.global_model.query_topics(text, num_topics=n_query)
        predictSDGs = np.zeros(17)  
        for topicIndex, topicScore in zip(topic_nums, topic_scores):
            if only_positive and topicScore < 0: break
            if topicScore < 0.2: break
            predictSDGs += topicScore * self.topics_association[topicIndex]
        top = sorted(predictSDGs, reverse=True)
        sdgs = []; scores = []
        for ii in range(len(top)):
            if top[ii] >= score_threshold:
                sdgs.append(list(predictSDGs).index(top[ii]) + 1)
                scores.append(top[ii])

        return [predictSDGs, sdgs, scores, topic_scores]
               
    def print_model_summary(self):
        # print('####### Model summary:')
        print(' - Number of topics: ', self.global_model.get_num_topics())
        # topic_sizes, topic_nums = self.global_model.get_topic_sizes()
        # for topic, topicIndex in zip(topic_sizes, topic_nums):
        #     print(' - {} documents found related to topic {}'.format(topic, topicIndex + 1))
        # topic_words, word_scores, topic_nums = self.global_model.get_topics()
        
    def get_topics_from_model(self, model, n_top_words):
        # Returns the n_top_words for each of the n_topics with which a model has been trained
        word_dict = dict()
        topicsRaw = model.show_topics(num_topics=model.num_topics, num_words=n_top_words)
        topicsParsed = []
        for topic in topicsRaw:
            topicStr = topic[1]
            words = []
            for comb in topicStr.split(' + '):
                coef, word = comb.split('*')
                coef = float(coef)
                word = word.replace('"','')
                words.append([coef, word])
            topicsParsed.append(words)
        return topicsParsed
        


class BERTopic_classifier:
    paths=[]
    global_model=[]
    topics_association=[]
    verbose=False
    trainData=[]
    
    def __init__(self, paths, verbose=False):
        self.paths = paths
        self.verbose = verbose
 
    def load_global_model(self):
        self.global_model = BERTopic.load(self.paths["model"] + "BERTopic")
        
    def save_global_model(self):
        self.global_model.save(self.paths["model"] + "BERTopic")	

            
    def train_global_model(self, train_data, embedding_model="all-MiniLM-L6-v2", n_gram_range=(1,3), top_n_words=20, workers=8, calculate_probabilities=True, seed_topic_list=[]):
        # trains the model based on the training files
        # @param train_files corpus of documents as a list of strings
        # @param method "fast-learn", "learn" or "deep-learn"
        # @param workes number of parallel workers
        # @param min_count minimum number of documents where a word must be to be valid
        self.trainData = train_data
        corpus = train_data[0]; associated_sdgs = train_data[1]
        
        self.global_model = BERTopic(language="english", 
                        embedding_model=embedding_model, 
                        top_n_words=top_n_words, 
                        n_gram_range=n_gram_range, 
                        calculate_probabilities=calculate_probabilities, 
                        seed_topic_list=seed_topic_list, 
                        verbose=True)
        # TODO: 1 - Adjust the ngram range

        # TODO: try to generate a methodology for assigning labels to the training texts in order t oclassify them?
        # embeddings = sentence_model.encode(trainData[0], show_progress_bar=True)
        topics, probs = self.global_model.fit_transform(corpus)
        
        self.print_model_summary()
        self.save_global_model()
        
         
    def test_model(self, corpus, associated_SDGs,  stat_topics=-1, path_to_plot="", path_to_excel="", only_bad=False, score_threshold=3.0, only_positive=False):
        rawSDG = []; 
        predictedSDGs = []
        realSDGs = []
        scoresSDGs = []
        valids = []
        validsAny = []
        texts = []
        statsGlobal = []
        countPerSDG = np.zeros(17)
        countWellPredictionsPerSDG = np.zeros(17)
        
        numTopics = self.global_model.get_num_topics()
        if stat_topics < 0 or stat_topics > numTopics:
            print(' - n_stat topics too large... now -> ', numTopics)
            stat_topics = numTopics
        
        for text, sdgs in zip(corpus, associated_SDGs):
            raw_sdgs, predic, score, raw_topicsScores = self.map_text_to_sdgs(text, n_query=stat_topics, score_threshold=score_threshold, only_positive=only_positive)  
            
            validSingle = False; ii = 0
            for sdg in sdgs:
                countPerSDG[sdg - 1] += 1
                if sdg in predic:
                    validSingle = True
                    ii += 1
                    countWellPredictionsPerSDG[sdg - 1] += 1
            valid = False
            if ii == len(sdgs):
                valid = True
                
            if (only_bad and not(valid)) or not(only_bad):
                raw_sdgsAscii = ["x{}: {:.2f}".format(xx, topic) for topic, xx in zip(raw_sdgs, range(1,18))]
                raw_sdgsAscii = "|".join(raw_sdgsAscii)
                rawSDG.append(raw_sdgsAscii)
                
                stats = [min(raw_sdgs), np.mean(raw_sdgs), max(raw_sdgs)]
                statsAscii = "[{:.2f}, {:.2f}, {:.2f}]".format(stats[0], stats[1], stats[2])
                statsGlobal.append(statsAscii)
                predictedSDGs.append(predic)
                realSDGs.append(sdgs)
                scoresSDGs.append(score)
                texts.append(text)
            valids.append(valid)
            validsAny.append(validSingle)
            
        oks = [ok for ok in valids if ok == True]
        oksSingle = [ok for ok in validsAny if ok == True]
        print("- {:.2f} % valid global, {:.2f} % valid any, of {} files".format(len(oks) / len(valids) * 100, len(oksSingle) / len(valids) * 100, len(valids)))
        
        if len(path_to_excel) > 0:
            df = pd.DataFrame()
            df["text"] = texts
            df["real"] = realSDGs
            df["topics_association"] = rawSDG
            df["stats"] = statsGlobal
            df["prediction"] = predictedSDGs
            df["scores"] = scoresSDGs
            df["all_valid"] = valids
            df["any_valid"] = validsAny
            df.to_excel(path_to_excel)
        
    def map_model_topics_to_sdgs(self, path_csv="", normalize=False):
        # maps each internal topic with the SDGs. A complete text associated to each specific SDG is fetched. Then each topic is compared with each text and the text-associated sdg with the maximum score is selected as the SDG.
        topics = self.global_model.get_topics()
        nTopics = len(topics)
        self.topics_association = np.zeros((nTopics, 17))
        for ii in range(nTopics):   
            docs = self.global_model.get_representative_docs(self, topic=None)
            if num_docs < 0:
                numDocs = topic_sizes[ii]
            else:
                numDocs = num_docs
            documents, document_scores, document_ids = self.global_model.search_documents_by_topic(topic_num=ii, num_docs=numDocs)
            if normalize: document_scores = document_scores / sum(document_scores)
            sdgs = np.zeros(17)
            for id, score in zip(document_ids, document_scores):
                realSDG = associated_sdgs[id]
                for sdg in realSDG:
                    sdgs[sdg - 1] += score * 1
            self.topics_association[ii] = sdgs
            # if normalize:
            #     self.topics_association[ii] = sdgs / sum(sdgs)
            listAscii = ["x{}: {:.2f}".format(xx, topic) for topic, xx in zip(self.topics_association[ii], range(1,18))]
            print('Topic{:2d}: '.format(ii), ' | '.join(listAscii))
            
        if len(path_csv) > 4:
            # Then the mapping result is stored in a csv
            df = pd.DataFrame()
            col_names = []
            col_data = []
            sdgTitles = data.get_sdg_titles(self.paths["ref"])
            topic_words, word_scores, topic_nums = self.global_model.get_topics()
            # for sdg in sdgTitles:
            #     sdgTitle = sdgTitles[sdg]
            #     colName = "{} - {}".format(sdg, sdgTitle)
            #     colWords = []
            #     sdgInt = list(sdgTitles.keys()).index(sdg) + 1
            #     for ii, index in zip(self.topics_association, range(nTopics)):
            #         if ii == sdgInt:
            #             words = list(topic_words[index])
            #             colWords.append(words[0:30])
            #     df[colName] = colWords[0]
            for words, index in zip(topic_words, topic_nums):
                df["topic{}".format(index)] = list(words)
            df.to_csv(path_csv)
            
    def map_text_to_sdgs(self, text, n_query, score_threshold, only_positive=False):
        topics_words, word_scores, topic_scores, topic_nums = self.global_model.query_topics(text, num_topics=n_query)
        topics, probs = self.global_model.fit_transform(text)
        predictSDGs = np.zeros(17)  
        for topicIndex, topicScore in zip(topic_nums, topic_scores):
            if only_positive and topicScore < 0: break
            if topicScore < 0.2: break
            predictSDGs += topicScore * self.topics_association[topicIndex]
        top = sorted(predictSDGs, reverse=True)
        sdgs = []; scores = []
        for ii in range(len(top)):
            if top[ii] >= score_threshold:
                sdgs.append(list(predictSDGs).index(top[ii]) + 1)
                scores.append(top[ii])

        return [predictSDGs, sdgs, scores, topic_scores]
               
    def print_model_summary(self):
        # print('####### Model summary:')
        info = self.global_model.get_topics()
        nTopics = len(info)
        print(' - Number of topics: ', nTopics)
        
    def get_topics_from_model(self, model, n_top_words):
        # Returns the n_top_words for each of the n_topics with which a model has been trained
        word_dict = dict()
        topicsRaw = model.show_topics(num_topics=model.num_topics, num_words=n_top_words)
        topicsParsed = []
        for topic in topicsRaw:
            topicStr = topic[1]
            words = []
            for comb in topicStr.split(' + '):
                coef, word = comb.split('*')
                coef = float(coef)
                word = word.replace('"','')
                words.append([coef, word])
            topicsParsed.append(words)
        return topicsParsed