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
warnings.filterwarnings('ignore')

    
class LDA_classifier(LdaModel):
    paths=[]
    topics_association=[]
    verbose=False
    train_data=[]
    
    def set_conf(self, paths, verbose=False):
        self.paths = paths
        self.verbose = verbose
 
    def load_global_model(self, n_topics):
        pass
        # model = tools.load_obj(self.paths["model"] + "model_{}topics.pickle".format(n_topics))
        # vectorizer = tools.load_obj(self.paths["model"] + "vect_{}topics.pickle".format(n_topics))
        # self.global_model = [model, vectorizer]
        
    def test_model(self, database, path_excel, abstract=True, kw=False, intro=False, body=False, concl=False):
        predictedSDGs = []
        realSDGs = []
        texts = []
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
            texts.append(text)
            realSDGs.append(sdgs)
            scoresSDGs.append(score)
            valids.append(valid)
            validsAny.append(validSingle)
            files.append(file)
            abstracts.append(abstracts)
            
        df = pd.DataFrame()
        df["file"] = files
        # df["abstract"] = abstracts
        df["texts"] = texts
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
           
    def train_model(self, train_data, iterations=100, workers=8):
        # Trains a PAM model
        # @param trainData corpus of texts (array). They must be passed as texts, they are tokenized internally
        # @param n_topics number of topics for the model
        # @return model
        self.train_data = train_data
        corpus = train_data[0]; associated_sdgs = train_data[1]
        for text in corpus:
            self.add_doc(text.split(' '))
            
        self.train(iter=iterations, workers=workers)
        
    def print_summary(self, top_words, path_csv=""):
        count = self.get_count_by_topics()
        count_ascii = ["x{}: {}".format(list(count).index(nWords), nWords) for nWords in count]
        count_ascii = "Words per topic -> " + "|".join(count_ascii)
        print(count_ascii)
        
        nTopics = self.k
        topicsWords = [[] for ii in range(nTopics)]
        dfTopics = pd.DataFrame()
        for topicIndex in range(nTopics):
            words_prob = self.get_topic_words(topicIndex, top_n=top_words)
            for elem in words_prob:
                topicsWords[topicIndex].append("{:.3f}:{}".format(elem[1], elem[0]))
            topicName = "Topic{}".format(topicIndex)
            dfTopics[topicName] = topicsWords[topicIndex]
        print(dfTopics)
        
        if len(path_csv) > 0:
            try:
                dfTopics.to_csv(path_csv)
            except:
                print('CSV IS OPENED... ABORTING TOPICS EXPORT')
            
    def map_model_topics_to_sdgs(self, path_csv="", normalize=False):
        # maps each internal topic with the SDGs. A complete text associated to each specific SDG is fetched. Then each topic is compared with each text and the text-associated sdg with the maximum score is selected as the SDG.
        self.topics_association = np.zeros((self.k, 17))
        for text, labeled_sdgs in zip(self.train_data[0], self.train_data[1]):
            topicDistribution = self.infer_text(text, iterations=100)
            meanSub = np.mean(topicDistribution)
            for (subIndex, subScore) in zip(range(self.k), topicDistribution):
                if 15 in labeled_sdgs:
                    a=32
                # if subScore < meanSub: continue
                for sdg in labeled_sdgs:
                    print(topicDistribution[sdg - 1])
                    tmp = np.zeros(17)
                    tmp[sdg - 1] = 1
                    self.topics_association[subIndex] += subScore * tmp
        for ii in range(self.k2):
            # if normalize:
            # self.topics_association[ii] = self.topics_association[ii] / sum(self.topics_association[ii])
            listAscii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), self.topics_association[ii])]
            print('Topic{:2d}: '.format(ii), '|'.join(listAscii))
            
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
            # df.to_csv(path_csv
            
    def infer_text(self, text, iterations=100):
        doc = self.make_doc(text)
        result = self.infer(doc, iter=iterations)
        topicDistribution = result[0]
        return topicDistribution
        
        