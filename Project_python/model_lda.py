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
    dict=[]
    verbose=False
    train_data=[]
    
    def set_conf(self, paths, dict, verbose=False):
        self.paths = paths
        self.dict = dict
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

    def print_summary(self, top_words, path_csv=""):
        nTopics = len(self.get_topics())
        topicsWords = [[] for ii in range(nTopics)]
        dfTopics = pd.DataFrame()
        words_prob = self.show_topics(num_topics=nTopics, num_words=top_words, log=False, formatted=False)
        for topicIndex in range(nTopics):
            distribution = words_prob[topicIndex]
            for elem in distribution[1]:
                topicsWords[topicIndex].append("{:.3f}:{}".format(elem[1], elem[0]))
            topicName = "Topic{}".format(topicIndex)
            dfTopics[topicName] = topicsWords[topicIndex]

        topic = 0
        maxTopicsPerLine = 7
        while(1):
            if topic + maxTopicsPerLine > nTopics:
                print(dfTopics.iloc[:, topic:])
                break
            else:
                print(dfTopics.iloc[:, topic:(topic + maxTopicsPerLine)])
            topic += maxTopicsPerLine
        
        if len(path_csv) > 0:
            try:
                dfTopics.to_csv(path_csv)
            except:
                print('CSV IS OPENED... ABORTING TOPICS EXPORT')
            
    def map_model_topics_to_sdgs(self, train_data, path_csv="", normalize=False, verbose=False):
        # maps each internal topic with the SDGs. A complete text associated to each specific SDG is fetched. Then each topic is compared with each text and the text-associated sdg with the maximum score is selected as the SDG.
        nTopics = len(self.get_topics())
        self.topics_association = np.zeros((nTopics, 17))
        for text, labeled_sdgs in zip(train_data[0], train_data[1]):
            topics, probs = self.infer_text(text)
            for (topicIndex, score) in zip(topics, probs):
                # if subScore < meanSub: continue
                for sdg in labeled_sdgs:
                    tmp = np.zeros(17)
                    tmp[sdg - 1] = 1
                    self.topics_association[topicIndex] += score * tmp
        sum_per_topic = np.zeros(17)
        for ii in range(nTopics):
            if normalize:
                self.topics_association[ii] = self.topics_association[ii] / sum(self.topics_association[ii])
                sum_per_topic += self.topics_association[ii]
            if verbose:
                listAscii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), self.topics_association[ii])]
                print('Topic{:2d}: '.format(ii), '|'.join(listAscii))
        listAscii = ["x{}:{:.2f}".format(xx, sdg) for xx, sdg in zip(range(1,18), sum_per_topic)]
        return [sum_per_topic, listAscii]
            
        # if len(path_csv) > 4:
        #     # Then the mapping result is stored in a csv
        #     df = pd.DataFrame()
        #     col_names = []
        #     col_data = []
        #     sdgTitles = data.get_sdg_titles(self.paths["ref"])
        #     topic_words, word_scores, topic_nums = self.global_model.get_topics()
        #     # for sdg in sdgTitles:
        #     #     sdgTitle = sdgTitles[sdg]
        #     #     colName = "{} - {}".format(sdg, sdgTitle)
        #     #     colWords = []
        #     #     sdgInt = list(sdgTitles.keys()).index(sdg) + 1
        #     #     for ii, index in zip(self.topics_association, range(nTopics)):
        #     #         if ii == sdgInt:
        #     #             words = list(topic_words[index])
        #     #             colWords.append(words[0:30])
        #     #     df[colName] = colWords[0]
        #     for words, index in zip(topic_words, topic_nums):
        #         df["topic{}".format(index)] = list(words)
            # df.to_csv(path_csv
            
    def infer_text(self, text):
        bow = self.dict.doc2bow(text)
        result = self.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None)
        topics = [elem[0] for elem in result]
        probs = [elem[1] for elem in result]
        return [topics, probs]
        
        