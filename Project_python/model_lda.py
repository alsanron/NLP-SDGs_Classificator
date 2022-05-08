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
 
    def save_model(self):
        self.save(self.paths["model"] + "lda")
        
    def load_model():
        return LDA_classifier.load(conf.get_paths()["model"] + "lda")
         
    def test_model(self, corpus, sdgs, path_to_plot="", path_to_excel="", only_bad=False, score_threshold=3.0, only_positive=False,     segmentize=-1):
        rawSDG = []; rawSDGseg = []
        predictedSDGs = []
        realSDGs = []
        valids = []
        validsAny = []
        texts = []
        statsGlobal = []
        countPerSDG = np.zeros(17)
        countWellPredictionsPerSDG = np.zeros(17)
        
        for text, labeled_sdgs in zip(corpus, sdgs):
            if segmentize > 0:
                # then the documents are divided
                text_segments = [text]
                textLength = len(text)
                if textLength > segmentize:
                    text_segments = []; index = 0
                    while(1):
                        if index + segmentize > textLength:
                            text_segments.append(text[index:])
                            break
                        else:
                            if index + segmentize + 200 > textLength:
                                text_segments.append(text[index:])
                                break
                            else:
                                text_segments.append(text[index:(index + segmentize)])
                        index += segmentize
                raw_sdgs = np.zeros(17)
                for segment in text_segments:
                    raw_sdgs += self.map_text_to_sdgs(segment, only_positive=only_positive)  
                raw_sdgs /= len(text_segments)
                raw_sdgs_seg = raw_sdgs
            else: 
                raw_sdgs_seg = np.zeros(17) 
                raw_sdgs = self.map_text_to_sdgs(text, only_positive=only_positive) 
            raw_sdgs_filt = raw_sdgs < 0.05
            for prob, index, filt in zip(raw_sdgs, range(len(raw_sdgs)), raw_sdgs_filt):
                if filt: 
                    prob = raw_sdgs[index]
                    raw_sdgs[index] = 0.0
                    raw_sdgs += prob * raw_sdgs / sum(raw_sdgs)
                    
            predic_sdgs = [list(raw_sdgs).index(sdgScore) + 1 for sdgScore in raw_sdgs if sdgScore > score_threshold]
            validSingle = False; ii = 0
            for sdg in labeled_sdgs:
                countPerSDG[sdg - 1] += 1
                if sdg in predic_sdgs:
                    validSingle = True
                    ii += 1
                    countWellPredictionsPerSDG[sdg - 1] += 1
            valid = False
            if ii == len(labeled_sdgs):
                valid = True
                
            if (only_bad and not(valid)) or not(only_bad):
                raw_sdgsAscii = ["x{}: {:.2f}".format(xx, topic) for topic, xx in zip(raw_sdgs, range(1,18))]
                raw_sdgsAscii = "|".join(raw_sdgsAscii)
                
                raw_sdgsAsciiseg = ["x{}: {:.2f}".format(xx, topic) for topic, xx in zip(raw_sdgs_seg, range(1,18))]
                raw_sdgsAsciiseg = "|".join(raw_sdgsAsciiseg)
                
                stats = [min(raw_sdgs), np.mean(raw_sdgs), max(raw_sdgs)]
                statsAscii = "[{:.2f}, {:.2f}, {:.2f}]".format(stats[0], stats[1], stats[2])
                
                rawSDG.append(raw_sdgsAscii)
                rawSDGseg.append(raw_sdgsAsciiseg)
                statsGlobal.append(statsAscii)
                predictedSDGs.append(predic_sdgs)
                realSDGs.append(labeled_sdgs)
                texts.append(text)
            valids.append(valid)
            validsAny.append(validSingle)
            
        oks = [ok for ok in valids if ok == True]
        oksSingle = [ok for ok in validsAny if ok == True]
        perc_valid_global = len(oks) / len(valids) * 100; perc_valid_any = len(oksSingle) / len(valids) * 100
        print("- {:.2f} % valid global, {:.2f} % valid any, of {} files".format(perc_valid_global, perc_valid_any, len(valids)))
        
        if len(path_to_excel) > 0:
            df = pd.DataFrame()
            df["text"] = texts
            df["labeled_sdgs"] = realSDGs
            df["sdgs_association"] = rawSDG
            df["sdgs_segmentated"] = rawSDGseg
            df["stats"] = statsGlobal
            df["predict_sdgs"] = predictedSDGs
            df["all_valid"] = valids
            df["any_valid"] = validsAny
            df.to_excel(path_to_excel)
            
        return [rawSDG, perc_valid_global, perc_valid_any]

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
            
    def map_model_topics_to_sdgs(self, train_data, top_n_words=30, path_csv="", normalize=False, verbose=False):
        # maps each internal topic with the SDGs. A complete text associated to each specific SDG is fetched. Then each topic is compared with each text and the text-associated sdg with the maximum score is selected as the SDG.
        self.train_data = train_data
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
                norm_topics = self.topics_association[ii] / sum(self.topics_association[ii])
                for nn in norm_topics:
                    if nn < 0.1: 
                        norm_topics[list(norm_topics).index(nn)] = 0.0
                norm_topics = norm_topics / sum(norm_topics)
                self.topics_association[ii] = norm_topics
                
                sum_per_topic += self.topics_association[ii]
            if verbose:
                listAscii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), self.topics_association[ii])]
                print('Topic{:2d}: '.format(ii), '|'.join(listAscii))
        listAscii = ["x{}:{:.2f}".format(xx, sdg) for xx, sdg in zip(range(1,18), sum_per_topic)]
        if verbose:
            print('GLOBAL: ' + '|'.join(listAscii))
         
        if len(path_csv) > 4:
            # Then the mapping result is stored in a csv
            df = pd.DataFrame()
            topics_words = []
            for ii in range(nTopics):
                topics_words.append(self.get_topic_terms(ii, topn=top_n_words))
            topic_words_ascii = [[] for ii in range(nTopics)]
            for words in topics_words:
                topicIndex = topics_words.index(words)
                for word in words:
                    topic_words_ascii[topicIndex].append("{:.3f}:{}".format(word[1], self.dict.id2token[word[0]]))
            
            # all the top SDGs with an score > 0.1 are considered for that topic
            for topicIndex in range(nTopics):
                topSDGs = sorted(self.topics_association[topicIndex], reverse=True)
                title = []
                for sdg in topSDGs:
                    if sdg < 0.1: break
                    sdgIndex = list(self.topics_association[topicIndex]).index(sdg)
                    ass_sdg = self.topics_association[topicIndex][sdgIndex]
                    title.append("{:.2f}*SDG{}".format(ass_sdg, sdgIndex + 1))
                title = ",".join(title)
                df[title] = topic_words_ascii[topicIndex]

            df.to_csv(path_csv)
                      
        return [sum_per_topic, listAscii]
            
    def map_text_to_sdgs(self, text, min_threshold=0, only_positive=True):
        topics, probs = self.infer_text(text)
        sdgs = np.zeros(17)
        for topic, prob in zip(topics, probs):
            if (prob < min_threshold) or (prob < 0 and only_positive): continue
            sdgs += prob * self.topics_association[topic]
        return sdgs
    
    def infer_text(self, text):
        bow = self.dict.doc2bow(text)
        result = self.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None)
        topics = [elem[0] for elem in result]
        probs = [elem[1] for elem in result]
        return [topics, probs]
        
        