# functions used for testing different model configurations
from signal import valid_signals
import tools
import pandas as pd
import numpy as np
import data
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import tools
import warnings
warnings.filterwarnings('ignore')
 
class NMF_classifier:
    paths=[]
    model=[]
    vectorizer=[]
    topics_association=[]
    train_data=[]
    verbose=False
    
    def __init__(self, paths, verbose=False):
        self.paths = paths
        self.verbose = verbose
    
    def load(self, n_topics):
        self.model = tools.load_obj(self.paths["model"] + "model_nmf_{}topics.pickle".format(n_topics))
        self.model.verbose = self.verbose
        self.vectorizer = tools.load_obj(self.paths["model"] + "vect_nmf_{}topics.pickle".format(n_topics))
         
    def save(self):
        nTopics = self.model.n_components
        tools.save_obj(self.model, self.paths["model"] + "model_nmf_{}topics.pickle".format(nTopics))
        tools.save_obj(self.vectorizer, self.paths["model"] + "vect_nmf_{}topics.pickle".format(nTopics))
            
    def train(self, train_data, n_topics, ngram, min_df=1):
        self.train_data = train_data
        # the corpus should be preprocessed before
        self.vectorizer = TfidfVectorizer(min_df=min_df, # They have to appear in at least x documents
                                    encoding='utf-8',
                                    ngram_range=ngram, # min-max
                                    )
        vectorized_data = self.vectorizer.fit_transform(train_data[0])
        self.model = NMF(n_components=n_topics, random_state=5, verbose=False, max_iter=2000)
        self.model.fit(vectorized_data) 
        
    def test_model(self, corpus, associated_SDGs, score_threshold=0.2, segmentize=-1, filter_low=False, path_to_plot="", path_to_excel="", normalize=True, expand_factor=1.0):
        rawSDG = []; rawSDGseg = []
        predictedSDGs = []
        realSDGs = []
        valids = []
        validsAny = []
        texts = []
        statsGlobal = []
        countPerSDG = np.zeros(17)
        countWellPredictionsPerSDG = np.zeros(17)
        maxSDG = 0.0
        for text, labeled_sdgs in zip(corpus, associated_SDGs):
            if segmentize > 0:
                # then the documents are divided
                text_segments = tools.segmentize_text(text, segment_size=segmentize)
                raw_sdgs = np.zeros(17)
                for segment in text_segments:
                    raw_sdgs += self.map_text_to_sdgs(segment, filter_low=filter_low, normalize=normalize, expand_factor=expand_factor)  
                raw_sdgs /= len(text_segments)
                raw_sdgs_seg = raw_sdgs
            else: 
                raw_sdgs_seg = np.zeros(17) 
                raw_sdgs = self.map_text_to_sdgs(text, filter_low=filter_low, normalize=normalize, expand_factor=expand_factor) 
                    
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
            maxLocal = max(raw_sdgs)
            if maxLocal > maxSDG: maxSDG = maxLocal
                
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
        print('Max found: {:.3f}'.format(maxSDG))
        
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
            
        return [rawSDG, perc_valid_global, perc_valid_any, maxSDG]
            
    def map_model_topics_to_sdgs(self, n_top_words, normalize=True, path_csv=""):
        # Maps each new topic of the general NMF model to an specific SDG obtained from training 17 models
        nTopics = self.model.n_components
        
        self.topics_association = np.zeros((nTopics, 17))
        for text, labeled_sdgs in zip(self.train_data[0], self.train_data[1]):
            topicScores = self.infer_text(text)
            for topicIndex, score in zip(range(nTopics), topicScores):
                # if subScore < meanSub: continue
                for sdg in labeled_sdgs:
                    tmp = np.zeros(17)
                    tmp[sdg - 1] = 1
                    self.topics_association[topicIndex] += score * tmp
        sum_per_topic = np.zeros(17)
        for ii in range(nTopics):
            if normalize:
                norm_topics = self.topics_association[ii] / sum(self.topics_association[ii])
                topics_to_delete = norm_topics < 0.1
                for nn, delete, index in zip(norm_topics, topics_to_delete, range(len(norm_topics))):
                    if delete: 
                        norm_topics[index] = 0.0
                norm_topics = norm_topics / sum(norm_topics)
                self.topics_association[ii] = norm_topics
                
                sum_per_topic += self.topics_association[ii]
            if self.verbose:
                listAscii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), self.topics_association[ii])]
                print('Topic{:2d}: '.format(ii), '|'.join(listAscii))
        listAscii = ["x{}:{:.2f}".format(xx, sdg) for xx, sdg in zip(range(1,18), sum_per_topic)]
        if self.verbose:
            print('GLOBAL: ' + '|'.join(listAscii))
    
        if len(path_csv) > 4:
            # Then the mapping result is stored in a csv
            dfMap = pd.DataFrame()
            rows = []
            for ii in range(nTopics):
                listAscii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), self.topics_association[ii])]
                rows.append('|'.join(listAscii))
            dfMap["topics_association_map"] = rows
            dfMap.to_excel(self.paths["out"] + "NMF/" + "topics_map.xlsx")
            
            df = pd.DataFrame()
            topic_words_ascii = []
            for ii in range(nTopics):
                topic_words_ascii.append(self.get_topic_terms(ii, topn=n_top_words))
            
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

    def map_text_to_sdgs(self, text, filter_low=True, normalize=True, expand_factor=1.0):
        query_words_vect = self.vectorizer.transform([text])
        topicFeats = self.model.transform(query_words_vect)[0]
        
        for word in text.split(' '):
            query_words_vect = self.vectorizer.transform([word])
            topicFeats = self.model.transform(query_words_vect)[0]
        
        sdgs_score = np.zeros(17)
        for topicScore in topicFeats:
            sdgs_score += topicScore * self.topics_association[list(topicFeats).index(topicScore)] 
        sdgs_score *= expand_factor
        
        if filter_low:
            raw_sdgs_filt = sdgs_score < 0.05
            for prob, index, filt in zip(sdgs_score, range(len(sdgs_score)), raw_sdgs_filt):
                if filt: 
                    prob = sdgs_score[index]
                    sdgs_score[index] = 0.0
                    sdgs_score += prob * sdgs_score / sum(sdgs_score)
        
        if normalize:
            sdgs_score = sdgs_score / sum(sdgs_score)
        
        return sdgs_score
    
    def infer_text(self, text):
        query_words_vect = self.vectorizer.transform([text])
        topicScores = self.model.transform(query_words_vect)[0]
        return topicScores
      
    def get_topic_terms(self, topic, topn):
        # Returns the n_top_words for each of the n_topics with which a model has been trained
        feat_names = self.vectorizer.get_feature_names_out()
        words_ids = self.model.components_[topic].argsort()[:-topn - 1:-1]
        words = [feat_names[key] for key in words_ids]
        return words
      