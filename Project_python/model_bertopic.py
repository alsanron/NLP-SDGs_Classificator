from signal import valid_signals
import tools
import pandas as pd
import numpy as np
import conf
import data
import matplotlib.pyplot as plt
import tools
import warnings
import hashlib
from bertopic import BERTopic
warnings.filterwarnings('ignore')


class BERTopic_classifier:
    paths=[]
    model=[]
    nTopics=[]
    topics_association=[]
    verbose=False
    trainData=[]
    
    def __init__(self, paths):
        self.paths = paths
            
    def train_model(self, train_data, 
                    embedding_model="all-MiniLM-L6-v2", # Others: all-MiniLM-L6-v2, all-MiniLM-L12-v2, all-mpnet-base-v2
                    n_gram_range=(1,3), # the default parameter is (1,1)
                    top_n_words=10, min_topic_size=10, # default parameters
                    nr_topics=None, # reduce the number of topics to this number
                    diversity=None, # value can be used between 0, 1
                    calculate_probabilities=True, 
                    seed_topic_list=None,
                    verbose=True
                    ):
        # trains the model based on the training files
        # @param train_files corpus of documents as a list of strings
        # @param method "fast-learn", "learn" or "deep-learn"
        # @param workes number of parallel workers
        # @param min_count minimum number of documents where a word must be to be valid
        self.trainData = train_data
        corpus = train_data[0]; associated_sdgs = train_data[1]
        
        self.model = BERTopic(language="english", 
                        embedding_model=embedding_model, 
                        top_n_words=top_n_words, 
                        min_topic_size=min_topic_size,
                        n_gram_range=n_gram_range, 
                        nr_topics=nr_topics,
                        calculate_probabilities=calculate_probabilities, 
                        seed_topic_list=seed_topic_list, 
                        verbose=verbose)
        
        topics, probs = self.model.fit_transform(corpus, 
                       embeddings=None, # use the sentence-transformer model
                       y=None # target class for semisupervised. not applicable
                       ) 
        self.nTopics = len(self.model.get_topics())
        if topics.count(-1) > 0: self.nTopics -= 1 # there is 1 outlier topic
        counts_per_topic = ["T{}: ".format(ii) + str(topics.count(ii)) for ii in range(0, self.nTopics)]
        print('## nDocs per topic: ' + " | ".join(counts_per_topic))
        
        self.map_model_topics_to_sdgs(associated_sdgs, topics, probs, normalize=True, verbose=True)  
         
    def test_model(self, corpus, associated_SDGs,  stat_topics=-1, path_to_plot="", path_to_excel="", 
                   only_bad=False, score_threshold=3.0, only_positive=True, filter_low=True, 
                   expand_factor=1.0, normalize=False):
        rawSDG = []; rawSDGseg = []
        predictedSDGs = [];  realSDGs = []
        scoresSDGs = []
        valids = []; validsAny = []
        texts = []
        statsGlobal = []
        countPerSDG = np.zeros(17)
        countWellPredictionsPerSDG = np.zeros(17)
        probs_per_sdg = [[] for ii in range(1,18)]
        maxSDG = 0.0
        
        topics, probs = self.model.transform(corpus) # transforms the entire corpus
        
        for text, sdgs, prob in zip(corpus, associated_SDGs, probs):
            raw_sdgs, predic, score = self.map_text_to_sdgs_with_probs(prob, score_threshold=score_threshold, only_positive=only_positive, expand_factor=expand_factor, filter_low=filter_low, normalize=normalize)  
            
            maxLocal = max(raw_sdgs)
            if maxLocal > maxSDG: maxSDG = maxLocal
            
            validSingle = False; ii = 0
            for sdg in sdgs:
                countPerSDG[sdg - 1] += 1
                probs_per_sdg[sdg - 1].append(raw_sdgs[sdg - 1])
                if sdg in predic:
                    validSingle = True
                    ii += 1
                    countWellPredictionsPerSDG[sdg - 1] += 1
            valid = False
            if ii == len(sdgs):
                valid = True
                
            if (only_bad and not(valid)) or not(only_bad):
                raw_sdgsAscii = ["x{}: {:.3f}".format(xx, topic) for topic, xx in zip(raw_sdgs, range(1,18))]
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
        perc_global = len(oks) / len(valids) * 100
        perc_single = len(oksSingle) / len(valids) * 100
        print("- {:.2f} % valid global, {:.3f} % valid any, of {} files".format(perc_global, perc_single, len(valids)))
        print('Max found: {:.3f}'.format(maxSDG))
        
        for probs, index in zip(probs_per_sdg, range(len(probs_per_sdg))):
            probs_per_sdg[index] = np.mean(probs_per_sdg[index])
        
        if len(path_to_excel) > 0:
            df = pd.DataFrame()
            df["text"] = texts
            df["real"] = realSDGs
            df["sdgs_association"] = rawSDG
            df["stats"] = statsGlobal
            df["prediction"] = predictedSDGs
            df["scores"] = scoresSDGs
            df["all_valid"] = valids
            df["any_valid"] = validsAny
            df.to_excel(path_to_excel)
            
        return predictedSDGs, maxSDG
        
    def map_model_topics_to_sdgs(self, associated_sdgs, topics, probs, path_csv="", normalize=True, verbose=True):
        # maps each internal topic with the SDGs. A complete text associated to each specific SDG is fetched. Then each topic is compared with each text and the text-associated sdg with the maximum score is selected as the SDG.
        self.topics_association = np.zeros((self.nTopics, 17))
        for sdgs, main_topic, prob in zip(associated_sdgs, topics, probs):
            if main_topic < 0: 
                continue # outlier topic is discarded
            tmp = np.zeros(17)
            for sdg in sdgs:
                tmp[sdg - 1] = 1
            for topic_index, prob_topic in zip(range(self.nTopics), prob):
                self.topics_association[topic_index, :] += tmp * prob_topic
            
        sum_per_topic = np.zeros(17)
        for topic_index in range(self.nTopics):
            if normalize: 
                tmp = self.topics_association[topic_index, :]
                self.topics_association[topic_index, :] = tmp / sum(tmp)
            sum_per_topic += self.topics_association[topic_index, :]
            if verbose:
                listAscii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), self.topics_association[topic_index, :])]
                print('Topic{:2d}: '.format(topic_index), '|'.join(listAscii))
        
        if verbose:
            listAscii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), sum_per_topic)]
            final_sum = 'Sum total: ' + '|'.join(listAscii)
            print(final_sum)   
            
        if len(path_csv) > 4:
            all_topics = topic_model.get_topics()
            
            raise ValueError('Association map to csv not supported')
            
    def map_text_to_sdgs(self, text, score_threshold, only_positive=False, 
                         expand_factor=3, filter_low=True, normalize=False):
        topics, probs = self.model.transform(text)
        [predictSDGs, sdgs, scores] = self.map_text_to_sdgs_with_probs(probs, score_threshold=score_threshold, only_positive=only_positive, expand_factor=expand_factor, filter_low=filter_low, normalize=normalize)
                
        return [predictSDGs, sdgs, scores]
    
    def map_text_to_sdgs_with_probs(self, probs, score_threshold, only_positive=False, 
                         expand_factor=3, filter_low=True, normalize=False):
        predictSDGs = np.zeros(17)  
        for topicIndex, topicScore in zip(range(self.nTopics), probs):
            if only_positive and topicScore < 0: continue
            predictSDGs += topicScore * self.topics_association[topicIndex]
            
        predictSDGs *= expand_factor
        
        if filter_low:
            raw_sdgs_filt = predictSDGs < 0.05
            for prob, index, filt in zip(predictSDGs, range(len(predictSDGs)), raw_sdgs_filt):
                if filt: 
                    prob = predictSDGs[index]
                    predictSDGs[index] = 0.0
                    predictSDGs += prob * predictSDGs / sum(predictSDGs)
                    
        if normalize: predictSDGs = predictSDGs / sum(predictSDGs)
        
        top = sorted(predictSDGs, reverse=True)
        sdgs = []; scores = []
        for ii in range(len(top)):
            if top[ii] >= score_threshold:
                sdgs.append(list(predictSDGs).index(top[ii]) + 1)
                scores.append(top[ii])
                
        return [predictSDGs, sdgs, scores]
         
                
    def print_summary(self, path_csv=""):
        topic_words_scores = [[] for ii in range(self.nTopics)]
        dfTopics = pd.DataFrame()
        all_topics = self.model.get_topics()
        for topic_index in range(self.nTopics):
            topic_words = all_topics[topic_index]
            for topic_tuple in topic_words:
                word = topic_tuple[0]; score = topic_tuple[1]
                word_str = "{:.3f}:{}".format(score, word)
                topic_words_scores[topic_index].append(word_str)   
                
            topicName = "Topic{}".format(topic_index)
            dfTopics[topicName] = topic_words_scores[topic_index]
        
        if len(path_csv) > 0:
            dfTopics.to_csv(path_csv)