# functions used for testing different model configurations
from signal import valid_signals
import tools
import pandas as pd
import numpy as np
import conf
import data
from sklearn.preprocessing import normalize
import tools
import warnings
from top2vec import Top2Vec
warnings.filterwarnings('ignore')
        
class Top2Vec_classifier:
    paths=[]
    model=[]
    train_data=[]
    topics_association=[]
    verbose=False
    
    def __init__(self, paths, verbose=False):
        self.paths = paths
        self.verbose = verbose
 
    def load(self, train_data):
        self.model = Top2Vec.load(self.paths["model"] + "model_top2vec")
        self.train_data = train_data
        
    def save(self):
        self.model.save(self.paths["model"] + "model_top2vec")
            
    def train(self, train_data, embedding_model="doc2vec", ngram=True, method="learn", workers=8, min_count=2, tokenizer=False):
        # trains the model based on the training files
        # @param train_files corpus of documents as a list of strings
        # @param method "fast-learn", "learn" or "deep-learn"
        # @param workes number of parallel workers
        # @param min_count minimum number of documents where a word must be to be valid
        self.train_data = train_data
        corpus = train_data[0]
        
        self.model = Top2Vec(documents=corpus, embedding_model=embedding_model, min_count=min_count, ngram_vocab=ngram, speed=method, workers=workers, document_chunker="sequential", use_embedding_model_tokenizer=tokenizer)
        
        self.print_model_summary()
        
    def test_model(self, corpus, associated_SDGs, path_to_plot="", path_to_excel="", only_bad=False,
                   score_threshold=3.0,  only_positive=False, filter_low=False, expand_factor=1.0, version=1):
        rawSDG = []; rawSDGseg = []
        predictedSDGs = []
        realSDGs = []
        scoresSDGs = []
        valids = []
        validsAny = []
        texts = []
        statsGlobal = []
        countPerSDG = np.zeros(17)
        countWellPredictionsPerSDG = np.zeros(17)
        probs_per_sdg = [[] for ii in range(1,18)]
        
        numTopics = self.model.get_num_topics()
        stat_topics = numTopics
        for text, sdgs in zip(corpus, associated_SDGs):
            raw_sdgs, predic, score, raw_topicsScores = self.map_text_to_sdgs(text, score_threshold=score_threshold, only_positive=only_positive, version=version)  
            
            raw_sdgs *= expand_factor
            if filter_low:
                raw_sdgs_filt = raw_sdgs < 0.05
                for prob, index, filt in zip(raw_sdgs, range(len(raw_sdgs)), raw_sdgs_filt):
                    if filt: 
                        prob = raw_sdgs[index]
                        raw_sdgs[index] = 0.0
                        raw_sdgs += prob * raw_sdgs / sum(raw_sdgs)
            
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
            
        return [perc_global, perc_single, probs_per_sdg]
        
    def map_model_topics_to_sdgs(self, path_csv="", normalize=False, version=1):
        # maps each internal topic with the SDGs. A complete text associated to each specific SDG is fetched. Then each topic is compared with each text and the text-associated sdg with the maximum score is selected as the SDG.
        if version == 1:
            nTopics = self.model.get_num_topics()
            topic_sizes, topics_num = self.model.get_topic_sizes()
            self.topics_association = np.zeros((nTopics, 17))
            for text, labeled_sdgs in zip(self.train_data[0], self.train_data[1]):
                topics, probs = self.infer_text(text)
                for topicIndex, score in zip(topics, probs):
                    # if subScore < meanSub: continue
                    if score < 0.0: continue
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
                if self.verbose:
                    listAscii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), self.topics_association[ii])]
                    print('Topic{:2d}: '.format(ii), '|'.join(listAscii))  
            if self.verbose:
                listAscii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), sum_per_topic)]
                final_sum = 'Sum total: ' + '|'.join(listAscii)
                print(final_sum) 
        else:
            sum_per_topic = np.zeros(17)
            nTopics = self.model.get_num_topics()
            topic_sizes, topics_num = self.model.get_topic_sizes()
            self.topics_association = np.zeros((nTopics, 17))
            for topicIndex in range(nTopics):   
                numDocs = topic_sizes[topicIndex] # all the associated documents to that topic
                documents, document_scores, document_ids = self.model.search_documents_by_topic(topic_num=topicIndex, num_docs=numDocs)
                if normalize: 
                    document_scores = document_scores / sum(document_scores)
                    
                sdgs = np.zeros(17)
                for docId, score in zip(document_ids, document_scores):
                    labeled_sdgs = self.train_data[1][docId]
                    for sdg in labeled_sdgs:
                        sdgs[sdg - 1] += score * 1
                
                if normalize:
                    norm_topics = sdgs / sum(sdgs)
                    for nn in norm_topics:
                        if nn < 0.1: 
                            norm_topics[list(norm_topics).index(nn)] = 0.0
                    norm_topics = norm_topics / sum(norm_topics)
                    sdgs = norm_topics
                    
                self.topics_association[topicIndex] = sdgs     
                sum_per_topic += sdgs
                if self.verbose:
                    listAscii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), self.topics_association[topicIndex])]
                    print('Topic{:2d}: '.format(topicIndex), '|'.join(listAscii))
            if self.verbose:
                listAscii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), sum_per_topic)]
                final_sum = 'Sum total: ' + '|'.join(listAscii)
                print(final_sum) 
            
        if len(path_csv) > 4:
            # Then the mapping result is stored in a csv
            topic_words, word_scores, topic_nums = self.model.get_topics()
            df = pd.DataFrame()

            topic_words_ascii = [[] for ii in range(nTopics)]
            for words, scores, topicIndex in zip(topic_words, word_scores, range(nTopics)):
                for word, score in zip(words, scores):
                    topic_words_ascii[topicIndex].append("{:.3f}:{}".format(score, word))
            
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
                
        return [sum_per_topic, final_sum]
            
    def map_text_to_sdgs(self, text, score_threshold, only_positive=False, version=1):
        if version == 1: # then map with topics
            numTopics = self.model.get_num_topics()
            topics_words, word_scores, probabilities, topic_nums = self.model.query_topics(text, num_topics=numTopics)
            predictSDGs = np.zeros(17)  
            for topicIndex, topicScore in zip(topic_nums, probabilities):
                if only_positive and (topicScore < 0): break
                predictSDGs += topicScore * self.topics_association[topicIndex]
        elif version == 2: # then map with documents
            documents , probabilities , doc_ids = self.model.query_documents(text, num_docs=100)
            predictSDGs = np.zeros(17)  
            for docIndex, docScore in zip(doc_ids, probabilities):
                if only_positive and (docScore < 0): break
                sdgs = np.zeros(17)
                labeled_sdgs = self.train_data[1][docIndex]
                for sdg in labeled_sdgs:
                    sdgs[sdg - 1] = 1
                predictSDGs += docScore * sdgs
            
        top = sorted(predictSDGs, reverse=True)
        sdgs = []; scores = []
        for ii in range(len(top)):
            if top[ii] >= score_threshold:
                sdgs.append(list(predictSDGs).index(top[ii]) + 1)
                scores.append(top[ii])

        return [predictSDGs, sdgs, scores, probabilities]
    
    def infer_text(self, text):
        nTopics = self.model.get_num_topics()
        t1, t2, topic_scores , topic_nums = self.model.query_topics(text, nTopics)
        return [topic_nums, topic_scores]
               
    def print_model_summary(self):
        # print('####### Model summary:')
        print(' - Number of topics: ', self.model.get_num_topics())

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
        

   