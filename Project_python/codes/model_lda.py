from signal import valid_signals
import tools
import pandas as pd
import numpy as np
from gensim.models import LdaModel
from sklearn.preprocessing import normalize
import tools
import warnings
warnings.filterwarnings('ignore')
    
# Class associated to the Latent-Dirichlet Allocation model
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
         
    def test_model(self, corpus, sdgs, path_to_plot="", path_to_excel="", only_bad=False, score_threshold=3.0, only_positive=False,     segmentize=-1, filter_low=False, expand_factor=1.0, normalize=True):
        rawSDG = []; rawSDGseg = []
        predictedSDGs = []
        realSDGs = []
        valids = []
        validsAny = []
        texts = []
        statsGlobal = []
        pred = []
        countPerSDG = np.zeros(17)
        countWellPredictionsPerSDG = np.zeros(17)
        maxSDG = 0.0
        for text, labeled_sdgs in zip(corpus, sdgs):
            if segmentize > 0:
                # then the documents are divided
                text_segments = tools.segmentize_text(text, segment_size=segmentize)
                raw_sdgs = np.zeros(17)
                for segment in text_segments:
                    raw_sdgs += self.map_text_to_sdgs(segment, only_positive=only_positive, filter_low=filter_low, normalize=normalize, expand_factor=expand_factor) 
                raw_sdgs /= len(text_segments)
                raw_sdgs_seg = raw_sdgs
            else: 
                raw_sdgs_seg = np.zeros(17) 
                raw_sdgs = self.map_text_to_sdgs(text, only_positive=only_positive, filter_low=filter_low, normalize=normalize, expand_factor=expand_factor) 

            # raw_sdgs *= expand_factor
            
            maxLocal = max(raw_sdgs)
            if maxLocal > maxSDG: maxSDG = maxLocal
                    
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
                pred.append(predic_sdgs)
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

            
        return [rawSDG, perc_valid_global, perc_valid_any, maxSDG, pred]

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
                tmp = np.zeros(17)
                for sdg in labeled_sdgs:
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
         
        listSDGsOut =  '|'.join(listAscii)
        if len(path_csv) > 4:
            dfMap = pd.DataFrame()
            rows = []
            sum_per_sdg = np.zeros(17)
            for ii in range(nTopics):
                listAscii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), self.topics_association[ii])]
                sum_per_sdg += self.topics_association[ii]
                rows.append('|'.join(listAscii))
            sum_ascii = ["x{}:{:.3f}".format(xx, sdg) for xx, sdg in zip(range(1,18), sum_per_sdg)]
            rows.append('|'.join(sum_ascii))
            dfMap["topics_association_map"] = rows
            dfMap.to_excel(self.paths["out"] + "LDA/" + "topics_map.xlsx")

            np.savetxt(self.paths["out"] + "LDA/" + "topics_map.csv", self.topics_association, delimiter=",")
            
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
                      
        return [sum_per_topic, listSDGsOut]
            
    def map_text_to_sdgs(self, text, min_threshold=0, only_positive=True, filter_low=True, normalize=True, expand_factor=1.0):
        text = self.convert_text(text)
        
        topics, probs = self.infer_text(text)
        sdgs = np.zeros(17)
        for topic, prob in zip(topics, probs):
            if (prob < min_threshold) or (prob < 0 and only_positive): continue
            sdgs += prob * self.topics_association[topic]
        sdgs *= expand_factor
        if filter_low:
            raw_sdgs_filt = sdgs < 0.05
            for prob, index, filt in zip(sdgs, range(len(sdgs)), raw_sdgs_filt):
                if filt: 
                    prob = sdgs[index]
                    sdgs[index] = 0.0
                    sdgs += prob * sdgs / sum(sdgs)  
                
        if normalize:
            sdgs = sdgs / sum(sdgs)
            
        return sdgs
    
    def map_text_to_topwords(self, text, top_n):
        if isinstance(text, str): text = text.split(' ')
        elif isinstance(text, list): text = text
        else: raise ValueError('Text type is not valid')

        words_collection = []
        topics, probs = self.infer_text(text)
        for topic, prob in zip(topics, probs):
            words = self.get_topic_terms(topic)
            for pair in words:
                wrd = self.dict.id2token[pair[0]]
                score = pair[1]
                words_collection.append((wrd, score * prob))
       
        def sort_method(elem):
            return elem[1]

        words_collection.sort(key=sort_method, reverse=True)    
        return words_collection[:top_n]
    
    def infer_text(self, text):
        bow = self.dict.doc2bow(text)
        result = self.get_document_topics(bow, minimum_probability=None, minimum_phi_value=None)
        topics = [elem[0] for elem in result]
        probs = [elem[1] for elem in result]
        return [topics, probs]
    
    def convert_text(self, text):
        if isinstance(text, str): text = text.split(' ')
        elif isinstance(text, list): text = text
        else: raise ValueError('Text type is not valid')
        return text
        
        