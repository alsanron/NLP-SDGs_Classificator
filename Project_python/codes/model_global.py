# functions used for testing different model configurations
from cmath import isnan
from logging import error
from multiprocessing.sharedctypes import Value
import data
import conf
import pandas as pd
import numpy as np
import tools
from scipy.special import softmax
         
         
class Global_Classifier:
    paths=[]
    nmf=[]; lda=[]; top2vec=[]; bertopic=[]
    verbose=False
    
    def __init__(self, paths, verbose=False):
        self.paths = paths
        self.verbose = verbose
 
    def load_models(self):
        self.nmf = tools.load_obj(self.paths["model"] + "nmf.pickle")
        print('# Loaded nmf...')
        
        self.lda = tools.load_obj(self.paths["model"] + "lda.pickle")
        print('# Loaded lda...')
        
        self.top2vec = tools.load_obj(self.paths["model"] + "top2vec.pickle")
        print('# Loaded top2vec...')
        
        self.bertopic = tools.load_obj(self.paths["model"] + "bertopic.pickle")
        print('# Loaded bertopic...')
        
    def test_model(self, raw_corpus, corpus, associated_SDGs=[], path_to_plot="", path_to_excel="", only_bad=False,
                   score_threshold=3.0,  only_positive=False, filter_low=False, only_main_topic=False):
        rawSDG = []; realSDGs = []; predic = []; scores = []; texts = []
       
        def parse_line(sdgs):
            sdgsAscii = ["x{}: {:.3f}".format(xx, topic) for topic, xx in zip(sdgs, range(1,18))]
            sdgsAscii = "|".join(sdgsAscii)
            sdgsAscii += "\n"
            return sdgsAscii
        if len(associated_SDGs) == 0: associated_SDGs = [[-1] for ii in range(len(corpus))]
        
        print('# Transforming corpus for bertopic...')
        topics, texts_Bertopic = self.bertopic.model.transform(corpus) # transforms the entire corpus for BERTopic. Faster than individual transforms.
        
        print('# Texts are being analysed...')
        for raw_text, text, textBERTopic, sdgs in zip(raw_corpus, corpus, texts_Bertopic, associated_SDGs):
            [nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs, bert_raw_sdgs] = self.map_text_to_sdgs(text, textBERTopic, score_threshold=score_threshold, 
                                                                               only_positive=only_positive, version=1, filter_low=filter_low, 
                                                                               normalize=False, normalize_threshold=-1)  
            
            bert_raw_sdgs = bert_raw_sdgs[0]
            
            def limit_values(array, min:float, max:float):
                rtArray = array
                for val, ind in zip(array, range(len(array))):
                    if val < min: rtArray[ind] = min
                    elif val > max: rtArray[ind] = max
                    elif isnan(val): rtArray[ind] = 0.0
                return rtArray
            
            minVal = 0; maxVal = 0.5
            nmf_raw_sdgs = limit_values(nmf_raw_sdgs, minVal, maxVal); lda_raw_sdgs = limit_values(lda_raw_sdgs, minVal, maxVal)
            top_raw_sdgs = limit_values(top_raw_sdgs, minVal, maxVal); bert_raw_sdgs = limit_values(bert_raw_sdgs, minVal, maxVal)
            
            concat_array = np.array([nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs, bert_raw_sdgs])
            filt_mean = np.zeros(17)
            for ii in range(17):
                counter = 0.0; tmp = 0.0
                for val in concat_array[:, ii]:
                    if val >= 0.0:
                        counter += 1; tmp += val
                if counter > 0: tmp /= counter
                filt_mean[ii] = tmp
                
            # predict_sdgs, scores_sdgs = self.get_identified_sdgs(nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs)
            predict_sdgs, scores_sdgs = self.get_identified_sdgs_mean(nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs, bert_raw_sdgs, filt_mean, only_main=only_main_topic)
                       
            rawSDG.append("NMF -> "+ parse_line(nmf_raw_sdgs) + "LDA -> " + parse_line(lda_raw_sdgs) + 
                          "TOP2VEC -> " + parse_line(top_raw_sdgs) + "BERTOPIC -> " + parse_line(bert_raw_sdgs) + 
                          "MEAN -> " + parse_line(filt_mean))
            predic.append(predict_sdgs); scores.append(scores_sdgs)
            realSDGs.append(sdgs)
            
            if len(raw_corpus) == 0: texts.append(text)
            else: texts.append(raw_text)
        
        if len(path_to_excel) > 0:
            df = pd.DataFrame()
            df["text"] = texts
            df["real"] = realSDGs
            
            def compare_second(elem): return elem[1]
            
            # parses the predict with the scores
            predic_str = []
            for pred, sc in zip(predic, scores):
                tpls = [(pp, ss) for pp, ss in zip(pred, sc)]
                tpls.sort(reverse=True, key=compare_second)
                
                pstr = ["{:.2f}:{}".format(elem[1], elem[0]) for elem in tpls]
                predic_str.append(", ".join(pstr))
                
            df["predict"] = predic_str
            df["sdgs_association"] = rawSDG
            # df = df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
            df.to_excel(path_to_excel)
            
        self.get_statistics_from_test(actual_sdgs=associated_SDGs, predict_sdgs=predic)
            
        return predic, scores, predic_str
    
    def get_statistics_from_test(self, actual_sdgs, predict_sdgs):
        count = 0; ok = 0; wrong = 0
        for act_sdg, pred_sdg in zip(actual_sdgs, predict_sdgs):
            for sdg in act_sdg:
                count += 1
                if sdg in pred_sdg: ok +=1
                else: wrong += 1
        print('# Results: OK: {:.2f} %'.format(ok / float(count) * 100.0))
        
        if not (ok + wrong) == count: raise ValueError('Something went wrong')
        
    def get_identified_sdgs(self, nmf, lda, top2vec):
        identified = []; scores = []
        for sdg in range(1, 18):
            index = sdg - 1; 
            predic = np.array([nmf[index], lda[index], top2vec[index]])
            # if any(predic >= 0.3):
            #     identified.append(sdg)
            if np.count_nonzero(predic >= 0.15) >= 2:
                values = [value for value in predic if value >= 0.15]
                identified.append(sdg)
                scores.append(np.mean(values))
            else: pass # not identified
        return identified, scores
    
    def get_identified_sdgs_mean(self, nmf, lda, top2vec, bertopic, mean_vec, only_main=False):
        identified = []; scores = []
        for sdg, predic in zip(range(1, 18), mean_vec):
            index = sdg - 1; 
            tmp = np.array([nmf[index], lda[index], top2vec[index], bertopic[index]])
            
            flag_mean = predic >= 0.12
            flag_count_low = np.count_nonzero(tmp >= 0.1) >= 2
            # flag_coun_high = np.count_nonzero(tmp >= 0.35) >= 1 and np.count_nonzero(tmp >= 0.1) >= 2
            
            # if flag_mean or flag_count_low or flag_coun_high:
            if flag_mean and flag_count_low:
                identified.append(sdg)
                scores.append(predic)
            else: pass # not identified
        
        if only_main:
            pairs = [(ii, jj) for ii, jj in zip(identified, scores)]
            def sort_sdgs(x):
                return x[1]
            sorted(pairs, key=sort_sdgs, reverse=True)
            identified = []; scores = []
            if len(pairs) > 0:
                identified = [pairs[0][0]]; scores = [pairs[0][1]]
        return identified, scores
            
           
    def map_text_to_sdgs(self, text, textBertopic, score_threshold, only_positive=False, version=1, filter_low=True, normalize=True, normalize_threshold=0.25):
        scale_factor = 1.4
        top_raw_sdgs, top_predic, top_score, top_raw_topicsScores = self.top2vec.map_text_to_sdgs(text, score_threshold=score_threshold, only_positive=only_positive,
                                                                                                  version=version, expand_factor=1.56*scale_factor, 
                                                                                                  filter_low=filter_low, normalize=normalize, 
                                                                                                  normalize_threshold=normalize_threshold)  
            
        nmf_raw_sdgs = self.nmf.map_text_to_sdgs(text, filter_low=filter_low, normalize=normalize, expand_factor=4.0*scale_factor)  
        
        lda_raw_sdgs = self.lda.map_text_to_sdgs(text, only_positive=only_positive, filter_low=filter_low, normalize=normalize, expand_factor=1.3*scale_factor) 
        
        bert_raw_sdgs = self.bertopic.map_text_to_sdgs_with_probs(textBertopic, score_threshold=score_threshold, only_positive=only_positive, expand_factor=1.0*scale_factor,
                                                                      filter_low=filter_low, normalize=normalize)  

        return [nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs, bert_raw_sdgs]
    