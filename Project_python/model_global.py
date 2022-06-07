# functions used for testing different model configurations
from logging import error
import data
import conf
import pandas as pd
import model_top2vec
import model_nmf
import model_lda
import numpy as np
import tools
         
class Global_Classifier:
    paths=[]
    nmf=[]; lda=[]; top2vec=[]
    verbose=False
    
    def __init__(self, paths, verbose=False):
        self.paths = paths
        self.verbose = verbose
 
    def load_models(self):
        self.nmf = tools.load_obj(self.paths["model"] + "nmf.pickle")
        self.lda = tools.load_obj(self.paths["model"] + "lda.pickle")
        self.top2vec = tools.load_obj(self.paths["model"] + "top2vec.pickle")
        
    def test_model(self, corpus, associated_SDGs, path_to_plot="", path_to_excel="", only_bad=False,
                   score_threshold=3.0,  only_positive=False, filter_low=False):
        rawSDG = []; realSDGs = []; predic = []; texts = []
       
        def parse_line(sdgs):
            sdgsAscii = ["x{}: {:.3f}".format(xx, topic) for topic, xx in zip(sdgs, range(1,18))]
            sdgsAscii = "|".join(sdgsAscii)
            sdgsAscii += "\n"
            return sdgsAscii
        
        for text, sdgs in zip(corpus, associated_SDGs):
            [nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs] = self.map_text_to_sdgs(text, score_threshold=score_threshold, only_positive=only_positive, version=1, filter_low=filter_low, normalize=False, normalize_threshold=-1)  
            predict_sdgs = self.get_identified_sdgs(nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs)
            
            rawSDG.append("NMF -> "+ parse_line(nmf_raw_sdgs) + "LDA -> " + parse_line(lda_raw_sdgs) + "TOP2VEC -> " + parse_line(top_raw_sdgs))
            predic.append(predict_sdgs)
            realSDGs.append(sdgs)
            texts.append(text)
            
            

        # oks = [ok for ok in valids if ok == True]
        # oksSingle = [ok for ok in validsAny if ok == True]
        # perc_global = len(oks) / len(valids) * 100
        # perc_single = len(oksSingle) / len(valids) * 100
        # print("- {:.2f} % valid global, {:.3f} % valid any, of {} files".format(perc_global, perc_single, len(valids)))
        # print('Max found: {:.3f}'.format(maxSDG))
        
        # for probs, index in zip(probs_per_sdg, range(len(probs_per_sdg))):
        #     probs_per_sdg[index] = np.mean(probs_per_sdg[index])
        
        if len(path_to_excel) > 0:
            df = pd.DataFrame()
            df["text"] = texts
            df["real"] = realSDGs
            df["predict"] = predic
            df["sdgs_association"] = rawSDG
            df.to_excel(path_to_excel)
            
        return [1]
        
    def get_identified_sdgs(self, nmf, lda, top2vec):
        identified = []
        for sdg in range(1, 18):
            index = sdg - 1; 
            predic = np.array([nmf[index], lda[index], top2vec[index]])
            if any(predic >= 0.3):
                identified.append(sdg)
            elif np.count_nonzero(predic >= 0.15) >= 2:
                identified.append(sdg)
            else: pass # not identified
        return identified
            
            
           
    def map_text_to_sdgs(self, text, score_threshold, only_positive=False, version=1, filter_low=True, normalize=True, normalize_threshold=0.25):
        scale_factor = 1.15
        top_raw_sdgs, top_predic, top_score, top_raw_topicsScores = self.top2vec.map_text_to_sdgs(text, score_threshold=score_threshold, only_positive=only_positive, version=version, expand_factor=1.56*scale_factor, filter_low=filter_low, normalize=normalize, normalize_threshold=normalize_threshold)  
            
        nmf_raw_sdgs = self.nmf.map_text_to_sdgs(text, filter_low=filter_low, normalize=normalize, expand_factor=4.0*scale_factor)  
        
        lda_raw_sdgs = self.lda.map_text_to_sdgs(text, only_positive=only_positive, filter_low=filter_low, normalize=normalize, expand_factor=1.0*scale_factor) 

        return [nmf_raw_sdgs, lda_raw_sdgs, top_raw_sdgs]
    