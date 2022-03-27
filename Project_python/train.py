# NLP training models
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import tools

def train_model(trainData, validData, model="nmf"):
    print("hola")
    
    
def train_nmf(trainData, n_topics, ngram=(1,1), alpha_w=0.0):
    tokens = []
    for text in trainData:
        tokens.append(" ".join(tools.lemmatize_text(text)))
    
    vectorizer = TfidfVectorizer(min_df=2, # They have to appear in at least x documents
                                 stop_words='english', # Remove all stop words from English
                                 encoding='utf-8',
                                 ngram_range=ngram, # min-max
                                 #token_pattern=u'(?u)\b\w*[a-zA-Z]\w*\b' 
                                 )
    vectorized_data = vectorizer.fit_transform(tokens)
    model_nmf = NMF(n_components=n_topics, random_state=5, verbose=0, alpha_W=alpha_w)
    model_nmf.fit(vectorized_data)
    
    return [model_nmf, vectorizer]


def get_topics(model, vectorizer, n_top_words, n_topics):
    # Returns the n_top_words for each of the n_topics with which a model has been trained
    feat_names = vectorizer.get_feature_names_out()
    
    word_dict = dict()
    for ii in range(n_topics):    
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[ii].argsort()[:-n_top_words - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # {:02d}'.format(ii + 1)] = words
        
    return pd.DataFrame(word_dict)

    
