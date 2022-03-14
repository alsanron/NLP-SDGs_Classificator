# NLP training models
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import tools

def train_model(trainData, validData, model="nmf"):
    print("hola")
    
    
def train_nmf(trainData, validData, n_topics):
    tokens = []
    for data in trainData:
        text = data[0]
        tokens.append(" ".join(tools.lemmatize_text(text)))
    
    vectorizer = TfidfVectorizer(min_df=2, # They have to appear in at least x documents
                                 stop_words='english', # Remove all stop words from English
                                 encoding='utf-8',
                                 ngram_range=(1, 1), # min-max
                                 #token_pattern=u'(?u)\b\w*[a-zA-Z]\w*\b' 
                                 )
    vectorized_data = vectorizer.fit_transform(tokens)
    model_nmf = NMF(n_components=n_topics, random_state=5, verbose=1)
    model_nmf.fit(vectorized_data)
    
    return [model_nmf, vectorizer]
    
