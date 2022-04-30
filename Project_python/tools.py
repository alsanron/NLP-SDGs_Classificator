# module that contains the required functions with specific functionalities such as converting texts from pdf to txt or preprocessing input text
import os
import subprocess
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import gensim
from gensim.parsing.preprocessing import STOPWORDS
import pickle
import conf

def pdfs2txt(pdfPath): 
    # Changes the environment to the powershell for windows
    os.environ["COMSPEC"] = r"C:\WINDOWS\system32\WindowsPowerShell\v1.0\powershell.exe"
    bashCommand = "bash pdftotxt.sh {} {}".format(pdfPath, pdfPath)
    subprocess.call(bashCommand, shell=True)

def tokenize_text(text, min_word_length=3, lemmatize=True, stem=False, extended_stopwords=False):
    # Clears the text from stopwords and lemmatizes the text returning the tokens
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokens = gensim.parsing.strip_tags(text)
    tokens = gensim.parsing.strip_punctuation(tokens)
    tokens = gensim.parsing.strip_numeric(tokens)
    # tokens = gensim.parsing.remove_stopwords(tokens, stopwords=STOPWORDS)
    tokens = gensim.parsing.strip_multiple_whitespaces(tokens)
    tokens = gensim.utils.simple_preprocess(tokens, deacc=True, min_len=min_word_length)
    
    set = STOPWORDS
    if extended_stopwords:
        paths = conf.get_paths()
        with open(paths["ref"] + "stop_words.txt", 'r') as f:
            words = f.read().split(' ')
            f.close()
        own_set = frozenset(words)
        set = STOPWORDS.union(own_set)
    tokens = [token for token in tokens if not(token in set)]
    
    tokenizedText = []
    for token in tokens:
        newToken = token
        if lemmatize: newToken = lemmatizer.lemmatize(newToken)
        if stem: newToken = stemmer.stem(newToken)
        tokenizedText.append(newToken)
    tokenizedText = [token for token in tokenizedText if not(token in set)]
    return tokenizedText

def save_obj(obj, path):
    pickle.dump(obj, open(path, 'wb'))
    
def load_obj(path):
    obj = pickle.load(open(path, 'rb'))
    return obj
