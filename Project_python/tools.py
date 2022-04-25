# module that contains the required functions with specific functionalities such as converting texts from pdf to txt or preprocessing input text
import os
import subprocess
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import gensim
from gensim.parsing.preprocessing import STOPWORDS
import pickle

def pdfs2txt(pdfPath): 
    # Changes the environment to the powershell for windows
    os.environ["COMSPEC"] = r"C:\WINDOWS\system32\WindowsPowerShell\v1.0\powershell.exe"
    bashCommand = "bash pdftotxt.sh {} {}".format(pdfPath, pdfPath)
    subprocess.call(bashCommand, shell=True)

def tokenize_text(text, min_word_length=3, lemmatize=True, stem=False):
    # Clears the text from stopwords and lemmatizes the text returning the tokens
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokens = gensim.parsing.strip_tags(text)
    tokens = gensim.parsing.strip_punctuation(tokens)
    tokens = gensim.parsing.strip_numeric(tokens)
    tokens = gensim.parsing.remove_stopwords(tokens, stopwords=STOPWORDS)
    tokens = gensim.parsing.strip_multiple_whitespaces(tokens)
    tokens = gensim.utils.simple_preprocess(tokens, deacc=True, min_len=min_word_length)
    
    own_set = frozenset(['cent', 'billion', 'million', 'use'])
    set = STOPWORDS.union(own_set)
    tokens = [token for token in tokens if token not in set]
    
    tokenizedText = []
    for token in tokens:
        newToken = token
        if lemmatize: newToken = lemmatizer.lemmatize(newToken)
        if stem: newToken = stemmer.stem(newToken)
        tokenizedText.append(newToken)
    return tokenizedText

def save_obj(obj, path):
    pickle.dump(obj, open(path, 'wb'))
    
def load_obj(path):
    obj = pickle.load(open(path, 'rb'))
    return obj

text = "This article is on the Decision Tree algorithm in Machine Learning. In this article, I will try to cover everything related to it. First of all, let’s see what is classification. Classification: The process of dividing the data points into the data set into different groups or different categories by adding a label to it is called Classification. But how is it possible"
print(tokenize_text(text))