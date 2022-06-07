# module that contains the required functions with specific functionalities such as converting texts from pdf to txt or preprocessing input text
import os
from string import punctuation
import subprocess
import nltk
from nltk.stem.wordnet  import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Phrases
import pickle
import conf


def pdfs2txt(pdfPath): 
    # Changes the environment to the powershell for windows
    os.environ["COMSPEC"] = r"C:\WINDOWS\system32\WindowsPowerShell\v1.0\powershell.exe"
    bashCommand = "bash pdftotxt.sh {} {}".format(pdfPath, pdfPath)
    subprocess.call(bashCommand, shell=True)

def tokenize_text(text, min_word_length=3, punctuation=True, lemmatize=True, stem=False, stopwords=True, extended_stopwords=False):
    # Clears the text from stopwords and lemmatizes the text returning the tokens
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    tokens = gensim.parsing.strip_tags(text)  
    if punctuation: tokens = gensim.parsing.strip_punctuation(tokens)
    tokens = gensim.parsing.strip_multiple_whitespaces(tokens)
    tokens = gensim.utils.simple_preprocess(tokens, deacc=True, min_len=1)
    tokens = [token for token in tokens if not token.isnumeric()]
    
    for token in tokens:
        newToken = token
        if lemmatize: newToken = lemmatizer.lemmatize(newToken)
        if stem: newToken = stemmer.stem(newToken)
        tokens[tokens.index(token)] = newToken
        
    if stopwords:
        set = STOPWORDS
        if extended_stopwords:
            paths = conf.get_paths()
            with open(paths["ref"] + "stop_words.txt", 'r') as f:
                words = f.read().split(' ')
                f.close()
            own_set = frozenset(words)
            set = STOPWORDS.union(own_set)
            tokens = [token for token in tokens if not(token in set)]
        # just in case
        tokens = [token for token in tokens if not token in set]
    tokens = [token for token in tokens if len(token) > min_word_length]
    return tokens

def save_obj(obj, path):
    pickle.dump(obj, open(path, 'wb'))
    
def load_obj(path):
    obj = pickle.load(open(path, 'rb'))
    return obj

def segmentize_text(text, segment_size):
    text_segments = [text]
    textLength = len(text)
    if textLength > segment_size:
        text_segments = []; index = 0
        while(1):
            if index + segment_size > textLength:
                text_segments.append(text[index:])
                break
            else:
                if index + segment_size + 200 > textLength:
                    text_segments.append(text[index:])
                    break
                else:
                    text_segments.append(text[index:(index + segment_size)])
            index += segment_size
    return text_segments

def parse_sdgs_ascii_list(sdgs_ascii):
    
    sdgs = []
    for sdgAscii in sdgs_ascii:
        tmp = [int(sdg) for sdg in sdgAscii[1:-1].split(',') if len(sdg) > 0]
        if len(tmp) > 0: sdgs.append(tmp)
    return sdgs

text = "goal 1: end poverty in all its forms everywhere. more than 700 million people, or 10% of the world population, still live in extreme poverty and is struggling to fulfil the most basic needs like health, education, and access to water and sanitation, to name a few. the majority of people living on less than $1.90 a day live in sub saharan africa. worldwide, the poverty rate in rural areas is 17.2 per cent more than three times higher than in urban areas. having a job does not guarantee a decent living. in fact, 8 per cent of employed workers and their families worldwide lived in extreme poverty in 2018. poverty affects children disproportionately. one out of five children live in extreme poverty. ensuring social protection for all children and other vulnerable groups is critical to reduce poverty. poverty has many dimensions, but its causes include unemployment, social exclusion, and high vulnerability of certain populations to disasters, diseases and other phenomena which prevent them from being productive."

print(len(tokenize_text(text, punctuation=True, stopwords=True)))