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

# text = "The Boomerang system – engineering logic gate genetic device for detection and treatment of cancer:::Despite recent treatment advancements, cancer is still a major cause of mortality worldwide. One of the fundamental problems preventing the development of effective therapy is the difficulty to target cancer cells exclusively. In Boomerang, we're engineering a genetic device based on a simple concept of AND logic gate: the activation of our CRISPR/Cas9-based system is dependent on the existence of two cancer-specific promoters that control the expression of Cas9 and gRNA, and the combination of these two will occur only in cancer cells. CRISPR/Cas9 system allows several applications of Boomerang: 1) disruption of genes essential for cancer survival; and 2) activation of suicide genes, or color proteins for cancer cell detection (e.g., for complete surgical removal). Our system can be potentially designed according to unique characteristics of a patient's tumor, paving the way to personalized medicine. We hope that our strategy will change the approach to cancer treatments"

# print(' '.join(tokenize_text(text, punctuation=True, stopwords=True)))