# module that contains the required functions with specific functionalities such as converting texts from pdf to txt or preprocessing input text
import os
from string import punctuation
import subprocess
from turtle import color
import difflib
from typing import List
from nltk.stem.wordnet  import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Phrases
import pickle
import conf
import numpy as np
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

def preprocess_files(folderPath):
    pdfs = [file for file in os.listdir(folderPath) if file.endswith(".pdf")]
    for pdf in pdfs:
        newPdf = standarize_file_name(pdf)
        oldPath = folderPath + pdf
        newPath = folderPath + newPdf
        os.renames(oldPath, newPath)
    # Converts the pdfs to txt
    pdfs2txt(folderPath)
    
def check_dictionary_valid(filesDict):
    # Checks if 2 files have a very close name. This generally avoids having to compare all texts
    for file in filesDict.keys():
        closestName = difflib.get_close_matches(file, filesDict.keys(),n=2,cutoff=0.8)
        if len(closestName) > 1:
            showStr = "File with name: {} close to {}, should the process continue? (Y/N): ".format(file, closestName[1:])
            userInput = input(showStr)
            userInput = userInput.lower()
            if userInput == "y":
                continue
            else:
                raise Exception("Process exited by user...")
                     
def standarize_file_name(file_name, n_iter=3):
    # removes the rare caracters from the file name
    symbols = [",", " ", "&", ":", "-","__","___","?","¿","$"]
    newName = file_name.lower()
    for iteration in range(0, n_iter):
        for symbol in symbols:
            newName = newName.replace(symbol, "_")

    return newName

def pdfs2txt(pdfPath:str): 
    # Converts all the PDFs located in the $pdfPath$ into txt format
    # @param pdfPath path where the pdfs should be located
    os.environ["COMSPEC"] = r"C:\WINDOWS\system32\WindowsPowerShell\v1.0\powershell.exe"
    bashCommand = "bash pdftotxt.sh {} {}".format(pdfPath, pdfPath)
    subprocess.call(bashCommand, shell=True)

def tokenize_text(text:str, min_word_length:int=3, punctuation:bool=True, lemmatize:bool=True, stem:bool=True, stopwords:bool=True, extended_stopwords:bool=True):
    # Tokenizes the input text. First, it applies all the options.
    # @param text Input text to clear and tokenize
    # @param min_word_length Minimum length of the works to keep
    # @param punctuation  Remove ASCII punctuation characters with spaces in s
    # @param lemmatize Wordnetlemmatizer
    # @param stem PorterStemmer
    # @param stopwords Remove the frequent stopwords
    # @param extended_stopwords Use the list stop_words.txt
    
    lemmatizer = WordNetLemmatizer()
    # stemmer = PorterStemmer()
    
    tokens = gensim.parsing.strip_tags(text)  
    if punctuation: tokens = gensim.parsing.strip_punctuation(tokens)
    tokens = gensim.parsing.strip_numeric(tokens)
    tokens = gensim.parsing.strip_non_alphanum(tokens)
    if stem: tokens = gensim.parsing.stem_text(tokens)
    tokens = gensim.parsing.strip_multiple_whitespaces(tokens)
    tokens = gensim.utils.simple_preprocess(tokens, deacc=punctuation, min_len=min_word_length)
    for token, tokenIndex in zip(tokens, range(len(tokens))):
        newToken = token
        if lemmatize: newToken = lemmatizer.lemmatize(newToken)
        # if stem: newToken = stemmer.stem(newToken)
        tokens[tokenIndex] = newToken
        
    if stopwords:
        set = STOPWORDS
        if extended_stopwords:
            paths = conf.get_paths()
            with open(paths["ref"] + "stop_words.txt", 'r') as f:
                words = f.read().split(' ')
                f.close()
            own_set = frozenset(words)
            set = STOPWORDS.union(own_set)
            # tokens = [token for token in tokens if not(token in set)]
        tokens = [token for token in tokens if not token in set]
        
    # tokens = [token for token in tokens if len(token) > min_word_length]
    return tokens

def standarize_raw_text(text:str):
    # Preprocess a raw text so that all have the same format.
    # @warning It does not tokenize or apply any process for cleaning the text
    # @param text
    outText = text
    outText = outText.lower()
    outText = outText.replace("_", " ").replace("-", " ").replace("“", " ").replace("”", " ").replace("'", "").replace("’","").replace("–", " ")
    outText = gensim.parsing.strip_multiple_whitespaces(outText)
    return outText
    
def save_obj(obj, path:str):
    pickle.dump(obj, open(path, 'wb'))
    
def load_obj(path:str):
    obj = pickle.load(open(path, 'rb'))
    return obj

def segmentize_text(text:str, segment_size):
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

def parse_sdgs_ascii_list(sdgs_ascii:list):
    # Parses a list of SDGs from ascii to int
    # @param sdgs_ascii List of sdgs in ascii -> "[1,2,4]" = SDG1,2,4
    # return sdgs List of sdgs in int -> [1,2,4]
    sdgs = []
    for sdgAscii in sdgs_ascii:
        tmp = [int(sdg) for sdg in sdgAscii[1:-1].split(',') if len(sdg) > 0]
        if len(tmp) > 0: sdgs.append(tmp)
    return sdgs

def save_figure(fig:plt, path:str):
    if os.path.exists(path): 
            os.remove(path) # otherwise, old figures are not overwritten
    fig.savefig(path)     

def analyze_predict_real_sdgs(real_sdgs, predic_sdgs, path_out="", case_name="default", show=True):
    ok = np.zeros(17); wrong = np.zeros(17)
    for real, predic in zip(real_sdgs, predic_sdgs):
        for rr in real:
            if rr in predic: ok[rr - 1] += 1
            else: wrong[rr - 1] += 1
                  
    label_ticks = range(1,18)
    plt.figure(figsize=(8, 8))
    plt.bar(label_ticks, ok + wrong, color="red")
    plt.bar(label_ticks, ok, color="green")
    plt.xlabel('SDG')
    plt.ylabel("Number of times identified")
    plt.xticks(label_ticks)
    save_figure(plt, path_out + case_name + ".png")
    if show: plt.show()
    

def count_words(text:str):
    return len(text.split(' '))

def count_texts_per_sdg(sdgs:list[list[int]]):
    # Gets the count per sdg
    # @param sdgs List[List[int]]
    # @return List[int(17)], Str (formatted)
    countPerSdg = np.zeros(17)
    for sdgG in sdgs:
        for sdg in sdgG:
            countPerSdg[sdg - 1] += 1
    countPerSdgStr = ["SDG{}:{}".format(sdg, int(count)) for sdg, count in zip(range(1,18), countPerSdg)]
    countPerSdgStr = " | ".join(countPerSdgStr)
    
    return countPerSdg, countPerSdgStr

def count_meanwords_per_sdg(texts:list[str], sdgs:list[list[int]]):
    meanWords = np.zeros(17); count = np.zeros(17)
    for text, sdgG in zip(texts, sdgs):
        nWords = count_words(text)
        for sdg in sdgG: 
            meanWords[sdg - 1] += nWords
            count[sdg - 1] += 1
    for ii in range(17): 
        if count[ii] > 0: meanWords[ii] /= count[ii] # the mean
        else: meanWords[ii] = 0
    meanWordsStr = ["SDG{}:{}".format(sdg, int(count)) for sdg, count in zip(range(1,18), meanWords)]
    meanWordsStr = " | ".join(meanWordsStr)
    
    return meanWords, meanWordsStr

def search_for_repeated_texts(texts:list[str], ratio:float=0.8):
    textsOut = texts.copy(); textsIn = texts.copy(); it=0
    for text1 in textsOut:
        it += 1
        print("# Checked: {} out of {}".format(it, len(textsOut)))
        
        for text2, index in zip(textsIn, range(len(textsIn))):
            similarityRatio = SequenceMatcher(text1, text2).ratio()
            if similarityRatio > ratio:
                print("# Similarity {:.2f} between: \r\n Text1: {} \r\n Text2{}".format(similarityRatio, text1, text2))
        
        textsIn.pop(0) # the first text can be removed...
        
def get_ok_nok_SDGsidentified(sdgs_labelled:list[list[int]], sdgs_identified:list[list[int]]):
    ok = np.zeros(17); nok = np.zeros(17)
    for ii in range(len(sdgs_labelled)):
        for sdg in sdgs_labelled[ii]:
            if sdg in sdgs_identified[ii]: ok[sdg - 1] += 1
            else: nok[sdg - 1] += 1
    return ok, nok
        
def plot_ok_vs_nok_SDGsidentified(sdgs_labelled:list[list[int]], sdgs_identified:list[list[int]], path_out:str="", show:bool=False):
    ok, nok = get_ok_nok_SDGsidentified(sdgs_labelled, sdgs_identified)
    
    xlabel = [ii for ii in range(1, 18)]
    
    plt.figure(figsize=(8, 8))
    for xx in xlabel:
        plt.bar(xx, ok[xx - 1] + nok[xx - 1], width=0.3, alpha=0.5, color='green',)
        plt.bar(xx, ok[xx - 1]              , width=0.3, alpha=1.0, color='green')
        
    plt.xticks(xlabel)
    plt.xlabel('SDG')
    plt.ylabel("Number of texts")
    # plt.ylim(top=0.5)
    # plt.title('SDGs to identify: {}'.format(labeledSDGs[textIndex]))
    plt.legend(['Not identified', 'Identified'])
    if len(path_out) > 4: plt.savefig(path_out)
    if show: plt.show()