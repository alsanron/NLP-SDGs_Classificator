import os
import subprocess
from nltk.stem import WordNetLemmatizer
import gensim

def pdfs2txt(pdfPath): 
    # Changes the environment to the powershell for windows
    os.environ["COMSPEC"] = r"C:\WINDOWS\system32\WindowsPowerShell\v1.0\powershell.exe"
    bashCommand = "bash pdftotxt.sh {} {}".format(pdfPath, pdfPath)
    subprocess.call(bashCommand, shell=True)


# TODO A more advanced lemmatization should be able to infer what kind of word the context has and pass it to the lemmatize...
def lemmatize_text(text, min_word_length=3):
    # Clears the text from stopwords and lemmatizes the text returning the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = gensim.parsing.strip_tags(text)
    tokens = gensim.parsing.strip_punctuation(tokens)
    tokens = gensim.parsing.strip_numeric(tokens)
    tokens = gensim.parsing.remove_stopwords(tokens)
    tokens = gensim.parsing.strip_multiple_whitespaces(tokens)
    tokens = gensim.utils.simple_preprocess(tokens, deacc=True, min_len=min_word_length)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens
