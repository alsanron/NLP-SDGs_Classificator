# script for getting statistic and analyze those texts used in the training or validation phases
from json import tool
import sys
sys.path.insert(1, '../Project_python/')
import conf
import data
import tools
import numpy as np
import matplotlib.pyplot as plt
import warnings

paths = conf.get_paths()
raw_orgFiles, sdgs_orgFiles = data.get_sdgs_org_files(paths["SDGs_inf"])
raw_extraFiles, sdgs_extra = data.get_extra_manual_files(paths["ref"])


def test_tokenizer():
    inputText = "The “province of Cordoba”, Argentina, uses the SDGs as a framework to promote social inclusion and well-being; also this is invented. Providing affordable housing, addressing the gender gap in unemployment, reducing air pollution, and improving water quality are key priorities to advance regional development in the province. What should we do here? I can't hold it! Magic is shit #SDGs ~bye. The SDGs provide a holistic framework to address these challenges in an integrated way and can help to identify the drivers of social inclusion in the province of Cordoba. The province has undertaken a multi-stakeholder engagement process, which has led to five strategic lines of action for the achievement of the SDGs in Cordoba to: i) build a vision of multidimensional economic development for the province, ii) bridge the housing supply gap and foster sustainable construction, iii) generate decent work for the most excluded, iv) implement a sustainable water management system and v) deepen the process of coordination and transparency in policymaking. Some of the stopwords: ton use tonne party stockholm determined april communicated nation small key index form birth oda non person age. I can't belive this is happening right now. Here are other words: non-negative matrix, latent-dirichlet."
    
    outPath = "test/Out/test_tokenizer.txt"
    fp = open(outPath, 'w')
    fp.write("## Raw text: " + inputText + "\r\n \r\n")
    
    tmp = tools.standarize_raw_text(inputText)
    fp.write("## Standarize text: " + tmp + "\r\n \r\n")
    
    tmp = " ".join(tools.tokenize_text(inputText, min_word_length=3, punctuation=True, lemmatize=False, stem=True, stopwords=False, extended_stopwords=False))
    fp.write("## punctuation=True, lemmatize=False, stem=True, stopwords=False, extended_stopwords=False: " + tmp + "\r\n \r\n")
    
    tmp = " ".join(tools.tokenize_text(inputText, min_word_length=3, punctuation=True, lemmatize=True, stem=True, stopwords=False, extended_stopwords=False))
    fp.write("## punctuation=True, lemmatize=True, stem=True, stopwords=False, extended_stopwords=False: " + tmp + "\r\n \r\n")
    
    tmp = " ".join(tools.tokenize_text(inputText, min_word_length=3, punctuation=True, lemmatize=True, stem=True, stopwords=True, extended_stopwords=True))
    fp.write("## punctuation=True, lemmatize=True, stem=True, stopwords=True, extended_stopwords=True: " + tmp + "\r\n \r\n")
    
    fp.close()


test_tokenizer()