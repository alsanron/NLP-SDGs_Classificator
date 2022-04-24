import pandas as pd
import class_paper
import os
import difflib
from distutils.log import error
import pandas as pd
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from gensim.utils import simple_preprocess
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
import warnings
import json

# INPUTS
validDB = pd.read_excel('database.xlsx')
# filePath = ["SDG{}/".format(ii) for ii in range(1,18)]
filePath = ['SDG1/','SDG2/','SDG3/']
cosine_threshold = 0.75 # thresholds for similarity

warnings.filterwarnings("ignore")


def test_paper_class(database, file_path):
    fileNames = []
    valid_abs = []; valid_kw = []; valid_into = []; valid_concl = []; valid_body = []
    sim_abs = []; sim_kw = []; sim_into = []; sim_concl = []; sim_body = []
    threshold_sim = 0.8
    df = pd.DataFrame()
    for path in file_path:
        for fileName in os.listdir(path):
            if fileName.endswith('.xml'):
                if fileName in fileNames:
                    continue
                fileNames.append(fileName)
                file_path = path + fileName
                paper = class_paper.Paper()
                paper.extract_information(file_path)
                
                realAbstract = find_abstract_in_database(fileName)
                realKeywords = find_keywords_in_database(fileName)
                realIntro = find_introduction_in_database(fileName)
                realConclusion = find_conclusion_in_database(fileName)
                realBody = find_body_in_database(fileName)
                
                [flag, sim] = are_similar(paper.abstract, realAbstract)
                valid_abs.append(flag)  
                sim_abs.append(sim)
                    
                if len(realKeywords) > 10:
                    [flagKey, simKey] = are_similar(','.join(paper.keywords), realKeywords)
                    valid_kw.append(flagKey)
                else:
                    simKey = -1
                sim_kw.append(simKey)
                    
                [flagIntro, simIntro] = are_similar(paper.introduction, realIntro)
                valid_into.append(flagIntro) 
                sim_into.append(simIntro)
                
                if len(realBody) > 10:
                    [flagBody, simBody] = are_similar(paper.body, realBody)
                    valid_body.append(flagBody)
                else:
                    simBody = -1
                sim_body.append(simBody)
                    
                if len(realConclusion) > 10:
                    [flagConcl, simConcl] = are_similar(paper.conclusions, realConclusion)
                    valid_concl.append(flagConcl)
                else:
                    simConcl = -1
                sim_concl.append(simConcl)
                
                if simConcl < 0.2:
                    paper.extract_information(file_path)       
                print("File: {}, Abstract: {:.2f}, Keywords: {:.2f}, Intro: {:.2f}, Body: {:.2f}, Simconcl: {:.2f}".format(fileName, sim, simKey, simIntro, simBody, simConcl))
    df['Name'] = fileNames
    df['Abstract'] = sim_abs
    df['Keywords'] = sim_kw
    df['Introduction'] = sim_into
    df['Body'] = sim_body
    df['Conclusions'] = sim_concl
    df.to_csv('test_xml_results.csv')
    
    print("################################## STATS ##########################")
    print("- Abstracts: {:.2f} % of {}".format(len([valid for valid in valid_abs if valid == 1])/len(valid_abs) * 100, len(valid_abs)))
    print("- Keywords: {:.2f} % of {}".format(len([valid for valid in valid_kw if valid == 1])/len(valid_kw) * 100, len(valid_kw)))
    print("- Introduction: {:.2f} % of {}".format(len([valid for valid in valid_into if valid == 1])/len(valid_into) * 100, len(valid_into)))
    print("- Body: {:.2f} % of {}".format(len([valid for valid in valid_body if valid == 1])/len(valid_body) * 100, len(valid_body)))
    print("- Conclusions: {:.2f} % of {}".format(len([valid for valid in valid_concl if valid == 1])/len(valid_concl) * 100, len(valid_concl)))
                
def find_abstract_in_database(file_name):
    closest = difflib.get_close_matches(file_name, validDB.Paper, n=1, cutoff=0.8)
    if len(closest) > 0:
        abstract = validDB.Abstract[list(validDB.Paper).index(closest[0])]
        return abstract
    error('file not found')
    
def find_keywords_in_database(file_name):
    closest = difflib.get_close_matches(file_name, validDB.Paper, n=1, cutoff=0.8)
    if len(closest) > 0:
        keywords = validDB.Keywords[list(validDB.Paper).index(closest[0])]
        if isinstance(keywords, float):
            keywords = ''
        elif isinstance(keywords, str):
            pass
        return keywords
    error('file not found')
    
def find_introduction_in_database(file_name):
    closest = difflib.get_close_matches(file_name, validDB.Paper, n=1, cutoff=0.8)
    if len(closest) > 0:
        text = validDB.Introduction[list(validDB.Paper).index(closest[0])]
        if isinstance(text, float):
            text = ''
        elif isinstance(text, str):
            pass
        return text
    error('file not found')
    
def find_conclusion_in_database(file_name):
    closest = difflib.get_close_matches(file_name, validDB.Paper, n=1, cutoff=0.8)
    if len(closest) > 0:
        text = validDB.Conclusions[list(validDB.Paper).index(closest[0])]
        if isinstance(text, float):
            text = ''
        elif isinstance(text, str):
            pass
        return text
    error('file not found')
    
def find_body_in_database(file_name):
    closest = difflib.get_close_matches(file_name, validDB.Paper, n=1, cutoff=0.8)
    if len(closest) > 0:
        text = validDB.Body[list(validDB.Paper).index(closest[0])]
        if isinstance(text, float):
            text = ''
        elif isinstance(text, str):
            pass
        return text
    error('file not found')

def get_abstract(filePath):
    # returns the abstract of the document if it exists or [] if it is not found
    toc = doc.get_toc()
    if len(toc) > 0:
        for section in toc:
            sectionTitle = standarize_text(section[1])
            [flag, sim] = are_similar(sectionTitle, 'abstract')
            if flag:
                sectionPage = section[2] - 1
                page = doc.load_page(sectionPage)
                return search_abstract_in_page(page)
    # if it didnt return, then abstract not found or toc invalid
    return search_abstract_brut_force(doc)
    
def search_abstract_brut_force(doc):
    for page in doc:
        abstract = search_abstract_in_page(page)
        if len(abstract) > 10:
            return abstract
    return ""
        
def search_abstract_in_page(page):
    pars = page.get_text("blocks")
    for par in pars:
        if not(par[4].startswith("<image")):
            std_text = standarize_text(par[4])
            std_tex2 = par[4].replace(' ', '').lower()
            if ('abstract' in std_text) or ('abstract' in std_tex2):
                words = par[4].split(' ')
                if len(words) > min_words_abstract:   
                    return par[4]
                else:
                    text = pars[pars.index(par) + 1]
                    if text.endswith('\n'):
                        text = pars[pars.index(par) + 1]
                    return text
    return ""

def are_similar(text1, text2):
    # returns whether the two texts are similar or not
    text1 = standarize_text(text1)
    text2 = standarize_text(text2)
    texts = [text1, text2]
    vectorizer = CountVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    similarity = cosine_sim_vectors(vectors[0], vectors[1])
    flag = 0
    if similarity > cosine_threshold:
        flag = 1
    return [flag, similarity]
   
def standarize_text(text, lemmatize=False, pos="n"):
    tokens = simple_preprocess(text)
    if lemmatize:
        wnl = WordNetLemmatizer()
        tokens = [wnl.lemmatize(token, pos=pos) for token in tokens]
    return ' '.join(tokens)
    
def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]
    
def create_database(file_path=filePath):
    databaseDict = {}
    paths = file_path
    for path in file_path:
        print('Path: {} of {}'.format(paths.index(path) + 1, len(paths)))
        sdg = int(path[3:-1])
        for fileName in os.listdir(path):
            if fileName.endswith('.xml'):
                filenameCleaned = fileName.split('.')[0] 
                if filenameCleaned in databaseDict.keys():
                    databaseDict[filenameCleaned]["SDG"].append(sdg)
                else:
                    file_path = path + fileName
                    paper = class_paper.Paper()
                    paper.extract_information(file_path)
                    paperInfo = paper.export_as_dictionary()
                    paperInfo["SDG"] = [sdg]
                    databaseDict[filenameCleaned] = paperInfo
    jsonDump = json.dumps(databaseDict)
    with open("ext_database.json", "w", encoding='utf8') as f:
        f.write(jsonDump)
        f.close()


# test_paper_class(database=validDB, file_path=filePath)
# create_database(file_path=filePath)

    
