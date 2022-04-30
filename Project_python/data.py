# File that contains the functions for accesing the required data: training and validation

import os
import difflib
from attr import assoc
from numpy import empty
import pandas as pd
import json
import conf
import preprocess

def get_sdg_titles(refPath):
    # returns the title of each SDG as a dictionary, with key: SDGx, value = title.
    # for example: "SDG1":"No poverty"
    f = open(refPath + "SDG_titles.json")
    sdgs_title = json.load(f)
    f.close()
    return sdgs_title

# DATASET: the role of artificial intelligence in achieving the sustainable development goals. NATURE PAPER. The user can select independently: abstract, keywords, introduction, body or conclusions.
def get_nature_files(abstract=True, kw=False, intro=False, body=False, concl=False):
    paths = conf.get_paths()
    with open(paths["ref"] + "ext_database.json", "r") as f:
        json_dump = f.read()
        f.close()
    database = json.loads(json_dump)
    
    corpus = []; associatedSDGs = []
    for file in database:
        text = ""
        sdgs = database[file]["SDG"]
        if abstract:
            text += database[file]["abstract"]
        if kw:
            text += database[file]["keywords"]
        if intro:
            text += database[file]["introduction"]
        if body:
            text += database[file]["body"]
        if concl:
            text += database[file]["conclusions"]
        corpus.append(text)
        associatedSDGs.append(sdgs)
    print("- {} nature files were found".format(len(corpus)))
    return [corpus, associatedSDGs]
    
# DATASET: https://sdgs.un.org/
# - Goals definition
# - Goals progress - evolution section
def get_sdgs_org_files(refPath, sdg=-1, compact_per_sdg=False):
    # Returns an array where each elements consist of an array with the fields:
    # [0] abstract or text related to a SDG, [1]: array with the associated SDGs.
    if sdg > 0:
        sdgStart = "{:02d}".format(sdg)
    else:
        sdgStart = "" # it's always true

    if compact_per_sdg:
        sdgsPaths = [refPath + "SDGs_description/"]
        corpus = [""] * 17
        associatedSDGs = [[ii] for ii in range(1,17)]
        for path in sdgsPaths:
            for file in os.listdir(path):
                if file.endswith(".txt") and file.startswith(sdgStart):
                    f = open(path + file, 'r', encoding="utf8")
                    text = f.read()
                    f.close()
                    fileSDG = int(file.partition("_")[0])
                    corpus[fileSDG - 1] += text
    else:
        sdgsPaths = [refPath + "SDGs_description/",
                     refPath + "SDGs_progress/",
                     #refPath + "SDGs_targets/"
                    ]
        corpus = []; associatedSDGs = []
        for path in sdgsPaths:
            for file in os.listdir(path):
                if file.endswith(".txt") and file.startswith(sdgStart):
                    try:
                        f = open(path + file, 'r')
                        text = f.read()
                    except UnicodeError:
                        f = open(path + file, 'r', encoding="utf8")
                        text = f.read()
                    f.close()
                    fileSDG = [int(file.partition("_")[0])]
                    corpus.append(text)
                    associatedSDGs.append(fileSDG)
        sdgsPaths = [refPath + "SDGs_targets/",
                    #  refPath + "SDGs_description/",
                    #  refPath + "SDGs_progress/"
                    ]
        for path in sdgsPaths:
            texts = [""] * 17
            for file in os.listdir(path):
                if file.endswith(".txt") and file.startswith(sdgStart):
                    try:
                        f = open(path + file, 'r')
                        text = f.read()
                    except UnicodeError:
                        f = open(path + file, 'r', encoding="utf8")
                        text = f.read()
                    f.close()
                    fileSDG = int(file.partition("_")[0])
                    texts[fileSDG - 1] += " " + text
            for text, sdg in zip(texts, range(1,18)):
                corpus.append(text)
                associatedSDGs.append([sdg])
        
    nFiles = len(corpus)
    print("- {} sdgs files were found".format(nFiles))    
    return [corpus, associatedSDGs]

# DATASET: files from scopus classified as related to a sdg previously by the algorithm
def get_previous_classified_abstracts(refPath):
    # returns files that were previously classifies by the algorithm as valid
    absPath = refPath + "Abstracts/"
    abstracts = []
    for file in os.listdir(absPath):
        if file.endswith(".txt"):
            f = open(absPath + file, 'r', encoding="utf-8")
            text = f.read()
            f.close()
            abstracts.append(text)   
            
# DATASET: https://sdg-pathfinder.org/ files related to each SDG
def get_sdgs_pathfinder(refPath):
    csv = pd.read_csv(refPath + "ds_sdg_path_finder.csv")
    corpus = []; sdgs = []
    for text, sdgsAscii in zip(csv["text"], csv["SDGs"]):
        sdgsInt = [int(sdg) for sdg in (sdgsAscii.replace("[","").replace("]","")).split(",")]
        corpus.append(text)
        sdgs.append(sdgsInt)
    print("- {} texts in the pathfinder dataset".format(len(corpus)))
    return [corpus, sdgs]
