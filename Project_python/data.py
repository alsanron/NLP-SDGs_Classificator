# File that contains the functions for accesing the required data: training and validation

import os
import difflib
from numpy import empty
import pandas as pd
import json
import preprocess

def get_sdg_titles(refPath):
    # returns the title of each SDG as a dictionary, with key: SDGx, value = title.
    # for example: "SDG1":"No poverty"
    f = open(refPath + "SDG_titles.json")
    sdgs_title = json.load(f)
    f.close()
    return sdgs_title

# DATASET: the role of artificial intelligence in achieving the sustainable development goals
def get_nature_files(refPath, flag_preprocess=False):
    # Returns a dictionary where the keys are the name of each papers, and the values are an array where:
    # [0] = path of each file, [1] = SDGs to which the paper belongs to
    filesDict = dict()
    for folder in os.listdir(refPath):
        if os.path.isdir(refPath + folder) and not(folder == "Temp_out"):
            sdgId = int(folder.replace("SDG",""))
            folderPath = refPath + folder + "/"
            if flag_preprocess:
                preprocess.preprocess_files(folderPath)
            files = [file for file in os.listdir(folderPath) if file.endswith(".txt")]
            for file in files:
                if file in filesDict.keys():
                    filesDict[file][1].append(sdgId)
                else:
                    filesDict[file] = [folderPath + file, [sdgId]]
    if flag_preprocess:
        # Only checks when new files are created
        preprocess.check_dictionary_valid(filesDict)
        
    nFiles = len(filesDict.keys())
    print("- Total of {} nature files were found".format(nFiles))          
    return filesDict

def get_nature_abstracts(refPath):
    # returns a dictionary where the keys are the name of each paper, and the values are an array where:
    # [0] = abstract of each file, [1] = SDGs to which the paper belongs to
    # warning: only returns those abstracts whose associated paper has been found classified in any of the SDGs folder
    cites = pd.read_csv(refPath + "cites_nature.csv")

    def get_closest_file(filesDict, fileName):
        # Checks if 2 files have a very close name. This generally avoids having to compare all texts
        closestName = difflib.get_close_matches(fileName, filesDict.keys(), n=1, cutoff=0.8)
        return closestName

    papers = []
    for [title, abstract] in zip(cites['Title'], cites['Abstract']):
        if title == "[No title available]" or abstract == "[No abstract available]":
            continue
        else:
            papers.append((preprocess.standarize_file_name(title) + ".txt", abstract))
    
    validationDict = get_nature_files(refPath=refPath)
    abstractsDict = dict()
    for paper in papers:
        getClose = get_closest_file(validationDict, paper[0])
        
        if len(getClose) == 1:
            fileName = getClose[0]
            dictValues = validationDict[fileName]
            abstractsDict[fileName] = [paper[1], dictValues[1]]

    print("Total of: {} abstracts from nature".format(len(abstractsDict)))
    return abstractsDict
    
# DATASET: https://sdgs.un.org/
# - Goals definition
# - Goals progress - evolution section
def get_sdgs_org_files(refPath, sdg=-1):
    # Returns an array where each elements consist of an array with the fields:
    # [0] abstract or text related to a SDG, [1]: array with the associated SDGs.
    sdgsPaths = [refPath + "SDGs_description/",
                 refPath + "SDGs_progress/"
                 ]
    if sdg > 0:
        sdgStart = "{:02d}".format(sdg)
    else:
        sdgStart = "" # it's always true
    
    files = []
    for path in sdgsPaths:
        for file in os.listdir(path):
            if file.endswith(".txt") and file.startswith(sdgStart):
                f = open(path + file, 'r', encoding="utf8")
                text = f.read()
                f.close()
                fileSDG = int(file.partition("_")[0])
                files.append([text, [fileSDG]])
      
    nFiles = len(files)
    
    print("- {} sdgs files were found".format(nFiles))    
    return files

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
    data = []
    for [text, sdgsAscii] in zip(csv["text"], csv["SDGs"]):
        sdgsInt = [int(sdg) for sdg in (sdgsAscii.replace("[","").replace("]","")).split(",")]
        data.append([text, sdgsInt])
    print("- {} texts in the pathfinder dataset".format(len(data)))
    return data