# Functions that return the training and validation data used for the models training

import os
import difflib
from numpy import empty
import pandas as pd
import tools
import json
import preprocess

def get_sdg_titles(refPath):
    # returns the title of each SDG as a dictionary, with key: SDGx, value = title.
    # for example: "SDG1":"No poverty"
    f = open(refPath + "SDG_titles.json")
    sdgs_title = json.load(f)
    f.close()
    return sdgs_title

def get_nature_files(refPath, preprocess=True):
    # Returns a dictionary where the keys are the name of each papers, and the values are an array where:
    # [0] = path of each file, [1] = SDGs to which the paper belongs to
    filesDict = dict()
    for folder in os.listdir(refPath):
        if os.path.isdir(refPath + folder) and not(folder == "Temp_out"):
            sdgId = int(folder.replace("SDG",""))
            folderPath = refPath + folder + "/"
            if preprocess:
                preprocess_files(folderPath)
            files = [file for file in os.listdir(folderPath) if file.endswith(".txt")]
            for file in files:
                if file in filesDict.keys():
                    filesDict[file][1].append(sdgId)
                else:
                    filesDict[file] = [folderPath + file, [sdgId]]
    if preprocess:
        # Only checks when new files are created
        check_dictionary_valid(filesDict)
        
    nFiles = len(filesDict.keys())
    print("- {} validation files were found".format(nFiles))          
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

    print("Total of: {} papers".format(len(abstractsDict)))
    return abstractsDict
    
def get_training_files(refPath, sdg=-1, abstracts=False):
    sdgsPaths = [refPath + "SDGs_description/",
                 refPath + "SDGs_progress/"
                 ]
    if sdg > 0:
        sdgStart = "{:02d}".format(sdg)
    else:
        sdgStart = "" # it's always true
    
    filesTraining = []
    for path in sdgsPaths:
        for file in os.listdir(path):
            if file.endswith(".txt") and file.startswith(sdgStart):
                f = open(path + file, 'r', encoding="utf8")
                text = f.read()
                f.close()
                filesTraining.append(text)
    if abstracts:
        # Then the abstracts are also returned
        absPath = refPath + "Abstracts/"
        for file in os.listdir(absPath):
            if file.endswith(".txt"):
                f = open(absPath + file, 'r', encoding="utf-8")
                text = f.read()
                f.close()
                filesTraining.append(text)     
            
    nFiles = len(filesTraining)
    
    print("- {} training files were found".format(nFiles))    
    return filesTraining
        
             
def preprocess_files(folderPath):
    pdfs = [file for file in os.listdir(folderPath) if file.endswith(".pdf")]
    for pdf in pdfs:
        newPdf = standarize_file_name(pdf)
        oldPath = folderPath + pdf
        newPath = folderPath + newPdf
        os.renames(oldPath, newPath)
    # Converts the pdfs to txt
    tools.pdfs2txt(folderPath)
    
    
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

