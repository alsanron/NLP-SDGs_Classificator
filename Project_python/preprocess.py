# Functions that return the training and validation data used for the models training

import os
import difflib
from numpy import empty
import pandas as pd
import tools
import temp

def get_validation_files(refPath, preprocess=True):
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
    
    abstractsFiles = temp.get_validation_files(refPath)
    newDict = dict()
    for paperName, abstract in abstractsFiles:
        newDict[paperName] = filesDict[paperName]
    
    
    # nFiles = len(filesDict.keys())
    # print("- {} validation files were found".format(nFiles))          
    # return filesDict
    nFiles = len(newDict.keys())
    print("- {} validation files were found".format(nFiles))          
    return newDict



def get_training_files(refPath, sdg=-1, abstracts=False):
    sdgsPath = refPath + "SDGs_description/"
    if sdg > 0:
        sdgStart = "{:02d}".format(sdg)
    else:
        sdgStart = "" # it's always true
    
    filesTraining = []
    for file in os.listdir(sdgsPath):
        if file.endswith(".txt") and file.startswith(sdgStart):
            f = open(sdgsPath + file, 'r', encoding="ascii")
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

    # First renames the files accordingly
    symbols = [",", " ", "&", ":", "-","__"]
    for pdf in pdfs:
        newPdf = pdf.lower()
        for symbol in symbols:
            newPdf = newPdf.replace(symbol, "_")
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

