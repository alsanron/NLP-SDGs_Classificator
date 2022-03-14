# Functions that return the training and validation data used for the models training

import os
import difflib
from numpy import empty
import pandas as pd
import tools


def get_validation_files(preprocess=True, refPath="ref/Validation/"):
    filesDict = dict()
    for folder in os.listdir(refPath):
        if os.path.isdir(refPath + folder):
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



def get_training_files(refPath="ref/Training/"):
    ercData = pd.read_csv(refPath + "ERC.csv", delimiter=",")
    nFiles = len(ercData["Abstract"])
    filesTraining = []
    for ii in range(0, nFiles):
        filesTraining.append([ercData["Abstract"][ii], ercData["Author Keywords"][ii]])
    
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


# filesDict = get_validation_files(preprocess=False)
# print(filesDict.values())

