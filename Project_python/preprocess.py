# Functions that return the training and validation data used for the models training

import os
import difflib
from numpy import empty
import pandas as pd
import tools

             
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

