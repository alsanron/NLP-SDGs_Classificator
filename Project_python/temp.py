# Loads the abstract cited in the nature publication
import pandas as pd
import os
import difflib

def get_validation_files(ref_path):

    cites = pd.read_csv(ref_path + "cites_nature.csv")

    def preprocess_name(file_name):
        # First renames the files accordingly
        symbols = [",", " ", "&", ":", "-","__"]
        newName = file_name.lower()
        for symbol in symbols:
            newName = newName.replace(symbol, "_")
        return newName

    def get_validation_files(refPath):
        # Returns a dictionary where the keys are the name of each papers, and the values are an array where:
        # [0] = path of each file, [1] = SDGs to which the paper belongs to
        filesDict = dict()
        for folder in os.listdir(refPath):
            if os.path.isdir(refPath + folder) and not(folder == "Temp_out"):
                sdgId = int(folder.replace("SDG",""))
                folderPath = refPath + folder + "/"
                files = [file for file in os.listdir(folderPath) if file.endswith(".txt")]
                for file in files:
                    if file in filesDict.keys():
                        filesDict[file][1].append(sdgId)
                    else:
                        filesDict[file] = [folderPath + file, [sdgId]]
        nFiles = len(filesDict.keys())
        print("- {} validation files were found".format(nFiles))          
        return filesDict

    def get_closest_file(filesDict, fileName):
        # Checks if 2 files have a very close name. This generally avoids having to compare all texts
        closestName = difflib.get_close_matches(fileName, filesDict.keys(), n=1, cutoff=0.8)
        # print("File: {} -> Aprox: {}".format(fileName, closestName))
        return closestName

    papers = []
    for [title, abstract] in zip(cites['Title'], cites['Abstract']):
        if title == "[No title available]" or abstract == "[No abstract available]":
            continue
        else:
            papers.append((preprocess_name(title) + ".txt", abstract))
    
    validationDict = get_validation_files(refPath=ref_path)
    newPapers = []
    for paper in papers:
        # value = validationDict[paper[0]]
        getClose = get_closest_file(validationDict, paper[0])
        
        if len(getClose) == 1:
            newPapers.append((getClose[0], paper[1]))

    print("Total of: {} papers".format(len(newPapers)))
    return newPapers
