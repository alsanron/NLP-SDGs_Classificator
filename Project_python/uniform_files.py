import conf
import os
from os.path import isdir
import pandas as pd

folder2clear = ["Manual_selected/", "SDGs_Information/", "Extra_files/"]
pathRef = conf.get_paths()["ref"]

def parse_text(txt):
    txt = txt.lower()
    txt = txt.replace("'", " ")
    txt = txt.replace("-", " ")
    txt = txt.replace("\n", " ")
    return txt


def parse_directory(path):
    for file in os.listdir(path):
        if isdir(path + file):
            newPath = path + file + "/"
            parse_directory(newPath)
        else:
            try: 
                fp = open(path + file)
                txt = fp.read(); fp.close()
                txt = parse_text(txt)
                fp = open(path + file, "w")
                fp.write(txt)
                fp.close()
            except:
                print('## not parsed: ' + path + file)
                
def parse_excel(path):
    csv = pd.read_excel(path)
    texts = list(csv["text"])
    parsedTexts = [parse_text(txt) for txt in texts]
    csv["text"] = parsedTexts
    csv.to_excel(path)

# for folder in folder2clear:
#     parse_directory(pathRef + folder)
    
parse_excel(pathRef + "test2set_thresholds.xlsx")
