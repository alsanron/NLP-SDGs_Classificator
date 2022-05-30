import conf
import os
from os.path import isdir

folder2clear = ["Manual_selected/", "SDGs_Information/", "Extra_files/"]
pathRef = conf.get_paths()["ref"]


def parse_directory(path):
    for file in os.listdir(path):
        if isdir(path + file):
            newPath = path + file + "/"
            parse_directory(newPath)
        else:
            try: 
                fp = open(path + file)
                txt = fp.read(); fp.close()
                txt = txt.lower()
                txt = txt.replace("'", " ")
                txt = txt.replace("-", " ")
                txt = txt.replace("\n", " ")
                
                fp = open(path + file, "w")
                fp.write(txt)
                fp.close()
            except:
                print('## not parsed: ' + path + file)

for folder in folder2clear:
    parse_directory(pathRef + folder)
