# Comparison of the text to avoid repetition
import os
from difflib import SequenceMatcher

files = []
names = []

for folder in os.listdir():
    if os.path.isdir(folder):
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                f = open("{}/{}".format(folder, file), 'r', encoding='utf8')
                text = f.read()
                f.close()
                files.append(text)
                names.append(file)
                

for ii in range(0, len(files)):
    counter = 0
    for jj in range(0, len(files)):
        if files[ii] == files[jj]:
            if names[ii] != names[jj]:
                print("Warning found: {} -> {}".format(names[ii], names[jj]))

newFiles = []
for ii in range(0, len(files)):
    if len(newFiles) > 0:
        count = 0
        for file in newFiles:
            if file == files[ii]:
                # The files has already been registered
                count += 1
                break
        if count == 0:
            newFiles.append(files[ii])
    else:
        newFiles.append(files[ii])

print('Total of {} files'.format(len(newFiles)))