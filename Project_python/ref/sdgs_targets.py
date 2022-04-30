import pandas as pd

targets = pd.read_excel("ref/targets.xlsx")
names = list(targets.iloc[:,0]); texts = list(targets.iloc[:,1])
outPath = "ref/SDGs_Information/SDGs_targets/"
for name, text in zip(names, texts):
    nameParsed = str(name).replace(".","_")
    fileName = nameParsed + "_target.txt"
    with open(outPath + fileName, 'w') as f:
        f.write(text)
        f.close()