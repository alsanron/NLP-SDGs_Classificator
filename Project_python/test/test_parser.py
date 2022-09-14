import pandas as pd


file_id = 2
models = ["NMF", "LDA", "Top2Vec", "BERTopic"]
with open("test/data.txt") as fp:
    df = pd.DataFrame()
    for ii in range(4):
        sdgs = []
        line = fp.readline()
        for sdg in line[line.find('>')+1:].split('|'):
            score = float(sdg[sdg.find(':')+1:])
            sdgs.append(score)
        df[models[ii]] = sdgs
df = df.transpose()
df.to_csv("test/data_parsed{}.csv".format(file_id))