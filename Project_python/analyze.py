# script for analyzing and comparing data
import pandas as pd


path_top2vec = "out/test_abstract_top2vec.xlsx"
path_nmf = "out/test_abstract_nmf.xlsx"

top2vec = pd.read_excel(path_top2vec)
nmf = pd.read_excel(path_nmf)

texts = list(top2vec['text'])

def get_data(results):
    return [list(results['raw']), list(results['scores']), list(results['prediction'])]

df = pd.DataFrame()
df["text"] = texts
df["real"] = list(top2vec["real"])
labels = ["top2vec", "nmf"]; data = [top2vec, nmf]
for label, excel in zip(labels, data):
    raw, scores, prediction = get_data(excel)
    df["raw - " + label] = raw
    df["predic - " + label] = prediction
    
df.to_excel("out/test_abstract_comparison.xlsx")

