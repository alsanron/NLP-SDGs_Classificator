# Analysis of the results obtained with the Aero database

# Configures the project paths: they can be launched from any code
from cProfile import label
from pkgutil import iter_importers
import sys, os
sys.path.append(os.path.realpath('.'))
import conf
conf.import_paths()

# Configuration flags
identify_sdgs = True # true: all the texts are identified, false: it used previous stored data

# Imports required to work properly
from logging import error
import data
import conf
import pandas as pd
import model_global
import numpy as np
import tools
import matplotlib.pyplot as plt
import warnings
import os

# Loads all the datasets
print('# Loading datasets...')
paths = conf.get_paths()
ds_train = data.get_dataset(requires_update=False, filter=["org", "manual_extra"])
ds_valid_short = data.get_dataset(requires_update=False, filter=["nature_abstract"])
ds_valid_long = data.get_dataset(requires_update=False, filter=["nature_all"])

raw_orgFiles = ds_train["standard"]
raw_natureShort = ds_valid_short["standard"]
raw_natureExt = ds_valid_long["standard"]
   
orgFiles = ds_train["lem"]; sdgs_org = ds_train["sdgs"]
natureShort = ds_valid_short["lem"]; sdgs_natureShort = ds_valid_short["sdgs"]
natureLong = ds_valid_long["lem"]; sdgs_natureLong = ds_valid_long["sdgs"]


# Generate plot comparing the for results.
nmf = pd.read_excel(paths["out"] + "All/Individual/" + "test_nature_short_nmf.xlsx")
nmf = tools.parse_sdgs_ascii_list(list(nmf["prediction"]), append_always=True)

lda = pd.read_excel(paths["out"] + "All/Individual/" + "test_nature_short_lda.xlsx")
lda = tools.parse_sdgs_ascii_list(list(lda["prediction"]), append_always=True)

top2vec = pd.read_excel(paths["out"] + "All/Individual/" + "test_nature_short_top2vec.xlsx")
top2vec = tools.parse_sdgs_ascii_list(list(top2vec["prediction"]), append_always=True)

bertopic = pd.read_excel(paths["out"] + "All/Individual/" + "test_nature_short_bertopic.xlsx")
bertopic = tools.parse_sdgs_ascii_list(list(bertopic["prediction"]), append_always=True)

all_sdgs = [nmf, lda, top2vec, bertopic]

positions = [-0.15, -0.05, 0.05, 0.15]
labels = ["nmf", "lda", "top2vec", "bertopic"]
colors = ["green", "red", "orange", "blue"]
xlabel = [ii for ii in range(1, 18)]

plt.figure(figsize=(8, 8))
for ii in range(4):
    ok, nok = tools.get_ok_nok_SDGsidentified(sdgs_natureShort, all_sdgs[ii])
    for jj in range(16):
        xx = xlabel[jj]
        plt.bar(xx + positions[ii], ok[xx - 1] + nok[xx - 1], width=0.1, alpha=0.5, color=colors[ii])
        plt.bar(xx + positions[ii], ok[xx - 1]              , width=0.1, alpha=1.0, color=colors[ii])
    xx = xlabel[-1]
    plt.bar(xx + positions[ii], ok[xx - 1]              , width=0.1, alpha=1.0, color=colors[ii], label=labels[ii])

plt.xticks(xlabel)
plt.xlabel('SDG')
plt.ylabel("Number of times a SDG is identified")
# plt.ylim(top=0.5)
# plt.title('SDGs to identify: {}'.format(labeledSDGs[textIndex]))
plt.legend()
plt.savefig(paths["out"] + "All/Individual/sdgs_model_nature_short.png")
# plt.show()
