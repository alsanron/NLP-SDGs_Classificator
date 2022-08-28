# Configures the project paths: they can be launched from any code
from pkgutil import iter_importers
import sys, os
sys.path.append(os.path.realpath('.'))
import conf
conf.import_paths()


# Real imports required by the file for work properly
from logging import error
import data
import conf
import pandas as pd
import model_global
import numpy as np
import tools


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

print('# Loading models...')
model = model_global.Global_Classifier(paths=paths, verbose=True)
model.load_models()

print('# Testing train dataset...')
predic, scores = model.test_model(raw_corpus=raw_orgFiles, corpus=orgFiles, associated_SDGs=sdgs_org, 
                 path_to_plot="", 
                 path_to_excel=paths["out"] + "All/test_training.xlsx", 
                 only_bad=False, score_threshold=-1,  only_positive=True, filter_low=True)
tools.plot_ok_vs_nok_SDGsidentified(sdgs_org, predic, paths["out"] + "All/" + "sdgs_train.png")


print('# Testing validation dataset...')
predic, scores = model.test_model(raw_corpus=raw_natureShort, corpus=natureShort, associated_SDGs=sdgs_natureShort, 
                 path_to_plot="", 
                 path_to_excel=paths["out"] + "All/test_nature_short.xlsx", 
                 only_bad=False, score_threshold=-1,  only_positive=True, filter_low=True)
tools.plot_ok_vs_nok_SDGsidentified(sdgs_natureShort, predic, paths["out"] + "All/" + "sdgs_test.png")
