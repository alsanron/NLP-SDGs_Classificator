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

texts = ["excess mortality (mort) in china due to exposure to ambient fine particulate matter with aerodynamic diameter 2.5 m (pm2.5) was determined using an ensemble prediction of annual average pm2.5 in 2013 \
         by the community multiscale air quality (cmaq) model with four emission inventories and observation data fusing. estimated mort values due to adult ischemic heart disease, cerebrovascular disease, chronic \
         obstructive pulmonary disease, and lung cancer are 0.30, 0.73, 0.14, and 0.13 million in 2013, respectively, leading to a total mort of 1.3 million. source oriented cmaq modeling determined that industrial \
         and residential sources were the two leading sources of mort, contributing to 0.40 (30.5\%) and 0.28 (21.7\%) million deaths, respectively. additionally, secondary ammonium ion from agriculture, secondary organic aerosol, \
        and aerosols from power generation were responsible for 0.16, 0.14, and 0.13 million deaths, respectively. a 30\% mort reduction in china requires an average of 50\% reduction of pm2.5 throughout the country and a reduction \
        by 62\%, 50\%, and 38\% for the beijing tianjin hebei, jiangsu zhejiang shanghai, and pearl river delta regions, respectively. \
         reducing pm2.5 to the caaqs grade ii standard of 35 g m 3 would only lead to a small reduction in mortality, and a more stringent standard of lt;15 g m 3 would be needed for more remarkable reduction of mort.",
         
         "thanks to the continuous improvement of calculation resources, computational fluid dynamics (cfd) is expected to provide in the next few years a cost effective and accurate tool to improve the understanding of \
        the unsteady aerodynamics of darrieus wind turbines. this rotor type is in fact increasingly welcome by the wind energy community, especially in case of small size applications and/or non conventional installation sites.\
        in the present study, unique tow tank experimental data on the performance curve and the near wake structure of a darrieus rotor were used as a benchmark to validate the effectiveness of different cfd approaches. in particular, \
        a dedicated analysis is provided to assess the suitability, the effectiveness and the future prospects of simplified two dimensional (2d) simulations. the correct definition of the computational domain, the selection of the \
        turbulence models and the correction of simulated data for the parasitic torque components are discussed in this study. results clearly show that, (only) if properly set, two dimensional cfd simulations are able to provide \
        with a reasonable computational cost an accurate estimation of the turbine performance and also quite reliably describe the attended flow field around the rotor and its wake",
        
        
        "scramjet is found to be the efficient method for the space shuttle. in this paper, numerical simulation is performed to investigate the fundamental flow physics of the interaction between an array of fuel jets and multi \
        air jets in a supersonic transverse flow. hydrogen as a fuel is released with a global equivalence ratio of 0.5 in presence of micro air jets on a flat plate into a mach 4 crossflow. the fuel and air are injected through \
        streamwise aligned flush circular portholes. the hydrogen is injected through 4 holes with 7dj space when the air is injected in the interval of the hydrogen jets. the numerical simulation is performed by using the reynolds \
        averaged navier stokes equations with menters shear stress transport (sst) turbulence model. both the number of air jets and jet to freestream total pressure ratio are varied in a parametric study. the interaction of the fuel \
        and air jet in the supersonic flow present extremely complex feature of fuel and air jet. the results present various flow features depending upon the number and mass flow rate of micro air jets. these flow features were found \
        to have significant effects on the penetration of hydrogen jets. a variation of the number of air jets, along with the jet to freestream total pressure ratio, induced a variety of flow structure in the downstream of the fuel jets."
         
         "recharge assessment is of critical importance for groundwater resources evaluation in arid/semiarid areas, as these have typically limited surface water resources. there are several models for water balance evaluation. one of \
        them is wetspass, which has the ability to simulate spatially distributed recharge, surface runoff, and evapotranspiration for seasonally averaged conditions. this paper presents a modified methodology and model, wetspass m, in \
        which the seasonal resolution is downscaled to a monthly scale. a generalized runoff coefficient was introduced, enabling runoff estimation for different land use classes. wetspass m has been calibrated and validated with \
        observed streamflow records from black volta. base flow from simulated recharge was compared with base flow derived via a digital filter applied to the observed streamflow and has shown to be in agreement. previous studies \
        have concluded that for this basin, small changes in rainfall could cause a large change in surface runoff, and here a similar behavior is observed for recharge rates. an advantage of the new model is that it is applicable to \
        medium and large sized catchments. it is useful as an assessment tool for evaluating the response of hydrological processes to the changes in associated hydrological variables. since monthly data for streamflow and climatic \
        variables are widely available, this new model has the potential to be used in regions where data availability at high temporal resolution is an issue. the spatial temporal characteristics of the model allow distributed \
        quantification of water balance components by taking advantage of remote sensing data."
         ]

print('# Loading models...')
paths = conf.get_paths()
model = model_global.Global_Classifier(paths=paths, verbose=True)
model.load_models()

ntop_words = 20
models = ["NMF", "LDA", "Top2Vec", "BERTopic"]
print('# Analysing texts')
for text, ii in zip(texts, range(len(texts))):
    words_collection = []
    
    words_scores = model.nmf.map_text_to_topwords(text, ntop_words)
    words_collection.append(words_scores)
    
    words_scores = model.lda.map_text_to_topwords(text, ntop_words)
    words_collection.append(words_scores)
    
    words_scores = model.top2vec.map_text_to_topwords(text, ntop_words)
    words_collection.append(words_scores)
    
    words_scores = model.bertopic.map_text_to_topwords(text, ntop_words)
    words_collection.append(words_scores)
    
    df = pd.DataFrame()
    for col, index in zip(words_collection, range(len(words_collection))):
        column_data = ["{:.4f}:{}".format(pair[1], pair[0]) for pair in words_collection[index]]
        df[models[index]] = column_data
    df.to_csv(paths["out"] + "All/" + "word_collection{}.csv".format(ii))
