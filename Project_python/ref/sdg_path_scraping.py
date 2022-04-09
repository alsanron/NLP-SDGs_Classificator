
# Web scrapping code for obtaining all the files related to each SDG that are classified in the SDG-pathfinder page:
# ref: https://sdg-pathfinder.org/
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import difflib

outputPath = "ref/ds_sdg_path_finder.csv"
checkForDuplicates = False

sdgPaths = ["no-poverty",
            "zero-hunger",
            "good-health",
            "quality-education",
            "gender-equality",
            "clean-water",
            "affordable-energy",
            "decent-work-growth",
            "industry-innovation-and-infrastructure",
            "reduced-inequalities",
            "sustainable-cities",
            "responsible-consumption",
            "climate-action",
            "life-below-water",
            "life-on-land",
            "peace-justice-and-strong-institutions",
            "partnerships-for-the-goals"            
            ]
basePath = "https://sdg-pathfinder.org/sdg/"
delayOpenPage = 10 # time delay to open the web page s
showMore_class = "flex.font-rubik.font-bold.cursor-pointer"

sdgs = ["No poverty",
        "Zero hunger",
        "Good health and well-being",
        "Quality education",
        "Gender equality",
        "Clean water and sanitation",
        "Affordable and clean energy",
        "Decent work and economic growth",
        "Industry, innovation and infrastructure",
        "Reduced inequalities",
        "Sustainable cities and communities",
        "Responsible consumption and production",
        "Climate action",
        "Life below water",
        "Life on land",
        "Peace, justice and strong institutions",
        "Partnerships for the goals"
        ]

# Opens google chrome
driver = webdriver.Chrome(ChromeDriverManager().install())
time.sleep(2)

# For each abstract, it obtains the text and the associated SDGs
texts = []
associatedSDGs = []

for path in sdgPaths:
    absPath = basePath + path
    print('############# Path: {} being fetched'.format(absPath))
    driver.get(absPath)
    time.sleep(delayOpenPage)  # Allow xx seconds before opening the webpage
    iter = 1
    while True:
        show_more_buttons = driver.find_elements_by_class_name(showMore_class)
        last_button = show_more_buttons[-1]
        
        if iter == 1:
            n_new = len(show_more_buttons)
            n_prev = n_new
        else:
            n_new = len(show_more_buttons)
            if n_new > n_prev:
                # then its fine, new texts have been found
                n_prev = n_new
            else:
                break
            
        if last_button.aria_role == 'button':
            while last_button.is_displayed() == False:
                time.sleep(0.5)
                print('Waiting until button is displayed')
        last_button.click()
        time.sleep(1) # delay time between show more buttons
        iter += 1
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    print('# Page expansion completed')

    for abstract in soup.find_all(class_="abstract"):
        abs = abstract.text
        auxItems = abstract.parent.parent.parent.find_all(class_="flex")
        textSDGs = []
        for ii in range(0, len(auxItems)):
            text = auxItems[-(ii + 1)].text
            if text in sdgs:
                textSDGs.append(sdgs.index(text) + 1)
            else:
                break
        if textSDGs.count(textSDGs[0]) == len(textSDGs):
            # then the abstract is associated to a single SDG
            textSDGs = [textSDGs[0]]
            
        if checkForDuplicates:
            closesText = difflib.get_close_matches(abs, texts, n=1, cutoff=0.85)
            if len(closesText) == 0:
                # if the text hasnt already been stored, then it is appended
                texts.append(abs)
                associatedSDGs.append(textSDGs)
        else:
            if abs not in texts:
                # if the text hasnt already been stored, then it is appended
                texts.append(abs)
                associatedSDGs.append(textSDGs)
    print('# Abstract extraction completed')
            
df = pd.DataFrame()
df['text'] = texts
df['SDGs'] = associatedSDGs
df.to_csv(outputPath)

print('## FINISH: {} TEXTS HAVE BEEN IDENTIFIED'.format(len(texts)))
