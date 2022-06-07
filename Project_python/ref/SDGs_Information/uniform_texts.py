# Uniforms all the files to have the same number of words
import pandas as pd
import os

target_words_per_text = 150 # it will try to get a text with the closest number of words >= 150

list_texts = [[] for ii in range(18)] # 17 lists, 1 per sdg
texts = ["" for ii in range(18)]
for dir in os.listdir():
    if os.path.isdir(dir):
        for file in os.listdir(dir):
            path = "{}/{}".format(dir, file)
            if not file.endswith(".txt"): continue
            with open(path, 'r') as fp:
                txt = fp.read(); fp.close()
                sdg = int(file.split("_")[0])
                texts[sdg - 1] += " " + txt

def add_sentence_to_text(text, sentence):
    sentence = sentence.replace("’", "'")
    text += sentence + ". "
    return text

def count_words(text):
    for char in '-.,;\n':
        text = text.replace(char,' ')
    word_list = [word for word in text.split(' ') if not word.isnumeric()]
    return len(word_list)

df_texts = []; df_sdg = [] # list of texts and sdgs to save
for text, sdg in zip(texts, range(1,18)):
    sentences = text.split('.'); nSentences = len(sentences)
    
    tmp = ""
    for sentence, index in zip(sentences, range(nSentences)):
        if (nSentences - (index + 1)) <= 2:
            tmp = add_sentence_to_text(tmp, sentence)
        elif count_words(tmp) >= target_words_per_text:
            df_texts.append(tmp); df_sdg.append([sdg])
            tmp = ""
        else:
            tmp = add_sentence_to_text(tmp, sentence)
    df_texts.append(tmp); df_sdg.append([sdg])
df = pd.DataFrame()
df["text"] = df_texts
df["sdg"] = df_sdg
df.to_excel('sdg_texts.xlsx')
            