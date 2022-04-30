# File for removing the non-useful words from each topic
import pandas as pd
import conf

paths = conf.get_paths()
# csvPath = paths["out"] + "topics_nmf_global_monogram.csv"
# csv = pd.read_csv(csvPath)
# stopWords = []
# for ii in range(1, 18):
#     sdg = csv.columns[ii]
#     words = list(csv.iloc[:, ii])
    
#     while 1:
#         print('#### ', sdg)
#         for word in words:
#             print('({}) {}'.format(words.index(word), word))
        
#         usrInput = input('Enter a word number to remove: ')
#         if usrInput.isnumeric():
#             index = int(usrInput)
#             word2remove = words[index]
#             if not(word2remove in stopWords):
#                 stopWords.append(word2remove)
#             words.remove(word2remove)
#             print('- word: {} removed'.format(word2remove))
#         else:
#             if len(usrInput) == 0:
#                 break
        
words = 'cent, people, day, person, cash, line, million, height, case, year, minimum, lower, million, half, data, stress, managed, people, billion, basin, final, billion, increased, likely, real, caput, average, share, high, intensity, medium, billion, remittance, total, average, caput, ton, use, tonne, party, stockholm, determined, april, communicated, nation, cent, small, use, key, index, form, data, birth, year, half'
wordsList = [word for word in words.replace(',', '').split(' ')]  
wordsNew = []
for word in wordsList:
    if not(word in wordsNew):
        wordsNew.append(word)
with open(paths["ref"] + "stop_words.txt", 'w') as f:
    f.write(' '.join(wordsNew))
    f.close()
        