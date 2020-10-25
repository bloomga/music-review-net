import numpy as np
import pandas as pd
import sys
from string import punctuation
from string import digits
import nltk

#to get this nltk dataset, first we must download it
#only needs to be done once for machine
try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

fname = 'metacritic_reviews_test'

df = pd.read_csv("datasets/" + fname +".csv")

#remove punctuation
df['review'] = df['review'].str.replace('[{}]'.format(punctuation), '')
df['review'] = df['review'].str.replace('[{}]'.format(digits), '')
df['review'] = df['review'].str.lower()
        
for index, row in df.iterrows():
    #remove "best new music" or "best new reissue" from bnm/bnr p4k reviews
    if 'p4k' in fname or 'pitchfork' in fname:
        if row['best'] == 1:
            set_row = ' '.join(row['review'].split(' ')[3:])
            df.at[index, 'review'] = set_row 
            row['review'] = set_row

    #reduce length of review to 300
    #this will be standardized later to 250 words once low-info words removed
    #this also drastically speeds up the loops below
    #remove all stopwords except 'not'
    nonStopwords = []
    count = 0
    limit = 300
    sw = stopwords.words('english')
    for word in row['review'].split(' '):
        if count == limit:
            break
        if word not in sw or word == "not":
            nonStopwords.append(word)
            if word == 'not':
                limit += 1
  
    #concatonate 'not's
    finalWords = ''
    flag = False
    for x in range(len(nonStopwords)-1):
        if flag == False:
            if nonStopwords[x] == 'not':
                finalWords += ('not' + nonStopwords[x+1] + ' ')
                flag = True
            else:
                finalWords += (nonStopwords[x] + ' ')
        else:
            flag = False
    finalWords += (nonStopwords[x+1])

    df.at[index, 'review'] = finalWords


#Save to a new csv
newfname = fname + "Preprocessed"
df.to_csv("datasets/" + newfname +".csv")

