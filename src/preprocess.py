import numpy as np
import pandas as pd
import sys
from string import punctuation
from string import digits

import nltk
from nltk.corpus import stopwords

fname = 'testReviews'

df = pd.read_csv("datasets/" + fname +".csv")

#remove punctuation
df['review'] = df['review'].str.replace('[{}]'.format(punctuation), '')
df['review'] = df['review'].str.replace('[{}]'.format(digits), '')
df['review'] = df['review'].str.lower()


if 'p4k' in fname or 'pitchfork' in fname:
    for index, row in df.iterrows():
        if row['best'] == 1:
            df.it[index, 'review'] = ' '.join(row['review'].split(' ')[3:])  
            
for index, row in df.iterrows():
    #remove all stopwords except 'not'
    nonStopwords = []
    sw = stopwords.word('english')
    for word in row['review'].split(' '):
        if (not word in sw) or (word == 'not'):
            nonStopwords.append(word)
    
    #concatonate 'not's
    finalWords = ''
    x = 0
    while x < len(nonStopwords):
        if x < (len(nonStopwords) - 1):
            if nonStopwords[x] == 'not':
                finalWords += ('not ' + nonStopwords[x+1])
                x += 1
            else:
                finalWords += (nonStopwords[x] + ' ')
        else:
            finalWords += (nonStopwords[x])

    df.it[index, 'review'] = finalWords

#Save to a new csv
newfname = fname + "Preprocessed"
df.to_csv("datasets/" + newfname +".csv")

