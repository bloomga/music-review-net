import numpy as np
import pandas as pd
import sys
from string import punctuation
from string import digits

fname = 'p4kreviews'

df = pd.read_csv("datasets/" + fname +".csv")

#remove punctuation
df['review'] = df['review'].str.replace('[{}]'.format(punctuation), '')
df['review'] = df['review'].str.replace('[{}]'.format(digits), '')
df['review'] = df['review'].str.lower()


if 'p4k' in fname or 'pitchfork' in fname:
    for index, row in df.iterrows():
        if row['best'] == 1:
            df.it[index, 'review'] = ' '.join(row['review'].split(' ')[3:])  
            
            

