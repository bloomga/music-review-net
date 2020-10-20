import numpy as np
import pandas as pd
import sys
from string import punctuation

#fname = sys.argv[0]
fname = 'p4kreviews.csv'

df = pd.read_csv(fname)

#remove punctuation
df['review'] = df['review'].str.replace('[{}]'.format(punctuation), '')
df['review'] = df['review'].str.lower()


if 'p4k' in fname or 'pitchfork' in fname:
    for index, row in df.iterrows():
        if row['best'] == 1:
            row['review'] = ' '.join(row['review'].split(' ')[3:])  
            
            

