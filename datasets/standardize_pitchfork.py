import requests 
from bs4 import BeautifulSoup
import pandas as pd
import unicodedata
from sklearn.preprocessing import StandardScaler
import numpy as np
import json

fname = 'p4kreviews'
df = pd.read_csv(fname + '.csv')
print(df)

#standardize data
scaler = StandardScaler()
scores = (np.asarray(df['score'])).reshape(-1,1)
scaler.fit(scores)
std_scores = ((scaler.transform(scores)).reshape(1,-1)).tolist()
std_scores = [item for sublist in std_scores for item in sublist]


with open(p4kStandardizedScores + '.json', "w") as fp:
    json.dump(std_scores, fp)
