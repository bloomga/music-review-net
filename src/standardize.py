#standardizes scores of either dataset and then outputs the scores as a json

import requests 
from bs4 import BeautifulSoup
import pandas as pd
import unicodedata
from sklearn.preprocessing import StandardScaler
import numpy as np
import json

fname = 'metacritic_reviews'
df = pd.read_csv('datasets/' + fname + '.csv')
print(df)

#standardize data
scaler = StandardScaler()
scores = (np.asarray(df['score'])).reshape(-1,1)
scaler.fit(scores)
std_scores = ((scaler.transform(scores)).reshape(1,-1)).tolist()
std_scores = [item for sublist in std_scores for item in sublist]


with open("obj/" + fname + "StandardizedScores.json", "w") as fp:
    json.dump(std_scores, fp)
