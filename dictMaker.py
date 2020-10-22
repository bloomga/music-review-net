import pandas as pd
import numpy as np
import operator
import json

#we do this step after preprocessing, but before training. 
#class containing the word frequency dictionary and its correspinding score
class review:
    def __init__(self, wordFreq, score,ID):
        self.dict = dictionary
        self.score = score
        self.id = ID

fname = 'testReviews'
df = pd.read_csv(fname + '.csv')

#will return a dictionary of word fequencies for each review
#TODO ignore empty entries
#to parse entire data sheet set range to 1766
dictionary = {}
low_dict = {}
high_dict = {}
low_bound = 4
high_bound = 7

for index, row in df.iterrows():

    text = row['review']
    
    if isinstance(text, str):
        pass
    else:
        text = " "
        row['review'] = text
    text = text.split()
    #determine low info words and create freq dict
    for j in text:
        dictionary[j] = dictionary.get(j, 0)+1
        if row['score'] >= high_bound:
            high_dict[j] = high_dict.get(j, 0)+1
        elif row['score'] <= low_bound:
            low_dict[j] = low_dict.get(j, 0)+1
        

#remove low info words from dictionary and reviews
low_info_words = []
for key in high_dict:
    if key in low_dict:
        info_score = high_dict[key] / low_dict[key]
        if info_score <= 2 and info_score >= 0.5:
            dictionary.pop(key)
            low_info_words.append(key)
    
    
sorted_dict = dict( sorted(dictionary.items(), key=operator.itemgetter(1),reverse=True))
ordered_key_list = list(sorted_dict.keys())
word_to_int = {w:i+1 for i, w in enumerate(ordered_key_list)}

scores = list(df['score'])

encoded_reviews = list()
for index, row in df.iterrows():
    encoded_review = list()
    text = row['review']
    text = text.split()
    count = 0
    for word in text:
        if count == 250:
            break
        if word in low_info_words:
            pass
        elif word in word_to_int.keys():
            encoded = word_to_int[word]
            count += 1
            encoded_review.append(encoded)
    while(count < 250):
        encoded_review.append(0)
    encoded_reviews.append(encoded_review)
print(encoded_reviews)
print(scores)
print(word_to_int)
#save lists of encoded reviews and encoding dict

with open("encoded" + fname + '.json', "w") as fp:
    json.dump(encoded_reviews, fp)
with open("encoded" + fname + 'Scores.json', "w") as fp:
    json.dump(scores, fp)
with open(fname + 'Dict.json', "w") as fp:
    json.dump(word_to_int, fp)

