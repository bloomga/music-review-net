import pandas as pd
import numpy as np
import operator
import json
import sys

#to only be run on non standardized score review csv's as they have same reviews as standardized score reviews
#we do this step after preprocessing, but before training. 

#class containing the word frequency dictionary and its correspinding score
#unused as of now
class review:
    def __init__(self, wordFreq, score,ID):
        self.dict = dictionary
        self.score = score
        self.id = ID

fname = str(sys.argv[1])
df = pd.read_csv("datasets/" + fname + '.csv')


dictionary = {}
low_dict = {}
high_dict = {}
low_bound = 4
high_bound = 7


for index, row in df.iterrows():

    text = row['review']

    #confirm that the review is not empty
    if isinstance(text, str):
        pass
    else:
        text = " "
        df.at[index, 'review'] = text
    text = text.split()
    #determine low info words and create word count dicts
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

print(low_info_words)  
#creates word to integer encoding dict
dictionary = dict( sorted(dictionary.items(), key=operator.itemgetter(1),reverse=True))
ordered_key_list = list(dictionary.keys())
word_to_int = {w:i+1 for i, w in enumerate(ordered_key_list)}

scores = list(df['score'])

#encodes reviews
encoded_reviews = list()
for index, row in df.iterrows():
    encoded_review = list()
    text = row['review']
    text = text.split()
    count = 0
    for word in text:
        if count == 250:
            break
        #dont include low info words in encoded review
        if word in low_info_words:
            pass
        elif word in word_to_int.keys():
            encoded = word_to_int[word]
            count += 1
            encoded_review.append(encoded)
    #pads with zeroes so reviews all have same length
    while(count < 250):
        encoded_review.append(0)
        count += 1
    encoded_reviews.append(encoded_review)

#save lists of encoded reviews and encoding dict
with open("obj/encoded" + fname + '.json', "w") as fp:
    json.dump(encoded_reviews, fp)
with open("obj/" + fname + 'Scores.json', "w") as fp:
    json.dump(scores, fp)
with open("obj/" + fname + 'Dict.json', "w") as fp:
    json.dump(word_to_int, fp)

