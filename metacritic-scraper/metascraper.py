import requests 
from bs4 import BeautifulSoup
import pandas as pd
import unicodedata
from sklearn.preprocessing import StandardScaler
import numpy as np
import json

review_dict = {'source':[], 'album':[], 'artist':[], 'date':[], 'review':[], 'score':[]}

album_list = []

for page in range(0,30):
    url = 'https://www.metacritic.com/browse/albums/release-date/available/date?page='+str(page)
    user_agent = {'User-agent': 'Mozilla/5.0'}
    response  = requests.get(url, headers = user_agent)
    soup = BeautifulSoup(response.text, 'html.parser')
    container = soup.find_all('td', class_='clamp-summary-wrap')
    for album in container:
        title = album.find('h3').text
        title = title.strip()
        title = title.lower()
        title = title.replace(' ', '-')
        title = title.replace('.','')
        title = title.replace('-&', '')
        title = title.replace("'", '')
        title = title.replace('[', '')
        title = title.replace(']', '')
        title = title.replace(':', '')
        title = title.replace('?', '')
        title = title.replace('!', '')
        title = ''.join((c for c in unicodedata.normalize('NFD', title) if unicodedata.category(c) != 'Mn'))
        artist = album.find('div', class_='artist').text
        artist = artist.strip()
        artist = artist[3:]
        artist = artist.lower()
        artist = artist.replace(' ', '-')
        artist = artist.replace('.','')
        artist = artist.replace('-&', '')
        artist = artist.replace("'", '')
        artist = ''.join((c for c in unicodedata.normalize('NFD', artist) if unicodedata.category(c) != 'Mn'))
        album_list.append((title, artist))

for album_info in album_list:
    (title, artist) = album_info
    url = 'https://www.metacritic.com/music/'+str(title)+'/'+str(artist)+'/critic-reviews'
    user_agent = {'User-agent': 'Mozilla/5.0'}
    response  = requests.get(url, headers = user_agent)
    soup = BeautifulSoup(response.text, 'html.parser')
    for review in soup.find_all('div', class_='review_content'):
        if review.find('div', class_='source') == None:
                       break
        if review.find('div', class_='source').find('a') == None:
            review_dict['source'].append(review.find('div', class_='source').text)
        else:
            review_dict['source'].append(review.find('div', class_='source').find('a').text)
        review_dict['album'].append(title)
        review_dict['artist'].append(artist)
        review_dict['date'].append(review.find('div', class_='date').text)
        review_dict['score'].append(float((review.find('div', class_='review_grade').find_all('div')[0].text)) / 10)
        if review.find('span', class_='blurb blurb_expanded'):
            review_dict['review'].append(review.find('span', class_='blurb blurb_expanded').text)
        else:
            review_dict['review'].append(review.find('div', class_='review_body').text)

#convert to csv
meta_reviews = pd.DataFrame(review_dict)
print(meta_reviews)
compression_opts = dict(method='zip', archive_name='metacritic_reviews.csv')  
meta_reviews.to_csv('metacritic_reviews.zip', index=False, compression=compression_opts) 

#standardize data
scaler = StandardScaler()
scores = (np.asarray(review_dict['score'])).reshape(-1,1)
scaler.fit(scores)
std_scores = ((scaler.transform(scores)).reshape(1,-1)).tolist()
std_scores = [item for sublist in std_scores for item in sublist]
 

with open("metacriticStandardizedScores.json, "w") as fp:
    json.dump(std_scores, fp)
