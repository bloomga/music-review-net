import requests 
from bs4 import BeautifulSoup
import pandas as pd
import unicodedata

review_dict = {'source':[], 'date':[], 'rating':[], 'review':[]}

album_list = []

for page in range(0,1):
    url = 'https://www.metacritic.com/browse/albums/release-date/new-releases/date?page='+str(page)
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
        review_dict['date'].append(review.find('div', class_='date').text)
        review_dict['rating'].append(review.find('div', class_='review_grade').find_all('div')[0].text)
        if review.find('span', class_='blurb blurb_expanded'):
            review_dict['review'].append(review.find('span', class_='blurb blurb_expanded').text)
        else:
            review_dict['review'].append(review.find('div', class_='review_body').text)


meta_reviews = pd.DataFrame(review_dict)
print(meta_reviews)

