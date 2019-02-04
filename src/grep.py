# grep information from user's board

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options


from random import randint
from datetime import datetime
import os
import time
import pandas as pd
import numpy as np

import wget
import shutil

import unidecode
import unicodedata

from collections import Counter


import math                                                                                                                    
import nltk
import os

import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import pandas as pd
import sys

import urllib


nltk.download('stopwords')
nltk.download('punkt')



def grepPinterest(url):
    #driver = webdriver.Chrome('/home/yingru/Documents/Project/Insight/Pinterest/chromedriver')
    #driver = webdriver.Chrome('/home/ubuntu/Pinterest_final/webApp/src/chromedriver')
    #driver = webdriver.PhantomJS()

    options = webdriver.ChromeOptions()
    driver_path = '/home/ubuntu/bin/chromedriver-linux/chromedriver'
    #driver_path = '/home/ubuntu/bin/chromedriver'

    options.add_argument('--disable-dev-shm-usage')

    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--single-process')
    driver = webdriver.Chrome(driver_path, chrome_options=options)

    driver.get(url)

    time.sleep(2)
    pageOutput = driver.page_source
    soup = BeautifulSoup(pageOutput, 'xml')

    board = soup.find('div', {'data-test-id': 'pinGrid'})
    pins = board.find_all('div', {'class': 'Grid__Item'})

    pinId = []
    pinUrl = []
    pinTags = []
    pinTitle = []
    pinDescription = []
    pinOrigComment = []

    for pin in pins:
        try:
            dum = pin.find('div', {'class': 'GrowthUnauthPinImage'})
            pinId.append(dum.find('a')['href'])
            pinUrl.append(dum.find('a').find('div').find()['src'])

            _dumC = dum.find('a')['title']
            #pinOrigComment.append(str(_dumC).encode('utf-8'))
            _dumC = str(_dumC).encode('ascii', 'ignore')
            pinOrigComment.append(_dumC.decode('ascii'))

            _dumT = pin.find('h3').text
            #pinTitle.append(str(_dumT).encode('utf-8'))
            _dumT = str(_dumT).encode('ascii', 'ignore')
            pinTitle.append(_dumT.decode('ascii'))


            dum = pin.find('div', {'data-test-id': 'vasetags'})
            _tags = ''
            for _ in dum.find_all('a')[:-1]:
                _tags += _.text + ','
                #_tags.append(_.text)
            #pinTags.extend(_tags)
            pinTags.append(_tags)

            #pinDescription.append(str(_desc).encode('utf-8'))
            
            _desc = pin.find('p', {'data-test-id': 'desc'}).text
            _desc = str(_desc).encode('ascii', 'ignore')
            pinDescription.append(_desc.decode('ascii'))
        
        except:
            continue

    data = pd.DataFrame([pinId, pinUrl, pinTags, pinTitle, pinDescription], 
                        index = ['pinId', 'url', 'tags', 'title', 'description'])

    driver.close()
    return data.T

    #return pinId, pinUrl, pinTags, pinDescription, pinTitle, pinOrigComment




def grepSearch(query, scope='pins'):
    """
    search a query based on key works on pinterest
    """
    q = '%20'.join(query.split())
    url = 'https://www.pinterest.com/search/{}/?q={}&rs=typed'.format(scope, q)
    #print(url)
    return grepPinterest(url)





def grepImage(pinUrl, desDir):
    """
    given ain list of url, download the images to folder
    """
    if not os.path.exists(desDir):
        os.mkdir(desDir)
        
    for url in pinUrl:
        try:
            os.system('wget {}'.format(url))
            name = url.split('/')[-1]
            shutil.move(name, '{}/{}'.format(desDir, name))
        except:
            continue


#########################################################################
# now analyze the grepped labels
def remove_string_special_characters(s):
    stripped = re.sub('[^\w\s]', '', s)
    stripped = re.sub('_', '', stripped)
    
    stripped = re.sub('\s+', ' ', stripped)
    stripped = stripped.strip()
    return stripped

    

def NaiveBayesCount(data):
    """
    data: pandas.DataFrame [pinId, url, tags, title, description]
    """
    stop_words = set(stopwords.words('english'))

    sentence = '.'.join([i for i in data.description if i])
    sentence = sent_tokenize(sentence)
    sentenceClean = [remove_string_special_characters(s) for s in sentence]


    total_words = []
    for sent in sentenceClean:
        words = word_tokenize(sent)
        filter_words = [w.lower() for w in words if not w in stop_words]
        total_words += filter_words

    counts = dict(Counter(total_words))
    result = pd.DataFrame(counts, index=['freq']).T
    result.sort_values(by=['freq'], axis=0, ascending=False, inplace=True)
    return result





if __name__ == '__main__':
    userName = input('Please enter your username: ')
    boardName = input('Please enter your board name: ')
    url =  'https://www.pinterest.com/{}/{}/'.format(userName, boardName)
    folder = './board/{}'.format(boardName)
    data = grepPinterest(url)
    print('pinId: ', data.pinId)
    print('pinTags: ', data.tags)
    print(' ')
    print('pinDescription: ', data.description)
    print('pinTitle: ', data.title)
    result = NaiveBayesCount(data)
    print(result)
