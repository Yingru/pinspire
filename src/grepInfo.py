'''
A class to scraping information from Pinterest through headless chrome
'''

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options

from collections import Counter
from random import randint
from datetime import datetime
import os, sys
import time
import math
import shutil
import string
import re

import pandas as pd
import numpy as np
import wget
import unidecode
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

import urllib
import argparse

# chrome driver (change the setup so that we can use headless chrome)
DRIVER_PATH = '/home/yingru/Documents/Project/Insight/Pinterest/Pinterest_final/webApp/src/chromedriver'
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--single-process')

class PinterestBoard(object):
    def __init__(self, userName, boardName, folder):
        '''
        Input -- userName, boardName [string, string]: pinterest username and boardname that user provides
        '''
        self.url = 'https://www.pinterest.com/{}/{}/'.format(userName, boardName)
        self.folder = '{}/{}'.format(folder, boardName)
        self.data = []

    def grep(self):
        '''
        grep information from Pinterest website
        Return -- self.data [pandas.DataFrame (pinId, pinUrl, pinTags, pinTitle, pinDescription)]
               -- pinId: unique Id associated with pin in Pinterest
               -- pinUrl: url for pin images
               -- pinTags: tags related to the pin 
               -- pinTitle: title related to the pin
               -- pinDescription: some description related to the pin 
        '''
        driver = webdriver.Chrome(DRIVER_PATH, chrome_options = options)
        driver.get(self.url)
        time.sleep(1)

        # now scraping
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
                _dumC = str(_dumC).encode('ascii', 'ignore')  # encode into ascii #pinOrigComment.append(str(_dumC).encode('utf-8'))
                pinOrigComment.append(_dumC.decode('ascii'))

                _dumT = pin.find('h3').text
                _dumT = str(_dumT).encode('ascii', 'ignore')
                pinTitle.append(_dumT.decode('ascii'))

                dum = pin.find('div', {'data-test-id': 'vasetags'})
                _tags = ''
                for _ in dum.find_all('a')[:-1]:
                    _tags += _.text + ','
                pinTags.append(_tags)

                _desc = pin.find('p', {'data-test-id': 'desc'}).text
                _desc = str(_desc).encode('ascii', 'ignore')
                pinDescription.append(_desc.decode('ascii'))

            except:
                continue

        data = pd.DataFrame([pinId, pinUrl, pinTags, pinTitle, pinDescription],
                            index = ['pinId', 'url', 'tage', 'title', 'description'])
        self.data = data.T
        driver.close()
        return self.data

    def grepImage(self):
        '''
        download images associated with the pin to local dst: self.folder
        '''
        if not len(self.data):
            raise ValueError('Please load board information first!  PinterestBoard.grep()')

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        for url in self.data['url']:
            try:
                os.system('wget {}'.format(url))
                name = url.split('/')[-1]
                shutil.move(name, '{}/{}'.format(self.folder, name))
            except:
                continue
        
        self.data['imageName'] = self.data['url'].apply(lambda x: x.split('/')[-1])
        print('done! :)')


    def naiveBayesCount(self):
        '''
        return the Naive Bayes analyze of the pin description
        '''
        if not len(self.data):
            raise ValueError('Please load board information first! PinterestBoard.grep()')
       
        self.data['label'] = self.data['description'].apply(lambda x: sent2word(x))

        total_words = []
        for words in self.data['label']:
            total_words += words

        counts = dict(Counter(total_words))
        result = pd.DataFrame(counts, index=['freq']).T
        result.sort_values(by=['freq'], axis=0, ascending=False, inplace=True)

        return result


def create_freq_dict(sents):
    '''
    find the frequency of the key words in sents
    '''
    i = 0
    freqDict_list = []
    for sent in sents:
        i += 1
        freq_dict = {}
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(sent)
        filter_words = [w for w in words if not w in stop_words]
        for word in filter_words:
            word = word.lower()
            if word in freq_dict:
                freq_dict[word] += 1
            else:
                freq_dict[word]= 1
            
            temp = {'doc_id': i, 'freq_dict': freq_dict}
        
        freqDict_list.append(temp)
    return freqDict_list


# the function to get TF-IDF score
def computeTF(doc_info, freqDict_list):
    TF_scores = []
    for tempDict in freqDict_list:
        id = tempDict['doc_id']
        for k in tempDict['freq_dict']:
            temp = {'doc_id': id,
                    'TF_score': tempDict['freq_dict'][k]/doc_info[id-1]['doc_length'],
                    'key': k
                   }
            TF_scores.append(temp)
    return TF_scores

def computeIDF(doc_info, freqDict_list):
    IDF_scores = []
    counter = 0
    for dic in freqDict_list:
        counter += 1
        for k in dic['freq_dict'].keys():
            count = sum([k in tempDict['freq_dict'] for tempDict in freqDict_list])
            temp = {'doc_id': counter, 'IDF_score':math.log(len(doc_info)/count), 
                   'key': k}
            IDF_scores.append(temp)
            
    return IDF_scores

def computeTFIDF(TF_scores, IDF_scores):
    TFIDF_scores = []
    for j in IDF_scores:
        for i in TF_scores:
            if j['key'] == i['key'] and j['doc_id'] == i['doc_id']:
                temp = {'doc_id': j['doc_id'],
                       'TFIDF_score': j['IDF_score'] * i['TF_score'],
                       'key': i['key']}
        TFIDF_scores.append(temp)
    return TFIDF_scores 


# remove the special characters from the description and tokenize the sentence
def remove_string_special_characters(s):
    stripped = re.sub('[^\w\s]', '', s)
    stripped = re.sub('_', '', stripped)
    
    stripped = re.sub('\s+', ' ', stripped)
    stripped = stripped.strip()
    return stripped


# convert the description sentence to words
def sent2word(s):
    '''
    Input -- s [string]: description
    Returns -- list[str]: cleaned key words
    '''
    stop_words = set(stopwords.words('english'))

    if s:
        sentence = sent_tokenize(s)
        sentenceClean = [remove_string_special_characters(_) for _ in sentence]

        totalWords = []
        for sent in sentenceClean:
            words = word_tokenize(sent)
            filter_words = [w.lower() for w in words if not w in stop_words and not w.isnumeric() and not len(w) == 1]
            totalWords += filter_words

        return totalWords
    else:
        return []
        
    
## query from pinterest based on key words
def grepSearch(query, scope='pins'):
    '''
    Input -- query (string)
          -- scope (string): the scope where the search can happen
    '''
    q = '%20'.join(query.split())
    url = 'https://www.pinterest.com/search/{}/?q={}&rs=typed'.format(scope, q)  # works so far for seledium wt chrome
    return grepPinterest(url)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'description')
    parser.add_argument('--user', default = 'xuyingru02', type=str)
    parser.add_argument('--board', default='art', type=str)

    args = parser.parse_args()

    board = PinterestBoard(args.user, args.board, '../board')
    board.grep()
    board.grepImage()
    print(board.naiveBayesCount()[:5])
    print(board.data['label'])
