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


def grepPinterest(url):
    driver = webdriver.Chrome('/home/yingru/Documents/Project/Insight/Pinterest/chromedriver')
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

    return data.T

    #return pinId, pinUrl, pinTags, pinDescription, pinTitle, pinOrigComment




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



if __name__ == '__main__':
    userName = input('Please enter your username: ')
    boardName = input('Please enter your board name: ')
    url =  'https://www.pinterest.com/{}/{}/'.format(userName, boardName)
    folder = './board/{}'.format(boardName)
    data = grepPinterest(url)
    #data.description.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
    #pinId, pinUrl, pinTags, pinDescription, pinTitle, pinOrigComment  = grepPinterest(url)
    print('pinId: ', data.pinId)
    print('pinTags: ', data.tags)
    print(' ')
    print('pinDescription: ', data.description)
    print('pinTitle: ', data.title)
    #grepImage(pinUrl, folder)
