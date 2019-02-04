from flask import render_template
from flask import request
from pureMix import app
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from inflection import singularize 

import os, sys
from collections import Counter

from src import grep
#from src import labels
from src import content
from src import pattern

app.var = {}

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', 
            title = 'Home', user= {'nickname': 'Miguel'},
            )

@app.route('/behind_the_scene')
def behindTheScene():
    return render_template('behind_the_scene.html')


@app.route('/input', methods=['GET', 'POST'])
def job_input():
    if request.method == 'GET':
        return render_template('input.html')
    else:
        input_key = request.form['input_key']
        input_username = request.form['input_username']
        input_boardname = request.form['input_boardname']
        url = 'https://www.pinterest.com/{}/{}/'.format(input_username, input_boardname)

        data = grep.grepPinterest(url)
        app.var['input_key'] = request.form['input_key']
        app.var['input_username'] = request.form['input_username']
        app.var['input_boardname'] = request.form['input_boardname']
        app.var['url'] = url
        app.var['data'] = data

        return render_template('input_display.html',
                            data = data,
                            input_key = input_key,
                            input_username = input_username,
                            input_boardname = input_boardname)


@app.route('/output', methods=['GET', 'POST'])
def job_output():
    dataLabel = grep.NaiveBayesCount(app.var['data'])
    
    # now scripting all the images
    imageUrl = app.var['data']['url']
    print(imageUrl)
    if not os.path.exists('board'):
        os.mkdir('board')
    imageFolder = 'board/{}'.format(app.var['input_boardname'])
    if not os.path.exists(imageFolder):
        os.mkdir(imageFolder)

    grep.grepImage(imageUrl, imageFolder)
    # done with grep, now analyze


    # -------- labels ------------------
    app.var['label'] = dataLabel.index.tolist()[:5]

    # -------- image processing --------
    try:
        #result_pattern = pattern.predict_pattern(imageFolder, weights = '/home/yingru/Documents/Project/Insight/Pinterest/clothing-pattern-dataset/checkoutpoints/ResNet50_model_weights.h5')
        result_pattern = pattern.predict_pattern(imageFolder, weights = './weights/ResNet50_model_weights.h5')

        result_content = content.predict_content(imageFolder)
        result_content.update(result_pattern)
        dataDL = pd.DataFrame(result_content, index=['prob']).T
        dataDL.sort_values(by='prob', axis=0, ascending=False, inplace=True)
        app.var['label2'] = dataDL.index.tolist()[:5]
    except:
        dataDL = []
        app.var['label2'] = []
        

    if len(dataLabel) and len(dataDL):
        result = {'keyword': app.var['input_key'],
                  'username': app.var['input_username'],
                  'boardname': app.var['input_boardname'], 
                  'label': dataLabel.index.tolist()[:5], 
                  'label2': dataDL.index.tolist()[:5]}

    elif len(dataLabel):
        result = {'keyword': app.var['input_key'],
                 'username': app.var['input_username'],
                 'boardname': app.var['input_boardname'],
                 'label': dataLabel.index.tolist()[:5],
                 'label2': dataLabel.index.tolist()[5:10]}
    else:
        result = {'keyword': app.var['input_key'],
                 'username': app.var['input_username'],
                 'boardname': app.var['input_boardname'],
                 'label': dataLabel.index.tolist()[:5],
                 'label2': dataLabel.index.tolist()[5:10]}
 
        

    return render_template('output.html',
                            data=result) 



@app.route('/output_final', methods=['GET', 'POST'])
def job_output2():
    try:
        for _ in app.var['label'] + app.var['label2']:
            if request.form.get(_):
                select = str(_)

        query = '{} {}'.format(app.var['input_key'], select)
        q = '%20'.join(query.split())
        scope = 'pins'
        url = 'https://www.pinterest.com/search/{}/?q={}&rs=typed'.format(scope, q)
        print('final output url: ', url)
        data = grep.grepPinterest(url)

    except:
        pass

    return render_template('output_display.html',
                        query = query,
                        data = data,
                        url = url,
                        input_key = app.var['input_key'],
                        input_username = app.var['input_username'],
                        input_boardname = app.var['input_boardname'])

