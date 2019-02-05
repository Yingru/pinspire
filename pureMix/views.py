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
        dum = request.form['input_username']
        input_username = dum.split('/')[0]
        input_boardname = dum.split('/')[1]
        
        if not os.path.exists('board'):
            os.mkdir('board')
        imageFolder = 'board/{}'.format(input_boardname)
        if not os.path.exists(imageFolder):
            os.mkdir(imageFolder)

        print('input: ', input_username, input_boardname)
        url = 'https://www.pinterest.com/{}/{}/'.format(input_username, input_boardname)

        data = grep.grepPinterest(url)
        data.to_csv('{}/dataLabel_{}.csv'.format(imageFolder, input_boardname))
        app.var['input_key'] = request.form['input_key']
        app.var['input_username'] = input_username
        app.var['input_boardname'] = input_boardname
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
    dataLabel.to_csv('{}/freq_dataLabel_{}.csv'.format(imageFolder, app.var['input_boardname']))

    grep.grepImage(imageUrl, imageFolder)
    # done with grep, now analyze
    

    # -------- labels ------------------
    app.var['label'] = dataLabel.index.tolist()[:5]

    # -------- image processing --------
    try:
        print('now start with image process: ')
        result_pattern = pattern.predict_pattern(imageFolder, weights = './weights/ResNet50_model_weights.h5')
        print('image pattern classification done successfully: ')
        result_content = content.predict_content(imageFolder)
        print('image content classification done successfully: ')
        result_content.update(result_pattern)
        dataDL = pd.DataFrame(result_content, index=['prob']).T
        dataDL.sort_values(by='prob', axis=0, ascending=False, inplace=True)
        dataDL.to_csv('{}/prob_dataDL_{}.csv'.format(imageFolder, app.var['input_boardname']))
        app.var['label2'] = dataDL.index.tolist()[:5]
        print('successful on the image classification!')
    except:
        dataDL = []
        app.var['label2'] = []
        

    if len(dataLabel) and len(dataDL):
        result = {'keyword': app.var['input_key'],
                  'username': app.var['input_username'],
                  'boardname': app.var['input_boardname'], 
                  'label': dataLabel.index.tolist()[:5] + dataDL.index.tolist()[:5]}

    elif len(dataLabel):
        result = {'keyword': app.var['input_key'],
                 'username': app.var['input_username'],
                 'boardname': app.var['input_boardname'],
                 'label': dataLabel.index.tolist()[:10]}
    else:
        result = {'keyword': app.var['input_key'],
                 'username': app.var['input_username'],
                 'boardname': app.var['input_boardname'],
                 'label': ['Error, give us another chance']}

    return render_template('output.html',
                            input_key = app.var['input_key'],
                            input_username = app.var['input_username'],
                            input_boardname = app.var['input_boardname'],
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
        imageFolder = 'board/{}'.format(app.var['input_boardname'])
        data.to_csv('{}/dataLabel_{}_{}.csv'.format(imageFolder, app.var['input_key'], select))

    except:
        pass

    return render_template('output_display.html',
                        query = query,
                        data = data,
                        url = url,
                        input_key = app.var['input_key'],
                        input_username = app.var['input_username'],
                        input_boardname = app.var['input_boardname'])


@app.route('/demo_input', methods=['GET', 'POST'])
def job_input_demo():
    if request.method == 'GET':
        return render_template('demo_input.html')
    else:
        input_key = request.form.get('input_key')
        dum = request.form.get('input_username')
        if len(input_key) == 0:
            input_key = 'phone case'
        if len(dum) == 0:
            dum = 'xuyingru02/art'

        input_username = dum.split('/')[0]
        input_boardname = dum.split('/')[1]

        print('input: ', input_username, input_boardname)
        url = 'https://www.pinterest.com/{}/{}/'.format(input_username, input_boardname)
        dataPath = 'board/{}/dataLabel_{}.csv'.format(input_boardname, input_boardname)

       
        data = pd.read_csv(dataPath)
        app.var['input_key'] = input_key
        app.var['input_username'] = input_username
        app.var['input_boardname'] = input_boardname
        app.var['url'] = url
        app.var['data'] = data
        print(data['url'])

        return render_template('demo_input_display.html',
                            data = data,
                            input_key = input_key,
                            input_username = input_username,
                            input_boardname = input_boardname)

@app.route('/demo_output', methods=['GET', 'POST'])
def job_output_demo():
    imageFolder = 'board/{}'.format(app.var['input_boardname'])
    dataLabel = pd.read_csv('{}/freq_dataLabel_{}.csv'.format(imageFolder, app.var['input_boardname']))
    dataDL = pd.read_csv('{}/prob_dataDL_{}.csv'.format(imageFolder, app.var['input_boardname']))
    app.var['label'] = list(dataLabel['Unnamed: 0'][:5]) + list(dataDL['Unnamed: 0'][:5])
    print(app.var['label'])
    result = {'keyword': app.var['input_key'],
              'username': app.var['input_username'],
              'boardname': app.var['input_boardname'], 
              'label': app.var['label']}

    return render_template('demo_output.html',
                            input_key = app.var['input_key'],
                            input_username = app.var['input_username'],
                            input_boardname = app.var['input_boardname'],
                            data=result) 


@app.route('/demo_output_final', methods=['GET', 'POST'])
def job_output2_demo():
    try:
        for _ in app.var['label']:
            print(_)
            if request.form.get(_):
                select = str(_)

        print('first what??', len(select))
        print('second what??', app.var['input_key'], len(app.var['input_key']))
        query = '{} {}'.format(app.var['input_key'], select)
        q = '%20'.join(query.split())
        scope = 'pins'
        url = 'https://www.pinterest.com/search/{}/?q={}&rs=typed'.format(scope, q)
        print('final output url: ', url)

        imageFolder = 'board/{}'.format(app.var['input_boardname'])
        data = pd.read_csv('{}/dataLabel_{}_{}.csv'.format(imageFolder, app.var['input_key'], select))
        print('final!!!', data)
    except:
        pass 

    return render_template('demo_output_display.html',
                        url = url,
                        query = query,
                        data = data,
                        input_key = app.var['input_key'],
                        input_username = app.var['input_username'],
                        input_boardname = app.var['input_boardname'])
