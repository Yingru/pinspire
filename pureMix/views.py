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

from pins import pinspire
from pins import grep
from pins import labels

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
    #dataLabel, dataDL = pinspire.pinspire(app.var['input_key'], app.var['input_username'], app.var['input_boardname'])
    #dataLabel, dataDL = None, None
    dataLabel = labels.naiveBayesCount(app.var['data'])
    
    # now scripting all the images
    imageUrl = app.var['data']['url']
    print(imageUrl)
    if not os.path.exists('board'):
        os.mkdir('board')
    imageFolder = 'board/{}'.format(app.var['input_boardname'])
    if not os.path.exists(imageFolder):
        os.mkdir(imageFolder)

    grep.grepImage(imageUrl, imageFolder)
    dataDL = pinspire.pinspire_simpler(imageFolder)
        
    app.var['label'] = dataLabel.index.tolist()[:5]
    if len(dataDL):
        app.var['label2'] = dataDL
    
    
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
                 'label': dataLabel.index.tolist()[:5]}
    else:
        result = {'keyword': app.var['input_key'],
                  'username': app.var['input_username'],
                  'boardname': app.var['input_boardname']
        }


    return render_template('output.html',
                            data=result) 



@app.route('/output_final', methods=['GET', 'POST'])
def job_output2():
    #dataLabel, dataDL = pinspire.pinspire(app.var['input_key'], app.var['input_username'], app.var['input_boardname'])
    #dataLabel, dataDL = None, None
    for _ in app.var['label']:
        if request.form.get(_): 
            select = str(_)

    return render_template('output_display.html',
                            input_key= app.var['input_key'],
                            input_username= app.var['input_username'], 
                            input_boardname= app.var['input_boardname'],
                            label= str(select)) 



'''
@app.route('/output', methods=['GET', 'POST'])
def jobs_output():
    ######################
    ### user input 
    ######################
    input_key = request.args.get('input_key')
    input_username = request.args.get('input_username')
    input_boardname = request.args.get('input_boardname')
    url = 'https://www.pinterest.com/{}/{}/'.format(input_username, input_boardname)
    data = grep.grepPinterest(url)
    
    return render_template('output.html',
                    data = data,
                    input_key = input_key,
                    input_username = input_username,
                    input_boardname = input_boardname)

@app.route('/output', methods=['GET', 'POST'])
def job_output2():
    input_key = request.args.get('input_key')
    input_username = request.args.get('input_username')
    input_boardname = request.args.get('input_boardname')
    url = 'https://www.pinterest.com/{}/{}/'.format(input_username, input_boardname)
    data = grep.grepPinterest(url)
    
    return render_template('output2.html',
                    data = data,
                    input_key = input_key,
                    input_username = input_username,
                    input_boardname = input_boardname)

'''
