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

sys.path.append('/home/yingru/Documents/Project/Insight/Pinterest/clothing-pattern-dataset/webApp/pinspire/src/pins')
import grep



def remove_string_special_characters(s):
    stripped = re.sub('[^\w\s]', '', s)
    stripped = re.sub('_', '', stripped)
    
    stripped = re.sub('\s+', ' ', stripped)
    stripped = stripped.strip()
    return stripped


def get_doc(text_sents_clean):
    doc_info = []
    i = 0 
    for sent in text_sents_clean:
        i += 1
        count = count_words(sent)
        temp= {'doc_id': i, 'doc_length':count}
        doc_info.append(temp)
    return doc_info


def count_words(sent):
    count = 0
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sent)
    filter_words = [w for w in words if not w in stop_words]
    for word in filter_words:
        count += 1
        
    return count


def create_freq_dict(sents):
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


def tfidf(data):
    """
    data: [pinId, url, tags, title, description]
    """

    sentence = '.'.join([i for i in data.description if i])
    text_sents = sent_tokenize(sentence)
    text_sents_clean = [remove_string_special_characters(s) for s in text_sents]
    doc_info = get_doc(text_sents_clean)

    freqDict_list = create_freq_dict(text_sents_clean)
    TF_scores = computeTF(doc_info, freqDict_list)
    IDF_scores = computeIDF(doc_info, freqDict_list)

    TFIDF_scores = computeTFIDF(TF_scores, IDF_scores)

    TFIDF_scores = sorted(TFIDF_scores, key=lambda x: (x['doc_id'], x['TFIDF_score']))

    return TFIDF_scores


from nltk.stem.porter import PorterStemmer

def naiveBayesCount(data):
    stop_words = set(stopwords.words('english'))

    sentence = '.'.join([i for i in data.description if i])
    text_sents = sent_tokenize(sentence)
    text_sents_clean = [remove_string_special_characters(s) for s in text_sents]
 
    total_words = []
    for sent in text_sents_clean:
        words = word_tokenize(sent)
        filter_words = [w.lower() for w in words if not w in stop_words]
        total_words += filter_words

    porter = PorterStemmer()
    total_words = [porter.stem(w) for w in total_words]
    counts = dict(Counter(total_words))
    result = pd.DataFrame(counts, index=['freq']).T
    #result = result.reset_index().rename(columns={'index': 'key'})
    return result

    #return sorted(counts.items(), key=lambda x: x[1], reverse=True)
