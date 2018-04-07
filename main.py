# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 17:30:40 2018

@author: BBALAJA
"""
#from sklearn.metrics import roc_auc_score
import pandas
import numpy
import re
import sys
import gc

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import math
import string
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#from sklearn.model_selection import StratifiedKFold
#from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
#import pydot


data_path = "/Users/bryanbalajadia/Desktop/AV_Datahack_TwitterSentimentAnalysis"

train = pandas.read_csv(data_path + "/train.csv", quotechar='"', skipinitialspace=True, encoding='utf-8')


def clean_text(txt):
     """Preprocessing - Turning texts into clean tokens
     """
     # Ensure lowercase text encoding
     txt = str(txt).lower()
     # split tokens by white space
     tokens = txt.split()
     # remove tokens not encoded in ascii
     isascii = lambda s: len(s) == len(s.encode())
     tokens = [w for w in tokens if isascii(w)]
     # regex for punctuation filtering
     re_punc = re.compile('[%s]' % re.escape(string.punctuation))
     # remove punctuation from each word
     tokens = [re_punc.sub('', w) for w in tokens]
     # remove tokens that aren't alphanumeric
     tokens = [w for w in tokens if w.isalnum()]
     # regex for digits filtering
     re_digt = re.compile('[%s]' % re.escape(string.digits)) 
     # remove digits from each word
     tokens = [re_digt.sub('', w) for w in tokens]  
     # filter out stop words
     stop_words = set(stopwords.words('english'))
     tokens = [w for w in tokens if not w in stop_words]
     # filter out long tokens
     tokens = [w for w in tokens if len(w) < 30]
     # filter out short tokens
     tokens = [w for w in tokens if len(w) > 2]
     # stemming of words
     porter = PorterStemmer()
     tokens = [porter.stem(w) for w in tokens]
     return tokens
       
def add_tokens_vocab(text, vocab):
    """Creating vocabulary containing unique tokens from all texts
    dependency: clean_text
    """
    tokens = clean_text(text)         
    vocab.update(tokens)
          
def save_vocab(lines, filename):
    """Saving a list of items to a file; line-by-line
    """
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()
    
def load_vocab(filename):
    """Load doc into memory
    """
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
    
def token_to_line(text, vocab):
    """Clean text and return line of tokens
    dependency: clean_text
    """
    # clean text
    tokens = clean_text(text)
    # filter by vocabulary
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)
    
def process_texts(texts, vocab):
    """Clean texts to only contain tokens present in the vocab
    dependency: token_to_line
    """
    lines = list()  
    for text in texts:
        # load and clean the doc
        line = token_to_line(text, vocab)
        # add to list
        lines.append(line)
    return lines
    
def build_vocab(texts):
    # Define the vocabulary
    vocab = Counter()
    for text in texts:
        add_tokens_vocab(text, vocab)
    
    tokens = [w for w in vocab if not w in ("subject")]
        
    # save tokens to a vocabulary file; for later access in model build/predict  
    save_vocab(tokens, data_path + "/vocab.txt")
        
def _NLP(texts):
    """Main process for running NLP feature engineering
    dependency: build_vocab must be run first (ref: build_model)
    """       
    tokens = load_vocab(data_path + "/vocab.txt")
        
    # process texts
    texts = process_texts(texts, vocab = tokens)
        
    # Boolean features
    vect = CountVectorizer(binary=True)
    X = vect.fit_transform(texts)
    Y = pandas.DataFrame(X.A, columns = vect.get_feature_names())
    return Y

# Create Vocabulary file
build_vocab(train.tweet) 

NLP_feat = _NLP(train.tweet)


