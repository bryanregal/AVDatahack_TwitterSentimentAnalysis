# -*- coding: utf-8 -*-
"""
@author: BBALAJA
Analytics Vidhya Datahack : Twitter Sentiment Analysis
Algorithm: Deep Learning
"""

import pandas
import re
import string
import tensorflow as tf
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


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
    tokens = [w for w in tokens if len(w) > 1]
    # stemming of words
    porter = PorterStemmer()
    tokens = [porter.stem(w) for w in tokens]
    return tokens

def token_to_line(txt, vocab):
    """Clean text and return line of tokens
    dependency: clean_text
    """
    # clean text
    tokens = clean_text(txt)
    # filter by vocabulary
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def process_texts(texts, vocab):
    """Clean texts to only contain tokens present in the vocab
    dependency: token_to_line
    """
    lines = list()  
    for txt in texts:
        # load and clean the doc
        line = token_to_line(txt, vocab)
        # add to list
        lines.append(line)
    return lines

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

def add_tokens_vocab(txt, vocab):
    """Creating vocabulary containing unique tokens from all texts
    dependency: clean_text
    """
    tokens = clean_text(txt)         
    vocab.update(tokens)  
    
def build_vocab(texts):
    # define the vocabulary
    vocab = Counter()
    for txt in texts:
        add_tokens_vocab(txt, vocab)   
            
    # save tokens to a vocabulary file; for later access in model build/predict  
    save_vocab(vocab, data_path + "/vocab.txt")
    
# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
    
# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences
    padded = pad_sequences(encoded, maxlen=max_length, padding='post') 
    return padded

# define the metrics
def tf_auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

# define network
def define_model(vocab_size, max_length): 
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

  
#----------------------
# MAIN 
#----------------------
data_path = "/Users/bryanbalajadia/DataScience/GitHub_repos/AVDatahack_TwitterSentimentAnalysis/Data"
train_df = pandas.read_csv(data_path + "/train.csv", quotechar='"', skipinitialspace=True, encoding='utf-8')
test_df = pandas.read_csv(data_path + "/test.csv", quotechar='"', skipinitialspace=True, encoding='utf-8')


# get response
targets = train_df.label
        
# create Vocabulary file
build_vocab(train_df.tweet)   

# Load vocabulary
tokens = load_vocab(data_path + "/vocab.txt")

texts = process_texts(train_df.tweet, vocab = tokens)
max_length = max([len(s.split()) for s in texts])

tokenizer = create_tokenizer(texts)
vocab_size = len(tokenizer.word_index) + 1

Xtrain = encode_docs(tokenizer, max_length, texts)
Xtest = encode_docs(tokenizer, max_length, process_texts(test_df.tweet, vocab = tokens))

model = define_model(vocab_size, max_length)

# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf_auc_roc]) # summarize defined model
model.summary()

# checkpoint
filepath = data_path + "/weights.bestmodel.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_tf_auc_roc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# fit network
model.fit(Xtrain, targets, epochs=10, validation_split=0.10, verbose=2, callbacks=callbacks_list)


neural_net = define_model(vocab_size, max_length)
neural_net.load_weights(data_path + "/weights.bestmodel.hdf5")
neural_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf_auc_roc])

results = pandas.DataFrame(neural_net.predict(Xtest, verbose=0))

test_df["label"] = results.iloc[:,0]
test_df = test_df[["id", "label"]]

test_df.to_csv(data_path + "/submission.csv", encoding='utf-8',index=False)





