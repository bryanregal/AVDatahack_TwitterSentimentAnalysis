# -*- coding: utf-8 -*-
"""
@author: BBALAJA
Analytics Vidhya Datahack : Twitter Sentiment Analysis
Algorithm: Deep Learning
"""

import pandas
import numpy
import re
import sys

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

class TSA(object):
    
    def __init__(self):

        self.data_path = "/Users/bryanbalajadia/DataScience/GitHub_repos/AVDatahack_TwitterSentimentAnalysis/Data"
        self.data_source = "train.csv"
        self.inputs_temp = "inputs_template.csv"
        self.class_weight = {0 : 100., 1: 8.} # handle for data imbalance
        self.neural_net = None
        
    def clean_text(self, txt):
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

    def add_tokens_vocab(self, txt, vocab):
        """Creating vocabulary containing unique tokens from all texts
        dependency: clean_text
        """
        tokens = self.clean_text(txt)         
        vocab.update(tokens)  
    
    def save_vocab(self, lines, filename):
        """Saving a list of items to a file; line-by-line
        """
        data = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()
    
    def load_vocab(self, filename):
        """Load doc into memory
        """
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text
    
    def token_to_line(self, txt, vocab):
        """Clean text and return line of tokens
        dependency: clean_text
        """
        # clean text
        tokens = self.clean_text(txt)
        # filter by vocabulary
        tokens = [w for w in tokens if w in vocab]
        return ' '.join(tokens)
    
    def process_texts(self, texts, vocab):
        """Clean texts to only contain tokens present in the vocab
        dependency: token_to_line
        """
        lines = list()  
        for txt in texts:
            # load and clean the doc
            line = self.token_to_line(txt, vocab)
            # add to list
            lines.append(line)
        return lines
    
    def build_vocab(self, texts):
         # Define the vocabulary
        vocab = Counter()
        for txt in texts:
            self.add_tokens_vocab(txt, vocab)   
            
        # save tokens to a vocabulary file; for later access in model build/predict  
        self.save_vocab(vocab, self.data_path + "/vocab.txt")
    # fit a tokenizer
    def create_tokenizer(self, lines):
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

    def _NLP(self, texts):
        """Main process for running NLP feature engineering
        dependency: build_vocab must be run first (ref: build_model)
        """       
        tokens = self.load_vocab(self.data_path + "/vocab.txt")
        
        # process texts
        texts = self.process_texts(texts, vocab = tokens)
        
        max_length = max([len(s.split()) for s in texts])
        print('Maximum length: %d' % max_length)
        
        tokenizer = self.create_tokenizer(texts)
        Xtrain = encode_docs(tokenizer, max_length, texts)
        return Xtrain, max_length
        
    
    def build_input_table(self, table):
        print("================================")
        print("Table Shape" + str(table.shape))
        print("Adding calculated fields...")
        
        inputs = self._NLP(table.tweet)

        print("Inputs table shape: "+str(inputs.shape))
        print("================================")
        
        return inputs

    # define the model
    def define_model(self, input_length): 
        # define network
        model = Sequential()
        model.add(Embedding(vocab_size, 100, input_length=max_length))
        model.add(Flatten())
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model
    
    # define the metrics
    def auc_roc(self, y_true, y_pred):
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
     
     # Main process for MODEL BUILD   
    def build_model(self):
        # fix random seed for reproducibility
        numpy.random.seed(2018)
        
        source_file = self.data_path + "/" + self.data_source
        print("Loading data from file: " + source_file)

        table = pandas.read_csv(source_file, quotechar='"', skipinitialspace=True, encoding='utf-8')
        
        # get response
        targets = table.label
        
        # create Vocabulary file
        self.build_vocab(table.tweet)   
        
        # build inputs frame
        inputs = self.build_input_table(table)

        # save the input template
        template = inputs.iloc[0:0]
        template.to_csv(self.data_path + "/" + self.inputs_temp, encoding='utf-8',index=False)
        
        print("Defining the neural network architecture...")
        # Define the network model
        model = self.define_model(inputs.shape[1])               
        # Compile network
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[self.auc_roc])  
        # summarize defined network model
        model.summary()     
        # checkpoint
        filepath = self.data_path + "/weights.bestmodel.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_auc_roc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        
        print("Building network...")
        # Fit the model
        model.fit(numpy.array(inputs), numpy.array(targets), validation_split=0.10, epochs=300, batch_size=30000, callbacks=callbacks_list, verbose=0, class_weight = self.class_weight)

    def get_template(self):
        return pandas.read_csv(self.data_path  + "/" + self.inputs_temp, quotechar='"', skipinitialspace=True, encoding='utf-8')

    def load_neural_net(self):
        # Loads the current model into memory for further processing
        file_to_load = self.data_path + "/weights.bestmodel.hdf5"
        print("Loading "+ file_to_load)
        inputs_size = self.get_template().shape[1]
        self.neural_net = self.define_model(inputs_size)
        self.neural_net.load_weights(self.data_path + "/weights.bestmodel.hdf5")
        self.neural_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=[self.auc_roc])
 
    def conform_features(self, original_columns, inputs):
        # create a dataframe in exact same structure as NN training set
        #original_columns should be array of strings
        common_columns = numpy.intersect1d(original_columns,inputs.keys())
        missing_columns = numpy.setdiff1d(original_columns, inputs.keys())

        new_table = inputs[common_columns]
        missing_columns_table = pandas.DataFrame(data=0, index=numpy.arange(len(inputs.index)),
                                                 columns=missing_columns)
        # put the tables together
        full_table = pandas.concat([new_table, missing_columns_table], axis=1)
        # set original column order
        full_table = full_table[original_columns]
        return full_table
    
    # Main process for running TEST FILE predictions
    def predict_from_file(self):
        print("Loading data for prediction...")
        table = pandas.read_csv(self.data_path + "/test.csv", quotechar='"', skipinitialspace=True, encoding='utf-8')

        print("Loading the neural network features template...")
        template = self.get_template()
        print("Number of neurons (neuralnet inputs layer): " + str(template.shape[1]))
             
        print("Building the for prediction data inputs table...")
        inputs = self.build_input_table(table)
        print("Conforming inputs table to the neuralnet features....")
        inputs = self.conform_features(template.keys(),inputs)
        print (inputs.shape)
        
        print("Running predictions...")
        results = pandas.DataFrame(self.neural_net.predict(inputs, verbose=0))
        table["label"] = results.iloc[:,0]
        table = table[["id", "label"]]
        
        print("Writing predictions to a submission file...")
        table.to_csv(self.data_path + "/submission.csv", encoding='utf-8',index=False)
        
        print("...Prediction process COMPLETED.")
        return table


if __name__ == '__main__':

    if len(sys.argv) >= 2:
        
        engine = TSA()
        if sys.argv[1].upper() == "BUILD":
            engine.build_model()
        elif sys.argv[1].upper() == "PREDICT":
            engine.load_neural_net()
            engine.predict_from_file()
        else:
            print("Unknown parameter - use BUILD to create a model; PREDICT for running predictions")

    else:
        print("Pass command BUILD, PREDICT")
