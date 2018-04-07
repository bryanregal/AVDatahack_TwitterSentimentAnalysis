### Word Embedding + Convolutional Neural Network (CNN) Model for Twitter Sentiment Analysis
Analytics Vidhya, Twitter Sentiment Analysis Practice Problem  

##### OBJECTIVE
The main goal is to utilize NLP and ML to identify the tweets which are hate tweets and which are not. For the complete competition details, see: https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/

###### 1. Data Preparation
Without getting too much into the details, the following are the steps I've used to prepare the data for sentiment analysis:

  1.1 Data Cleaning -- split tokens by white space, removal of tokens which are not encoded in ascii format, removal of punctuations from each token, removal of non-alphanumeric tokens, removal of digits from each token, removal of known stop words, removal of tokens that have a character length ≤ 1 or ≥  30, and word stemming via Porter stemmer
    
  1.2 Word Vector Representation -- Each tweet is converted into a vector representation. The process involved developing a word vocabulary from all the tweets available in the train data, then reloading each tweet, cleaning it, filtering out tokens not in the predefined vocabulary, and finally returning the tweet as a string of white space separated tokens.

###### 2. CNN with Embedding Layer
Word embedding is used while training a CNN. The full neural network model structure with word embedding and convolutional
layers is given by:
Defining the neural network model...
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, 21, 150)           4819800   
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 14, 32)            38432     
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 7, 32)             0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 224)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 30)                6750      
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 31        
=================================================================
Total params: 4,865,013
Trainable params: 4,865,013
Non-trainable params: 0
_________________________________________________________________



###### 3. Model Evaluation

###### 4. Test set Prediction and Leaderboard Result
