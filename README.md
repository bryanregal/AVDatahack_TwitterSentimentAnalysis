### Word Embedding + Convolutional Neural Network (CNN) Model for Twitter Sentiment Analysis
Analytics Vidhya, Twitter Sentiment Analysis Practice Problem  

##### OBJECTIVE
The main goal is to utilize NLP and ML to identify the tweets which are hate tweets and which are not. For the complete competition details, see: https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/

###### 1. Data Preparation
Without getting too much into the details, the following are the steps I've used to prepare the data for sentiment analysis:
  1.1 Data Cleaning \n
    1.1.1 split tokens by white space
    1.1.2 Removal of tokens which are not encoded in ascii format
    1.1.3 Removal of punctuations from each token
    1.1.4 Removal of non-alphanumeric tokens
    1.1.5 Removal of digits from each token
    1.1.6 Removal of known stop words
    1.1.7 Removal of tokens that have a character length ≤ 1 or ≥  30
    1.1.8 Word stemming via Porter stemmer
    
  1.2 Word Vector Representation
    Each tweet is converted into a vector representation. The process involved developing a word vocabulary from all the tweets available in the train data, then reloading each tweet, cleaning it, filtering out tokens not in the predefined vocabulary, and finally returning the tweet as a string of white space separated tokens.

###### 2. CNN with Embedding Layer
Word embedding is used while training a CNN. The full neural network model structure with word embedding and convolutional
layers is given by:



###### 3. Model Evaluation

###### 4. Test set Prediction and Leaderboard Result
