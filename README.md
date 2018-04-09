### Word Embedding + Convolutional Neural Network (CNN) Model for Twitter Sentiment Analysis
Analytics Vidhya, Twitter Sentiment Analysis Practice Problem  

##### OBJECTIVE
The objective is to utilize NLP and ML to detect hate speech in tweets. For the complete competition details, see: https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/

###### 1. Data Preparation
Without getting too much into the details, following are the steps I've used to prepare the data for sentiment analysis:

  1.1 Data Cleaning -- split tokens by white space, removal of tokens which are not encoded in ascii format, removal of punctuations from each token, removal of non-alphanumeric tokens, removal of digits from each token, removal of known stop words, removal of tokens that have a character length ≤ 1 or ≥  30, and word stemming via Porter stemmer
    
  1.2 Word Vector Representation -- Each tweet is converted into a vector representation. The process involved developing a word vocabulary from all the tweets available in the train data, then reloading each tweet, cleaning it, filtering out tokens not in the predefined vocabulary, and finally returning the tweet as a string of white space separated tokens.

###### 2. CNN with Embedding Layer
Word embedding is used while training a CNN. The full neural network model structure with word embedding and convolutional
layers is given by:
![alt text](https://github.com/BBalajadia/AVDatahack_TwitterSentimentAnalysis/blob/master/Data/nnmodel.png)

