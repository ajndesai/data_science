#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:14:41 2020

@author: arundesai
"""

import argparse
import re 
import tweepy 
import string
import nltk
from tweepy import OAuthHandler 
from textblob import TextBlob 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TwitterClient(object): 
    ''' 
    Generic Twitter Class for sentiment analysis. 
    '''
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret): 
        ''' 
        Class constructor or initialization method. 
        '''
        # keys and tokens from the Twitter Dev Console 
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.emoji_pattern = re.compile(
                            u"(\ud83d[\ude00-\ude4f])|"  # emoticons
                            u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
                            u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
                            u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
                            u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
                            "+", flags=re.UNICODE)

        #HappyEmoticons
        self.emoticons_happy = set([
            ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
            ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
            '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
            'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
            '<3'
            ])
    
        # Sad Emoticons
        self.emoticons_sad = set([
            ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
            ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
            ':c', ':{', '>:\\', ';('
            ])
    
        self.emoticons = self.emoticons_happy.union(self.emoticons_sad)
        # attempt authentication 
        try: 
            # create OAuthHandler object 
            self.auth = OAuthHandler(self.consumer_key, self.consumer_secret) 
            # set access token and secret 
            self.auth.set_access_token(self.access_token, self.access_token_secret) 
            # create tweepy API object to fetch tweets 
            self.api = tweepy.API(self.auth) 
        except: 
            print("Error: Authentication Failed") 

    def clean_tweet(self, tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        lst = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 
        print(lst)
        return lst

    def clean_tweets(self, tweet):
     
        stop_words = set(stopwords.words('english'))
        
    #after tweepy preprocessing the colon symbol left remain after      #removing mentions
        tweet = re.sub(r':', '', tweet)
        tweet = re.sub(r'‚Ä¶', '', tweet)
    #replace consecutive non-ASCII characters with a space
        tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    #remove emojis from tweet
        tweet = self.emoji_pattern.sub(r'', tweet)
        
        tweet = re.sub(r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})", "", tweet)
        word_tokens = word_tokenize(tweet)
    #filter using NLTK library append it to a string
#        filtered_tweet = [w for w in word_tokens if not w in stop_words]
        filtered_tweet = []
    #looping through conditions
        for w in word_tokens:
    #check tokens against stop words , emoticons and punctuations
            if w not in stop_words and w not in self.emoticons and w not in string.punctuation:
                filtered_tweet.append(w)
        return ' '.join(filtered_tweet)
        #print(word_tokens)
        #print(filtered_sentence)return tweet

    def get_tweet_sentiment(self, tweet): 
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(self.clean_tweet(tweet)) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'

    def get_user_timeline_tweets(self, userid):
        tweets = self.api.user_timeline(user_id=userid, lang="en", count=10, tweet_mode="extended")
        return tweets
        
    def get_tweets(self, query, count = 10): 
        ''' 
        Main function to fetch tweets and parse them. 
        '''
        # empty list to store parsed tweets 
        tweets = [] 

        try: 
            # call twitter api to fetch tweets 
            fetched_tweets = self.api.search(q = query, count = count) 

            # parsing tweets one by one 
            for tweet in fetched_tweets: 
                # empty dictionary to store required params of a tweet 
                parsed_tweet = {} 

                # saving text of tweet 
                parsed_tweet['text'] = tweet.text 
                # saving sentiment of tweet 
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 

                # appending parsed tweet to tweets list 
                if tweet.retweet_count > 0: 
                    # if tweet has retweets, ensure that it is appended only once 
                    if parsed_tweet not in tweets: 
                        tweets.append(parsed_tweet) 
                else: 
                    tweets.append(parsed_tweet) 

            # return parsed tweets 
            return tweets 

        except tweepy.TweepError as e: 
            # print error (if any) 
            print("Error : " + str(e))
            
    def extract_NN(self, sent):
        grammar = r"""
                    NBAR:
                        # Nouns and Adjectives, terminated with Nouns
                        {<NN.*>*<NN.*>}
                
                    NP:
                        {<NBAR>}
                        # Above, connected with in/of/etc...
                        {<NBAR><IN><NBAR>}
                    """
        chunker = nltk.RegexpParser(grammar)
        ne = set()
        chunk = chunker.parse(nltk.pos_tag(nltk.word_tokenize(sent)))
        for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
            ne.add(' '.join([child[0] for child in tree.leaves()]))
        return ne
    
    def getTweetFultext(self, tweetObj):
        if tweetObj:
            return [tweet.full_text for tweet in tweetObj]
        return []
    
    def sentiment_scores(self, sentence): 
  
        # Create a SentimentIntensityAnalyzer object. 
        sid_obj = SentimentIntensityAnalyzer() 
      
        # polarity_scores method of SentimentIntensityAnalyzer 
        # oject gives a sentiment dictionary. 
        # which contains pos, neg, neu, and compound scores. 
        sentiment_dict = sid_obj.polarity_scores(sentence) 
          
        print("Overall sentiment dictionary is : ", sentiment_dict) 
        print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
        print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
        print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
      
        print("Sentence Overall Rated As", end = " ") 
      
        # decide sentiment as positive, negative and neutral 
        if sentiment_dict['compound'] >= 0.05 : 
            print("Positive") 
      
        elif sentiment_dict['compound'] <= - 0.05 : 
            print("Negative") 
      
        else : 
            print("Neutral") 
            
def retreiveTweetsWithSentiments(api):
        # calling function to get tweets 
    tweets = api.get_tweets(query = 'Narendra Modi', count = 200) 

    # picking positive tweets from tweets 
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
    # percentage of positive tweets 
    print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
    # picking negative tweets from tweets 
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
    # percentage of negative tweets 
    print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))) 

    neutraltweets = [tweet for tweet in tweets if tweet['sentiment'] == 'neutral'] 
    # percentage of negative tweets 
    print("Neutral tweets percentage 1: {} %".format(100*len(neutraltweets)/len(tweets))) 

    # percentage of neutral tweets 
    print("Neutral tweets percentage 2: {} % ".format(100*(len(tweets) - len(ntweets) - len(ptweets))/len(tweets))) 

    # printing first 5 positive tweets 
    print("\n\nPositive tweets:") 
    for tweet in ptweets[:10]: 
        print(tweet['text']) 

    # printing first 5 negative tweets 
    print("\n\nNegative tweets:") 
    for tweet in ntweets[:10]: 
        print(tweet['text']) 
        
    # printing first 5 negative tweets 
    print("\n\\nNeutral tweets:") 
    for tweet in neutraltweets[:10]: 
        print(tweet['text']) 


def main(): 
    # creating object of TwitterClient Class 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-consumer_key', type=str, required=True, help="Twitter consumer key")
    parser.add_argument('-consumer_secret', type=str, required=True, help="Twitter consumer key secret")
    parser.add_argument('-access_token', type=str, required=True, help="Twitter access token")
    parser.add_argument('-access_token_secret', type=str, required=True, help="Twitter access token secret")
    args = parser.parse_args()
    
    if(not args.consumer_key or not args.consumer_secret or not args.access_token or not args.access_token_secret):
        print(args.consumer_key)
        print(args.consumer_secret)
        print(args.access_token)
        print(args.access_token_secret)
        print('Not all arguments passed or are none')
        sys.exit(-1)

    api = TwitterClient(args.consumer_key, args.consumer_secret, args.access_token, args.access_token_secret) 
#    retreiveTweetsWithSentiments(api)
    #@IndiGo6E, @AirAsiaIndian, @airindiain, @airvistara
    tweets1 = api.get_user_timeline_tweets('@EatFit')
    tweets2 = api.get_user_timeline_tweets('@SwiggyCares')
    tweets3 = api.get_user_timeline_tweets('@Swiggy.in')
    tweets4 = api.get_user_timeline_tweets('@airvistara')
    
    combined_tweets = []
    combined_tweets.extend(api.getTweetFultext(tweets1))
    combined_tweets.extend(api.getTweetFultext(tweets2))
    combined_tweets.extend(api.getTweetFultext(tweets3))
    combined_tweets.extend(api.getTweetFultext(tweets4))
    
    filtered_tweets=[]
    filtered_tweets.extend(api.clean_tweets(tweet) for tweet in combined_tweets)
    
    all_tweets = ' '.join(filtered_tweets)
    entities = api.extract_NN(all_tweets)
    
    print(type(entities))
    print(entities)
    print (' '.join(entities))
    
    api.sentiment_scores(' '.join(entities))

if __name__ == "__main__": 
    # calling main function 
    main() 
