#!/usr/bin/env python
# coding: utf-8

# # Step [1]: Prepare libraries and data

# ## [1.1] Include important libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import nltk
from operator import itemgetter
import arabic_reshaper
import re
import json
import requests
from ar_corrector.corrector import Corrector
import qalsadi.lemmatizer


# ## [1.2] Download data

# In[ ]:


# get_ipython().system("wget 'https://drive.google.com/uc?export=download&id=1KepfzAhJ7dloG8XaWQf0ovQipDHYS8aI' -O 'final_data.zip'")


# In[ ]:


# get_ipython().system('unzip final_data.zip')


# ## [1.3] read data from csv file

# In[2]:


# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')
# valid = pd.read_csv('valid.csv')
# train.head()


# # Step [2]: Text Analysis

# ## [2.1] Tweets per class

# In[3]:


# count = train['label'].value_counts()
# plt.hist(weights=count, x=['unknown','positive', 'irrelevant','negative'])
# plt.show()
# count


# ## [2.2] Finding collocations (n-grams)

# In[4]:


def find_bigrams(tweet, work = 1):
    """find collocations in tweet function.
    Input:
        tweet: a string containing a tweet
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        bigrams

    """
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_words(tweet)
    finder.apply_freq_filter(10)
    # return the 10 n-grams with the highest PMI
    collocations = finder.nbest(bigram_measures.raw_freq, 10)
#     for i in finder.score_ngrams(bigram_measures.pmi):
#         print (i)
    return collocations


# In[5]:


def find_trigrams(tweet, work = 1):
    """find collocations in tweet function.
    Input:
        tweet: a string containing a tweet
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        trigrams

    """
    trigramm_measures = nltk.collocations.TrigramAssocMeasures()
    finder = nltk.collocations.TrigramCollocationFinder.from_words(tweet)
    finder.apply_freq_filter(10)
    # return the 10 n-grams with the highest PMI
    collocations = finder.nbest(trigramm_measures.raw_freq, 10)
#     for i in finder.score_ngrams(trigramm_measures.pmi):
#         print (i)
    return collocations


# In[6]:


def find_quadgrams(tweet, work = 1):
    """find collocations in tweet function.
    Input:
        tweet: a string containing a tweet
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        quadgrams

    """
    quadgram_measures = nltk.collocations.QuadgramAssocMeasures()
    finder = nltk.collocations.QuadgramCollocationFinder.from_words(tweet)
    finder.apply_freq_filter(10)
    # return the 10 n-grams with the highest PMI
    collocations = finder.nbest(quadgram_measures.raw_freq, 10)
#     for i in finder.score_ngrams(trigramm_measures.pmi):
#         print (i)
    return collocations


# In[7]:


# words = str(train.tweet.values.tolist()).split(' ')
# co = find_bigrams(words)
# print(co)
# co = find_trigrams(words)
# print(co)
# co = find_quadgrams(words)
# print(co)


# ## [2.3] Finding collocations (n-grams) for each label
# 

# In[8]:


# words_0 = str(train[train.label == 0].tweet.values.tolist()).split(' ')
# co = find_bigrams(words_0)
# print(co)
# co = find_trigrams(words_0)
# print(co)
# co = find_quadgrams(words_0)
# print(co)


# In[9]:


# words_1 = str(train[train.label == 1].tweet.values.tolist()).split(' ')
# co = find_bigrams(words_1)
# print(co)
# co = find_trigrams(words_1)
# print(co)
# co = find_quadgrams(words_1)
# print(co)


# In[10]:


# words_2 = str(train[train.label == 2].tweet.values.tolist()).split(' ')
# co = find_bigrams(words_2)
# print(co)
# co = find_trigrams(words_2)
# print(co)
# co = find_quadgrams(words_2)
# print(co)


# In[11]:


# words_3 = str(train[train.label == 3].tweet.values.tolist()).split(' ')
# co = find_bigrams(words_3)
# print(co)
# co = find_trigrams(words_3)
# print(co)
# co = find_quadgrams(words_3)
# print(co)


# ## [2.4] Histogram for tweet length

# In[12]:


# wx = {}
# i=0
# for t in train.tweet.values.tolist() :
#     if len(str(t)) in wx:
#         wx[len(str(t))] += 1
#     else:
#         wx[len(str(t))] = 1


# In[13]:


# w = [] 
# x = [] 
# for k in wx:
#     x.append(k)
#     w.append(wx[k])


# In[14]:


# plt.hist(x=x, weights=w, bins = int(len(x)))
# plt.title('Number of tweet vs. tweets length')
# plt.xlabel='Number of tweets'
# plt.ylabel='tweets length'
# plt.show()


# ## [2.5] Trending hashtags

# In[15]:


def trending_hashtags(words, maxN, work=1):
    """Process tweet function.
    Input:
        words: words of the tweets
        maxN: max number of trending hashtags returned
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        trending hashtags

    """
    hashtags = {}
    for word in words:
        if word in hashtags and word and word[0] == '#':
            hashtags[word] += 1
        else:
            hashtags[word] = 1
    trend = dict(sorted(hashtags.items(), key = itemgetter(1), reverse = True)[:maxN])
    
    w = [] 
    x = [] 
    for k in trend:
        x.append(k)
        w.append(trend[k])
    
    plt.hist(x=x, weights=w, bins = int(len(x)))
    plt.title('Number of tweet vs. tweets length')
    plt.xticks(rotation=90)
    plt.show()
    
    return trend


# In[16]:


# trending_hashtags(words, 10)


# ## [2.6] Trending hashtags for each label

# In[17]:


# trending_hashtags(words_0, 10)


# In[18]:


# trending_hashtags(words_1, 10)


# In[19]:


# trending_hashtags(words_2, 10)


# In[20]:


# trending_hashtags(words_3, 10)


# ## [3.1] Remove URLs & mentions

# # اولا نحذف الأسطر التي تحوي قيم فارغة

# In[21]:


# tweets = []
# for t in train.tweet.values:
#     if isinstance(t, str):
#         tweets.append(t)


# In[22]:


def remove_urls(tweets, work=1):
    """Process tweet function.
    Input:
        tweets: list of tweets
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        cleaned_tweets: tweet after removing urls

    """
    if work == 0:
        return tweets
    cleaned_tweets = []
    for t in tweets:
        cleaned_tweets.append(re.sub(r"http\S+", " ",t))
    return cleaned_tweets


# In[23]:


def remove_mentions(tweets, work=1):
    """Process tweet function.
    Input:
        tweets: list of tweets
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        cleaned_tweets: tweet after removing mentions

    """
    if work == 0:
        return tweets
    cleaned_tweets = []
    for t in tweets:
        cleaned_tweets.append(re.sub("@([a-zA-Z0-9_]{1,50})", " ",t))
    return cleaned_tweets


# In[24]:


# tweets_no_urls = remove_urls(tweets)
# tweets_no_url_mentions = remove_mentions(tweets_no_urls)
# tweets_no_url_mentions[:5]


# ## لا نحذف الهاشتاغات لأنها جزء من محتوى النص تساعد على فهم المعنى 

# ## [3.2] Remove duplicated characters

# In[25]:


def remove_dup_cahrs(tweets, work=1):
    """Process tweet function.
    Input:
        tweets: list of tweets
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        cleaned_tweets: tweet after removing duplicated chars

    """
    if work == 0:
        return tweets
    cleaned_tweets = []
    for t in tweets:
        cleaned_tweets.append(re.sub(r'(.)\1+', r'\1', t))
    return cleaned_tweets


# ## لتوحيد الكلمات التي تملك نفس المعنى

# In[26]:


# tweets_no_dup_cahrs = remove_dup_cahrs(tweets_no_url_mentions)


# In[27]:


# tweets_no_url_mentions[23]


# In[28]:


# tweets_no_dup_cahrs[23]


# ## [3.3] Unify words

# In[29]:


def spell_check(tweets, work=1):
    """Process tweet function.
    Input:
        tweets: list of tweets
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        cleaned_tweets: tweet after spell_check

    """
    if work == 0:
        return tweets
    
    corr = Corrector()
    cleaned_tweets = []
    for tweet in tweets:
        cleaned_tweet = ''
        for word in tweet.split(' '):
            ch = corr.spell_correct(word)
            if ch != True:
                word = ch[0][0]
            cleaned_tweet += word + ' '
        cleaned_tweets.append(cleaned_tweet)
    return cleaned_tweets


# ## خطوة مفيدة ولكن عملية التصحيح ليست دقيقة وقد تغير المعنى المقصود من النص

# In[30]:


# spell_check(tweets[:2])


# ## [3.4] Unify Numbers

# In[31]:


def replace_numbers(tweets, work=1):
    """Process tweet function.
    Input:
        tweets: list of tweets
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        cleaned_tweets: tweet after removing duplicated chars

    """
    if work == 0:
        return tweets
    cleaned_tweets = []
    for t in tweets:
        cleaned_tweets.append(re.sub(r"\d+", "NUM", t))
    return cleaned_tweets


# تكون الأرقام مهمة في الإحصائيات

# يمكن عدم حذف الأرقام في حال كانت نسبة مئوية

# In[32]:


# tweets_replaced_numbers = replace_numbers(tweets_no_dup_cahrs)


# In[33]:


# tweets_no_dup_cahrs[7]


# In[34]:


# tweets_replaced_numbers[7]


# ## [3.5] Remove stop words

# In[35]:


def remove_stop_words(tweets, work=1):
    """Process tweet function.
    Input:
        tweets: list of tweets
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        cleaned_tweets: tweet after removing stop words

    """
    if work == 0:
        return tweets
    cleaned_tweets = []
    for tweet in tweets:
        tweet_no_stopwords = ''
        for word in tweet.split(' '):
            if not word in nltk.corpus.stopwords.words('arabic'):
                tweet_no_stopwords = tweet_no_stopwords + ' ' + word
        cleaned_tweets.append(tweet_no_stopwords)
    return cleaned_tweets


# ## حذفها ليس بالأمر الجيد دائما لأنه ممكن أن يتغير المعنى عند حذفها في بعض الحالات

# In[36]:


def remove_emojis(tweets, work=1):
    """Process tweet function.
    Input:
        tweets: list of tweets
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        cleaned_tweets: tweet after removing emojis

    """
    if work == 0:
        return tweets
    cleaned_tweets = []
    for tweet in tweets:
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
        cleaned_tweets.append(emoji_pattern.sub(r'',tweet ))
    return cleaned_tweets


# In[37]:


# tweets_no_stopwords = remove_stop_words(tweets_replaced_numbers[:10])


# In[38]:


# tweets_no_emojis = remove_emojis(tweets_no_stopwords)


# In[39]:


# tweets_replaced_numbers[7]


# In[40]:


# tweets_no_emojis[7]


# ## [3.6] Keep arabic text

# In[57]:


def keep_arabic(tweets, work=1):
    """Process tweet function.
    Input:
        tweets: list of tweets
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        cleaned_tweets: tweet after keeping arabic words and hashtags

    """
    if work == 0:
        return tweets
    cleaned_tweets = []
    pattern = re.compile("([\u0627-\u064a]+) | #\w+")
    for tweet in tweets:
        tweet_ar = ''
        for word in tweet.split(' '):
            if pattern.match(word):
                tweet_ar = tweet_ar + ' ' + word
        cleaned_tweets.append(tweet_ar)
    return cleaned_tweets


# ## [3.7] Lemmatization

# In[80]:


def lemmatize(tweets, work=1):
    """Process tweet function.
    Input:
        tweets: list of tweets
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        cleaned_tweets: tweet after Lemmatization

    """
    if work == 0:
        return tweets
    cleaned_tweets = []
    lemmer = qalsadi.lemmatizer.Lemmatizer()
    for tweet in tweets:
        lemmas = lemmer.lemmatize_text(tweet)
        tweet_lem = ''
        for word in lemmas:
            tweet_lem = tweet_lem + ' ' + word
        cleaned_tweets.append(tweet_lem)
    return cleaned_tweets


# In[81]:


# tweet_lemmatized = lemmatize(tweets_no_emojis[7:8])
# print(tweets_no_emojis[7:8])
# tweet_lemmatized


# ## [3.8] Normalization

# In[82]:


def normalize(tweets, work=1):
    """Process tweet function.
    Input:
        tweets: list of tweets
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        cleaned_tweets: tweet after Normalization

    """
    if work == 0:
        return tweets
    cleaned_tweets = []
    for tweet in tweets:
        for form in  ('أ', 'إ', 'آ',):
            tweet = tweet.replace(form, 'ا')
        
        tweet = tweet.replace('ة', 'ه')
        
        tweet = tweet.replace('ئ', 'ء')
        tweet = tweet.replace('ؤ', 'ء')
        
#         tweet = tweet.replace('ي', 'ى')
        

        cleaned_tweets.append(tweet)
    return cleaned_tweets


# In[83]:


# norm = normalize(tweets[2463:2464])
# print(tweets[2463:2464])
# norm


# # خطوة مهمة لكي لا تكون الداتا Sparse 

# #### بحيث لا توجد طريقة موحدة عند كتابة الأحرف المذكورة

# #### يمكن أيضا إزالة الحركات والتنوين

# ## Extra: diacritic removal

# In[84]:


def diacritic_removal(tweets, work=1):
    """Process tweet function.
    Input:
        tweets: list of tweets
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        cleaned_tweets: tweet after Normalization

    """
    if work == 0:
        return tweets
    
    DIACRITICS = (
    'َ',  # Fatha
    'ُ',  # Damma
    'ِ',  # Kasra
    'ً',  # Tanween Fath
    'ٌ',  # Tanween Damm
    'ٍ',  # Tanween Kase
    'ْ',  # Sokoon
    'ّ',  # Shadda
    'ـ', # Tatweel
    )
    
    cleaned_tweets = []
    for tweet in tweets:
        for diacritic in DIACRITICS:
            tweet = tweet.replace(diacritic, '')
        cleaned_tweets.append(tweet)
    return cleaned_tweets


# In[85]:


# t = diacritic_removal(norm)
# print(norm)
# t


# In[86]:


def remove_duplicate_tweets(tweets, work=1):
    """Process tweet function.
    Input:
        tweets: list of tweets
        work: binary value take 1 by default working as on/off for the function, if work=0 the function will return
     the tweet without changing
    Output:
        tweets after removing duplicated tweets

    """
    if work == 0:
        return tweets
    
    return list(set(tweets))


# In[87]:


def preprocess(tweet, flags=[1,1,1,1,1,1,1,1,1,1]):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
        flags: list of "work" values for all functions will called here.
    Output:
        cleaned_tweet: tweet after apply all cleaning and normlizaing functions

    """
    tweet = remove_urls(tweet, flags[0])
    tweet = remove_mentions(tweet, flags[1])
    tweet = remove_dup_cahrs(tweet, flags[2])
    tweet = spell_check(tweet, flags[3])
    tweet = replace_numbers(tweet, flags[4])
    tweet = remove_stop_words(tweet, flags[5])
    tweet = remove_emojis(tweet, flags[6])
    tweet = lemmatize(tweet, flags[7])
    tweet = diacritic_removal(tweet, flags[8])
    tweet = remove_duplicate_tweets(tweet, flags[9])
    ...
    return tweet

