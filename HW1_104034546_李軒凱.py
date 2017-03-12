import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
from nltk import word_tokenize
from textblob import TextBlob as tb
import math
from operator import itemgetter
import warnings
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
warnings.filterwarnings("ignore", category=DeprecationWarning,
module="pandas", lineno=570)
import numpy as np
import io
import os
import csv
import sys
import re
import codecs
from collections import Counter
import os
import csv
import pandas as pd
from pandas import Series,DataFrame
from sklearn import cluster, datasets, metrics
from sklearn.cluster import SpectralClustering
pathProg = 'C:\\Python27'
os.chdir(pathProg)

stopwords = set(stopwords.words('english')) 

import string
cc = string.punctuation
dd ='--'
for symbol in cc:
    stopwords.add(symbol)
    stopwords.update(cc)
    stopwords.add("--")
    stopwords.add("'s")
    stopwords.add("'ve")
    stopwords.add("'re")
    stopwords.add("n't")
    stopwords.add("``")
    stopwords.add("''")
   
bb=[]
file = open (pathProg + '/building_global_community.txt','r')
f = file.read()

bb = f.lower()
bb = word_tokenize(bb)

bb = [w for w in bb if w not in stopwords]

bb = Counter(bb)

aaa=[]

for sent in nltk.pos_tag_sents(word_tokenize(sent) for sent in bb):


 ab=[]
 ab.append(sent[0][0])
 ab.append(sent[0][1])
 k=bb[sent[0][0]]
 ab.append(k)

 aa = (' '.join('{1}:{0}'.format(pos, word) for pos, word in sent))
 aaa.append(ab)

bb = bb.most_common()

with open('wordcount.csv', 'w') as csvfile:
 fieldnames = ['word', 'count']
 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

 writer.writeheader()
 for word, count in bb:
        writer.writerow({'word': word, 'count': count})


l=sorted(aaa, key=lambda tup: tup[1])



with open('wordcount-bonus.csv', 'w') as csvfile:

 fieldnames = ['pos','word', 'count']
 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

 writer.writeheader()
 for word,pos, count in l:
        writer.writerow({'pos': pos,'word': word, 'count': count})

file = open(pathProg + '/wordcount-bonus.csv', 'r')
f = pd.read_csv(file)

f = f.groupby('pos')

for word, count in f :

 f2 = count.sort_values(by = 'count',ascending=False)
 f2 = f2.set_index('pos')

 f2 = pd.DataFrame(f2)
 print (f2)








