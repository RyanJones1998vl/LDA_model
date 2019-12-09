"""
import pandas as pd
import csv 
data=pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)
data_text=data[['headline_text']]
data_text['index']=data_text.index
documents=data_text[:10000]

documents_=[]
with open('content.csv') as csv_file:
   reader=csv.reader(csv_file)
   for row in csv_file:
           documents_.append(row)

"""
"""
import gensim 
from collections import defaultdict
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy
numpy.random.seed(2018)

import nltk
nltk.download('wordnet')

stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    yield result


processed_docs = preprocess(documents_[1])

dictionary = gensim.corpora.Dictionary(processed_docs)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

from gensim import corpora, models
from pprint import pprint


lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=10, id2word=dictionary, passes=10)



for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

"""
import csv
import re
stopwords=[]
with open('vietnamese-stopwords.txt','r') as file:
        lines = file.readlines()
        for line in lines:
                stopwords.append(line.replace('\n', '').replace(' ', '_'))

file.close()

data=[]
with open('content.csv', 'r') as file:
        sentences = csv.reader(file, delimiter="\n")
        for sentence in sentences:
                data.append(''.join(sentence))

unspacedDictionary=[]
with open('dict.txt', 'r') as file:
        words=file.readlines()
        for word in words:
                unspacedDictionary.append(word.replace('\n', ''))

spacedDictionary=[word.replace('_', ' ') for word in unspacedDictionary]
'''
data=data[:20]

data.append("Hello and " + spacedDictionary[1])
data.append("Hello and " + spacedDictionary[3])
'''
_documents=[re.sub('\"', ' ', line) for line in data]
_documents=[re.sub('\d', ' ', line) for line in _documents]
_documents=[re.sub('  ', ' ', line) for line in _documents]
_documents=[re.sub("/", " ", line) for line in _documents]
_documents=[re.sub('\xa0', ' ', line) for line in _documents]

documents=_documents
'''
for line in _documents:
        out=line
        for w in spacedDictionary:
                if w in out:
                        re.sub(w, unspacedDictionary[spacedDictionary.index(w)], out)
        documents.append(out)
'''
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=False))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in texts]

_data_words=list(sent_to_words(documents))
data_words= remove_stopwords(_data_words)
id2word = corpora.Dictionary(data_words)
corpus = [id2word.doc2bow(text) for text in data_words]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='asymmetric',
                                           per_word_topics=True)

topics = lda_model.print_topics()
for topic in topics:
    print(topic)