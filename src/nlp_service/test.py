# -*- coding: utf-8 -*-
from scipy import linalg
from sklearn import mixture
from sklearn.cluster import DBSCAN
import os
import re
from scipy import spatial
import numpy as np
from nltk.corpus import stopwords
import time
from textblob import TextBlob
from gensim.models.keyedvectors import KeyedVectors
from pattern3.fr import singularize
from nltk.tokenize import ToktokTokenizer
from nltk.tokenize import sent_tokenize

word_vectors = KeyedVectors.load_word2vec_format('non-lem.bin', binary=True)
# word_vectors = KeyedVectors.load_word2vec_format('lem.bin', binary=True)
np.seterr(all='raise')
stoppers = stopwords.words('french')

def extract(filename):
    returnList = {}
    file = open('../../text_bk/' + filename, encoding="ISO-8859-1")
    factMatch = re.compile('\[\d+\](.*\n){0,3}?(?=\[\d+\]|\n)', re.M)
    lines = file.read()
    for match in factMatch.findall(lines):
        sentence = re.sub('\[\d+\]\s*', '',
                          match).strip().replace('\n', '').replace('\t', '')
        for sent in sent_tokenize(sentence):
            vector = vectorize(sent)
            if vector is not None and np.sum(vector) != 0:
                returnList[sent] = vector
    return returnList


def vectorize(sentence):
    tok = ToktokTokenizer()
    vec = np.zeros(500)
    numWords = 0
    for words in tok.tokenize(sentence):
        for word in words.split('-'):
            word = word.lower().strip('.0123456789')
            if word not in stoppers:
                newWord = np.zeros(500)
                try:
                    newWord = word_vectors.wv[word]
                    numWords += 1
                    vec = np.add(vec, newWord)
                except:
                    try:
                        newWord = word_vectors.wv[singularize(word)]
                        numWords += 1
                        vec = np.add(vec, newWord)
                    except:
                        errorWords.add(word)
    if numWords < 1:
        return None
    return np.divide(vec, numWords)


def getClustered(sentences):
    X = np.matrix(list(sentences.values()))
    ms = DBSCAN()
    ms.fit(X)
    labels = ms.labels_
    import pdb; pdb.set_trace()

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    for i, sent in enumerate(sentences.keys()):
            f = open('outputs/' + str(labels[i]), 'a')
            f.write('{:.140}'.format(sent) + '\n')
            f.close()

    print("number of estimated clusters : %d" % n_clusters_)
    return ms


sentences = {}
errorWords = set()
start = time.time()
j = 0
for i in os.listdir("../../text_bk"):
    if 'AZ-51' in i and j < 3000:
        j += 1
        intermed = time.time()
        sentences.update(extract(i))

done = time.time()
fileError = open('error_words', 'w')
for err in errorWords:
    fileError.write(err + '\n')
fileError.close()
print(errorWords)
elapsed = done - start
print('Vector time:')
print(elapsed)

ms = getClustered(sentences)


done = time.time()
elapsed = done - start

print('End time:')
print(elapsed)
