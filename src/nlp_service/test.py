# -*- coding: utf-8 -*-
from scipy import linalg
from sklearn import mixture
from sklearn.cluster import MeanShift, estimate_bandwidth
import os
import re
from scipy import spatial
import numpy as np
from nltk.corpus import stopwords
import time
from textblob import TextBlob
from gensim.models.keyedvectors import KeyedVectors
from pattern3.fr import singularize

word_vectors = KeyedVectors.load_word2vec_format('test.bin', binary=True)
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
        vector = vectorize(sentence)
        if vector is not None and np.sum(vector) != 0:
            returnList[sentence] = vector
    return returnList


def vectorize(sentence):
    vec = np.zeros(200)
    numWords = 0
    for word in TextBlob(sentence).words.lower():
        if word not in stoppers:
            newWord = np.zeros(200)
            try:
                word = singularize(word)
                if word == 'locateur':
                    word = 'locataire'
                if not re.match('\d', word) and re.match('\S', word):
                    numWords += 1
                    newWord = word_vectors.wv[word]
                    vec = np.add(vec, newWord)
            except:
                errorWords.add(word)
    if numWords < 1:
        return None
    return np.divide(vec, numWords)


def findClosest(key1):
    maxi = 0
    actKey = ''
    for key2 in sentences.keys():
        # try:
        if key1 != key2:
            ke1 = sentences[key1]
            ke2 = sentences[key2]
            k1 = np.transpose(sentences[key1])
            k2 = np.transpose(sentences[key2])
            dist = 1 - spatial.distance.cosine(ke1, ke2)
            if dist > maxi:
                maxi = dist
                actKey = key2
        # except as e:
        #     print(e)
        #     print(key1, '---------' ,key2, sentences[key1], sentences[key2])
        #     exit()
    return actKey


def getClustered(sentences):
    X = np.matrix(list(sentences.values()))
    ms = MeanShift()
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)
    return ms

sentences = {}
errorWords = set()
start = time.time()
j = 0
for i in os.listdir("../../text_bk"):
    if 'AZ-51' in i and j < 10000:
        j += 1
        intermed = time.time()
        sentences.update(extract(i))

done = time.time()
elapsed = done - start
print('Vector time:')
print(elapsed)

ms = getClustered(sentences)
# test = 'Le locataire doit payer les dommages'
# inp = vectorize(test)

# import pdb; pdb.set_trace()
# ms.predict(inp)

start = done

for sent, vector in sentences.items():
    lab = ms.predict([vector])
    f = open('outputs/' + str(lab[0]), 'a')
    f.write(str(lab[0]) + ', ' + '{:.200}'.format(sent) + '\n')
    f.close()
# print(errorWords)
# key1 = next(iter(sentences.keys()))

# print('------------------------- SEARCH PHRASES')
# print(key1)
# print('------------------------- START BEST RESULT PHRASE')
# findClosest(key1):
# print(actKey)
# print('------------------------- END RESULT')
# test = 'Le locataire doit payer les dommages'
# sentences[test] = vectorize(test)
# print(findClosest(test))

# X = np.array()
# for val in sentences.values():
#     X = n

done = time.time()
elapsed = done - start

print('End time:')
print(elapsed)
