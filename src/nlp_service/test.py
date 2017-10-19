# -*- coding: utf-8 -*-
import os, re, mmap, nltk
from scipy import spatial
import numpy as np
from nltk.stem.snowball import FrenchStemmer
from textblob import TextBlob
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from pattern3.fr import singularize
word_vectors = KeyedVectors.load_word2vec_format('test.bin', binary=True)
import time
np.seterr(all='raise')
f = open('outputs', 'w')
def extract(filename):
    returnList = {}
    file = open('../../../text_bk/' + filename, encoding = "ISO-8859-1")
    factMatch = re.compile('\[\d+\](.*\n){0,3}?(?=\[\d+\]|\n)',re.M)
    lines = file.read()
    for match in factMatch.findall(lines):
        sentence = re.sub('\[\d+\]\s*', '', match).strip().replace('\n','').replace('\t','')
        f.write(sentence + '\n')
        vector = vectorize(sentence)
        if vector is not None and np.sum(vector) != 0:
            returnList[sentence] = vector
    return returnList

def vectorize(sentence):
    vec = np.zeros(200)
    numWords = 0
    for word in TextBlob(sentence).words.lower():
        newWord = np.zeros(200)
        try:
            word = singularize(word)
            if word == 'locateur':
                word = 'locataire'
            if not re.match('\d', word) and re.match('\S', word):
                numWords += 1
                newWord = word_vectors.wv[word]
                vec = np.add(vec,newWord)
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

sentences = {}
errorWords = set()
start = time.time()
j = 0
for i in os.listdir("../../../text_bk"):
    if 'AZ-51' in i and j < 300:
        j += 1
        intermed = time.time()
        sentences.update(extract(i))

done = time.time()
elapsed = done - start
print(elapsed)
# print(errorWords)
# key1 = next(iter(sentences.keys()))

# print('------------------------- SEARCH PHRASES')
# print(key1)
# print('------------------------- START BEST RESULT PHRASE')
# findClosest(key1):
# print(actKey)
# print('------------------------- END RESULT')
test = 'Le locataire doit payer les dommages'
sentences[test] = vectorize(test)
print(findClosest(test))