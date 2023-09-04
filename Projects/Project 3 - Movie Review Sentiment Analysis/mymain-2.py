#####################################
# Load libraries
# Load your vocabulary and training data
#####################################

import pandas as pd
import sklearn
import string
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import metrics
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
import csv

words = pd.read_table("myvocab.txt")
word_list = list(words['0'])

df = pd.read_table("train.tsv")
sentiment_list = list(df['sentiment'])
corpus = []

for document in df['review']:
    clean = document.translate(str.maketrans('','',string.punctuation)).lower()
    words = clean.split()
    filtered = []
    for word in words:
        if word[len(word) - 2] == 'b' and word[len(word) - 1] == 'r':
                word = word[0:(len(word) - 2)]
        filtered.append(word)
    new_string = ' '.join(filtered)
    corpus.append(new_string)

vectorizer = CountVectorizer(ngram_range = (1,2))
Xc = vectorizer.fit_transform(corpus)
names = vectorizer.get_feature_names_out()
names = list(names)
inds = np.zeros((len(word_list)))

for i in range(len(word_list)):
    if word_list[i] in names:
        inds[i] = names.index(word_list[i])
    else:
        inds[i] = -1

traindata = []

for i in range(len(word_list)):
    if inds[i] < 0:
        traindata.append(np.zeros((len(corpus))))
    else:
        traindata.append(np.array(Xc[:,inds[i]].toarray()).flatten())

traindata = np.array(traindata).T
trainlabels = np.array(sentiment_list)


#####################################
# Train a binary classification model
#####################################

model = Sequential()
model.add(Dense(20, input_shape=(980,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', metrics=[metrics.AUC()])
model.fit(traindata, trainlabels, epochs=20, verbose=2)

#####################################
# Load test data, and
# Compute prediction
#####################################

df = pd.read_table("test.tsv")
sentiment_list = list(df['sentiment'])
corpus = []


for document in df['review']:
    clean = document.translate(str.maketrans('','',string.punctuation)).lower()
    words = clean.split()
    filtered = []
    for word in words:
        if word[len(word) - 2] == 'b' and word[len(word) - 1] == 'r':
                word = word[0:(len(word) - 2)]
        filtered.append(word)
    new_string = ' '.join(filtered)
    corpus.append(new_string)

vectorizer = CountVectorizer(ngram_range = (1,2))
Xc = vectorizer.fit_transform(corpus)
names = vectorizer.get_feature_names_out()
names = list(names)
inds = np.zeros((len(word_list)))

for i in range(len(word_list)):
    if word_list[i] in names:
        inds[i] = names.index(word_list[i])
    else:
        inds[i] = -1

testdata = []

for i in range(len(word_list)):
    if inds[i] < 0:
        testdata.append(np.zeros((len(corpus))))
    else:
        testdata.append(np.array(Xc[:,inds[i]].toarray()).flatten())

testdata = np.array(testdata).T
testlabels = np.array(sentiment_list)

Ypred = model.predict(testdata)

#####################################
# Store your prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predicted probs
#####################################

dict_out = {'id': list(df['id']), 'prob': list(Ypred.flatten())}

df_out = pd.DataFrame.from_dict(dict_out)

df_out.to_csv('mysubmission.txt', index = False, quoting = csv.QUOTE_NONNUMERIC)


