# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 14:07:09 2022

@author: faikm
"""

import numpy as np
import tensorflow as tf
import random
import json
import pandas as pd
import re
import string

from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
from tensorflow import keras
from keras import layers, Sequential
from tensorflow.python.keras.layers import Activation




delete_list = ["[", "]", """""""", '"', "''", ',', "'", 
               '&quot',':::::', '***','-','̇‘','’', '“', '”',  '…','°', '´', ' –', '‘ ', '3','(', ')']

with open('text.txt', encoding=('utf-8')) as fin, open('cleaned.txt', 'w+' ,encoding=('utf-8')) as fout:
    for line in fin: 
        for word in delete_list:
            line = line.replace(word, "")
        fout.write(line)
fin.close()
fout.close()

text = open('cleaned.txt', 'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]

characters = sorted(set(text))
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_length = 40
STEP_SIZE = 3
sentences = []
next_char = []
for i in range(0, len(text) - SEQ_length, STEP_SIZE):
    sentences.append(text[i: i + SEQ_length])
    next_char.append(text[i + SEQ_length])
    
x = np.zeros((len(sentences), SEQ_length, len(characters)), dtype=(np.bool_))
y = np.zeros((len(sentences), len(characters)), dtype=(np.bool_))
for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1
    
    
    
model = Sequential()
model.add(LSTM(128,
               input_shape=(SEQ_length,
                            len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01))

model.fit(x, y, batch_size=256, epochs=4)



def sample(pred, temperature=1):
    pred = np.asarray(pred).astype('float64')
    pred = np.log(pred) / temperature
    exp_pred = np.exp(pred)
    pred = exp_pred / np.sum(exp_pred)
    probas = np.random.multinomial(1, pred, 1)
    return  np.argmax(probas)

start_index = random.randint(0, len(text) - SEQ_length - 1)
generated =''
sentence = text[start_index: start_index + SEQ_length]
generated += sentence
for i in range(45):
    x_predictions = np.zeros((1, SEQ_length, len(characters)))
    for t, char in enumerate(sentence):
        x_predictions[0, t, char_to_index[char]] = 1
    predictions = model.predict(x_predictions, verbose=0)[0]
   
    next_index = sample(predictions,
                                 0.6)
    next_character = index_to_char[next_index]

    generated += next_character
    sentence = sentence[1:] + next_character
    
print(generated)
    

    
    
    
    
    
    