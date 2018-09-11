import pandas as pd
import numpy as np
import re

from keras.utils import to_categorical
from dl_with_keras.util import read_glove_vecs

def clean_and_convert(cont):
    cont = re.sub(r'@[^\s]*', '', cont)
    cont = re.sub(r'\.*', '', cont)
    cont = re.sub(r'[^a-zA-Z0-9\s\']', ' ', cont)
    cont = re.sub(r'n\'t', ' not', cont)
    cont = re.sub(r'\'ve', '  have', cont)
    cont = re.sub(r'\'s', '  is', cont)
    return cont

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
#print('word_to_index size : ' + str(len(word_to_index)))

train_data = pd.read_csv('sentiment_train_data.csv')
#print(train_data.describe())

sentiment_list = train_data.sentiment.unique()
sentiment_map = dict(map(lambda t: (t[1], t[0]), enumerate(sentiment_list)))
#print(sentiment_map)

y_train = [sentiment_map[sentiment] for sentiment in train_data.sentiment]
x_train = [clean_and_convert(cont.lower()) for cont in train_data.content]
print(max([len(c) for c in x_train]))
#print(y_train[1:5])










