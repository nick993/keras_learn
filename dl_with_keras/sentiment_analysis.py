import pandas as pd
import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Activation
from keras.models import Model
import re

from keras.utils import to_categorical
from dl_with_keras.util import read_glove_vecs

def clean(cont):
    cont = re.sub(r'@[^\s]*', '', cont)
    cont = re.sub(r'\.*', '', cont)
    cont = re.sub(r'[^a-zA-Z0-9\s\']', ' ', cont)
    cont = re.sub(r'n\'t', ' not', cont)
    cont = re.sub(r'\'ve', '  have', cont)
    cont = re.sub(r'\'s', '  is', cont)
    return cont

def check_percent_of_unknowns(x_train, word_to_idx):
    count = 0
    total_count = 0
    unk_index = word_to_idx['unk']
    for sent in x_train:
        for no in sent:
            if no == unk_index:
                count += 1
            total_count += 1

    return (1.0 * count)/total_count



def convert_to_nos(cont, word_to_idx):
    splitted_sent = cont.split()
    return [word_to_idx[word] if word_to_idx.__contains__(word) == True else word_to_idx['unk'] for word in splitted_sent]


def zero_append_vec(x_train, max_vec_len):
    cnt = 0
    for sent in x_train:
        if cnt % 1000 == 0:
            print('zero append vec ind : ' + str(cnt))
        len_diff = max_vec_len - len(sent)
        sent = np.pad(sent, (0, len_diff), 'constant')
        cnt += 1


def pre_trained_emb_model(word_to_vec_map, word_to_idx):
    vocab_len = len(word_to_index)
    emb_len = word_to_vec_map['the'].shape[0]

    embed_matrix = np.zeros((vocab_len, emb_len))

    for word, index in word_to_idx.items():
        embed_matrix[index:] = word_to_vec_map[word]

    embedding_layer = Embedding(input_shape=vocab_len, output_dim=emb_len, trainable=False)

    embedding_layer.build((None,))

    embedding_layer.set_weights([embed_matrix])

    return embedding_layer


def create_model(inp_shape, output_dim, word_to_vec_map, word_to_idx):
    input = Input(shape=inp_shape, dtype='int32')

    embedding_layer = pre_trained_emb_model(word_to_vec_map, word_to_idx)
    embeddings = embedding_layer(input)

    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(output_dim)(X)
    X = Activation('softmax')(X)

    model = Model(inputs=input, outputs=X)

    return model



word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
#print('word_to_index size : ' + str(len(word_to_index)))

train_data = pd.read_csv('sentiment_train_data.csv')
#print(train_data.describe())

sentiment_list = train_data.sentiment.unique()
sentiment_map = dict(map(lambda t: (t[1], t[0]), enumerate(sentiment_list)))
#print(sentiment_map)

y_train = [sentiment_map[sentiment] for sentiment in train_data.sentiment]
x_train = [convert_to_nos(clean(cont.lower()), word_to_index) for cont in train_data.content]
max_vec_len = max([len(c) for c in x_train])
max_output = max(y_train)




print('Unknowns fractions in training_set ' + str(check_percent_of_unknowns(x_train, word_to_index)))
print('Max vec len : ' + str(max_vec_len))
print('Unique output : ' + str(max_output))
y_train = to_categorical(y_train, num_classes=max_output + 1)
zero_append_vec(x_train, max_vec_len)

model = create_model(inp_shape=(max_vec_len,), output_dim=max_output, word_to_vec_map=word_to_vec_map, word_to_idx=word_to_index)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=50, shuffle=True)


#print(y_train[1:5])










