import os
from urllib.request import urlretrieve
from keras.preprocessing import sequence
from keras.layers import Input, Embedding, Reshape, Dense, merge, Dot
from keras.models import Model
import zipfile
import numpy as np
import tensorflow as tf
import re
from collections import Counter




# Preparing the data
def maybe_download(file_name, url, expected_bytes):
    if not os.path.exists(file_name):
        file_name, _ = urlretrieve(url + file_name, file_name)
    statinfo = os.stat(file_name)
    if statinfo.st_size == expected_bytes:
        print('Found and Verified', file_name)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + file_name)

    return file_name

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def format_and_clean_words(words):
    cleaned_words = []
    for word in words:
        replaced = re.sub('[^a-zA-Z0-9]', '', word)
        if (replaced and replaced != ''):
            cleaned_words.append(replaced.lower())
    return cleaned_words

def read_data_alice(filename):
    words = []
    fin = open(filename, 'r')
    for line in fin:
        # line = line.strip().decode('ascii', 'ignore').encode('utf-8')
        line = line.strip()
        if line and len(line) != 0:
            words = words + line.split()
    return words


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(Counter(words).most_common(n_words - 1))
    word_dict = dict()
    for word, _ in count:
        word_dict[word] = len(word_dict)
    data = list()
    unk_count = 0
    for word in words:
        if word in word_dict:
            index = word_dict[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    rev_dict = dict(zip(word_dict.values(), word_dict.keys()))
    return data, count, word_dict, rev_dict

vocab_size = 2500
data, count, dictionary, reverse_dictionary = build_dataset(format_and_clean_words(read_data_alice('alice_in_wonderland.txt')), vocab_size)
vocab_size = min(vocab_size, reverse_dictionary.__len__())
#data, count, dictionary, reverse_dictionary = build_dataset(read_data('text8.zip'), vocab_size)

print(data[0:7])

window_size = 5
vector_dim = 100
epochs = 1000000
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

sampling_table = sequence.make_sampling_table(vocab_size)
couples, labels = sequence.skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype='int32')
word_context = np.array(word_context, dtype='int32')
print(couples[:10], labels[:10])

# create inp variables
inp_target = Input((1, ))
inp_context = Input((1, ))
embedding = Embedding(input_dim=vocab_size, output_dim=vector_dim, input_length=1, name='embedding')
target = embedding(inp_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(inp_context)
context = Reshape((vector_dim, 1))(context)

similarity = Dot(axes=0, normalize=True)([target, context])
dot_prod = Dot(axes=1, normalize=False)([target, context])
dot_prod = Reshape((1, ))(dot_prod)

output = Dense(1, activation='sigmoid')(dot_prod)
model = Model(inputs = [inp_target, inp_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='adam')

validation_model = Model(inputs=[inp_target, inp_context], output=similarity)
counter = 1

class SimilarityCallback:
    def run_sim(self, counter=0):
        model.save('alice_txt_model' + str(counter) + '.h5')
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_words_idx):
        sim = np.zeros((vocab_size, ))
        in_arr1 = np.zeros((1, ))
        in_arr2 = np.zeros((1, ))
        for i in range(vocab_size):
            in_arr1[0, ] = valid_words_idx
            in_arr2[0, ] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

sim_cb = SimilarityCallback()

arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))
    if cnt > 0 and cnt % 10000 == 0:
        sim_cb.run_sim(cnt)

'''
url = 'http://mattmahoney.net/dc/'
filename = maybe_download('text8.zip', url, 31344016)
vocabulary = read_data(filename)
print('Vocab Size', len(vocabulary))
print(vocabulary[:7])
'''