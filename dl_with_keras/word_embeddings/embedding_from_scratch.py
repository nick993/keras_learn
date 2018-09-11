import numpy as np
import nltk
from keras.preprocessing.text import Tokenizer, one_hot
from sklearn.preprocessing.data import OneHotEncoder
from sklearn.model_selection import train_test_split

np.random.seed(12)

BATCH_SIZE = 128
NO_EPOCHS = 20

lines = []
fin = open('alice_in_wonderland.txt', 'r')
for line in fin:
    #line = line.strip().decode('ascii', 'ignore').encode('utf-8')
    line = line.strip()
    if len(line) != 0:
        lines.append(line)

fin.close()
sents = nltk.sent_tokenize(" ".join(lines))

tokenizer = Tokenizer(5000)
tokens = tokenizer.fit_on_texts(sents)
vocab_size = len(tokenizer.word_counts) + 1

xs = []
ys = []
for sent in sents:
    embedding = one_hot(sent, vocab_size)
    triples = list(nltk.trigrams(embedding))
    w_lefts = [x[0] for x in triples]
    w_centers = [x[1] for x in triples]
    w_rights = [x[2] for x in triples]
    xs.extend(w_centers)
    ys.extend(w_lefts)
    xs.extend(w_centers)
    ys.extend(w_rights)

ohe = OneHotEncoder(n_values=vocab_size)
X = ohe.fit_transform(np.array(xs).reshape(-1, 1)).todense()
Y = ohe.fit_transform(np.array(ys).reshape(-1, 1)).todense()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3,
                                                random_state=42)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)


print('End')

