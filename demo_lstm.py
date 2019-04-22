import pandas as pd
import numpy as np

data = pd.read_csv("data/kaggle_corpus/ner_dataset.csv", encoding="latin1")

data = data.fillna(method="ffill")

words = list(set(data["Word"].values))
n_words = len(words)


tags = list(set(data["Tag"].values))
n_tags = len(tags)


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sentences = getter.sentences
# print(sentences[0])

# import matplotlib.pyplot as plt
# plt.style.use("ggplot")

# #plot sentence distribution of the dataset
# plt.hist([len(s) for s in sentences],bins = 50)
# plt.show()

#build dictionary of words,tags
max_len = 50
word2idx = {w: i for i, w in enumerate(words)} # {word: word_id}
tag2idx = {t: i for i, t in enumerate(tags)} # {tag: tag_id}

# print(word2idx.keys())
# exit()
# print(tag2idx.keys())

# print(sentences)
# print(sentences[0])
'''
[('Thousands', 'NNS', 'O'), ('of', 'IN', 'O'), ('demonstrators', 'NNS', 'O'), ('have',
'VBP', 'O'), ('marched', 'VBN', 'O'), ('through', 'IN', 'O'), ('London', 'NNP', 'B-geo'
), ('to', 'TO', 'O'), ('protest', 'VB', 'O'), ('the', 'DT', 'O'), ('war', 'NN', 'O'), (
'in', 'IN', 'O'), ('Iraq', 'NNP', 'B-geo'), ('and', 'CC', 'O'), ('demand', 'VB', 'O'),
('the', 'DT', 'O'), ('withdrawal', 'NN', 'O'), ('of', 'IN', 'O'), ('British', 'JJ', 'B-
gpe'), ('troops', 'NNS', 'O'), ('from', 'IN', 'O'), ('that', 'DT', 'O'), ('country', 'N
N', 'O'), ('.', '.', 'O')]
'''
# print(word2idx)
# exit()

from keras.preprocessing.sequence import pad_sequences
#[
# [[word_id],...[word_id]] # 1 sent
# ]
X = [[word2idx[w[0]] for w in s] for s in sentences]
y = [[tag2idx[w[2]] for w in s] for s in sentences]
# print(X[:2])
'''
[
    [1636, 14562, 523, 2647, 15895, 21545, 9779, 24781, 12282, 31440, 18371, 1665, 11858,
    5637, 20017, 31440, 1913, 14562, 24509, 32086, 2963, 25885, 4951, 25841], [5176, 19649,
     20753, 9249, 23473, 24781, 9816, 1063, 24781, 15847, 21292, 25388, 14562, 31440, 28976
    , 372, 34765, 21884, 26702, 33577, 12662, 28650, 11646, 34584, 25841]
]
'''
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
# print(n_words) #35178
# print(n_tags) # 17

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


y = [to_categorical(i,num_classes = n_tags) for i in y]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = 0.1)

#fit a LSTM network with an embedding layer
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout,Bidirectional

#max_len = 50
#input_length = sequence length# # of batches or # of instances??
#input_dim = # of features
# eg.
#   32 = input_shape/batch_size
#   (32,10,16) = a batch of 32 samples, where each sample is a sequence of 10 vectors of 16 dimensions.
input = Input(shape=(max_len,)) #dimension of input = max_len = 50
model = Embedding(input_dim = n_words, output_dim =50,input_length = max_len)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences = True, recurrent_dropout =0.1))(model)
out = TimeDistributed(Dense(n_tags, activation = "softmax"))(model) # softmax output

model = Model(input,out)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
# print(y_tr[0])
# print(type(y_tr))
# exit()
history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)


hist = pd.DataFrame(history.history)


# # plot result
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,12))
# plt.plot(hist["acc"])
# plt.plot(hist["val_acc"])
# plt.show()

# evalutation

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

test_pred = model.predict(X_te, verbose=1)

idx2tag = {i: w for w, i in tag2idx.items()}


def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out


pred_labels = pred2label(test_pred)
test_labels = pred2label(y_te)

print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))

print(classification_report(test_labels, pred_labels))

#look at some of the prediction
i = 1927
p = model.predict(np.array([X_te[i]]))
p = np.argmax(p, axis=-1)
true = np.argmax(y_te[i], -1)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_te[i], true, p[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(words[w-1], tags[t], tags[pred]))

# inference with bilstm-crf

test_sentence = ["Hawking", "was", "a", "Fellow", "of", "the", "Royal", "Society", ",", "a", "lifetime", "member",
                 "of", "the", "Pontifical", "Academy", "of", "Sciences", ",", "and", "a", "recipient", "of",
                 "the", "Presidential", "Medal", "of", "Freedom", ",", "the", "highest", "civilian", "award",
                 "in", "the", "United", "States", "."]

x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                            padding="post", value=0, maxlen=max_len)

p = model.predict(np.array([x_test_sent[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(test_sentence, p[0]):
    print("{:15}: {:5}".format(w, tags[pred]))





