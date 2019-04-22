import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, f1_score

dframe = pd.read_csv("data/kaggle_corpus/ner.csv", encoding = "ISO-8859-1" ,error_bad_lines = False)

dframe.dropna(inplace = True)
# print(dframe[dframe.isnull().any(axis=1)].size) # 0
# exit()

# keep = lemma, pos
dframe=dframe.drop(['Unnamed: 0', 'next-lemma', 'next-next-lemma', 'next-next-pos',
       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
       'prev-prev-word', 'prev-shape', 'prev-word'],axis=1)

dframe = dframe[:5000]

class SentenceGetter(object):
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w,t) for w,t in zip( s['word'].values.tolist(), s['tag'].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            n_sent = float(self.n_sent)
            # print(type(self.grouped))
            # exit()
            s = self.grouped["{:f}".format(n_sent)]
            self.n_sent +=1
            return s
        except:
            return None

getter = SentenceGetter(dframe)
sentences = getter.sentences


##############################3
### Declair variables
##############################3

word2idx = {w:i for i, w in enumerate(set(dframe.word.values))}
tag2idx = {t:i for i, t in enumerate(set(dframe.tag.values))}
X = [[word2idx[w[0]] for w in s] for s in sentences ]

from keras.preprocessing.sequence import pad_sequences

maxlen = max([len(s) for s in sentences])
n_words = len(set(dframe.word.values))
n_tags = len(set(dframe.tag.values))


# max number of words in a sent 140
X = pad_sequences(maxlen= maxlen, sequences=X, padding = "post", value = n_words -1)

# for each word, pad, its feture to be

y = [[tag2idx[w[1]] for w in s] for s in sentences ]
y = pad_sequences(maxlen= maxlen, sequences=y, padding = "post", value = tag2idx["O"])

from keras.utils import to_categorical
y =[to_categorical(i,num_classes=n_tags ) for i in y ] #indies to vector
# print(np.array(X).shape) # (224, 41)
# print(np.array(y).shape) # (224,41,16)
# exit()

# 224,41, 1


#########################3
### old data preprocessing
#########################

vectorizer = DictVectorizer(sparse= False)
X_df = dframe[["lemma", "pos"]]

X_feature = vectorizer.fit_transform(X_df.to_dict("records")) #numpy.ndarray

#append y_feature to y and x_feature to x
X = np.array(X)

# print(X_feature.shape) # (5000,1439) -> (224,41, 1439)
# print(y.shape)         # (224,41,16)
# print(X.shape)         # (224,41)
exit()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

##############
## build the model
##############

# clf = Perceptron(verbose= 10, n_jobs = -1, n_iter = 5)
# all_classes = list(set(y))
# clf.partial_fit(x_train,y_train, all_classes)
#
# print(f1_score(clf.predict(x_test), y_test, average = "micro"))

from keras.models import Model,Input
from keras.layers import LSTM,Embedding,Dense, TimeDistributed,Dropout, Bidirectional

tags2idx = {t: i for i,t in enumerate(dframe.tag.values)}
words2idx = {w:i for i,w in enumerate(dframe.word.values)}

feature_len = x_train.shape[1] # 15445
n_tags = len(set(dframe.tag.values))

input = Input(shape=(feature_len, ))

#what should be the size of my output_dim?s
model = Embedding(input_dim = feature_len, output_dim= 128, input_length =feature_len)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units = 100, return_sequences=True, recurrent_dropout = 0.1))(model)
out = TimeDistributed(Dense(n_tags, activation = "softmax"))(model)

# model = Bidirectional(LSTM(units = 100, return_sequences=False, recurrent_dropout = 0.1))(model)
# out = Dense(n_tags, activation = "softmax")(model)

#feed in parameter for input and output
model = Model(input, out)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics =["accuracy"])
#save the model

print(model.summary())
exit()

history = model.fit(x_train, np.array(y_train), batch_size=32, epochs = 1, validation_split= 0.2, verbose = 1)

