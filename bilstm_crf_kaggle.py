import numpy as np
import pandas as pd

dframe = pd.read_csv("data/kaggle_corpus/ner.csv", encoding = "ISO-8859-1" ,error_bad_lines = False)
# print(dframe.tail())
# print(dframe.columns)
'''
Index(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
       'next-word', 'pos', 'prev-iob', 'prev-lemma', 'prev-pos',
       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
       'prev-prev-word', 'prev-shape', 'prev-word', 'sentence_idx', 'shape',
       'word', 'tag'],
'''
dataset=dframe.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
       'prev-prev-word', 'prev-shape', 'prev-word',"pos"],axis=1)

# print(dataset.head)
# print(dataset.columns)
'''
Index(['sentence_idx', 'shape', 'word', 'tag'], dtype='object')
'''
# print(dataset.head())
# exit()

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

getter = SentenceGetter(dataset)
sentences = getter.sentences

####################
### declair variables, parameter
####################
maxlen = max([len(s) for s in sentences]) #max length sentences

words = list(set(dataset["word"].values))# uniq_word
# print(words)
# exit()
n_words = len(words) # len of uniq_word

tags = list(set(dataset["tag"].values))
n_tags = len(tags)

word2idx = {w: i for i,w in enumerate(words)}
tag2idx = {t:i for i,t in enumerate(tags)}

# print(word2idx['Thousands'])
# print(tag2idx["O"])

#############
###preprocessing
#############
from keras.preprocessing.sequence import pad_sequences

#convert word in sentence to id using word2idx
X = [[word2idx[w[0]] for w in s] for s in sentences ]
# print(np.array(X).shape) # (35177,)
# print(len(np.array(X)[0])) # 48
# exit()

#pad to maxlength
#   >What should I padd words list with?

# max number of words in a sent 140
X = pad_sequences(maxlen= maxlen, sequences=X, padding = "post", value = n_words -1)

y = [[tag2idx[w[1]] for w in s] for s in sentences ]
y = pad_sequences(maxlen= maxlen, sequences=y, padding = "post", value = tag2idx["O"])

# print(np.array(X).shape) # (28141,140)
# print(np.array(y).shape) # (28141,140)
# exit()

from keras.utils import to_categorical
y =[to_categorical(i,num_classes=n_tags ) for i in y ] #indies to vector
# print(y[0])
# exit()

from sklearn.model_selection import train_test_split

#split test, train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# print(X_train[1])
# print(np.array(X_train).shape) # (28141,140)
# print(np.array(y_train).shape) # (28141,140,18)
# exit()

# #################
# ## load existing model
# #################
#
# import tensorflow as tf
# from tensorflow.keras.models import model_from_json
#
# with open('model/NER_kaggle/bilstm_crf_architecture.json', 'r') as f:
#     model = model_from_json(f.read())
#
# model.load_weights('model/NER_kaggle/bilstm_crf_weight.h5')
# # model.load_weights("callback/tmp/weights.hdf5")
#
# #lets use it to predict
#
# x = model.predict(X_test[:20], verbose =1 )
#
# import numpy as np
# pred_result = [np.argmax(pred_list) for pred_list in x[1] ]
# # print(len(x[1]))
# # print(x[1].shape) # (140,18)
# # print(pred_result)

######################
###callback variablea
######################
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, History, BaseLogger
#only log the last value
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class AccHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


tensorboard = TensorBoard(log_dir='./callback/NER_kaggle/traning.tensorboard', histogram_freq=0,
                          write_graph=True, write_images=False)


checkpointer = ModelCheckpoint(filepath='./callback/NER_kaggle/ModelCheckpoints/weights.hdf5',
                               verbose=1,
                               save_best_only=True)

csvlogger = CSVLogger(filename = './callback/NER_kaggle/trainning.log', separator=',', append=False)

history_loss = LossHistory()
history_acc = AccHistory()


#################
## building model with keras
#################
from keras.models import Model,Input
from keras.layers import LSTM,Embedding,Dense, TimeDistributed,Dropout, Bidirectional
# print(maxlen) #140
# exit()


input = Input(shape=(maxlen, ))
#How should i select output_dim
model = Embedding(input_dim = n_words, output_dim=maxlen , input_length =maxlen)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units = 100, return_sequences=True, recurrent_dropout = 0.1))(model)
# model = Bidirectional(LSTM(units = 100, return_sequences=False, recurrent_dropout = 0.1))(model)

out = TimeDistributed(Dense(n_tags, activation = "softmax"))(model)

#feed in parameter for input and output
model = Model(input, out)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics =["accuracy"])
#save the model

###################
#inspect outputs

# print(model.summary())
# exit()
#####################
# from keras import backend as K
# inp = model.input
outputs = [layer.output for layer in model.layers]
print(outputs)
exit()
# functor = K.function([inp, K.learning_phase()], outputs)
#
# test = np.random.random([2,2])[np.newaxis,:]
# layer_outs = functor([tes t,1])
# print(layer_outs)
# exit()

history = model.fit(X_train, np.array(y_train), batch_size=32, epochs = 7, validation_split= 0.2, verbose = 1,
                    callbacks=[tensorboard, history_loss, history_acc, checkpointer, csvlogger])

###############
##save the model
###############

print("saving..")
#save model weight
model.save_weights('model/NER_kaggle/bilstm_crf_weight.h5')

with open('model/NER_kaggle/bilstm_crf_architecture.json', 'w') as f:
    print("writing to file...")
    f.write(model.to_json())

