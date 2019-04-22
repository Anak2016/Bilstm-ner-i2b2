import tensorflow as tf
from tensorflow.keras.models import model_from_json
################
# loading model's architecture and weight
################
# from keras.dataset import mnist
mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

with open('model/sentdex_lstm.json', 'r') as f:
    model = model_from_json(f.read())

model.load_weights('model/sentdex_lstm_demo.h5')
# model.load_weights("callback/tmp/weights.hdf5")

#lets use it to predict

x = model.predict(x_test[:20], verbose =1 )
# label
# print(x.shape) #(20,10)

# index_min = minx(xrange(len(values)), key=values.__getitem__)
import numpy as np
pred_result = [np.argmax(pred_list) for pred_list in x ]

print(pred_result)

################
# loading checking
# ################0
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
#
# mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
# (x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test
#
# model = Sequential()
#
# # model.add(Dense(10, input_dim = 28 , activation='relu'))
# # model.add(Dense(10, input_dim = 28 , activation='relu'))
# #
# # model.add(Dense(10, activation='softmax'))
#
# opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
#
# model.load_weights("callback/tmp/weights.hdf5")
# # Compile model
# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=opt,
#     metrics=['accuracy'],
# )
#
# history_callback = model.fit(x_train,
#           y_train,
#           epochs=1
#           )
# import pandas as pd
# print(pd.DataFrame(history_callback.history))
