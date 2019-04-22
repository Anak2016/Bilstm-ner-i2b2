import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM
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

tensorboard = TensorBoard(log_dir='./callback/traning.tensorboard', histogram_freq=0,
                          write_graph=True, write_images=False)

checkpointer = ModelCheckpoint(filepath='./callback/tmp/weights.hdf5',
                               verbose=1,
                               save_best_only=True)

csvlogger = CSVLogger(filename = './callback/trainning.log', separator=',', append=False)

#how does Baselogger, and History work?
# base


mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

x_train = x_train/255.0
x_test = x_test/255.0

# print(x_train.shape) # (60000,28,28)
# print(y_train[:20]) # (60000,)
# exit()
x_train = x_train[:500]
y_train = y_train[:500]
# x_test = x_test

model = Sequential()

# IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

# print(x_train.shape[1:]) # (28,28)
# print(model.output_shape) # (None, 28, 128)

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

# print(model.output_shape) # (None, 128)

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

print(model.output_shape) # (None, 32)

model.add(Dense(10, activation='softmax'))

print(model.output_shape) # (None, 10)
# print(model.summary())
# exit()

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

history_loss = LossHistory()
history_acc = AccHistory()


history_callback = model.fit(x_train,
          y_train,
          epochs=500,
          # validation_data=(x_test, y_test),

          # matrics = ["accuracy"],
          callbacks=[tensorboard, history_loss,history_acc,checkpointer, csvlogger]
          # callbacks=[history_loss,history_acc,checkpointer, csvlogger]
          # callbacks=[history_loss,history_acc]
          )

# save the model
# from keras.models import load_model
print("saving..")
model.save_weights('model/sentdex_lstm_demo.h5')
with open('model/sentdex_lstm.json', 'w') as f:
    print("writing to file...")
    f.write(model.to_json())

# with open('sentdex_lstm.json', 'w') as f:
#     print("writing to file...")
#     f.write(model.to_json())

#
# print("here is history.losses")
# print(history_loss.losses)
#
#
# print("here is history_callback.losses")
# print(history_callback.history["loss"]) # result = avg loss, but how? that's BaseLogger class right?
#
# print("here is history.acc")
# print(history_acc.acc)
#
# print("here is history_callback.acc")
# print(history_callback.history["acc"])
#
# print("pd.Dataframe(history_callback.history)")
# import pandas as pd
# print(pd.DataFrame(history_callback.history))
#
