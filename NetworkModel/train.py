import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
import numpy as np

import pickle

# import trianing dataset
x = pickle.load(open("./data/SR-ARE-train/names_onehots.pickle", "rb"))

x_test = pickle.load(open("./data/SR-ARE-test/names_onehots.pickle", "rb"))

y = []
with open("./data/SR-ARE-train/names_labels.txt", "r") as file:
    for line in file:
        y.append(int(line.split(',')[-1][0]))

y_test = []
with open("./data/SR-ARE-test/names_labels.txt", "r") as file:
    for line in file:
        y_test.append(int(line.split(',')[-1][0]))


y = np.array(list(y))
y_test = np.array(list(y_test))

# oversampling the data


def oversampling(x, y):
    # divide the data into two group: one with labels 1 and one with labels 0
    boolLabels = y != 0
    oneLables = y[boolLabels]
    zeroLables = y[~boolLabels]
    oneValue = x[boolLabels]
    zeroValue = x[~boolLabels]

    # add new positive examples
    index = np.arange(len(oneValue))
    choices = np.random.choice(index, len(zeroValue)-len(oneValue))
    newValue = np.concatenate([oneValue, oneValue[choices]], axis=0)
    newLables = np.concatenate([oneLables, oneLables[choices]], axis=0)

    # resamble the data
    resambledValue = np.concatenate([newValue, zeroValue], axis=0)
    resambledLables = np.concatenate([newLables, zeroLables], axis=0)

    # shuffle the data
    order = np.arange(len(resambledLables))
    np.random.shuffle(order)
    resambledValue = resambledValue[order]
    resambledLables = resambledLables[order]

    return resambledValue, resambledLables


def undersampling(x, y):
    # divide the data into two group: one with labels 1 and one with labels 0
    boolLabels = y != 0
    oneLables = y[boolLabels]
    zeroLables = y[~boolLabels]
    oneValue = x[boolLabels]
    zeroValue = x[~boolLabels]

    # extract random negitive examples
    index = np.arange(len(zeroValue))
    choices = np.random.choice(index, len(oneValue))
    newValue = zeroValue[choices]
    newLables = zeroLables[choices]

    # resamble the data
    resambledValue = np.concatenate([newValue, oneValue], axis=0)
    resambledLables = np.concatenate([newLables, oneLables], axis=0)

    # shuffle the data
    order = np.arange(len(resambledLables))
    np.random.shuffle(order)
    resambledValue = resambledValue[order]
    resambledLables = resambledLables[order]

    return resambledValue, resambledLables


smile_oversmapled, labels_oversmapled = oversampling(x['onehots'], y)
smile_undersmapled, labels_undersmapled = undersampling(x['onehots'], y)
x_test_oversmapled, y_test_oversmapled = oversampling(
    x_test['onehots'], y_test)
x_test_undersmapled, y_test_undersmapled = undersampling(
    x_test['onehots'], y_test)


# building the network
model = Sequential()

# first layer
model.add(Conv1D(64, (3), activation='relu', input_shape=[70, 325]))
model.add(MaxPooling1D(pool_size=(2)))

# second layer
model.add(Conv1D(64, (3), activation='relu'))
model.add(MaxPooling1D(pool_size=(2)))

# third layer
model.add(Conv1D(32, (3), activation='relu'))
model.add(MaxPooling1D(pool_size=(2)))

# fourth layer
model.add(Conv1D(16, (3), activation='relu'))

# dense layer
model.add(GlobalMaxPooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))  # dropout
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))  # dropout

model.add(Dense(1, activation='sigmoid', kernel_initializer='random_uniform',
                bias_initializer='zeros'))


# adjust learning rate
STEPS_PER_EPOCH = 12138//32
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    1e-4,
    decay_steps=STEPS_PER_EPOCH*5,
    decay_rate=1,
    staircase=False)

opt = keras.optimizers.Adam(lr_schedule)

# model compile
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# model training
model.fit(smile_oversmapled, labels_oversmapled, batch_size=32, epochs=25)
model.fit(x_test_oversmapled, y_test_oversmapled, batch_size=32, epochs=25)

# retrain
model.fit(smile_oversmapled, labels_oversmapled, batch_size=32, epochs=25)
model.fit(x_test_oversmapled, y_test_oversmapled, batch_size=32, epochs=25)

# model evaluating
results = model.evaluate(x_test['onehots'], y_test, batch_size=32)
print(results)

# model saving
model.save('test.h5')
