import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow import keras
import numpy as np

import pickle
# load model
model = keras.models.load_model("test.h5")

# import testing dataset
x_test = pickle.load(open("../SR-ARE-score/names_onehots.pickle", "rb"))

txtList = (model.predict(
    x_test['onehots'], batch_size=x_test['onehots'].shape[0]) > 0.5).astype(int)
txtList = txtList.tolist()

# output the results
file = open("labels.txt", "w")
for line in txtList:
    file.write(str(line[0])+'\n')
file.close()
