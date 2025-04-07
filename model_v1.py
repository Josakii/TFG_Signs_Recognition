import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     TimeDistributed, LSTM, Dropout, Dense)

class SignLanguageModel(Model):
    def __init__(self, input_shape, num_classes):
        super(SignLanguageModel, self).__init__()
        self.cnn1 = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))
        self.pool1 = TimeDistributed(MaxPooling2D((2, 2)))
        self.cnn2 = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))
        self.pool2 = TimeDistributed(MaxPooling2D((2, 2)))
        self.flatten = TimeDistributed(Flatten())
        self.lstm = LSTM(128, return_sequences=False)
        self.dropout = Dropout(0.5)
        self.dense1 = Dense(64, activation='relu')
        self.output_layer = Dense(num_classes, activation='softmax')
        self.build((None, *input_shape))  # build the model with dynamic batch size

    def call(self, x):
        x = self.cnn1(x)
        x = self.pool1(x)
        x = self.cnn2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.lstm(x)
        x = self.dropout(x)
        x = self.dense1(x)
        return self.output_layer(x)
