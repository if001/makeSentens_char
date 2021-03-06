import numpy as np
#import matplotlib.pylab as plt
from keras.models import Model

from keras.layers import Input, LSTM, RepeatVector
from keras.models import Sequential
from keras.models import load_model
# from keras.layers.wrappers import TD

from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector, concatenate, Dropout, Bidirectional, SimpleRNN
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.layers          import merge, multiply
from keras.optimizers import Adam,SGD,RMSprop

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.normalization import BatchNormalization as BN

from keras import regularizers


import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# mylib
import lib
#import cons


class Net():
    def __init__(self):
            self.cs = lib.Const.SaveConst()


    def weightController(self, model, flag, fname):
        if flag == "save":
            print("save" + self.cs.weight_save_dir + fname)
            model.save(self.cs.weight_save_dir + fname)
        if flag == "load":
            print("load" + self.cs.weight_save_dir + fname)
            # model.load(self.cs.weight_save_dir + fname)
            model.load_weights(self.cs.weight_save_dir + fname)


    def load_model(self, fname):
            print("load model" + self.cs.weight_save_dir + fname)
            return load_model(self.cs.weight_save_dir + fname)


class Lstm(Net):
    """This is a test program."""

    def __init__(self):
        super().__init__()
        self.cl = lib.Const.LearningConst()
        # self.tb_cb = TensorBoard(log_dir="./tb.log", histogram_freq=0)


    def make_net(self, dict_len):
        """ make net by reference to Keras official doc """

        input_dim = dict_len
        output_dim = dict_len

        encoder_inputs = Input(shape=(None, input_dim))
        layer_outputs = LSTM(output_dim, return_sequences=True)(encoder_inputs)

        return Model(encoder_inputs, layer_outputs)


    def model_complie(self, model):
        """ complie """
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        # loss = 'mean_squared_error'
        loss = 'categorical_crossentropy'
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy'])

        model.summary()


    def train(self, model, train_data, teach_data):
        """ Run training """
        loss = model.fit(train_data, teach_data,
                                      batch_size = self.cl.batch_size,
                                      epochs=1)

        return loss


class Lstm2(Net):
    """This is a test program."""

    def __init__(self):
        super().__init__()
        self.cl = lib.Const.LearningConst()
        # self.tb_cb = TensorBoard(log_dir="./tb.log", histogram_freq=0)


    def make_net(self, dict_len):
        """ make net by reference to Keras official doc """

        input_dim = dict_len
        output_dim = dict_len
        latent_dim = self.cl.latent_dim

        encoder_inputs = Input(shape=(None, input_dim))
        layer_outputs = LSTM(latent_dim, return_sequences=True)(encoder_inputs)
        layer_outputs = Dropout(0.5)(layer_outputs)
        layer_outputs = LSTM(output_dim, return_sequences=True)(layer_outputs)
        layer_outputs = Dropout(0.5)(layer_outputs)
        return Model(encoder_inputs, layer_outputs)


    def model_complie(self, model):
        """ complie """
        # optimizer = 'rmsprop'
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        loss = 'categorical_crossentropy'
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy'])

        model.summary()


    def train(self, model, train_data, teach_data):
        """ Run training """
        loss = model.fit(train_data, teach_data,
                         batch_size = self.cl.batch_size,
                         epochs=1,
                         validation_split=0.2)
        return loss


def main():
    lstm = Lstm()
    lstm.make_net(10)
    lstm.model_complie()


if __name__ == "__main__":
    main()

