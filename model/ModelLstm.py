import numpy as np
#import matplotlib.pylab as plt
from keras.models import Model

from keras.layers import Input, LSTM, RepeatVector
from keras.models import Sequential
# from keras.layers.wrappers import TD

from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector, concatenate, Dropout, Bidirectional
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.layers          import merge, multiply
from keras.optimizers import Adam,SGD,RMSprop

from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization as BN

from keras import regularizers


import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# mylib
#import lib
import cons

class Lstm():
    """This is a test program."""

    def __init__(self):
        super().__init__()
        self.lc = cons.Const.LearningConst()
        self.cs = cons.Const.SaveConst()

    def make_net(self, dict_len):
        """ make net by reference to Keras official doc """

        input_dim = dict_len
        output_dim = dict_len

        encoder_inputs = Input(shape=(None, input_dim))
        #encoder_outputs, state_h, state_c = LSTM(self.lc.latent_dim)(encoder_inputs)
        hidden_outputs = LSTM(output_dim, return_sequences=True)(encoder_inputs)
        layer_outputs = LSTM(output_dim, return_sequences=True)(hidden_outputs)
        # encoder_outputs, state_h, state_c = LSTM(self.lc.latent_dim, return_state=True)(encoder_inputs)
        # layer_outputs = Dense(output_dim)(encoder_outputs)
        return Model(encoder_inputs, layer_outputs)

    def model_complie(self, model):
        """ complie """
        optimizer = 'rmsprop'
        #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        # optimizer = 'Adam'
        loss = 'mean_squared_error'
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy'])

        model.summary()


    def train(self, model, train_data, teach_data):
        """ Run training """
        loss = model.fit(train_data, teach_data,
                                      batch_size = self.lc.batch_size,
                                      epochs=5)
        # loss = model.fit(train_data, teach_data,
        #                               batch_size = self.lc.batch_size,
        #                               epochs=1,
        #                               validation_split = 0.2)
        return loss


    def waitController(self, model, flag, fname):
        if flag == "save":
            print("save" + self.sc.wait_save_dir+fname)
            model.save(self.sc.wait_save_dir+fname)
        if flag == "load":
            print("load" + self.sc.wait_save_dir+fname)
            model.load(self.sc.wait_save_dir+fname)


def main():
    lstm = Lstm()
    lstm.make_net(10)
    lstm.model_complie()


if __name__ == "__main__":
    main()

