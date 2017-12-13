import numpy as np

import lib
import model

#from enum import Flag, auto

class DictFlag():
    Make = "make"
    Load = "load"


class Trainer():
    def __init__(self):
        "docstring"
        self.projct_path = lib.SetProject.get_path()
        self.char_dict = []
        self.char_lines = []

    def make_net(self, input_dim):
        self.lstm = model.ModelLstm.Lstm()
        model_lstm = self.lstm.make_net(input_dim)
        self.lstm.model_complie(model_lstm)
        return model_lstm


    def train(self, model_lstm, train_data, teach_data):
        # print("tr",train_data)
        # print("te",teach_data)
        # print("--")
        # print("tr",train_data.shape)
        # print("te",teach_data.shape)

        for tr,te in train_data, teach_data:
            tr = np.array([tr])
            te = np.array([te])
            print("tr",tr.shape)
            print("te",te.shape)            
            self.lstm.train(model_lstm, tr, te)


def main():
    tr = Trainer()
    md = lib.MakeData.MakeData()

    ds = lib.DataShaping.DataSaping()
    l = ds.load_file(tr.projct_path+"/lib/", "test.txt")
    tr.char_lines = ds.make_char_line(l)

    dict_dir = tr.projct_path+"/lib/"
    dict_fname = "dict.txt"

    flag = DictFlag.Make
    if flag == DictFlag.Make :
        tr.char_dict = ds.make_dict(tr.char_lines)
        ds.save_dict(tr.char_dict, dict_dir, dict_fname)
    if flag == DictFlag.Load :
        tr.char_dict = ds.load_dict(dict_dir, dict_fname)

    print("dict len :", len(tr.char_dict))
    train_data = md.make_data(tr.char_dict, tr.char_lines)
    teach_data = md.make_teach_data(tr.char_dict, tr.char_lines)
    model_lstm = tr.make_net(input_dim=len(tr.char_dict))
    tr.train(model_lstm, train_data, teach_data)


if __name__ == "__main__":
   main()
