import numpy as np

import lib
import model

#from enum import Flag, auto

class DictFlag():
    Make = "make"
    Load = "load"

class TrainFlag():
    train = "train"
    make = "make"

class Trainer():
    def __init__(self):
        "docstring"
        self.projct_path = lib.SetProject.get_path()
        self.cl = lib.Const.LearningConst()
        self.cs = lib.Const.SaveConst()

        self.char_dict = []
        self.char_lines = []

    def make_net(self, input_dim):
        self.lstm = model.ModelLstm.Lstm()
        model_lstm = self.lstm.make_net(input_dim)
        self.lstm.model_complie(model_lstm)
        return model_lstm


    def train(self, model_lstm, train_data, teach_data):
        for tr,te in zip(train_data, teach_data):
            tr = np.array([tr])
            te = np.array([te])
            print("tr",tr.shape)
            print("te",te.shape)
            self.lstm.train(model_lstm, tr, te)



            
    def make_sentens(self, model_lstm, char_dict):
        one_hot = [0] * len(char_dict)
        one_hot[char_dict.index("。")] = 1
        vec = np.array([[one_hot]])

        for i in range(10):
            sentens = ""
            while(True):
                vec = model_lstm.predict(vec)
                vec_index = list(vec[0][0]).index(max(vec[0][0]))
                char = char_dict[vec_index]
                sentens += char
                if char == "BOS": break
                if char == "。": break
            print(">> ",sentens)


def main():
    tr = Trainer()
    md = lib.MakeData.MakeData()

    ds = lib.DataShaping.DataSaping()
    l = ds.load_file(tr.cs.train_file)
    tr.char_lines = ds.make_char_line(l)


    flag = DictFlag.Load
    if flag == DictFlag.Make :
        tr.char_dict = ds.make_dict(tr.char_lines)
        ds.save_dict(tr.char_dict, tr.cs.dict_fname)
    if flag == DictFlag.Load :
        tr.char_dict = ds.load_dict(tr.cs.dict_fname)

    print("dict len :", len(tr.char_dict))

    #t_flag = TrainFlag.train
    t_flag = TrainFlag.make
    if t_flag == TrainFlag.train:
        model_lstm = tr.make_net(input_dim=len(tr.char_dict))
        for char_line in tr.char_lines:
            print("train at ",tr.char_lines.index(char_line),"/",len(tr.char_lines))
            print(char_line)
            train_data = md.make_data_one(tr.char_dict, char_line)
            teach_data = md.make_teach_data_one(tr.char_dict, char_line)
            tr.train(model_lstm, train_data, teach_data)
            if tr.char_lines.index(char_line)  % 10:
                tr.lstm.weightController(model_lstm, "save", tr.cs.weight_fname)

    if t_flag == TrainFlag.make:
        model_lstm = tr.make_net(input_dim=len(tr.char_dict))
        tr.lstm.weightController(model_lstm, "load", tr.cs.weight_fname)
        tr.make_sentens(model_lstm, tr.char_dict)

if __name__ == "__main__":
   main()
