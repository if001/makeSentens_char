import numpy as np

import lib
# from DataShaping import DataSaping

class MakeData():
    def __init__(self):
        "docstring"
        cl = lib.Const.LearningConst()

    def make_data_one(self, char_dict, char_line):
        train_char_lines_vec = []
        char_line_vec = []
        for char in char_line:
            one_hot = [0] * len(char_dict)
            one_hot[char_dict.index(char)] = 1
            char_line_vec.append(one_hot)
        train_char_lines_vec.append(char_line_vec)
        return np.array(train_char_lines_vec)


    def make_teach_data_one(self, char_dict, char_line):
        teach_char_lines_vec = []

        char_line_vec = []
        for char in char_line[1:]:
            one_hot = [0] * len(char_dict)
            one_hot[char_dict.index(char)] = 1
            char_line_vec.append(one_hot)
        one_hot = [0] * len(char_dict)
        one_hot[char_dict.index("。")] = 1
        char_line_vec.append(one_hot)
        teach_char_lines_vec.append(char_line_vec)

        return np.array(teach_char_lines_vec)


    def make_data(self, char_dict, char_line):
        train_char_lines_vec = []
        char_line_vec = []
        for char in char_line:
            one_hot = [0] * len(char_dict)
            one_hot[char_dict.index(char)] = 1
            char_line_vec.append(one_hot)
        train_char_lines_vec.append(char_line_vec)
        return train_char_lines_vec


def main():
    ds = DataSaping()

    l = ds.load_file("./", "test.txt")
    char_lines = ds.make_char_line(l)

    # char_dict = ds.make_dict(char_lines)
    # ds.save_dict(char_dict, "./", "dict.txt")
    char_dict = ds.load_dict("./", "dict.txt")

    md = MakeData()
    md.make_data(char_dict, char_lines)



if __name__ == "__main__":
   main()
