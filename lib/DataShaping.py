

import numpy as np
#import lib
import random
import os


class DataSaping():
    def load_file(self, dname, fname):
        with open(dname + fname, 'r') as f:
            lines = f.readlines()
        return lines


    def make_char_line(self, lines):
        __split_lines = []
        for line in lines:
            __bos = line.split(' ')[0]
            line = list(line[3:])
            line = [__bos] + line
            while(' ' in line): line.remove(' ')
            while('\n' in line): line.remove('\n')
            __split_lines.append(line)
        return __split_lines


    def flatten(self, nested_list):
        """2重のリストをフラットにする関数"""
        return [e for inner_list in nested_list for e in inner_list]


    def make_dict(self, split_line):
        __l = split_line[::]
        return list(set(self.flatten(__l)))


    def save_dict(self, d, dname, fname):
        if (os.path.exists(dname)):
            print("over write " + dname + fname)
            with open(dname + fname, 'w') as f:
                for value in d :
                    f.write(value + " ")
        else:
            print("write " + dname + fname)
            with open(dname + fname, 'a') as f:
                for value in d :
                    f.write(value + " ")


    def load_dict(self, dname, fname):
        __dict = self.load_file(dname, fname)[0].split(" ")
        while(' ' in __dict): __dict.remove(' ')
        return __dict


def main():
    ds = DataSaping()
    l = ds.load_file("./", "test.txt")
    sl = ds.make_char_line(l)
    print(sl)
    d = ds.make_dict(sl)
    ds.save_dict(d, "./", "dict.txt")
    d = ds.load_dict("./", "dict.txt")
    print("d", d)


if __name__ == "__main__":
   main()
