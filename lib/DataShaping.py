

import numpy as np
#import lib
import random
import os


class DataSaping():
    def load_file(self, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
        return lines


    def flatten(self, nested_list):
        """2重のリストをフラットにする関数"""
        return [e for inner_list in nested_list for e in inner_list]


    def make_char_line(self, lines):
        __split_lines = []
        for line in lines:
            __bos = line.split(' ')[0]
            line = list(line)
            line = [__bos] + line
            while(' ' in line): line.remove(' ')
            while('' in line): line.remove('')
            __split_lines.append(line)

        while(['\n'] in __split_lines): __split_lines.remove(['\n'])
        while(['。','。'] in __split_lines): __split_lines.remove(['。','。'])
        while([' '] in __split_lines): __split_lines.remove([' '])
        while([''] in __split_lines): __split_lines.remove([''])
        while(['。'] in __split_lines): __split_lines.remove(['。'])
        
        return self.flatten(__split_lines)


    def make_dict(self, split_line):
        __l = split_line[::]
        return list(set(__l))


    def save_dict(self, d, fname):
        if (os.path.exists(fname)):
            print("over write " + fname)
            with open(fname, 'w') as f:
                for value in d :
                    f.write(value + " ")
        else:
            print("write " + fname)
            with open(fname, 'a') as f:
                for value in d :
                    f.write(value + " ")


    def load_dict(self, fname):
        with open(fname, 'r') as f:
            __lines = f.read()
        
        __dict = __lines.split(" ")
        
        while(' ' in __dict): __dict.remove(' ')
        return __dict


def main():
    ds = DataSaping()
    l = ds.load_file("./re_re_oshieto_tabisuru_otoko2.txt")
    sl = ds.make_char_line(l)
    print(sl)
    d = ds.make_dict(sl)
    print(d)
    # ds.save_dict(d, "./dict.txt")
    # d = ds.load_dict("./dict.txt")
    # print("d", d)


if __name__ == "__main__":
   main()
