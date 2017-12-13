"""
定数用
"""

#import pylab as plt
import lib

class Const():
    def __init__(self):
        """ valiable setting"""
        self.batch_size = 32
        self.learning_num = 100
        self.check_point = 30

        """ directory setting"""
        self.project_path = lib.SetProject.get_path()


        self.wait_save_dir = self.project_path + '/nn/wait/'
        self.dict_fdir = self.project_path + "/aozora_text/files/"
        self.dict_fname = "files_all_rnp.txt"

        self.train_file = self.project_path + "/aozora_text/files/files_all_rnp.txt"


