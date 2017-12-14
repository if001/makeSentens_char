"""
定数用
"""

#import pylab as plt
import lib

class LearningConst():
    def __init__(self):
        """ valiable setting"""
        self.latent_dim = 512
        self.batch_size = 64
        self.learning_num = 100
        self.check_point = 30


class SaveConst():
    def __init__(self):
        """ directory setting"""
        self.project_path = lib.SetProject.get_path()

        self.weight_save_dir = self.project_path + '/model/weight/'
        self.weight_fname = 'param.hdf5'

        self.dict_fname = self.project_path + "/dict/dict.txt"
        self.train_file = self.project_path + "/aozora_text/files/files_all_rnp2.txt"
