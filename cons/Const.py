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
        self.project_dir = lib.SetProject.get_path()

        """ seq2seq """
        # self.seq2seq_wait_save_dir = self.project_dir+'/nn/wait/'
        self.wait_save_dir = self.project_dir+'/nn/wait/'
        self.train_file = self.project_dir+"/aozora_text/files/files_all_rnp.txt"
        self.train_file = self.project_dir+"/aozora_text/files/files_all_rnp2.txt"
