#単語をベクトル化

import gensim
from gensim.models import word2vec
#from gensim import models as mod
#import pylab as plt

import numpy as np



class MyWord2Vec():
    def __init__(self):
        self.word_feat_len = 5


    def train(self,fname,saveflag="save"):
        sentences = gensim.models.word2vec.Text8Corpus(fname)
        #model = gensim.models.word2vec.Word2Vec(sentences, size=200, window=5, workers=4, min_count=5)
        self.model = gensim.models.word2vec.Word2Vec(sentences, size=self.word_feat_len, window=5, workers=4, min_count=1)
        # if saveflag == "save":
        #     print("save "+self.word2vec_wait)
        #     self.model.save(self.word2vec_wait)

    def load_model(self): pass
        # 読み込み
        # print("load "+self.word2vec_wait)
        # self.model = word2vec.Word2Vec.load(self.word2vec_wait)

    def get_vector(self,st):
        return self.model.wv[st]

    def get_similar_vector(self,st):
        __st = self.model.most_similar(positive=st, topn=1)[0][0]  
        return self.model.wv[__st]


    def get_word(self,vec):
        return self.model.most_similar( [ vec ], [], 1)[0][0]


    def get_similar_words(self,st,top):
        # 類似ワード出力
        results = self.model.most_similar(positive=st, topn=top)
        for result in results:
            print(result[0], '\t', result[1])
        return results


    # def get_most_similar_word(self,vec):
    #     return self.model.most_similar(positive=st, topn=1)[0][0]




def plot(vec):
    t = range(len(vec))
    plt.plot(t,vec)
    plt.show()


def main():
    net = MyWord2Vec()

    #net.train(const.dict_train_file,"not save")
    net.train("../aozora_text3/files/files_all_rnp.txt","not save")

    net.load_model()


    print("in 怪盗")
    vec = net.get_vector("怪盗")
    print(vec)
    word = net.get_word(vec)
    print("word",word)


    print("in 怪盗")
    vec = net.get_similar_vector("怪盗")
    print(vec)
    word = net.get_word(vec)
    print("word",word)


    # print("in 怪盗")
    # vec = [ 0.57944483, 0.85978442, 0.03766631, 0.16827765, -1.56712174 ]
    # word = net.get_word(vec)
    # print(word)
    

if __name__ == "__main__":
    main()
