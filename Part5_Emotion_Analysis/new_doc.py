#!/usr/bin/env pythonw
# encoding: utf-8
import sys
import os
# 简单分词工具
import jieba
# 带词义检查的分词工具
import jieba.posseg as pseg
# 数据缓存工具
import cPickle
import numpy as np
import sklearn
import platform
import re
import traceback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from gensim.models.word2vec import Word2Vec

reload(sys)
sys.setdefaultencoding('utf-8')

# 启用并行计算，加快分词系统的计算
jieba.enable_parallel = True
base_path = os.getcwdu()
train_path = os.path.join(base_path, 'htl')
train_cached_path = os.path.join(base_path, 'htl_cached')


def detectOS():
    os = (platform.system()).lower()
    if os == 'windows':
        return '\\'
    else:
        return '/'

_dir = detectOS()

# 检测并且补全一个路径
def detectpath(path):
    if (not os.path.exists(path)):
        detectpath(_dir.join(path.split(_dir)[0:-1]))
        os.mkdir(path)
        return False
    else:
        if len(os.listdir(path)) == 0:
            return False
        else:
            return True

# 加载一个路径下所有的文件格式
def loadtext(path):
    for child in os.listdir(path):
        # 忽略 Windows 和 macOS 的系统文件
        if os.path.isfile(os.path.join(path, child)) and child != '.DS_Store' and child != 'Desktop.ini':
            full_child_path = os.path.join(path, child)
            fin = open(full_child_path, 'r')
            fil_data = fin.read()
            fil_data = fil_data.decode('utf-8', 'ignore').strip()
            fin.close()
            yield (child, fil_data)


def LoadAndMakePickle(train_tag='pos'):
    try:
        files = loadtext(os.path.join(train_path, train_tag))
        finally_data_list = []
        finally_wordposseg_list = []
        detectpath(os.path.join(train_cached_path, train_tag))
        line_idx = 0
        for filename, filedata in files:
            filedata.replace('\n', '')
            dat_0 = re.findall(u'[0-9A-Za-z\u4e00-\u9fff]*', filedata)
            dat_1 = ''.join(dat_0)
            dat_2 = pseg.cut(dat_1)
            word = []
            pos = []
            for w, p in dat_2:
                word.append(w)
                pos.append(p)

            finally_data_list.append(word)
            # wordline = ' '.join(word)
            # finally_data_list.append(wordline)
            finally_wordposseg_list.append((word, pos))
            line_idx += 1
            if line_idx % 100 == 0:
                print('已经处理' + str(line_idx) + '文件。')
        data_dump = open(os.path.join(train_cached_path, train_tag + _dir + 'linedata.tmp'), 'wb')
        wordpos_dump = open(os.path.join(train_cached_path, train_tag + _dir + 'wposdata.tmp'), 'wb')
        cPickle.dump(finally_data_list, data_dump)
        cPickle.dump(finally_wordposseg_list, wordpos_dump)
        data_dump.close()
        wordpos_dump.close()
        return finally_data_list, finally_wordposseg_list
    except:
        traceback.print_exc()


def LoadPickle(load_tag='pos'):
    try:
        detectpath(os.path.join(train_cached_path, load_tag))
        data_dump = open(os.path.join(train_cached_path, load_tag + _dir + 'linedata.tmp'), 'rb')
        wordpos_dump = open(os.path.join(train_cached_path, load_tag + _dir + 'wposdata.tmp'), 'rb')
        finally_data_list = cPickle.load(data_dump)
        finally_wordposseg_list = cPickle.load(wordpos_dump)
        data_dump.close()
        wordpos_dump.close()
        return finally_data_list, finally_wordposseg_list
    except:
        traceback.print_exc()


# 就是将这一整篇文章所有的单词全部一起计算
# 然后计算平均值，作为整一篇文章的向量值。
def BuildWordVector(clf, text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in text:
        try:
            vec += clf[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


class AutoMean(object):
    def __init__(self):
        self.Clear()

    def Clear(self):
        self._iter = 0.0
        self._sum = 0.0

    def Add(self, val):
        if (self._iter == 0):
            self._sum = val
        else:
            self._sum *= (self._iter) / (self._iter + 1)
            self._sum += val / (self._iter + 1)
        self._iter += 1

    def Value(self):
        return self._sum

    def Total(self):
        return self._sum * self._iter

if __name__ == '__main__':
    pos_data, pos_wordposs = None, None
    neg_data, neg_wordposs = None, None
    # pos_data 存放所有积极情绪的文档
    # neg_data 存放所有消极情绪的文档
    if not detectpath(train_cached_path):
        pos_data, pos_wordposs = LoadAndMakePickle('pos')
        neg_data, neg_wordposs = LoadAndMakePickle('neg')
    else:
        pos_data, pos_wordposs = LoadPickle('pos')
        neg_data, neg_wordposs = LoadPickle('neg')

    # use 1 for positive sentiment, 0 for negative sentiment
    print('加载完成!')

    # 设 y = 1 表示积极，y = 0 表示消极。建立目标值
    y = np.concatenate((np.ones(len(pos_data)), np.zeros(len(neg_data))))
    mean = AutoMean()
    for i in range(10):
        # 首先将 pos_data 和 neg_data 进行 concatenate，然后通过
        # sklearn.model_selection 下的 train_test_split 函数进行
        # 随机的样本分割 
        x_train, x_test, y_train, y_test = \
            train_test_split(np.concatenate((pos_data, neg_data)), y, test_size=0.4)

        n_dim = 300

        # 初始化模型并构建词汇
        imdb_w2v = Word2Vec(size=n_dim, min_count=10)
        # 根据训练集设置词汇
        imdb_w2v.build_vocab(x_train)
        
        # 训练集 首先通过 imdb_w2v 训练样本，然后导出句子对应的词向量，计算每个评论的平均值。
        # 最后根据高斯分布进行调整
        imdb_w2v.train(x_train, total_examples=len(x_train) + len(x_test), epochs=imdb_w2v.iter)
        train_vecs = np.concatenate([BuildWordVectorç(imdb_w2v, text, n_dim) for text in x_train])
        train_vecs = scale(train_vecs)
        
        # 测试集
        imdb_w2v.train(x_test, total_examples=len(x_train) + len(x_test), epochs=imdb_w2v.iter)
        test_vecs = np.concatenate([BuildWordVector(imdb_w2v, text, n_dim) for text in x_test])
        test_vecs = scale(test_vecs)

        # lr = SGDClassifier(loss='log', penalty='l1')
        # lr.fit(train_vecs, y_train)
        clf = svm.SVC()
        clf.fit(train_vecs, y_train)
        prec = clf.score(test_vecs, y_test)
        # prec = lr.score(test_vecs, y_test)
        print('%.6f' % prec)
        mean.Add(prec)
    print('平均精确率：%.6f' % mean.Value())
