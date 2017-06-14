# encoding: utf-8

import sys
import os
# 简单分词工具
import jieba
# 带词义检查的分词工具
import jieba.posseg as pseg
import logging
import langconv

from gensim import corpora
reload(sys)
sys.setdefaultencoding('utf-8')

# 启用并行计算，加快分词系统的计算
jieba.enable_parallel = True
base_path = os.getcwdu()
train_data_path = os.path.join(base_path, 'htl')
train_cached_path = os.path.join(base_path, 'htl_cached')

class ChineseConv(object):
    def __init__(self):
        self._simpconv = langconv.Converter('zh-hans')
        self._tradconv = langconv.Converter('zh-hant')
        
    def s2t(self, sentence):
        return self._simpconv.convert(sentence)
    
    def t2s(self, sentence):
        return self._tradconv.convert(sentence)
    
cc = ChineseConv()

def AdvancedSplit(data, sep_symb):
    if isinstance(data, unicode) or isinstance(data, str):
        for piece in data.split(sep_symb):
            s = piece.strip()
            if len(s) > 0:
                yield s
    elif isinstance(data, list):
        for word in data:
            for piece in word.split(sep_symb):
                s = piece.strip()
                if len(s) > 0:
                    yield s
    
def DocSplit(data, mode):
    if mode == 'd':
        c_data = AdvancedSplit(data, u'\n')
        c_data = AdvancedSplit(c_data, '.')
        c_data = AdvancedSplit(c_data, ';')
        c_data = AdvancedSplit(c_data, '?')
        c_data = AdvancedSplit(c_data, '!')
        c_data = AdvancedSplit(c_data, u'。')
        c_data = AdvancedSplit(c_data, u'；')
        c_data = AdvancedSplit(c_data, u'？')
        c_data = AdvancedSplit(c_data, u'！')
        return c_data
    elif mode == 's':
        c_data = AdvancedSplit(data, ',')
        c_data = AdvancedSplit(data, u'，')
        return c_data

def detectpath(path):
    if (not os.path.exists(path)):
        detectpath('/'.join(path.split('/')[0:-1]))
        os.mkdir(path)

def loadtext(path):
    for child in os.listdir(path):
        # 忽略 Windows 和 macOS 的系统文件
        if os.path.isfile(os.path.join(path, child)) and child != '.DS_Store' and child != 'Desktop.ini':
            full_child_path = os.path.join(path, child)
            fin = open(full_child_path, 'r')
            fil_data = fin.read()
            fil_data = fil_data.decode('utf-8').strip()
            fin.close()
            yield (child, fil_data)

def f(original_path, cached_path):
    detectpath(original_path)
    detectpath(cached_path)
    childs = os.listdir(cached_path)
    if len(childs) == 0:
        for (filename, data) in loadtext(original_path):
            print filename, data
            lines = DocSplit(data, 'd')
            for line in lines:
                
    else:
        pass

if __name__ == '__main__':
    f(train_data_path+'/pos', train_cached_path+'/pos')