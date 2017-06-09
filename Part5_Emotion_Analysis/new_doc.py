# encoding: utf-8
import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')
# 简单分词工具
import jieba
# 带词义检查的分词工具
import jieba.posseg as pseg
# 数据缓存工具
import cPickle
import sklearn
import platform
import re
import traceback
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec

# 本地文件，用于中文简体和繁体互相转换使用
import langconv
# 启用并行计算，加快分词系统的计算
jieba.enable_parallel = True
base_path = os.getcwdu()
train_path = os.path.join(base_path, 'htl')
train_cached_path = os.path.join(base_path, 'htl_cached')

class ChineseConv(object):
    def __init__(self):
        self._simpconv = langconv.Converter('zh-hans')
        self._tradconv = langconv.Converter('zh-hant')
        
    def s2t(self, sentence):
        return self._simpconv.convert(sentence)
    
    def t2s(self, sentence):
        return self._tradconv.convert(sentence)

#  定义一个全局的中文简体/翻译转换工具    
cc = ChineseConv()

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
        if len(os.listdir(path))==0:
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

def LoadAndMakePickle(train_tag = 'pos'):
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
            wordline = ' '.join(word)
            finally_data_list.append(wordline)
            finally_wordposseg_list.append((word, pos))
            line_idx += 1
            if line_idx % 100 == 0:
                print('已经处理'+str(line_idx)+'文件。')
        data_dump = open(os.path.join(train_cached_path, train_tag + _dir + 'linedata.tmp'), 'wb')
        wordpos_dump = open(os.path.join(train_cached_path, train_tag + _dir + 'wposdata.tmp'), 'wb')
        cPickle.dump(finally_data_list, data_dump)
        cPickle.dump(finally_wordposseg_list, wordpos_dump)
        data_dump.close()
        wordpos_dump.close()
        return finally_data_list, finally_wordposseg_list
    except:
        traceback.print_exc()
        
def LoadPickle(load_tag = 'pos'):
    try:
        detectpath(os.path.join(train_cached_path, load_tag))
        data_dump = open(os.path.join(train_cached_path, load_tag + _dir + 'linedata.tmp'), 'rb')
        wordpos_dump = open(os.path.join(train_cached_path, load_tag + _dir + 'wposdata.tmp'), 'rb')
        finally_data_list=cPickle.load(data_dump)
        finally_wordposseg_list=cPickle.load(wordpos_dump)
        data_dump.close()
        wordpos_dump.close()
        return finally_data_list, finally_wordposseg_list
    except:
        traceback.print_exc()
    
if __name__ == '__main__':
    pos_data, pos_wordposs = None, None
    neg_data, neg_wordposs = None, None
    if not detectpath(train_cached_path):
        pos_data, pos_wordposs = LoadAndMakePickle('pos')
        neg_data, neg_wordposs = LoadAndMakePickle('neg') 
    else:
        pos_data, pos_wordposs = LoadPickle('pos')
        neg_data, neg_wordposs = LoadPickle('neg')
    #现在我们已经获得了所有的数据，可以开始工作获得了
    print(u'加载完成!')
    
    