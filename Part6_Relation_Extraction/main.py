#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 03:14:10 2017

@author: X230大青椒
"""

# -*- coding: utf-8 -*-

import sys
import os
import langconv
import traceback
import cPickle
import string
from gensim.models import Word2Vec
import jieba
import re
reload(sys)
sys.setdefaultencoding('utf-8')

basepath = os.getcwdu()
data_path = os.path.join(basepath, 'data')
cached_path = os.path.join(basepath, 'cached')
# 全局声明数据文件和缓存文件
people_filename = os.path.join(data_path, 'people.txt')
people_cached = os.path.join(cached_path, 'people')
text_filename = os.path.join(data_path, 'sentence.txt')
text_cached = os.path.join(cached_path, 'sentence')

# jieba处理程序
jieba.load_userdict(os.path.join(data_path, 'people.txt'))

# 繁体转简体程序
class ChineseConv(object):
    def __init__(self):
        self._simpconv = langconv.Converter('zh-hans')
        self._tradconv = langconv.Converter('zh-hant')
        
    def s2t(self, sentence):
        return self._tradconv.convert(sentence)
    
    def t2s(self, sentence):
        return self._simpconv.convert(sentence)

cc = ChineseConv()

rel2id = {}
id2rel = {}

with open(os.path.join(data_path, 'relation_dict.txt')) as f:
    for line in f:
        line = line.decode('utf-8', 'ignore')
        key, s_val = line.split('\t')
        val = string.atoi(s_val)
        rel2id.setdefault(key, val)
        id2rel.setdefault(val, key)
        
def relation2id(rel):
    if rel2id.has_key(rel):
        return rel2id[rel]
    else:
        return -1

def dump_file(obj, filename):
    fout = open(filename, 'wb')
    cPickle.dump(obj, fout)
    fout.close()
    print("已经缓存文件 '%s'。" % filename)

def load_dump(filename):
    fin = open(filename, 'rb')
    obj = cPickle.load(fin)
    fin.close()
    print("已经从文件 '%s' 读入缓存。" % filename)
    return obj

def simp_chinese(path):
    file_idx = 0
    for base, dirs, files in os.walk(path):
        for file in files:
            if file != u'.DS_Store' and file != u'Desktop.ini':
                try:
                    fin = open(os.path.join(base, file), 'r')
                    msg = fin.read().decode('utf-8', 'ignore')
                    fin.close()
                    
                    msg = cc.t2s(msg)
                    
                    fout = open(os.path.join(base, file), 'w')
                    fout.write(msg)
                except:
                    traceback.print_exc()
                finally:
                    fout.close()
                    file_idx += 1
                    if file_idx % 100 == 0:
                        print(u'处理%d个文件' % file_idx)
    print(u'处理%d个文件' % file_idx)

def get_people():
    try:
        fp_in = open(people_filename, 'r')
        dat = set()
        for line in fp_in:
            line = line.decode('utf-8', 'ignore')
            dat.add(line.strip())
        fp_in.close()
        
        # 缓存数据
        dump_file(dat, people_cached)
        return dat
    except:
        traceback.print_exc()

def get_relation(filename):
    try:
        fin = open(filename, 'r')
        for line in fin:
            line = line.decode('utf-8', 'ignore')
            n1, n2, rel = line.split('\t', 2)
            n1 = n1.strip()
            n2 = n2.strip()
            rel = rel.strip()
            yield (n1, n2, rel)
        fin.close()
    except:
        traceback.print_exc()

def set_relation(n1, n2, rela, obj = {}):
    obj.setdefault(n1, {})
    obj.setdefault(n2, {})
    obj[n1].setdefault(n2, relation2id(rela))
    obj[n2].setdefault(n1, relation2id(rela))
    return obj

def load_sentence_line():
    fin = open(text_filename, 'r')
    for line in fin:
        line = line.decode('utf-8', 'ignore')
        dat_0 = re.findall(u'[0-9A-Za-z\u4e00-\u9fff]+', line)
        dat_1 = ' '.join(dat_0)
        dat_2 = []
        for w in jieba.cut(dat_1):
            if len(w.strip()) > 0:
                dat_2.append(w.strip())
        yield dat_2
    fin.close()
    
def extract_name(sen_list, peo_set):
    people_dict = dict()
    for i in range(len(sen_list)):
        word = sen_list[i]
        if word in peo_set:
            people_dict.setdefault(word, set())
            people_dict[word].add(i)
    return people_dict

def cutWordAndFindName():
    print('计算文本并进行名字分类')
    texts, text_users = [], []
    data = (texts, text_users)
    line_idx = 0
    for line_data in load_sentence_line():
        texts.append(line_data)
        text_users.append(extract_name(line_data, people_set))
        line_idx += 1
        if (line_idx % 5000 == 0):
            print("处理 %d 个文件" % line_idx)
    print("处理 %d 个文件" % line_idx)
    return data

def parseList(main, alt):
    res = []
    for i in range(min((len(main), len(alt)))):
        if alt[i] != 0:
            res.append(main[i])
    return res

# 采用全新的方式处理一句话中存在多个人的关系的问题
def build_study(text_data, rela, window = 3, ignore_rel = False):
    texts = text_data[0]
    text_users = text_data[1]
    for i in range(len(texts)):
        line = texts[i]
        line_users = text_users[i].keys()
        line_userDict = text_users[i]
        for j in range(len(line_users) - 1):
            for k in range(j+1, len(line_users)):
                # j, k 分别都是本次的句子中需要的人的内容
                
                vct = [1] * len(line)
                
                rel = -1
                if not ignore_rel:
                    try:
                        rel = rela[line_users[j]][line_users[k]]
                    except:
                        pass
                # 与这些人有关的内容需要保留下来，而其他
                # 关系则不需要保留，直接删除。遍历其他人名
                
                for l in range(len(line_users)):
                    if l not in (j, k):
                        for sym in line_userDict[line_users[l]]:
                            for m in range(max(0, sym-window+1), min(len(line), sym+window)):
                                vct[m] = 0
                # 因为有可能还是把主要关键人给删了，因此还要重新补上
                for l in (j, k):
                    for sym in line_userDict[line_users[l]]:
                        vct[sym] = 1
                sent_item = parseList(line, vct)
                yield (line_users[j], line_users[k]), sent_item, rel

def s_print(l):
    for w in l:
        print w

if __name__ == '__main__':
    # 在程序的开始部分读入了名字，这样分次器就可以
    # 正确把人名字给分割开来。
    
    # load people
    people_set = None
    if not os.path.exists(people_cached):
        people_set = get_people()
    else:
        people_set = load_dump(people_cached)

    # load text and set data
    textdata = None
    if not os.path.exists(text_cached):
        textdata = cutWordAndFindName()
        dump_file(textdata, text_cached)
    else:
        textdata = load_dump(text_cached)
    
    # load relation
    train_relation = {}
    for n1, n2, rel in \
    get_relation(os.path.join(data_path, 'train_relation.txt')):
        set_relation(n1, n2, rel, train_relation)
    
    # 如果关闭，那么永远不会读取缓存数据，也不会生成缓存数据
    accept_tt_cache = True
    
    trainfile_path = os.path.join(cached_path, 'train')
    testfile_path = os.path.join(cached_path, 'test')
    Users, X, y = None, None, None
    if not accept_tt_cache or not os.path.exists(trainfile_path):
        Users, X, y = [], [], []
        line_idx = 0
        for namegroup, sent, target in build_study(textdata, train_relation):
            if (y != -1):
                Users.append(namegroup)
                X.append(sent)
                y.append(target)
                line_idx += 1
            if (line_idx % 5000 == 0):
                print("处理 %d 个文件" % line_idx)
        print("处理 %d 个文件" % line_idx)
        if accept_tt_cache:
            dump_file((Users, X, y), trainfile_path)
    else:
        (Users, X, y) = load_dump(trainfile_path)
    