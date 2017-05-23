# encoding: utf-8

import sys
import logging
import traceback
import os
import jieba
import codecs
import re
# import math
# import string
import numpy as np
import random
# import matplotlib

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn import metrics
from collections import Counter

reload(sys);
sys.setdefaultencoding('utf-8')

basepath = os.getcwdu()
trainpath = os.path.join(basepath, u'data_offline')
testpath = os.path.join(basepath, u'data_test')
loadfilepath = os.path.join(basepath, u'')
#validpath = os.path.join(basepath, u'data_valid')

train_cachepath = os.path.join(basepath, u'cached_offline')
test_cachepath = os.path.join(basepath, u'cached_test')
#valid_cachepath = os.path.join(basepath, u'cached_valid')
output_tf =  os.path.join(basepath, u'out_tf.txt')
output_tfidf =  os.path.join(basepath, u'out_tfidf.txt')
output_km =  os.path.join(basepath, u'out_kmeans.txt')

# in this project, I want to start
# encapsulate all the functions I want to use
# in a single class so that I can easily
# manage all of it.
class Text_Cluster(object):
    def __init__(self):
        pass

    # 自动完成子目录
    def complete_subpath(self, path, subpath):
        if not isinstance(path, unicode):
            return ''
            
        if isinstance(subpath, unicode):
            return os.path.join(path, subpath)
        elif isinstance(subpath, list):
            ans = []
            for p in subpath:
                if isinstance(p, unicode):
                    ans.append(os.path.join(path, p))
            return ans
            
    def detect_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def deal_withChineseText(self, data_ucs):
        #remove all punctuation
        data_ucs_chs = ''.join(re.findall(u'[\u4e00-\u9fff]+', \
                                      data_ucs))
        list_eng =  re.findall(u'[a-zA-z\s]+', data_ucs)
        data_ucs_eng = ""
        for word in list_eng:
            word = word.strip()
            if len(word) > 0:
                data_ucs_eng += ' ' + word.strip()
        return data_ucs_chs + ' ' + data_ucs_eng.strip()
    
    # 将文件以 UTF-8 模式加载
    def load_fileUTF8(self, AbsFilePath):
        fin = open(AbsFilePath, 'r')
        fdata = fin.read()
        fin.close()
        data_ucs = fdata.decode('utf-8')
        return data_ucs
    
    # 加载用户词典
    def load_userdictfile(self, dict_file):
        jieba.load_userdict(dict_file)
    
    # 加载文件
    def load_processfile(self, process_file):
		# 语料库列表
        corpus_list = []
        try:
            fp = open(process_file, 'r')
            
            for line in fp:
                conline = line.strip()
                corpus_list.append(conline)
            return True, corpus_list
        except:
            logging.error(traceback.format_exc())
            return False, u'Failed to get given process_file.'
    
    # 加载目录
    def load_processfolder(self, process_path, ifdeal):
		# 语料库列表
        corpus_list = []
        file_list = []
        try:
            files = os.listdir(process_path)
            for pro_fil in files:
                process_file = os.path.join(process_path, pro_fil)
                if not os.path.isfile(process_file):
                    continue
                filedata = self.load_fileUTF8(process_file)
                # deal with Chinese text, remove 
                # all the punctuations
                if isinstance(ifdeal, bool) and ifdeal == True:
                    corpus_list.append(self.deal_withChineseText(filedata))
                else:
                    corpus_list.append(filedata)
                file_list.append(pro_fil)
            return True, corpus_list, file_list
        except:
            logging.error(traceback.format_exc())
            return False, u'Failed to get process files in given path.', u'No file names'
    
    # 分词程序
    def segerate_word(self, sentence):
        seg_word = jieba.cut(sentence)
        return " ".join(seg_word)
    
    # 输出文件
    def output_file(self, out_file, item):
        try:
            fw = open(out_file, 'a')
            fw.write('%s' % (item.encode('utf-8')))
            fw.close()
        except:
            logging.error(traceback.format_exc())
            return False, u'out file fail'
    
    def process_cluster(self, \
                        process_folder_path, \
                        cached_folder_path, \
                        tf_ResFileName, \
                        tfidf_ResFileName, \
                        n, \
                        cluster_ResFileName, \
                        valid_filename = u'',
                        sample_terms_count = 15,
                        sample_files_count = 3
                        ):
        try:
            sen_seg_list = []
            sen_file_list = []
            sen_file_dict = {}
            
            self.detect_path(cached_folder_path)
            cached_files = self.complete_subpath(cached_folder_path, 
                                                os.listdir(cached_folder_path))
            # here is the branch where cached files are not found.
            # We should never use such a thing to 
            if len(cached_files) == 0:
                print('Calculating...')
                flag, lines, flist = self.load_processfolder(process_folder_path, True)
                if flag == True:
                    sen_file_list = flist
                    for line in lines:
                        sen_seg_list.append(self.segerate_word(line))
                    for i in range(0, len(flist)):
                        self.output_file(\
                            os.path.join(cached_folder_path, sen_file_list[i]), \
                            sen_seg_list[i])
                else:
                    logging.error(u'Loading original files failed.')
                    return False, u"load error"
            else:
                 # loading cached files
                 print('loading...')
                 flag, lines, flist = self.load_processfolder(cached_folder_path, False)
                 if flag == True:
                     sen_seg_list = lines
                     sen_file_list = flist
                 else:
                     logging.error(u'Loading cached files failed.')
                     return False, u'load error'
            
            #calc the dict for file names
            for i in range(0, len(sen_file_list)):
                filename, ext = os.path.splitext(sen_file_list[i])
                sen_file_dict[filename] = i
            
            
            # begin learning
            tf_vectorizer = CountVectorizer()
            
            # 就是文章中所有的单词，全部都用一个数字进行标记
            # 然后，对于文章中的单词分别使用这些ID进行替换，而形成的新的数据列表。
            tf_matrix = tf_vectorizer.fit_transform(sen_seg_list)
            
            word_list = tf_vectorizer.get_feature_names()
            
            tfidf_transformer = TfidfTransformer()        
            tfidf_matrix = tfidf_transformer.fit_transform(tf_matrix)
            
            tf_weight = tf_matrix.toarray()
            tfidf_weight = tfidf_matrix.toarray()
    
            # 打印特征向量文本内容
            # print 'Features length: ' + str(len(word_list))
            '''
            tf_Res = codecs.open(tf_ResFileName, 'w', 'utf-8')
            word_list_len = len(word_list)
            for num in range(word_list_len):
                if num == word_list_len - 1:
                    tf_Res.write(word_list[num])
                else:
                    tf_Res.write(word_list[num] + '\t')
            tf_Res.write('\r\n')
            tf_Res.close()
            '''
                    
            # 输出tfidf矩阵
            '''
            tfidf_Res = codecs.open(tfidf_ResFileName, 'w', 'utf-8')
    
            for num in range(word_list_len):
                if num == word_list_len - 1:
                    tfidf_Res.write(word_list[num])
                else:
                    tfidf_Res.write(word_list[num] + '\t')
            tfidf_Res.write('\r\n')
            tfidf_Res.close()
            '''
            
            # 计算 KMeans 聚类算法
            tf_kmeans = KMeans(n_clusters = n)
            tf_kmeans.fit(tfidf_matrix)
    
            # print (Counter(tf_kmeans.labels_))  # 打印每个类多少人
            # 中心点
            # print(km.cluster_centers_)
            # 每个样本所属的簇
            
            clusterRes = codecs.open(cluster_ResFileName, 'w', 'utf-8') 
            # data_class = pd.read_table('id2class.txt',header=None)
            count = 1
            while count <= len(tf_kmeans.labels_):
                clusterRes.write(str(count) + '\t' + sen_file_list[count - 1] + \
                                 '\t' + str(tf_kmeans.labels_[count - 1]))
                clusterRes.write('\r\n')
                count = count + 1
            clusterRes.close()
            
            if isinstance(valid_filename, unicode) and \
                len(valid_filename) > 0 and \
                os.path.isfile(valid_filename):
                    valid_labels_s = [u''] * len(sen_file_list)
                    fval = codecs.open(valid_filename, 'r', encoding='utf-8')
                    for line in fval:
                        filename, val = line.split()
                        truefilename, ext = os.path.splitext(filename)
                        valid_labels_s[sen_file_dict[truefilename]] = val
                        tempVct = CountVectorizer()
                        valid_labels = tempVct.fit_transform(valid_labels_s)
                    fval.close()
                    
                    # calc the SS, SD, DS and DD of valid_labels and 
                    # tf_kmeans.labels_
                    
                    C = tf_kmeans.labels_
                    C_ = valid_labels.indices
                    
                    SS = 0
                    SD = 0
                    DS = 0
                    DD = 0
                    
                    for i in range(0, len(sen_file_list)):
                        for j in range(i+1, len(sen_file_list)):
                            if C[i] == C[j] and C_[i] == C_[j]:
                                SS += 1
                            elif C[i] == C[j] and C_[i] != C_[j]:
                                SD += 1
                            elif C[i] != C[j] and C_[i] == C_[j]:
                                DS += 1
                            else:
                                DD += 1
                    
                    list_keyword = [[] for i in range(n)]
                    
                    order_centorids = tf_kmeans.cluster_centers_.argsort()[:,::-1]
                    
                    for i in range(0, n):
                        for idx in order_centorids[i,:sample_terms_count]:
                            list_keyword[i].append(word_list[idx])
                    
                    list_file = [[] for i in range(n)]
                    for i in range(0, n):
                        id_list = np.where(tf_kmeans.labels_ == i)[0].tolist()
                        for j in random.sample(id_list, sample_files_count):
                            list_file[i].append(sen_file_list[j])
                    
                    return True, (SS, SD, DS, DD, sen_file_list, list_keyword, \
                                  Counter(tf_kmeans.labels_), \
                                  metrics.silhouette_score(tfidf_matrix, tf_kmeans.labels_, metric='euclidean'),
                                  list_file)
            
        except:
            logging.error(traceback.format_exc())
            return False, "process fail"
        
if __name__ == '__main__':
    txClst = Text_Cluster()
    agg_count = 4
    state, result = txClst.process_cluster(trainpath, train_cachepath, output_tf, \
                           output_tfidf, agg_count, output_km, \
                           os.path.join(basepath, 'id2class.txt'))
    if (state == True):
        a = float(result[0])
        b = float(result[1])
        c = float(result[2])
        d = float(result[3])
        m = len(result[4])
        JC = a/(a+b+c)
        FMI = np.sqrt(np.power(a, 2) / (a+b) / (a+c) )
        RAND = 2 * (a + d) / (float(m)*(float(m)-1))
        print(u'计算结果：\nJC：%f\nFMI：%f\nRAND：%f\n边缘值：%f' % (JC, FMI, RAND, result[7]))
        print '\n\n'
        dic = [[]] * agg_count
        # 随机从接下来的所有的数组里面，每个
        # 随机选出10个单词进行打印
        cluster_attr = result[5]
        counter = result[6]
        file_list = result[8]
        for i in range(0, agg_count):
            print(u'第%d个分类: 共计%d个文件' % (i, counter[i]))
            for word in cluster_attr[i]:
                print word + ' ',
            print
            print '随机参考文件:'
            for file_name in file_list[i]:
                print '\'' + file_name + '\'' + ' ',
            print '\n'