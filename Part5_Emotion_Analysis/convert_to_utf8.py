#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import os
import logging
import cchardet
reload(sys)
sys.setdefaultencoding('utf-8')

base_path = os.getcwdu()
data_path = os.path.join(base_path, u'htl')

def detnconv(path):
    if os.path.isfile(path):
        fin = open(path, 'r')
        msg = fin.read()
        fin.close()
        res = cchardet.detect(msg)
        encoding = res['encoding']
        try:
            if encoding.strip().lower() == 'utf-8':
                return
            new_msg = msg.decode(res['encoding'], 'ignore')
            fin = open(path ,'w')
            fin.write(new_msg.encode('utf-8'))
            fin.close()
        except:
            logging.error(path)
        
        
def cascadeScan(path):
    if os.path.isdir(path):
        for child in os.listdir(path):
            if os.path.isdir(os.path.join(path, child)):
                cascadeScan(os.path.join(path, child))
            else:
                detnconv(os.path.join(path, child))
                
cascadeScan(data_path)