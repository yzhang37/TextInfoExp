# encoding: utf-8
import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')

base_path = os.getcwdu()

for base, dirs, files in os.walk(base_path):
    flag = 0
    for file in files:
        if file == '.DS_Store':
            if (flag == 0):
                print('在'+base+'下：')
                flag = 1
            print('删除文件：' + file)
            os.remove(os.path.join(base, file))