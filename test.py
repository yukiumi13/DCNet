# -*-coding:utf-8 -*-
'''
@File    :   test.py
@Time    :   2021/07/30 18:32:09
@Author  :   Li Yang 
@Version :   1.0
@Contact :   liyang259@mail2.sysu.edu.cn
@License :   (C)Copyright 2020-2021, Li Yang
@Desc    :   None
'''

from main import *

samplePath = os.listdir('../testset/sample')
labelPath = os.listdir('../testset/label')
data1 = ImageDataset('../testset/sample/' + samplePath[0], '../testset/label/' + labelPath[0])
data = DataLoader(data1, batch_size=4, shuffle=True, pin_memory=False)
test(data)
