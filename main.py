#coding=utf-8#
'''
@File    :   main.py
@Time    :   2021/07/27 01:26:58
@Author  :   Li Yang 
@Version :   1.0
@Contact :   liyang259@mail2.sysu.edu.cn
@License :   (C)Copyright 2020-2021, Li Yang
@Desc    :   None
'''

import numpy as np
from config import Config as cg
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import sys
# sys.path.append('/home/fengtianyuan/code2')
from auxiliary_functions import *
from torch.utils.data import Dataset, DataLoader
from loadnpy import ImageDataset
import matplotlib.pyplot as plt
import torchvision
import torch
import cv2
#log_dir = cg.root_path + "/log"

def train(data):
    ssm = single_salicency_model(drop_rate=0.2, layers=12)
    # ssm = torch.nn.DataParallel(ssm, device_ids=[0, 1]) 分布式训练
    ssm.cuda()
    ssm.load_state_dict(torch.load('bmvc.pth'))
    ssm = ssm.train()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # 启用cuDNN库并自主选择convolution算法
    optimizer = torch.optim.SGD(ssm.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0001, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 30], 0.1)
    # lrdecay管理器
    min_loss = 10
    for epoch in range(30):
        for i, (imagedata, labeldata) in enumerate(data):
            xs = imagedata.cuda()
            ys = labeldata.cuda()
            xs1 = xs.clone().cpu()
            ys1 = ys.clone().cpu()
            trans = torchvision.transforms.ToPILImage()
            ysc = ys
            yp, logits_scale_64_3_upsampled_to_256_sigmoid, logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_64_1_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(xs)
            loss_64_3 , dice_64_3 = fused_loss(logits_scale_64_3_upsampled_to_256_sigmoid, ysc)
            loss_64_2 , dice_64_2 = fused_loss(logits_scale_64_2_upsampled_to_256_sigmoid, ysc)
            loss_64_1 , dice_64_1 = fused_loss(logits_scale_64_1_upsampled_to_256_sigmoid, ysc)
            loss_128 , dice_128 = fused_loss(logits_scale_128_upsampled_to_256_sigmoid, ysc)
            loss_256 , dice_256 = fused_loss(logits_scale_256_upsampled_to_256_sigmoid, ysc)
            loss_yp , dice_yp = fused_loss(yp, ysc)
            cross_entropy = loss_yp + loss_64_3 + loss_64_2 + loss_64_1 + loss_128 + loss_256
            MAE = torch.mean(torch.abs(yp - ysc))
            prec, recall, F_score = F_measure(ysc, yp)
            mask=torch.ones(cg.image_size,cg.image_size).cuda()
            background = torch.zeros(cg.image_size,cg.image_size).cuda()
            yp_threshold = torch.where(yp>0.3, mask, background)
            dice = diceCal(yp_threshold, ys)
            ls643 = yp_threshold.clone().cpu()
            ls643 = ls643[0,:,:,:]
            ls643_1 = torch.squeeze(ls643)
            ls643_2 = trans(ls643_1)
            ls644 = ys.clone().cpu()
            ls644_1 = torch.squeeze(ls644)
            ls644_2 = trans(ls644_1)
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(ls643_2, cmap='gray')
            plt.axis('off')
            plt.savefig('./currentSeg.jpg')
            plt.subplot(1,2,2)
            plt.imshow(ls644_2, cmap='gray')
            plt.axis('off')
            plt.savefig('./currentSegTruth.jpg')
            plt.show()
             
            if cross_entropy < min_loss:
                    min_loss = cross_entropy
                    print('保存参数 ')
                    torch.save(ssm.state_dict(), "./parameters/" +  str(epoch) + '_' + str(i) + ".pth")
            if i % 50 == 0:
                print('保存loss')
                torch.save({'epoch': epoch + 1, 'cross_loss': cross_entropy, 'mae': MAE, 'dice': dice},
                       "./loss/" + "loss" + str(epoch) + '_' + str(i) + ".pth")
            optimizer.zero_grad()
            cross_entropy.backward()
            optimizer.step()

            print('epoch=', epoch, "sampleNo.=", i, 'minloss=', min_loss,  'cross_entropy=', cross_entropy, 'mae=', MAE, 'prec=', prec, 'recall=', recall,
                  'fscore=', F_score, 'dice=', dice)
    scheduler.step()




def test(data):

    ssm = single_salicency_model(drop_rate=0.2, layers=12)
    ssm.cuda()
    ssm.load_state_dict(torch.load('bmvc_cv_single.pth'))
    ssm.eval()
    trans = torchvision.transforms.ToPILImage()
    for epoch in range(1):
        for i, (imagedata, labeldata) in enumerate(data):
            xs = imagedata.cuda()
            ys = labeldata.cuda()
            ysc = ys
            yp, logits_scale_64_3_upsampled_to_256_sigmoid, logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_64_1_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(
            xs)

            loss_64_3,_ = fused_loss(logits_scale_64_3_upsampled_to_256_sigmoid, ysc)
            loss_64_2,_ = fused_loss(logits_scale_64_2_upsampled_to_256_sigmoid, ysc)
            loss_64_1,_ = fused_loss(logits_scale_64_1_upsampled_to_256_sigmoid, ysc)
            loss_128,_ = fused_loss(logits_scale_128_upsampled_to_256_sigmoid, ysc)
            loss_256,_ = fused_loss(logits_scale_256_upsampled_to_256_sigmoid, ysc)
            loss_yp,_ = fused_loss(yp, ysc)
            cross_entropy = loss_yp + loss_64_3 + loss_64_2 + loss_64_1 + loss_128 + loss_256
            MAE = torch.mean(torch.abs(yp - ysc))
            prec, recall, F_score = F_measure(ysc, yp)
            ls643 = yp.clone().cpu()
            ls643 = ls643[0,:,:,:]
            ls643_1 = torch.squeeze(ls643)
            ls643_2 = trans(ls643_1)
            ls644 = ys.clone().cpu()
            ls644_1 = torch.squeeze(ls644)
            ls644_2 = trans(ls644_1)
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(ls643_2, cmap='gray')
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(ls644_2, cmap='gray')
            plt.axis('off')
            plt.show()
            print('Test','Cross Entropy=', cross_entropy , 'MAE=', MAE, 'Fscore=', F_score)



if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    samplePath = os.listdir('../IMG/sample')
    labelPath = os.listdir('../IMG/label')
    data1 = ImageDataset('../IMG/sample/' + samplePath[0], '../IMG/label/' + labelPath[0])
    data = DataLoader(data1, batch_size=4, shuffle=True, pin_memory=False)
    train(data)
    # test(data)
