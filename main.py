#coding=utf-8#
import numpy as np
from config import Config as cg
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3" 选择计算卡
import sys
# sys.path.append('/home/fengtianyuan/code2') # 添加搜索目录
from auxiliary_functions import *
from torch.utils.data import Dataset, DataLoader
from loadnpy import ImageDataset
import matplotlib.pyplot as plt
import torchvision
import torch
#log_dir = cg.root_path + "/log"

def train(data):
    ssm = single_salicency_model(drop_rate=0.2, layers=12)
    # ssm = torch.nn.DataParallel(ssm, device_ids=[0, 1]) 分布式训练
    ssm.cuda()
    # ssm.load_state_dict(torch.load('/home/fengtianyuan/log/ckpt_lumen29_103.pth'))
    ssm = ssm.train()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # 启用cuDNN库并自主选择convolution算法
    optimizer = torch.optim.SGD(ssm.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0001, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 30], 0.1)
    # lrdecay管理器
    min_loss = 10
    for epoch in range(400):
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
            if i == 0:
                ls643 = yp.clone().cpu()
                ls643 = ls643[0,:,:,:]
                ls643_1 = torch.squeeze(ls643)
                ls643_2 = trans(ls643_1)
                plt.imshow(ls643_2)
                plt.axis('off')
                plt.savefig('./test2.jpg')
                plt.show()
            if cross_entropy < min_loss:
                min_loss = cross_entropy
                print('cross_entropy < 10 ')
                torch.save(ssm.state_dict(), "./parameters/" +  str(epoch) + '_' + str(i) + ".pth")
            if i % 50 == 0:
                print('保存loss')
                torch.save({'epoch': epoch + 1, 'cross_loss': cross_entropy, 'mae': MAE, 'dice': dice_yp},
                       "./" + "loss" + str(epoch) + '_' + str(i) + ".pth")
            optimizer.zero_grad()
            cross_entropy.backward()
            optimizer.step()

            print('epoch=', epoch, "sampleNo.=", i, 'cross_entropy=', cross_entropy, 'mae=', MAE, 'prec=', prec, 'recall=', recall,
                  'fscore=', F_score, 'dice=', dice_yp)
    scheduler.step()




def test(data):

    ssm = single_salicency_model(drop_rate=0.2, layers=12)
    ssm = torch.nn.DataParallel(ssm, device_ids=[0, 1])
    ssm.cuda()
    # ssm.load_state_dict(torch.load('/home/fengtianyuan/ma_2_162.pth'))
    ssm.eval()
    for epoch in range(1):
        for i, (imagedata, labeldata) in enumerate(data):
            xs = imagedata.cuda()
            ys = labeldata.cuda()
            ysc = ys
            yp, logits_scale_64_3_upsampled_to_256_sigmoid, logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_64_1_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(
            xs)

            loss_64_3 = fused_loss(logits_scale_64_3_upsampled_to_256_sigmoid, ysc)
            loss_64_2 = fused_loss(logits_scale_64_2_upsampled_to_256_sigmoid, ysc)
            loss_64_1 = fused_loss(logits_scale_64_1_upsampled_to_256_sigmoid, ysc)
            loss_128 = fused_loss(logits_scale_128_upsampled_to_256_sigmoid, ysc)
            loss_256 = fused_loss(logits_scale_256_upsampled_to_256_sigmoid, ysc)
            loss_yp = fused_loss(yp, ysc)
            cross_entropy = loss_yp + loss_64_3 + loss_64_2 + loss_64_1 + loss_128 + loss_256
            MAE = torch.mean(torch.abs(yp - ysc))
            prec, recall, F_score = F_measure(ysc, yp)
            print('Test','Cross Entropy=', cross_entropy , 'MAE=', MAE, 'Fscore=', F_score)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    samplePath = os.listdir('../IMG/sample')
    labelPath = os.listdir('../IMG/label')
    data1 = ImageDataset('../IMG/sample/' + samplePath[0], '../IMG/label/' + labelPath[0])
    data = DataLoader(data1, batch_size=8, shuffle=True, num_workers=1, pin_memory=False)
    train(data)
    # test(data)