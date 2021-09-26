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

from PIL.Image import Image
import numpy as np
from config import Config as cg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
# sys.path.append('/home/fengtianyuan/code2')
from auxiliary_functions import *
from torch.utils.data import Dataset, DataLoader
from loadnpy import ImageDataset
import matplotlib.pyplot as plt
import torchvision
import torch
import cv2
import PIL
import torch.utils.tensorboard as tb
#log_dir = cg.root_path + "/log"

def train(data):
    # ssm = single_salicency_model(drop_rate=0.2, layers=12)
    # ssm = single_salicency_model_c(drop_rate=0.2, layers=12)
    trans = torchvision.transforms.ToPILImage()
    ssm = single_salicency_model_b(drop_rate=0.2, layers=12)
    ssm = torch.nn.DataParallel(ssm, device_ids=[0, 1])
    ssm.cuda()
    # ssm.load_state_dict(torch.load('ssm_e.pth'))
    ssm = ssm.train()
    # fine-tune
    '''
    for para in ssm.module.conv2d128.parameters():
        para.requires_grad=False
    for para in ssm.module.conv2d256.parameters():
        para.requires_grad=False
    for para in ssm.module.convlast.parameters():
        para.requires_grad=False
    for para in ssm.module.convlast2.parameters():
        para.requires_grad=False
    '''
    '''
    for para in ssm.module.conv2d.parameters():
        para.requires_grad = False
    for para in ssm.module.conv2d1.parameters():
        para.requires_grad = False
    for para in ssm.module.bac1.parameters():
        para.requires_grad = False
    for para in ssm.module.conv2d2.parameters():
        para.requires_grad = False
    for para in ssm.module.bac2.parameters():
        para.requires_grad = False
    for para in ssm.module.conv2d3.parameters():
        para.requires_grad = False
    for para in ssm.module.bac3.parameters():
        para.requires_grad = False
    for para in ssm.module.conv2d4.parameters():
        para.requires_grad = False
    for para in ssm.module.bac4.parameters():
        para.requires_grad = False
    for para in ssm.module.conv2d5.parameters():
        para.requires_grad = False
    for para in ssm.module.ppm64.parameters():
        para.requires_grad = False
    '''
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # 启用cuDNN库并自主选择convolution算法
    optimizer = torch.optim.SGD(ssm.parameters(), lr=0.0001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',verbose=True)

    # lrdecay管理器
    min_dice = 0.8
    min_dice_val = 0
    writer = tb.SummaryWriter('./events_ssm_f')
    train_idx = 0 
    for epoch in range(300):
        for i, (imagedata, labeldata) in enumerate(data):
            xs = imagedata.cuda()
            ys = labeldata.cuda()
            ''''
            xs1 = xs.clone().cpu()
            ys1 = ys.clone().cpu()
            trans = torchvision.transforms.ToPILImage()
            '''
            ysc = ys.unsqueeze(1)
            # yp, logits_scale_64_4_upsampled_to_256_sigmoid, logits_scale_64_3_upsampled_to_256_sigmoid,logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_64_1_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(xs)
            yp, logits_scale_64_3_upsampled_to_256_sigmoid,logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_64_1_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(xs)
            # yp, logits_scale_64_3_upsampled_to_256_sigmoid, logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(xs)
            # loss_64_4 , dice_64_4 = fused_loss(logits_scale_64_4_upsampled_to_256_sigmoid, ysc) 
            # loss_64_3 , dice_64_3 = fused_loss(logits_scale_64_3_upsampled_to_256_sigmoid, ysc)
            # loss_64_2 , dice_64_2 = fused_loss(logits_scale_64_2_upsampled_to_256_sigmoid, ysc)
            # loss_64_1 , dice_64_1 = fused_loss(logits_scale_64_1_upsampled_to_256_sigmoid, ysc)
            # loss_128 , dice_128 = fused_loss(logits_scale_128_upsampled_to_256_sigmoid, ysc)
            # loss_256 , dice_256 = fused_loss(logits_scale_256_upsampled_to_256_sigmoid, ysc)
            loss_yp , dice_yp = fused_loss(yp, ysc)
            cross_entropy = loss_yp
            # + loss_64_3 + loss_64_2 + loss_64_1 + loss_128 + loss_256
            writer.add_scalar('train/loss',cross_entropy,global_step=train_idx)
            '''
            MAE = torch.mean(torch.abs(yp - ysc))
            prec, recall, F_score = F_measure(ysc, yp)
            '''
            mask=torch.ones(cg.image_size,cg.image_size).cuda()
            background = torch.zeros(cg.image_size,cg.image_size).cuda()
            yp = torch.where(yp>0.5, mask, background)
            # dice = diceCal(yp_threshold, ys)
            # ls643 = yp_threshold.clone().cpu()
            dice = dice_unweighted(yp,ys)
            writer.add_scalar('train/dice',dice,global_step=train_idx)
            train_idx += 1
            # Visualization 
            '''
            yp_img = yp.clone().cpu()
            out = []
            out.append(yp_img)
            out.append(logits_scale_256_upsampled_to_256_sigmoid.clone().cpu())
            out.append(logits_scale_128_upsampled_to_256_sigmoid.clone().cpu())
            out.append(logits_scale_64_1_upsampled_to_256_sigmoid.clone().cpu())
            out.append(logits_scale_64_2_upsampled_to_256_sigmoid.clone().cpu())
            out.append(logits_scale_64_3_upsampled_to_256_sigmoid.clone().cpu())
            plt.figure()
            for j in range(0,yp_img.shape[0]):
                for lv, img in enumerate(out):
                    img = img[j,:,:,:]
                    img_1 = torch.squeeze(img)
                    img_c = img_1.unsqueeze(0)
                    writer.add_image('train/outputs/pred'+str(j)+'/level'+str(lv),img_c)
                    img_2 = trans(img_1)
                    plt.subplot(yp_img.shape[0],7,lv+1+j*7)
                    plt.imshow(img_2, cmap='gray')
                    plt.axis('off')
                    plt.title('S'+str(lv))
                gt_img = ys.clone().cpu()
                gt_img = gt_img[j,:,:]
                gt_img_c = gt_img.unsqueeze(0)
                writer.add_image('train/outputs/gt'+str(j),gt_img_c)
                gt_img_1 = torch.squeeze(gt_img)
                gt_img_2 = trans(gt_img_1)
                plt.subplot(yp_img.shape[0],7,7+j*7)
                plt.imshow(gt_img_2,cmap='gray')
                plt.axis('off')
                plt.title('G')
                plt.savefig('Out.svg')
            '''

            '''
            if dice > 0.8:
                plt.savefig('results/'+str(dice)+'.svg')
            '''
            '''
            if dice > min_dice:
                plt.savefig('Best.svg')
                min_dice = dice
                print('min_dice',min_dice)
            '''

             
            '''
            if i % 50 == 0:
                print('保存loss')
                torch.save({'epoch': epoch + 1, 'cross_loss': cross_entropy, 'dice': dice},
                       "./loss/" + "loss" + str(epoch) + '_' + str(i) + ".pth")
            '''
            optimizer.zero_grad()
            cross_entropy.backward()
            optimizer.step()

            print('epoch=', epoch, "sampleNo.=", i, 'cross_entropy=', cross_entropy, 'dice=', dice)
        
        val_data1 = ImageDataset('../testset/sample.npy', '../testset/label.npy')
        val_data = DataLoader(val_data1, batch_size=1, shuffle=True, pin_memory=False)
        dice_val_sum = 0
        sen_val_sum = 0
        spe_val_sum = 0 
        ppv_val_sum = 0
        voe_val_sum = 0
        rvd_val_sum = 0 
        for i, (imagedata, labeldata) in enumerate(val_data):
            xs = imagedata.cuda()
            ys = labeldata.cuda()
            ysc = ys.unsqueeze(1)
            yp,logits_scale_64_3_upsampled_to_256_sigmoid, logits_scale_64_2_upsampled_to_256_sigmoid,logits_scale_64_1_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(xs)
            # yp, logits_scale_64_4_upsampled_to_256_sigmoid, logits_scale_64_3_upsampled_to_256_sigmoid,logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_64_1_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(xs)
            # yp, logits_scale_64_3_upsampled_to_256_sigmoid, logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(xs)
            mask=torch.ones(cg.image_size,cg.image_size).cuda()
            background = torch.zeros(cg.image_size,cg.image_size).cuda()
            yp = torch.where(yp>0.5, mask, background)
            # dice = diceCal(yp_threshold, ys)
            # ls643 = yp_threshold.clone().cpu()
            dice_val_sum = dice_val_sum + dice_unweighted(yp,ys)
            sen_val_sum = sen_val_sum + sen(yp,ys)
            spe_val_sum = spe_val_sum + spe(yp,ys)
            ppv_val_sum = ppv_val_sum + ppv(yp,ys)
            voe_val_sum = voe_val_sum + voe(yp,ys)
            rvd_val_sum = rvd_val_sum + rvd(yp,ys)
            yp_img = yp.clone().cpu()
            yp_img = yp_img[0,:,:,:]
            yp_img_1 = torch.squeeze(yp_img)
            yp_img_2 = trans(yp_img_1)
            # yp_img_2 = yp_img_2.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            yp_img_2 = yp_img_2.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            yp_img_2 = yp_img_2.rotate(90)
            gt_img = ys.clone().cpu()
            gt_img = gt_img[0,:,:]
            gt_img_1 = torch.squeeze(gt_img)
            gt_img_2 = trans(gt_img_1)
            # gt_img_2 = gt_img_2.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            gt_img_2 = gt_img_2.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            gt_img_2 = gt_img_2.rotate(90)
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(yp_img_2, cmap='gray')
            plt.axis('off')
            plt.title('prediction')
            plt.subplot(1,2,2)
            plt.imshow(gt_img_2, cmap='gray')
            plt.axis('off')
            plt.title('ground truth')
            plt.savefig('testresults/test'+str(i)+'.jpg')
        dice_val = dice_val_sum/17
        sen_val = sen_val_sum/17
        spe_val = spe_val_sum/17
        ppv_val = ppv_val_sum/17
        voe_val = voe_val_sum/17
        rvd_val = rvd_val_sum/17

        if  dice_val > min_dice_val : 
            min_dice_val = dice_val
            print('保存参数 ')
            torch.save(ssm.state_dict(), "./parameters/" + "best" + '_f_' + str(min_dice_val) + ".pth")
        print('dice=',dice_val, 'sen=', sen_val, 'spe=', spe_val, 'ppv=', ppv_val, 'voe=', voe_val, 'rvd=', rvd_val)

        writer.add_scalar('validate/dice',dice_val,global_step=epoch) 
        writer.add_scalar('validate/sen',sen_val,global_step=epoch) 
        writer.add_scalar('validate/spe',spe_val,global_step=epoch) 
        writer.add_scalar('validate/ppv',ppv_val,global_step=epoch) 
        writer.add_scalar('validate/voe',voe_val,global_step=epoch) 
        writer.add_scalar('validate/rvd',rvd_val,global_step=epoch) 
        scheduler.step(dice_val)

def test(data):    
    ssm = single_salicency_model(drop_rate=0.2, layers=12)
    ssm.cuda()
    # ssm = torch.nn.DataParallel(ssm, device_ids=[0, 1])
    ssm.load_state_dict(torch.load('ssm_a.pth'))
    ssm.eval()
    trans = torchvision.transforms.ToPILImage()
    dice_val_sum = 0
    sen_val_sum = 0
    spe_val_sum = 0 
    ppv_val_sum = 0
    voe_val_sum = 0
    rvd_val_sum = 0 
    for i, (imagedata, labeldata) in enumerate(data):
        xs = imagedata.cuda()
        ys = labeldata.cuda()
        yp,logits_scale_64_3_upsampled_to_256_sigmoid, logits_scale_64_2_upsampled_to_256_sigmoid,logits_scale_64_1_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(xs)
        # yp, logits_scale_64_4_upsampled_to_256_sigmoid, logits_scale_64_3_upsampled_to_256_sigmoid,logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_64_1_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(xs)
        # yp, logits_scale_64_3_upsampled_to_256_sigmoid, logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(xs)
        mask=torch.ones(cg.image_size,cg.image_size).cuda()
        background = torch.zeros(cg.image_size,cg.image_size).cuda()
        yp = torch.where(yp>0.5, mask, background)
        # dice = diceCal(yp_threshold, ys)
        # ls643 = yp_threshold.clone().cpu()
        dice_val_sum = dice_val_sum + dice_unweighted(yp,ys)
        sen_val_sum = sen_val_sum + sen(yp,ys)
        spe_val_sum = spe_val_sum + spe(yp,ys)
        ppv_val_sum = ppv_val_sum + ppv(yp,ys)
        voe_val_sum = voe_val_sum + voe(yp,ys)
        rvd_val_sum = rvd_val_sum + rvd(yp,ys)
        yp_img = yp.clone().cpu()
        yp_img = yp_img[0,:,:,:]
        yp_img_1 = torch.squeeze(yp_img)
        yp_img_2 = trans(yp_img_1)
        # yp_img_2 = yp_img_2.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        yp_img_2 = yp_img_2.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        yp_img_2 = yp_img_2.rotate(90)
        gt_img = ys.clone().cpu()
        gt_img = gt_img[0,:,:]
        gt_img_1 = torch.squeeze(gt_img)
        gt_img_2 = trans(gt_img_1)
        # gt_img_2 = gt_img_2.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        gt_img_2 = gt_img_2.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        gt_img_2 = gt_img_2.rotate(90)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(yp_img_2, cmap='gray')
        plt.axis('off')
        plt.title('prediction')
        plt.subplot(1,2,2)
        plt.imshow(gt_img_2, cmap='gray')
        plt.axis('off')
        plt.title('ground truth')
        plt.savefig('testresults/test'+str(i)+'.jpg')
    dice_val = dice_val_sum/17
    sen_val = sen_val_sum/17
    spe_val = spe_val_sum/17
    ppv_val = ppv_val_sum/17
    voe_val = voe_val_sum/17
    rvd_val = rvd_val_sum/17

    print('dice=',dice_val, 'sen=', sen_val, 'spe=', spe_val, 'ppv=', ppv_val, 'voe=', voe_val, 'rvd=', rvd_val)

mod = 'test'

if __name__ == '__main__':
    
    if mod == 'train': 
        # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
        data1 = ImageDataset('../train_data_aug/sample.npy', '../train_data_aug/label.npy')
        data = DataLoader(data1, batch_size=8, shuffle=True, pin_memory=False)
        train(data)
        # test(data)

    if mod == 'test':
        data1 = ImageDataset('../testset/sample.npy', '../testset/label.npy')
        data = DataLoader(data1, batch_size=1, shuffle=True, pin_memory=False)
        test(data)
