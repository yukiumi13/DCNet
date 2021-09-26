from PIL.Image import Image
import numpy as np
from config import Config as cg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
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
    ssm = single_salicency_model(drop_rate=0.2, layers=12)
    ssm = torch.nn.DataParallel(ssm, device_ids=[0,1])
    ssm.cuda()
    ssm.load_state_dict(torch.load('bmvc_cv.pth'))
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
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # 启用cuDNN库并自主选择convolution算法
    optimizer = torch.optim.SGD(ssm.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0001,nesterov=True))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 30], 0.1)
    # lrdecay管理器
    min_loss = 10
    min_dice = 0.7
    writer = tb.SummaryWriter('./events_ablation')
    train_idx = 0 
    for epoch in range(300):
        for i, (imagedata, labeldata) in enumerate(data):
            xs = imagedata.cuda()
            ys = labeldata.cuda()
            xs1 = xs.clone().cpu()
            ys1 = ys.clone().cpu()
            trans = torchvision.transforms.ToPILImage()
            ysc = ys
            yp, logits_scale_64_3_upsampled_to_256_sigmoid, logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_64_1_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(xs)
            loss_64_3 , dice_64_3 = fused_loss(logits_scale_64_3_upsampled_to_256_sigmoid, ysc)
            # loss_64_2 , dice_64_2 = fused_loss(logits_scale_64_2_upsampled_to_256_sigmoid, ysc)
            # loss_64_1 , dice_64_1 = fused_loss(logits_scale_64_1_upsampled_to_256_sigmoid, ysc)
            # loss_128 , dice_128 = fused_loss(logits_scale_128_upsampled_to_256_sigmoid, ysc)
            # loss_256 , dice_256 = fused_loss(logits_scale_256_upsampled_to_256_sigmoid, ysc)
            # loss_yp , dice_yp = fused_loss(yp, ysc)
            cross_entropy = loss_64_3
            writer.add_scalar('train/loss',cross_entropy,global_step=train_idx)
            MAE = torch.mean(torch.abs(yp - ysc))
            prec, recall, F_score = F_measure(ysc, yp)
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
            if dice > min_dice:
                plt.savefig('Best.svg')
                min_dice = dice
                print('min_dice',min_dice)
            

             
            if cross_entropy < min_loss:
                    min_loss = cross_entropy
                    print('保存参数 ')
                    torch.save(ssm.state_dict(), "./parameters_ablation/" +  str(epoch) + '_' + str(i) + ".pth")
            if i % 50 == 0:
                print('保存loss')
                torch.save({'epoch': epoch + 1, 'cross_loss': cross_entropy, 'mae': MAE, 'dice': dice},
                       "./loss_ablation/" + "loss" + str(epoch) + '_' + str(i) + ".pth")
            optimizer.zero_grad()
            cross_entropy.backward()
            optimizer.step()

            print('epoch=', epoch, "sampleNo.=", i, 'minloss=', min_loss,  'cross_entropy=', cross_entropy, 'dice=', dice)
    scheduler.step()




def test(data):

    ssm = single_salicency_model(drop_rate=0.2, layers=12)
    ssm.cuda()
    ssm = torch.nn.DataParallel(ssm, device_ids=[0, 1, 2, 3])
    ssm.load_state_dict(torch.load('bmvc_current_distributed.pth'))
    ssm.eval()
    trans = torchvision.transforms.ToPILImage()
    for epoch in range(1):
        for i, (imagedata, labeldata) in enumerate(data):
            xs = imagedata.cuda()
            ys = labeldata.cuda()
            ysc = ys
            yp, logits_scale_64_3_upsampled_to_256_sigmoid, logits_scale_64_2_upsampled_to_256_sigmoid, logits_scale_64_1_upsampled_to_256_sigmoid, logits_scale_128_upsampled_to_256_sigmoid, logits_scale_256_upsampled_to_256_sigmoid = ssm(
            xs)

            loss_64_3, dice_64_3 = fused_loss(logits_scale_64_3_upsampled_to_256_sigmoid, ysc)
            loss_64_2,ice_64_2 = fused_loss(logits_scale_64_2_upsampled_to_256_sigmoid, ysc)
            loss_64_1,dice_64_1= fused_loss(logits_scale_64_1_upsampled_to_256_sigmoid, ysc)
            loss_128,dice_128= fused_loss(logits_scale_128_upsampled_to_256_sigmoid, ysc)
            loss_256,dice_256 = fused_loss(logits_scale_256_upsampled_to_256_sigmoid, ysc)
            loss_yp,dice_yp = fused_loss(yp, ysc)
            cross_entropy = loss_yp + loss_64_3 + loss_64_2 + loss_64_1 + loss_128 + loss_256
            MAE = torch.mean(torch.abs(yp - ysc))
            prec, recall, F_score = F_measure(ysc, yp)
            dice = dice_unweighted(yp,ys)
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
            print('Test ',i,' Dice=', dice)

mod = 'train'

if __name__ == '__main__':
    
    if mod == 'train': 
        # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
        data1 = ImageDataset('../IMG/sample.npy', '../IMG/label.npy')
        data = DataLoader(data1, batch_size=8, shuffle=True, pin_memory=False)
        train(data)
        # test(data)

    if mod == 'test':
        data1 = ImageDataset('testset/sample.npy', 'testset/label.npy')
        data = DataLoader(data1, batch_size=17, shuffle=True, pin_memory=False)
        test(data)
