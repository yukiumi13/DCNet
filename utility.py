#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 16:18:18 2021

@author: menglidaren
"""
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import gui

def myshow(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayViewFromImage(img)
    spacing = img.GetSpacing()
        
    ysize = nda.shape[0]
    xsize = nda.shape[1]
      
    figsize = (1 + 2*margin) * ysize / dpi, (1 + 2*margin) * xsize / dpi

    fig = plt.figure(title, figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    extent = (0, xsize*spacing[1], 0, ysize*spacing[0])
    
    t = ax.imshow(nda,
            extent=extent,
            interpolation='none',
            cmap='gray',
            origin='lower')
    
    if(title):
        plt.title(title)

def disp_images(images, fig_size, wl_list=None):
    if images[0].GetDimension()==2:
      gui.multi_image_display2D(image_list=images, figure_size=fig_size, window_level_list=wl_list)
    else:
      gui.MultiImageDisplay(image_list=images, figure_size=fig_size, window_level_list=wl_list)

def show_sample_label(img2show, label2show):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img2show, cmap="gray")
    plt.title('sample')
    plt.subplot(1,2,2)
    plt.imshow(label2show, cmap="gray")
    plt.title('label')