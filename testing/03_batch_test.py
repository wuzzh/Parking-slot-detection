import cv2 as cv 
import numpy as np
import scipy
from PIL import Image
import os
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import matplotlib
#%matplotlib inline
import pylab as plt
import normalization as NL
import visualize_tool as vt
import get_PAF_info as PAF
import get_heatmap_info as HEATMAP
import key_point_detection as KPD
import skimage.io
import scipy.io as sio

#--------------------------------------------------------------------------------------------------------------------------------
#init

param, model = config_reader()


if param['use_gpu']: 
    caffe.set_mode_gpu()
    caffe.set_device(param['GPUdeviceNumber']) # set to your device!
else:
    caffe.set_mode_cpu()
net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

img_path = param['img_path']

num_parts = model['np']
print "------num_parts------",num_parts
num_connection = model['nc']

#---------------------------------------------------------------------------------------------------------------------------------
#get key point and paf info

#H = 148 # 4 up 
#W = 48

#H = 300
#W = 96
H = 37
W = 12

i_w = 25
i_h = 25

iCount = 1
for img_dir in os.listdir(img_path):
    img_path_1 = img_path +'/'+ img_dir
    #if img_dir != 'parking_key_point0828_cd_L':
	#continue
    #print '----img_path_1',img_path_1

    for test_image in os.listdir(img_path_1):
    	iCount = iCount + 1
    	#if iCount !=8:
       	#	continue
    	#test_image = '000007.jpg'
    	#print '---test image name---', test_image
        #if test_image != '000003.jpg':
          # continue
        suffix = test_image.split('.');
        if suffix == 'db':
	    continue
    	img = cv.imread(img_path_1 + '/' + test_image)

    	multiplier = [x * model['boxsize'] / img.shape[0] for x in param['scale_search']]

    	tmap_avg = np.zeros((H, W, num_parts+1))
    	paf_avg = np.zeros((H, W, num_parts+1))
    	heatmap_avg = np.zeros((H, W, num_parts+1))

    	heatmap_conv6_3 = np.zeros((i_h, i_w, num_parts+1))

	marks = []
	marks_filename = test_image.split('.')[0] + '.mat'
    	KPD.key_point_detection(test_image,img, param, model, multiplier, net, paf_avg, heatmap_avg, heatmap_conv6_3, img_dir, marks)
	sio.savemat('./mat/'+marks_filename, {'marks':marks})
   #	break
   # break
#---------------------------------------------------------------------------------------------------------------------------------

    
   
    
