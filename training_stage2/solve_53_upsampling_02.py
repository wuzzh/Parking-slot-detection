import caffe
#import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

#weights = '/mnt/nfs/zzwu/02_train/zjz-multi-pose_bak/Realtime_Multi-Person_Pose_Estimation-master/training/parking_model/20181212_3_53_8x_07_iter_50000.caffemodel'
#weights = '/mnt/nfs/zzwu/02_train/zjz-multi-pose_bak/Realtime_Multi-Person_Pose_Estimation-master/training/parking_model/20181212_3_53_8x_07_iter_50000.caffemodel'
#weights = '/mnt/nfs/zzwu/02_train/zjz-multi-pose_bak/Realtime_Multi-Person_Pose_Estimation-master/training/parking_model/20190105_ca_53_8x_iter_50000.caffemodel'
#weights = '/mnt/nfs/zzwu/02_train/zjz-multi-pose_bak/Realtime_Multi-Person_Pose_Estimation-master/training/parking_model/20190107_2_ca_53_8x_iter_20000.caffemodel'
#weights = '/mnt/nfs/zzwu/02_train/zjz-multi-pose_bak/Realtime_Multi-Person_Pose_Estimation-master/training/parking_model/20190110_2_stereo_garrage_53_8x_iter_20000.caffemodel'
#weights = '/mnt/nfs/zzwu/02_train/zjz-multi-pose_bak/Realtime_Multi-Person_Pose_Estimation-master/training/parking_model/20190110_53_stere_garage_8x_07_upsampling_iter_50000.caffemodel'
#weights = '/mnt/nfs/zzwu/02_train/zjz-multi-pose_bak/Realtime_Multi-Person_Pose_Estimation-master/training/parking_model/20190114_2_stereo_garrage_53_8x_iter_20000.caffemodel'
weights = '/mnt/nfs/zzwu/02_train/zjz-multi-pose_bak/Realtime_Multi-Person_Pose_Estimation-master/training/parking_model_20190828/20190828_test_key_point_iter_50000.caffemodel'

# init
caffe.set_device(0)
#caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver53_upsampling_02.prototxt')
#solver = caffe.solver('solver.prototxt')

solver.net.copy_from(weights)
'''
# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)
'''
# scoring
#val = np.loadtxt('../../data/unit_20170710/train.txt', dtype=str)

'''
for _ in range(20):

    solver.step(1000000)
    score.seg_tests(solver, False, val, layer='score_32',gt='data')
'''

solver.step(50000)
