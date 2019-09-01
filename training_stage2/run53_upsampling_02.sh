#!/usr/bin/env sh
#/home/CN/zzwu/01_train/zjz-multi-pose/caffe_train/build/tools/caffe train --solver=solver_unit_17.prototxt --gpu=$1 --weights=../parking_model/20171114_unit_12_iter_110000.caffemodel 2>&1 | tee ./output_17_unit.txt


#python solve_53_upsampling_02.py  2>&1 | tee ./logs/20190107_53_ca_8x_upsampling.txt &
#python solve_53_upsampling_02.py  2>&1 | tee ./logs/20190110_53_stereo_garage_8x_upsampling.txt &
python solve_53_upsampling_02.py  2>&1 | tee ./logs_20190828/20190828_key_point_upsampling.txt &





