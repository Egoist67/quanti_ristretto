#!/bin/bash
GPU_ID=2
cur_date=`date +%Y-%m-%d-%H-%M-%S`
# log_file_name=./examples/warpctc_captcha/model/only_line2_longmao/line2_longmao_noAug-${cur_date}
log_file_name=./examples/warpctc_captcha/model/ft_energy_l1_l2_gan_aug/ft_energy_l1_l2_gan_aug-ft-${cur_date}
		./build/tools/caffe train \
    		-solver ./examples/warpctc_captcha/solver_12_3fc_all_ft.prototxt \
			-weights  ./examples/warpctc_captcha/model_4ft_iter_12000_bnscale011.caffemodel\
     		-gpu $GPU_ID 2>&1 | tee -a ${log_file_name}
