import os
import sys
import time
import shutil

net_src = "yolov3-tiny.prototxt"  #src_prototxt
model_src = "yolov3-tiny.caffemodel"  #32bits caffemodel
model_bnscale011 = "yolov3-tiny_bn011.caffemodel"  #bnscale011 caffemodel

os.system("python2 auto_bnscale011.py --net0="+net_src+ \
	" --model="+model_src+" --model_bn="+model_bnscale011)