#encoding utf-8
import sys
import os
import math
import numpy as np
sys.path.append("/rdata/qi.liu/code/End2endLP/ezai/caffe_base_ezai/python")
from caffe.proto import caffe_pb2
import caffe
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n0", "--net0", help="caffe net prototxt file name", required=True)
parser.add_argument("-m", "--model", help="caffe model file name", required=True)
parser.add_argument("-mb", "--model_bn", help="caffe model after merging bn and scale into one", required=True)
args = parser.parse_args()


if __name__ == "__main__":

	net = caffe.Net(args.net0, args.model, caffe.TEST)
	model62 =caffe_pb2.NetParameter()
	with open(args.model, 'r') as f:
		model62.ParseFromString(f.read())
	keys = net.params.keys()
	print 'keys: ', keys
	layers = model62.layer
	for L, layer in enumerate(layers):
		if (layer.type == 'BatchNorm'):
			mean_data = net.params[layer.name][0].data
			variance_data = net.params[layer.name][1].data
			scale_factor = net.params[layer.name][2].data[0]
			
			
			data = np.vstack((mean_data, variance_data))
			data /= scale_factor
			# data[1,:] += 1e-05
			data[1,:] += 1e-04

			data[1,:] = data[1, :]**0.5

			scale_layer = layers[L+1]
			weight_data = net.params[scale_layer.name][0].data
			bias_data = net.params[scale_layer.name][1].data
			print weight_data, bias_data
			weight_data /= data[1,:]
			bias_data -= weight_data * data[0,:]
			
			net.params[layer.name][0].data[...] = 0
			net.params[layer.name][1].data[...] = 1
			net.params[layer.name][2].data[...] = 1
	net.save(args.model_bn)
