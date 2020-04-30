#encoding utf-8
import sys
import math
sys.path.append("/home/yzzc/Work/lq/warpctcaffe_62/warpctcaffe_62/python")
from caffe.proto import caffe_pb2
import caffe
if __name__ == "__main__":

	caffemodel_filename = 'gge_68w.caffemodel'
	net = caffe.Net('deploy.prototxt', 'gge_68w.caffemodel', caffe.TEST)
	model62 =caffe_pb2.NetParameter()
	f=open(caffemodel_filename, 'r')
	model62.ParseFromString(f.read())
	f.close()
	keys = net.params.keys()
	print 'keys: ', keys
	layers = model62.layer
	for L in range(len(layers)):
		layer = layers[L]
		if (layer.type == 'BatchNorm'):
			mean_data = net.params[layer.name][0].data
			variance_data = net.params[layer.name][1].data
			scale_factor = net.params[layer.name][2].data[0]
			data = [mean_data, variance_data]
			max_ = -1000000
			min_ = 10000000
			bw_params = 8
			fl_params = 5
			max_data = (pow(2, bw_params - 1) - 1) * pow(2, -fl_params); #0.9X
			min_data = -pow(2, bw_params - 1) * pow(2, -fl_params);
			for i in range(2):
				for j in range(data[i].shape[0]):
					data[i][j] /= scale_factor
					if i == 1:
						data[i][j] += 1e-05
						data[i][j] = pow(data[i][j], 0.5)
					# data[i][j] = max(min(data[i][j], max_data), min_data);
					# data[i][j] /= pow(2, -fl_params);
					# rounding = layer.quantization_param.rounding_scheme;
					# if rounding == 0:
					# 	#print 'rounding 0'
					# 	data[i][j] = round(data[i][j]);
					# else:
					# 	#print 'rounding 1'
					# 	data[i][j] = floor(data[i][j] + RandUniform_cpu());
					# data[i][j] *= pow(2, -fl_params)
			scale_layer = layers[L+1]
			weight_data = net.params[scale_layer.name][0].data
			bias_data = net.params[scale_layer.name][1].data
			for i in range(weight_data.shape[0]):
				weight_data[i] /= data[1][i] 
				bias_data[i] -= weight_data[i] * data[0][i]
				data[0][i] = 0
				data[1][i] = 1

	net.save('gge_68w_bnscale011.caffemodel')