#encoding utf-8
import sys
sys.path.append("/home/.prog_and_data/shared/codes/ezai/caffe_stand_alone_qmake/warpctcaffe_62/python")
from caffe.proto import caffe_pb2
import caffe
if __name__ == "__main__":

	caffemodel_filename = 'lstm_ctc_new12_iter_100000.caffemodel'
	net = caffe.Net('lstm_ctc_newsmall.prototxt', 'lstm_ctc_new12_iter_100000.caffemodel', caffe.TEST)
	model62 =caffe_pb2.NetParameter()
	f=open(caffemodel_filename, 'r')
	model62.ParseFromString(f.read())
	f.close()
	keys = net.params.keys()
	print 'keys: ', keys
	layers = model62.layer
	for L in range(len(layers)):
		layer = layers[L]
		if (layer.type == 'Convolution'):
			weight_data = net.params[layer.name][0].data
			if (layer.name == 'conv1'):
				bw_params,fl_params = 6,5
			if (layer.name == 'conv2'):
			 	bw_params,fl_params = 7,6	
			if (layer.name == 'conv3'):
			 	bw_params,fl_params = 7,6		
			if (layer.name == 'conv4'):
			 	bw_params,fl_params = 7,6		
			if (layer.name == 'conv5'):
			 	bw_params,fl_params = 7,6		
			if (layer.name == 'conv6'):
			 	bw_params,fl_params = 7,6		
			if (layer.name == 'conv7'):
			 	bw_params,fl_params = 7,6		
			if (layer.name == 'conv8'):
			 	bw_params,fl_params = 7,6			
			max_data = (pow(2, bw_params - 1) - 1) * pow(2, -fl_params); #0.9X
			min_data = -pow(2, bw_params - 1) * pow(2, -fl_params);
			for i in range(weight_data.shape[0]):
				for j in range(weight_data.shape[1]):
					for p in range(weight_data.shape[2]):
						for q in range(weight_data.shape[3]):
							weight_data[i][j][p][q] = max(min(weight_data[i][j][p][q], max_data), min_data);
							weight_data[i][j][p][q] /= pow(2, -fl_params);
							rounding = layer.quantization_param.rounding_scheme;
							if rounding == 0:
								weight_data[i][j][p][q] = round(weight_data[i][j][p][q]);
							else:
								weight_data[i][j][p][q] = floor(weight_data[i][j][p][q] + RandUniform_cpu());
							weight_data[i][j][p][q] *= pow(2, -fl_params)

		if (layer.type == 'BatchNorm'):
			mean_data = net.params[layer.name][0].data
			variance_data = net.params[layer.name][1].data
			scale_factor = net.params[layer.name][2].data[0]
			data = [mean_data, variance_data]
			max_ = -10000000
			min_ = 100000000
			if (layer.name == 'bn_conv1'):
				# bw_bn_params,fl_bn_params = 8,5
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'bn_conv2'):
				# bw_bn_params,fl_bn_params = 8,5
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'bn_conv3'):
				# bw_bn_params,fl_bn_params = 8,5
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'bn_conv4'):
				# bw_bn_params,fl_bn_params = 8,5
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'bn_conv5'):
				# bw_bn_params,fl_bn_params = 8,5
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'bn_conv6'):
				# bw_bn_params,fl_bn_params = 8,5
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'bn_conv7'):
				# bw_bn_params,fl_bn_params = 8,5
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'bn_conv8'):
				# bw_bn_params,fl_bn_params = 8,5
				bw_scale_params,fl_scale_params = 8,5
			#max_data = (pow(2, bw_bn_params - 1) - 1) * pow(2, -fl_bn_params); #0.9X
			#min_data = -pow(2, bw_bn_params - 1) * pow(2, -fl_bn_params);
			for i in range(2):
				for j in range(data[i].shape[0]):
					data[i][j] /= scale_factor
					if i == 1:
						data[i][j] += 1e-05
						data[i][j] = pow(data[i][j], 0.5)
			scale_factor = 1					
					# data[i][j] = max(min(data[i][j], max_data), min_data);
					# data[i][j] /= pow(2, -fl_bn_params);
					# rounding = layer.quantization_param.rounding_scheme;
					# if rounding == 0:
					# 	data[i][j] = round(data[i][j]);
					# else:
					# 	data[i][j] = floor(data[i][j] + RandUniform_cpu());
					# data[i][j] *= pow(2, -fl_bn_params)
			scale_layer = layers[L+1]
			weight_data = net.params[scale_layer.name][0].data
			bias_data = net.params[scale_layer.name][1].data
			max_data = (pow(2, bw_scale_params - 1) - 1) * pow(2, -fl_scale_params); #0.9X
			min_data = -pow(2, bw_scale_params - 1) * pow(2, -fl_scale_params);
			for i in range(weight_data.shape[0]):
				weight_data[i] /= data[1][i] 
				if weight_data[i] > max_:
					max_ = weight_data[i]
				if weight_data[i] < min_:
					min_ = weight_data[i]
				bias_data[i] -= weight_data[i] * data[0][i]
				data[0][i] = 0
				data[1][i] = 1

				weight_data[i] = max(min(weight_data[i], max_data), min_data);
				weight_data[i] /= pow(2, -fl_scale_params);
				rounding = layer.quantization_param.rounding_scheme;
				if rounding == 0:
					weight_data[i] = round(weight_data[i]);
				else:
					weight_data[i] = floor(weight_data[i] + RandUniform_cpu());
				weight_data[i] *= pow(2, -fl_scale_params)
			print max_, min_

		if (layer.type == 'InnerProduct'):
			weight_data = net.params[layer.name][0].data
			bw_params = 6
			fl_params = 5
			max_data = (pow(2, bw_params - 1) - 1) * pow(2, -fl_params); #0.9X
			min_data = -pow(2, bw_params - 1) * pow(2, -fl_params);
			for i in range(weight_data.shape[0]):
				for j in range(weight_data.shape[1]):
					weight_data[i][j] = max(min(weight_data[i][j], max_data), min_data);
					weight_data[i][j] /= pow(2, -fl_params);
					rounding = layer.quantization_param.rounding_scheme;
					if rounding == 0:
						weight_data[i][j] = round(weight_data[i][j]);
					else:
						weight_data[i][j] = floor(weight_data[i][j] + RandUniform_cpu());
					weight_data[i][j] *= pow(2, -fl_params)

	net.save('zhuceng_bn.caffemodel')
