#encoding utf-8
import sys
import math
sys.path.append("/home/.prog_and_data/shared/codes/ezai/caffe_stand_alone_qmake/warpctcaffe_62/python")
from caffe.proto import caffe_pb2
import caffe
import numpy as np
if __name__ == "__main__":

	caffemodel_filename = 'bnscale_011.caffemodel'
	net = caffe.Net('lstm_ctc_newsmall_bnscale011.prototxt', 'bnscale_011.caffemodel', caffe.TEST)
	model62 =caffe_pb2.NetParameter()
	f=open(caffemodel_filename, 'r')
	model62.ParseFromString(f.read())
	f.close()
	keys = net.params.keys()
	print 'keys: ', keys
	layers = model62.layer
	print layers[0]
	for L in range(len(layers)):
		layer = layers[L]
		if (layer.type == 'Convolution'):
			weight_data = net.params[layer.name][0].data
			bias_data = net.params[layer.name][1].data
			if (layer.name == 'conv1'):
				bw_conv_params,fl_conv_params = 6,5
			if (layer.name == 'conv2'):
			 	bw_conv_params,fl_conv_params = 7,6	
			if (layer.name == 'conv3'):
			 	bw_conv_params,fl_conv_params = 7,6		
			if (layer.name == 'conv4'):
			 	bw_conv_params,fl_conv_params = 7,6		
			if (layer.name == 'conv5'):
			 	bw_conv_params,fl_conv_params = 7,6		
			if (layer.name == 'conv6'):
			 	bw_conv_params,fl_conv_params = 7,6		
			if (layer.name == 'conv7'):
			 	bw_conv_params,fl_conv_params = 7,6		
			if (layer.name == 'conv8'):
			 	bw_conv_params,fl_conv_params = 7,6			
			max_data = (pow(2, bw_conv_params - 1) - 1) * pow(2, -fl_conv_params); #0.9X
			min_data = -pow(2, bw_conv_params - 1) * pow(2, -fl_conv_params);
			sum_conv_weights_before,sum_conv_weights_after = 0,0
			sum_conv_bias_before,sum_conv_bias_after = 0,0
			for i in range(weight_data.shape[0]):
				for j in range(weight_data.shape[1]):
					for p in range(weight_data.shape[2]):
						for q in range(weight_data.shape[3]):
							#print "conv_params_before: ",weight_data[i][j][p][q]
							sum_conv_weights_before += weight_data[i][j][p][q]
							weight_data[i][j][p][q] = max(min(weight_data[i][j][p][q], max_data), min_data);
							weight_data[i][j][p][q] /= pow(2, -fl_conv_params);
							rounding = layer.quantization_param.rounding_scheme;
							if rounding == 0:
								weight_data[i][j][p][q] = np.rint(weight_data[i][j][p][q]);
							else:
								weight_data[i][j][p][q] = floor(weight_data[i][j][p][q] + RandUniform_cpu());
							weight_data[i][j][p][q] *= pow(2, -fl_conv_params)
							#print "conv_params_after: ",weight_data[i][j][p][q]
							sum_conv_weights_after += weight_data[i][j][p][q]
			for i in range(bias_data.shape[0]):
				#print "conv_bias_before: ",bias_data[i]
				sum_conv_bias_before += bias_data[i]
				bias_data[i] = max(min(bias_data[i], max_data), min_data);
				bias_data[i] /= pow(2, -fl_conv_params);
				rounding = layer.quantization_param.rounding_scheme;
				if rounding == 0:
					bias_data[i] = np.rint(bias_data[i]);
				else:
					bias_data[i] = floor(bias_data[i] + RandUniform_cpu());
				bias_data[i] *= pow(2, -fl_conv_params)
				#print "conv_bias_after: ",bias_data[i]
				sum_conv_bias_after += bias_data[i]
			print "sum_conv_weights: ",sum_conv_weights_before," ",sum_conv_weights_after
			print "sum_conv_bias: ",sum_conv_bias_before," ",sum_conv_bias_after
		if (layer.type == 'Scale'):
			weight_data = net.params[layer.name][0].data
			bias_data = net.params[layer.name][1].data
			#print "scale_bias_data.size: ",bias_data.shape[0]
			if (layer.name == 'scale_conv1'):
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'scale_conv2'):
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'scale_conv3'):
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'scale_conv4'):
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'scale_conv5'):
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'scale_conv6'):
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'scale_conv7'):
				bw_scale_params,fl_scale_params = 8,5
			if (layer.name == 'scale_conv8'):
				bw_scale_params,fl_scale_params = 8,5
			max_data = (pow(2, bw_scale_params - 1) - 1) * pow(2, -fl_scale_params); #0.9X
			min_data = -pow(2, bw_scale_params - 1) * pow(2, -fl_scale_params);
			sum_scale_weights_before,sum_scale_weights_after = 0,0
			sum_scale_bias_before,sum_scale_bias_after = 0,0
			for i in range(weight_data.shape[0]):
				#print "scale_params_before: ",weight_data[i]
				sum_scale_weights_before += weight_data[i]
				weight_data[i] = max(min(weight_data[i], max_data), min_data);
				weight_data[i] /= pow(2, -fl_scale_params);
				rounding = layer.quantization_param.rounding_scheme;
				if rounding == 0:
					weight_data[i] = np.rint(weight_data[i])
				else:
					weight_data[i] = floor(weight_data[i] + RandUniform_cpu());
				weight_data[i] *= pow(2, -fl_scale_params)
				#print "scale_params_after: ",weight_data[i]
				sum_scale_weights_after += weight_data[i]
			for i in range(bias_data.shape[0]):
				#print "scale_bias_before: ",bias_data[i]
				sum_scale_bias_before += bias_data[i]
				bias_data[i] = max(min(bias_data[i], max_data), min_data);
				bias_data[i] /= pow(2, -fl_scale_params);
				rounding = layer.quantization_param.rounding_scheme;
				if rounding == 0:
					bias_data[i] = np.rint(bias_data[i]);
				else:
					bias_data[i] = floor(bias_data[i] + RandUniform_cpu());
				bias_data[i] *= pow(2, -fl_scale_params)
				#print "scale_bias_after: ",bias_data[i]
				sum_scale_bias_after += bias_data[i]
			print "sum_scale_weights: ",sum_scale_weights_before," ",sum_scale_weights_after
			print "sum_scale_bias: ",sum_scale_bias_before," ",sum_scale_bias_after 

		if (layer.type == 'InnerProduct'):
			weight_data = net.params[layer.name][0].data
			bias_data = net.params[layer.name][1].data
			#print "fc_bias_data.size: ",bias_data.shape[0]
			if (layer.name == 'fc1'):
				bw_fc_params,fl_fc_params = 8,7
			max_data = (pow(2, bw_fc_params - 1) - 1) * pow(2, -fl_fc_params); #0.9X
			min_data = -pow(2, bw_fc_params - 1) * pow(2, -fl_fc_params);	
			sum_fc_weights_before,sum_fc_weights_after = 0,0
			sum_fc_bias_before,sum_fc_bias_after = 0,0	
			for i in range(weight_data.shape[0]):
				for j in range(weight_data.shape[1]):
					#print "fc_params_before: ",weight_data[i][j]
					sum_fc_weights_before += weight_data[i][j]
					weight_data[i][j] = max(min(weight_data[i][j], max_data), min_data);
					weight_data[i][j] /= pow(2, -fl_fc_params);
					rounding = layer.quantization_param.rounding_scheme;
					if rounding == 0:
						weight_data[i][j] = np.rint(weight_data[i][j]);
					else:
						weight_data[i][j] = floor(weight_data[i][j] + RandUniform_cpu());
					weight_data[i][j] *= pow(2, -fl_fc_params)
					#print "fc_params_after: ",weight_data[i][j]
					sum_fc_weights_after += weight_data[i][j]
			for i in range(bias_data.shape[0]):
				#print "fc_bias_before: ",bias_data[i]
				sum_fc_bias_before += bias_data[i]
				bias_data[i] = max(min(bias_data[i], max_data), min_data);
				bias_data[i] /= pow(2, -fl_fc_params);
				rounding = layer.quantization_param.rounding_scheme;
				if rounding == 0:
					bias_data[i] = np.rint(bias_data[i]);
				else:
					bias_data[i] = floor(bias_data[i] + RandUniform_cpu());
				bias_data[i] *= pow(2, -fl_fc_params)
				#print "fc_bias_after: ",bias_data[i]
				sum_fc_bias_after += bias_data[i]
			print "sum_fc_weights: ",sum_fc_weights_before," ",sum_fc_weights_after
			print "sum_fc_bias: ",sum_fc_bias_before," ",sum_fc_bias_after 
	net.save('fixedpoint_bnscale011.caffemodel')
