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
	print layers[0]
	for layer in layers:
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
					if data[i][j] > max_:
						max_ = data[i][j]
					if data[i][j] < min_:
						min_ = data[i][j]
					data[i][j] = max(min(data[i][j], max_data), min_data);
					data[i][j] /= pow(2, -fl_params);
					rounding = layer.quantization_param.rounding_scheme;
					if rounding == 0:
						#print 'rounding 0'
						data[i][j] = round(data[i][j]);
					else:
						#print 'rounding 1'
						data[i][j] = floor(data[i][j] + RandUniform_cpu());
					data[i][j] *= pow(2, -fl_params)
				print "max_, min_ ", max_, min_

		elif (layer.name in keys and layer.type != 'LSTM'):
			weight_data = net.params[layer.name][0].data
			#print type(net.params[layer.name][0].data.shape[0])
			shape = len(net.params[layer.name][0].data.shape)
			print '********',layer.name,shape
			if layer.type == 'Convolution':
				bw_params = 8
				fl_params = 7	
			elif layer.type == 'Scale':
				bw_params = 8
				fl_params = 5
			elif layer.type == 'InnerProduct':
				bw_params = 8
				fl_params = 7
			else:
				print 'error'
			max_data = (pow(2, bw_params - 1) - 1) * pow(2, -fl_params); #0.9X
			min_data = -pow(2, bw_params - 1) * pow(2, -fl_params);

			if shape == 4:
				for i in range(weight_data.shape[0]):
					for j in range(weight_data.shape[1]):
						for p in range(weight_data.shape[2]):
							for q in range(weight_data.shape[3]):
								weight_data[i][j][p][q] = max(min(weight_data[i][j][p][q], max_data), min_data);
								weight_data[i][j][p][q] /= pow(2, -fl_params);
								rounding = layer.quantization_param.rounding_scheme;
								if rounding == 0:
									#print 'rounding 0'
									weight_data[i][j][p][q] = round(weight_data[i][j][p][q]);
								else:
									#print 'rounding 1'
									weight_data[i][j][p][q] = floor(weight_data[i][j][p][q] + RandUniform_cpu());
								weight_data[i][j][p][q] *= pow(2, -fl_params)
				# print "after", net.params[layer.name][0].data[weight_data.shape[0] / 2][0][0][weight_data.shape[3] - 1]
			elif shape == 2:
				# print "before", net.params[layer.name][0].data[weight_data.shape[0] / 2][weight_data.shape[1] / 2]
				for i in range(weight_data.shape[0]):
					for j in range(weight_data.shape[1]):
						weight_data[i][j] = max(min(weight_data[i][j], max_data), min_data);
						weight_data[i][j] /= pow(2, -fl_params);
						rounding = layer.quantization_param.rounding_scheme;
						if rounding == 0:
							#print 'rounding 0'
							weight_data[i][j] = round(weight_data[i][j]);
						else:
							#print 'rounding 1'
							weight_data[i][j] = floor(weight_data[i][j] + RandUniform_cpu());
						weight_data[i][j] *= pow(2, -fl_params)
				# print "after", net.params[layer.name][0].data[weight_data.shape[0] / 2][weight_data.shape[1] / 2]
			elif shape == 1:
				for i in range(weight_data.shape[0]):
					weight_data[i] = max(min(weight_data[i], max_data), min_data);
					weight_data[i] /= pow(2, -fl_params);
					rounding = layer.quantization_param.rounding_scheme;
					if rounding == 0:
						#print 'rounding 0'
						weight_data[i] = round(weight_data[i]);
					else:
						#print 'rounding 1'
						weight_data[i] = floor(weight_data[i] + RandUniform_cpu());
					weight_data[i] *= pow(2, -fl_params)
	net.save('zhuceng_bn.caffemodel')