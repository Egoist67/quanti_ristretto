#encoding utf-8
import sys
sys.path.append("/home/.prog_and_data/shared/codes/ezai/caffe_stand_alone_qmake/warpctcaffe_62/python")
from caffe.proto import caffe_pb2
import caffe
if __name__ == "__main__":

	caffemodel_filename = 'lstm_ctc_newsmall.caffemodel'
	net = caffe.Net('lstm_ctc_newsmall.prototxt', 'lstm_ctc_newsmall.caffemodel', caffe.TEST)
	model62 =caffe_pb2.NetParameter()
	f=open(caffemodel_filename, 'r')
	model62.ParseFromString(f.read())
	f.close()
	keys = net.params.keys()
	print 'keys: ', keys
	layers = model62.layer
	cnt = 0
	zero_before = 0
	zero_after = 0
	for layer in layers:
		cnt1 = 0
		cnt2 = 0
		cnt3 = 0
		cnt4 = 0
		zero_before1 = 0
		zero_before2 = 0
		zero_before3 = 0
		zero_before4 = 0
		zero_after1 = 0
		zero_after2 = 0
		zero_after3 = 0
		zero_after4 = 0
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
			ave = 0
			for i in range(2):
				for j in range(data[i].shape[0]):
					ave += abs(data[i][j])
					cnt1 += 1
					if data[i][j] == 0:
						zero_before += 1
			
			ave = 0.0 * ave / cnt1
			print 'bn ave: ', ave
			for i in range(2):
				# print data[i].shape[0]
				# print "before", data[i][0], data[i][2], data[i][4]
				for j in range(data[i].shape[0]):
					if abs(data[i][j]) < ave:
						data[i][j] = 0
						zero_after1 += 1
					else:
						data[i][j] /= scale_factor
						if i == 1:
							data[i][j] += 1e-05
							data[i][j] = pow(data[i][j], 0.5)
						# print data[i][j],
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
				# print "after", data[i][0], data[i][2], data[i][4]
			# print "after", net.params[layer.name][0].data[weight_data.shape[0] / 2]
			# print "after", net.params[layer.name][0].data[weight_data.shape[0] / 2 + 1]

		elif (layer.name in keys and layer.type != 'LSTM'):
			# print ' layer { '
			# print ' name: "%s" ' % layer.name
			# print ' type: "%s" ' % layer.type
			# print ' rounding_scheme: "%s" ' % layer.quantization_param.rounding_scheme
			# print ' bw_params:"%s" ' % dir(layer.quantization_param)#.bw_params
			# print ' fl_params:"%s" ' % layer.quantization_param.fl_params
			# print ' fl_in:"%s" ' % layer.quantization_param.bw_layer_in
			# print ' fl_out:"%s" ' % layer.quantization_param.bw_layer_out
			# print '}'
			# print 'layer_class:"%s"'%dir(net.params[layer.name][0])#channels height num width
			weight_data = net.params[layer.name][0].data
			#print type(net.params[layer.name][0].data.shape[0])
			shape = len(net.params[layer.name][0].data.shape)
			#weight_cnt = weight_data.size
			#print 'weight_cout: ',weight_cnt
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
				ave = 0 
				for i in range(weight_data.shape[0]):
					for j in range(weight_data.shape[1]):
						for p in range(weight_data.shape[2]):
							for q in range(weight_data.shape[3]):
								ave += abs(weight_data[i][j][p][q])
								cnt2 += 1
								if weight_data[i][j][p][q] == 0:
									zero_before2 += 1
				
				ave = 0.0 * ave / cnt2
				print 'conv ave: ', ave
				# print "before", net.params[layer.name][0].data[weight_data.shape[0] / 2][0][0][weight_data.shape[3] - 1]
				for i in range(weight_data.shape[0]):
					for j in range(weight_data.shape[1]):
						for p in range(weight_data.shape[2]):
							for q in range(weight_data.shape[3]):
								if abs(weight_data[i][j][p][q]) < ave:
									weight_data[i][j][p][q] = 0
									zero_after2 += 1
								else:					
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
				ave = 0
				for i in range(weight_data.shape[0]):
					for j in range(weight_data.shape[1]):
						ave += abs(weight_data[i][j])
						cnt3 += 1
						if weight_data[i][j] == 0:
							zero_before3 += 1
				
				ave = 0.0 * ave / cnt3
				print 'fc ave: ', ave
				# print "before", net.params[layer.name][0].data[weight_data.shape[0] / 2][weight_data.shape[1] / 2]
				for i in range(weight_data.shape[0]):
					for j in range(weight_data.shape[1]):
						if abs(weight_data[i][j]) < ave:
							weight_data[i][j] = 0
							zero_after3 +=1
						else:							
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
				# print "before", net.params[layer.name][0].data[weight_data.shape[0] / 2]
				# print "before", net.params[layer.name][0].data[weight_data.shape[0] / 2 + 1]
				# print "max_data: ", max_data, "; min_data:", min_data
				# print "size: ", net.params[layer.name][2].data[0]
				ave = 0
				for i in range(weight_data.shape[0]):
					ave += abs(weight_data[i])
					cnt4 += 1
					if weight_data[i] == 0:
						zero_before4 += 1
				
				ave = 0.0 * ave / cnt4
				print 'scale ave: ', ave
				for i in range(weight_data.shape[0]):
					if abs(weight_data[i]) < ave:
						weight_data[i] = 0
						zero_after4 +=1
					else:						
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
		cnt = cnt + cnt1 + cnt2 + cnt3 + cnt4
		zero_before = zero_before + zero_before1 + zero_before2 + zero_before3 + zero_before4
		zero_after = zero_after + zero_after1 + zero_after2 + zero_after3 + zero_after4
				# print "after", net.params[layer.name][0].data[weight_data.shape[0] / 2]
				# print "after", net.params[layer.name][0].data[weight_data.shape[0] / 2 + 1]
	print 'cnt: ', cnt
	print 'zero_before: ', zero_before
	print 'zero_after: ', zero_after
	net.save('test.caffemodel')


			# bw_params = 8
			# if layer.type == 'Scale':
			# 	fl_params = 5
			# else:
			# 	fl_params = 7
			# for index in range(weight_cnt):
			# 	max_data = (pow(2, bw_params - 1) - 1) * pow(2, -fl_params); 
			# 	min_data = -pow(2, bw_params - 1) * pow(2, -fl_params);
			# 	print 'weight_data[index]: "%s"'%weight_data[index]
			# 	weight_data[index] = max(min(weight_data[index], max_data), min_data);

			# 	# Round data
			# 	weight_data[index] /= pow(2, -fl_params);
			# 	rounding = layer.quantization_param.rounding_scheme
			# 	if QuantizationParameter_Rounding_NEAREST == rounding:
			# 		weight_data[index] = round(weight_data[index]);
			# 	else:
			# 		weight_data[index] = floor(weight_data[index] + RandUniform_cpu());
			# 	weight_data[index] *= pow(2, -fl_params)
	#net.save('new62.caffemodel')
