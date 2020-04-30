#encoding utf-8
import sys
sys.path.append("/home/.prog_and_data/shared/codes/ezai/caffe_stand_alone_qmake/warpctcaffe_62/python")
from caffe.proto import caffe_pb2
import caffe
if __name__ == "__main__":

	caffemodel_filename = 'lstm_ctc_new12_12_3fc_all_mix_iter_60000_995.caffemodel'
	net = caffe.Net('lstm_ctc_newsmall12_12_3fc_all_train.prototxt', 'lstm_ctc_new12_12_3fc_all_mix_iter_60000_995.caffemodel', caffe.TEST)
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
			max_ = -1000000
			min_ = 10000000
			for i in range(weight_data.shape[0]):
				for j in range(weight_data.shape[1]):
					for p in range(weight_data.shape[2]):
						for q in range(weight_data.shape[3]):
							if weight_data[i][j][p][q] > max_:
								max_ = weight_data[i][j][p][q]
							if weight_data[i][j][p][q] < min_:
								min_ = weight_data[i][j][p][q]
			print 'layer.name: ',layer.name
			print 'max_: ',max_,'    min_: ',min_
		if (layer.type == 'BatchNorm'):
			mean_data = net.params[layer.name][0].data
			variance_data = net.params[layer.name][1].data
			scale_factor = net.params[layer.name][2].data[0]
			data = [mean_data, variance_data]
			max_ = -1000000
			min_ = 10000000
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
			print 'layer.name: ',layer.name
			print 'max_: ',max_,'    min_: ',min_
		if (layer.type == 'Scale'):
			weight_data = net.params[layer.name][0].data
			max_ = -1000000
			min_ = 10000000
			for i in range(weight_data.shape[0]):
				if weight_data[i] > max_:
					max_ = weight_data[i]
				if weight_data[i] < min_:
					min_ = weight_data[i]
			print 'layer.name: ',layer.name
			print 'max_: ',max_,'    min_: ',min_
		if (layer.type == 'InnerProduct'):
			weight_data = net.params[layer.name][0].data
			max_ = -1000000
			min_ = 10000000
			for i in range(weight_data.shape[0]):
				for j in range(weight_data.shape[1]):
					if weight_data[i][j] > max_:
						max_ = weight_data[i][j]
					if weight_data[i][j] < min_:
						min_ = weight_data[i][j]
			print 'layer.name: ',layer.name
			print 'max_: ',max_,'    min_: ',min_	