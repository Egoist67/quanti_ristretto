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
	layers = model62.layer
	for layer in layers:
		if (layer.type == 'BatchNorm'):
			net.params[layer.name][0].data = 0
			net.params[layer.name][1].data = 1
			net.params[layer.name][2].data[0] = 1
			mean_data = net.params[layer.name][0].data
			variance_data = net.params[layer.name][1].data
			scale_factor = net.params[layer.name][2].data[0]
			data = [mean_data, variance_data]
			

		if (layer.type = 'Scale'):
			net.params[layer.name][0].data = 1
			weight_data = net.params[layer.name][0].data
			#print type(net.params[layer.name][0].data.shape[0])
			
			
	net.save('bnscale_011.caffemodel')