#encoding utf-8
import sys
sys.path.append("/home/rhp/cmj_rhp_ssd_regrec/regrec_ssd/python")
from caffe.proto import caffe_pb2
import caffe
if __name__ == "__main__":

	caffemodel_filename = 'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'
	net = caffe.Net('model2text.prototxt', 'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel', caffe.TEST)
	model62 =caffe_pb2.NetParameter()
	f=open(caffemodel_filename, 'r')
	model62.ParseFromString(f.read())
	f.close()
	keys = net.params.keys()
	print 'keys: ', keys
	layers = model62.layer
	print layers[0]
