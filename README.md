# quanti_ristretto

1. fuse bn and scale
	python /weight_caffemodel/bnscale011_script.py

2. quantize the model
	sh /examples/ristretto/00_quantize_carplate_auto.sh

3. quantize src
	/src/caffe/ristretto/quantization.cpp