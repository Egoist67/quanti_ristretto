#!/usr/bin/env sh

build/tools/ristretto quantize \
	--model=../carplate_recognition/models/deploy.prototxt \
	--model_quantized=../carplate_recognition/models/auto_deploy.prototxt \
	--qt_script_name="python2 ../carplate_recognition/quantize_test.py" \
	--PerAccuracy=../carplate_recognition/quantize_results/out.txt \
	--AllAccuracy=../carplate_recognition/quantize_results/all_results.txt \
	--trimming_mode=dynamic_fixed_point \
	--gpu=2
