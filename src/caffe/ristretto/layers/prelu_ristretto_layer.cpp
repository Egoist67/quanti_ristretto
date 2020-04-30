#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"

#include "caffe/util/math_functions.hpp"
#include "ristretto/prelu_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
PReLURistrettoLayer<Dtype>::PReLURistrettoLayer(
      const LayerParameter& param) : PReLULayer<Dtype>(param),
      BaseRistrettoLayer<Dtype>() {
  this->precision_ = this->layer_param_.quantization_param().precision();
  this->rounding_ = this->layer_param_.quantization_param().rounding_scheme();
  switch (this->precision_) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
    this->bw_params_ = this->layer_param_.quantization_param().bw_params();
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
    this->fl_params_ = this->layer_param_.quantization_param().fl_params();
    break;
  case QuantizationParameter_Precision_MINIFLOAT:
    this->fp_mant_ = this->layer_param_.quantization_param().mant_bits();
    this->fp_exp_ = this->layer_param_.quantization_param().exp_bits();
    break;
  case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
    this->pow_2_min_exp_ = this->layer_param_.quantization_param().exp_min();
    this->pow_2_max_exp_ = this->layer_param_.quantization_param().exp_max();
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
    break;
  default:
    LOG(FATAL) << "Unknown precision mode: " << this->precision_;
    break;
  }
}

template <typename Dtype>
void PReLURistrettoLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // LOG(INFO) << "chensonglu";
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  PReLUParameter prelu_param = this->layer_param().prelu_param();
  int channels = bottom[0]->channels();
  channel_shared_ = prelu_param.channel_shared();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // LOG(INFO) << "chensonglu1";
    this->blobs_.resize(1);
    // LOG(INFO) << "chensonglu2";
    this->slope_quantized_.resize(1);
    // LOG(INFO) << "chensonglu3";
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
      this->slope_quantized_[0].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
      this->slope_quantized_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    // LOG(INFO) << "chensonglu4";
    shared_ptr<Filler<Dtype> > filler;
    if (prelu_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(prelu_param.filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(0.25);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    // LOG(INFO) << "chensonglu5";
    filler->Fill(this->blobs_[0].get());
    filler->Fill(this->slope_quantized_[0].get());
    // LOG(INFO) << "chensonglu6";
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Negative slope size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Negative slope size is inconsistent with prototxt config";
  }
  // LOG(INFO) << "chensonglu7";
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
  // LOG(INFO) << "chensonglu8";
}

template <typename Dtype>
void PReLURistrettoLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
  // LOG(INFO) << "chensonglu9";
}

template <typename Dtype>
void PReLURistrettoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // LOG(INFO) << "chensonglu10";
    ///////Quantize Input////////
  if (this->phase_ == TEST) {
    for (int i = 0; i < bottom.size(); ++i) {
      this->QuantizeLayerInputs_cpu(bottom[i]->mutable_cpu_data(),
          bottom[i]->count());
    }
  }
  // LOG(INFO) << "chensonglu11";
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  // const Dtype* slope_data = this->blobs_[0]->cpu_data();

  
  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }



  caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),this->slope_quantized_[0]->mutable_cpu_data());
  // LOG(INFO) << this->slope_quantized_[0]->mutable_cpu_data()[0];
  int rounding = this->phase_ == TEST ? this->rounding_ :QuantizationParameter_Rounding_STOCHASTIC;
  this->QuantizeWeights_cpu(this->slope_quantized_, rounding,false);
  const Dtype* slope_data = this->slope_quantized_[0].get()->cpu_data();



  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + slope_data[c] * std::min(bottom_data[i], Dtype(0));
  }

  ////////Quantize Output/////////
  if (this->phase_ == TEST) {
    this->QuantizeLayerOutputs_cpu(top[0]->mutable_cpu_data(),top[0]->count());
  }

}

template <typename Dtype>
void PReLURistrettoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* slope_data = this->blobs_[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;

  // Propagte to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* slope_diff = this->blobs_[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      slope_diff[c] += top_diff[i] * bottom_data[i] * (bottom_data[i] <= 0);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + slope_data[c] * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(PReLURistrettoLayer);
#endif

INSTANTIATE_CLASS(PReLURistrettoLayer);
REGISTER_LAYER_CLASS(PReLURistretto);

}  // namespace caffe
