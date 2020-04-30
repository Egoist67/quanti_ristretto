#include <algorithm>
#include <vector>
#include "ristretto/batch_norm_ristretto_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchNormRistrettoLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (this->phase_ == TEST) { /////////Quantize/////////
    for (int i = 0; i < bottom.size(); ++i) {
      //std::cout << "bn_in_quant befor: " << bottom[0]->cpu_data()[1] << " " << bottom[0]->cpu_data()[2] << std::endl;
      this->QuantizeLayerInputs_gpu(bottom[i]->mutable_cpu_data(),
          bottom[i]->count());
      //std::cout << "bn_in_quant after: " << bottom[0]->cpu_data()[1] << " " << bottom[0]->cpu_data()[2] << std::endl;
    }
  }
	
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(this->channels_*bottom[0]->shape(0));

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }


  if (this->use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[2]->cpu_data()[0];
    caffe_gpu_scale(this->variance_.count(), scale_factor,
        this->blobs_[0]->gpu_data(), this->mean_.mutable_gpu_data());
    caffe_gpu_scale(this->variance_.count(), scale_factor,
        this->blobs_[1]->gpu_data(), this->variance_.mutable_gpu_data());

    caffe_copy(this->mean_.count(), this->mean_.cpu_data(), this->weights_quantized_mean_[0]->mutable_cpu_data());
    caffe_copy(this->variance_.count(), this->variance_.cpu_data(), this->weights_quantized_variance_[0]->mutable_cpu_data());
    int rounding = this->phase_ == TEST ? this->rounding_ :
                  QuantizationParameter_Rounding_STOCHASTIC;
    this->QuantizeWeights_gpu(this->weights_quantized_mean_, rounding, false);
    //this->QuantizeWeights_gpu(this->weights_quantized_variance_, rounding, false);
  } else {
    // compute mean
    caffe_gpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim,
        1. / (num * spatial_dim), bottom_data,
        this->spatial_sum_multiplier_.gpu_data(), 0.,
        this->num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1.,
        this->num_by_chans_.gpu_data(), this->batch_sum_multiplier_.gpu_data(), 0.,
        this->mean_.mutable_gpu_data());
  }

  if (this->use_global_stats_) {
      // subtract mean
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1,
          this->batch_sum_multiplier_.gpu_data(), this->weights_quantized_mean_[0]->gpu_data(), 0.,
          this->num_by_chans_.mutable_gpu_data());
  } else {
      // subtract mean
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1,
          this->batch_sum_multiplier_.gpu_data(), this->mean_.gpu_data(), 0.,
          this->num_by_chans_.mutable_gpu_data());
  }
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num,
      spatial_dim, 1, -1, this->num_by_chans_.gpu_data(),
      this->spatial_sum_multiplier_.gpu_data(), 1., top_data);

  if (!this->use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_gpu_powx(top[0]->count(), top_data, Dtype(2),
        this->temp_.mutable_gpu_data());  // (X-EX)^2
    caffe_gpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim,
        1. / (num * spatial_dim), this->temp_.gpu_data(),
        this->spatial_sum_multiplier_.gpu_data(), 0.,
        this->num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1.,
        this->num_by_chans_.gpu_data(), this->batch_sum_multiplier_.gpu_data(), 0.,
        this->variance_.mutable_gpu_data());  // E((X_EX)^2)

    // compute and save moving average
    this->blobs_[2]->mutable_cpu_data()[0] *= this->moving_average_fraction_;
    this->blobs_[2]->mutable_cpu_data()[0] += 1;
    caffe_gpu_axpby(this->mean_.count(), Dtype(1), this->mean_.gpu_data(),
        this->moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
    int m = bottom[0]->count()/this->channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    caffe_gpu_axpby(this->variance_.count(), bias_correction_factor,
        this->variance_.gpu_data(), this->moving_average_fraction_,
        this->blobs_[1]->mutable_gpu_data());
  }

  // normalize variance
  caffe_gpu_add_scalar(this->variance_.count(), this->eps_, this->variance_.mutable_gpu_data());
  caffe_gpu_powx(this->variance_.count(), this->variance_.gpu_data(), Dtype(0.5),
      this->variance_.mutable_gpu_data());


  if (this->phase_ == TEST) { /////////Quantize/////////
      // normalize variance
      caffe_gpu_add_scalar(this->variance_.count(), this->eps_, this->weights_quantized_variance_[0]->mutable_gpu_data());
      caffe_gpu_powx(this->variance_.count(), this->weights_quantized_variance_[0]->gpu_data(), Dtype(0.5),
          this->weights_quantized_variance_[0]->mutable_gpu_data());
    int rounding = this->phase_ == TEST ? this->rounding_ : QuantizationParameter_Rounding_STOCHASTIC;
    this->QuantizeWeights_gpu(this->weights_quantized_variance_, rounding, false);
    // replicate variance to input size
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1,
        this->batch_sum_multiplier_.gpu_data(), this->weights_quantized_variance_[0]->gpu_data(), 0.,
        this->num_by_chans_.mutable_gpu_data());
  }else{
      // replicate variance to input size
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1,
          this->batch_sum_multiplier_.gpu_data(), this->variance_.gpu_data(), 0.,
          this->num_by_chans_.mutable_gpu_data());
  }
  
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num,
      spatial_dim, 1, 1., this->num_by_chans_.gpu_data(),
      this->spatial_sum_multiplier_.gpu_data(), 0., this->temp_.mutable_gpu_data());
  caffe_gpu_div(this->temp_.count(), top_data, this->temp_.gpu_data(), top_data);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(this->x_norm_.count(), top_data,
      this->x_norm_.mutable_gpu_data());
	  
  if (this->phase_ == TEST) { /////////Quantize/////////
    //std::cout << "bn_out_quant befor: " << top[0]->cpu_data()[1] << " " << top[0]->cpu_data()[2] << std::endl;
    this->QuantizeLayerOutputs_gpu(top[0]->mutable_gpu_data(),top[0]->count());///////Quantize///////
    //std::cout << "bn_out_quant after: " << top[0]->cpu_data()[1] << " " << top[0]->cpu_data()[2] << std::endl;
  }
}

template <typename Dtype>
void BatchNormRistrettoLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->gpu_diff();
  } else {
    caffe_copy(this->x_norm_.count(), top[0]->gpu_diff(), this->x_norm_.mutable_gpu_diff());
    top_diff = this->x_norm_.gpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  if (this->use_global_stats_) {
    caffe_gpu_div(this->temp_.count(), top_diff, this->temp_.gpu_data(), bottom_diff);
    return;
  }
  const Dtype* top_data = this->x_norm_.gpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count()/(this->channels_*bottom[0]->shape(0));
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_gpu_mul(this->temp_.count(), top_data, top_diff, bottom_diff);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim, 1.,
      bottom_diff, this->spatial_sum_multiplier_.gpu_data(), 0.,
      this->num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1.,
      this->num_by_chans_.gpu_data(), this->batch_sum_multiplier_.gpu_data(), 0.,
      this->mean_.mutable_gpu_data());

  // reshape (broadcast) the above
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1,
      this->batch_sum_multiplier_.gpu_data(), this->mean_.gpu_data(), 0.,
      this->num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->channels_ * num,
      spatial_dim, 1, 1., this->num_by_chans_.gpu_data(),
      this->spatial_sum_multiplier_.gpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_mul(this->temp_.count(), top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemv<Dtype>(CblasNoTrans, this->channels_ * num, spatial_dim, 1.,
      top_diff, this->spatial_sum_multiplier_.gpu_data(), 0.,
      this->num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, this->channels_, 1.,
      this->num_by_chans_.gpu_data(), this->batch_sum_multiplier_.gpu_data(), 0.,
      this->mean_.mutable_gpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, this->channels_, 1, 1,
      this->batch_sum_multiplier_.gpu_data(), this->mean_.gpu_data(), 0.,
      this->num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * this->channels_,
      spatial_dim, 1, 1., this->num_by_chans_.gpu_data(),
      this->spatial_sum_multiplier_.gpu_data(), 1., bottom_diff);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_gpu_axpby(this->temp_.count(), Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff);

  // note: this->temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_gpu_div(this->temp_.count(), bottom_diff, this->temp_.gpu_data(), bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormRistrettoLayer);


}  // namespace caffe
