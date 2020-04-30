#include <cfloat>
#include <vector>
#include <algorithm>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"

#include "ristretto/scale_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ScaleForward(const int n, const Dtype* in,
    const Dtype* scale, const int scale_dim, const int inner_dim,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index];
  }
}

template <typename Dtype>
__global__ void ScaleBiasForward(const int n, const Dtype* in,
    const Dtype* scale, const Dtype* bias,
    const int scale_dim, const int inner_dim, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index] + bias[scale_index];
  }
}

template <typename Dtype>
void ScaleRistrettoLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ///////Quantize Input////////
  if (this->phase_ == TEST) {
    for (int i = 0; i < bottom.size(); ++i) {
      this->QuantizeLayerInputs_gpu(bottom[i]->mutable_gpu_data(),
          bottom[i]->count());
    }
  }
	
  const int count = top[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  if (bottom[0] == top[0]) {
    // in-place computation; need to store bottom data before overwriting it.
    // Note that this is only necessary for Backward; we could skip this if not
    // doing Backward, but Caffe currently provides no way of knowing whether
    // we'll need to do Backward at the time of the Forward call.
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(),
               this->temp_.mutable_gpu_data());
  }
  
  /*
  const Dtype* scale_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  */

  Dtype* top_data = top[0]->mutable_gpu_data();
  
  ///////Quantize////////
  // Trim weights
  if(bottom.size() <= 1){
	  caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),this->scale_quantized_[0]->mutable_cpu_data());
	  if (this->scale_quantized_.size()>1) {
               //std::cout<<"this->scale_quantized_.size()>1: "<<this->scale_quantized_.size()<<std::endl;
		caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(),this->scale_quantized_[1]->mutable_cpu_data());
	  }
	  int rounding = this->phase_ == TEST ? this->rounding_ :QuantizationParameter_Rounding_STOCHASTIC;
         //std::cout<<"scale_bias_before: "<<this->scale_quantized_[1]->mutable_cpu_data()[1]<<" "<<this->scale_quantized_[1]->mutable_cpu_data()[2]<<std::endl;
	  this->QuantizeWeights_gpu(this->scale_quantized_, rounding,true);//this->bias_layer_);
         //std::cout<<"scale_bias_after: "<<this->scale_quantized_[1]->mutable_cpu_data()[1]<<" "<<this->scale_quantized_[1]->mutable_cpu_data()[2]<<std::endl;
  }
  
  // Trim weights
  /*if(bottom.size() <= 1){
	  caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
		  this->scale_quantized_[0]->mutable_cpu_data());
	  int rounding = this->phase_ == TEST ? this->rounding_ :
		  QuantizationParameter_Rounding_STOCHASTIC;
	  this->QuantizeWeights_gpu(this->scale_quantized_, rounding,
		  bottom.size() <= 1);
  }
  */

  ///////Quantize////////
  const Dtype* scale_quantized_data_ =
      ((bottom.size() > 1) ? bottom[1] : this->scale_quantized_[0].get())->cpu_data();
	  
  if (this->bias_layer_) {
    const Dtype* bias_data = this->blobs_[this->bias_param_id_]->gpu_data();
    ScaleBiasForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, scale_quantized_data_, bias_data, this->scale_dim_, this->inner_dim_,
        top_data);
  } else {
    ScaleForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, scale_quantized_data_, this->scale_dim_, this->inner_dim_, top_data);
  }
  
  ////////Quantize/////////
  if (this->phase_ == TEST) {
    this->QuantizeLayerOutputs_gpu(top[0]->mutable_gpu_data(),top[0]->count());///////Quantize///////
  }
}

template <typename Dtype>
void ScaleRistrettoLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->bias_layer_ &&
      this->param_propagate_down_[this->param_propagate_down_.size() - 1]) {
    this->bias_layer_->Backward(top, this->bias_propagate_down_, this->bias_bottom_vec_);
  }
  const bool scale_param = (bottom.size() == 1);
  Blob<Dtype>* scale = scale_param ? this->scale_quantized_[0].get() : bottom[1];  ///////Quantize///////
  if ((!scale_param && propagate_down[1]) ||
      (scale_param && this->param_propagate_down_[0])) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const bool in_place = (bottom[0] == top[0]);
    const Dtype* bottom_data = (in_place ? &(this->temp_) : bottom[0])->gpu_data();
    // Hack: store big eltwise product in bottom[0] diff, except in the special
    // case where this layer itself does the eltwise product, in which case we
    // can store it directly in the scale diff, and we're done.
    // If we're computing in-place (and not doing eltwise computation), this
    // hack doesn't work and we store the product in this->temp_.
    const bool is_eltwise = (bottom[0]->count() == scale->count());
    Dtype* product = (is_eltwise ? scale->mutable_gpu_diff() :
        (in_place ? this->temp_.mutable_gpu_data() : bottom[0]->mutable_gpu_diff()));
    caffe_gpu_mul(top[0]->count(), top_diff, bottom_data, product);
    if (!is_eltwise) {
      Dtype* sum_result = NULL;
      if (this->inner_dim_ == 1) {
        sum_result = product;
      } else if (this->sum_result_.count() == 1) {
        const Dtype* sum_mult = this->sum_multiplier_.gpu_data();
        Dtype* scale_diff = scale->mutable_cpu_diff();
        if (scale_param) {
          Dtype result;
          caffe_gpu_dot(this->inner_dim_, product, sum_mult, &result);
          *scale_diff += result;
        } else {
          caffe_gpu_dot(this->inner_dim_, product, sum_mult, scale_diff);
        }
      } else {
        const Dtype* sum_mult = this->sum_multiplier_.gpu_data();
        sum_result = (this->outer_dim_ == 1) ?
            scale->mutable_gpu_diff() : this->sum_result_.mutable_gpu_data();
        caffe_gpu_gemv(CblasNoTrans, this->sum_result_.count(), this->inner_dim_,
                       Dtype(1), product, sum_mult, Dtype(0), sum_result);
      }
      if (this->outer_dim_ != 1) {
        const Dtype* sum_mult = this->sum_multiplier_.gpu_data();
        if (this->scale_dim_ == 1) {
          Dtype* scale_diff = scale->mutable_cpu_diff();
          if (scale_param) {
            Dtype result;
            caffe_gpu_dot(this->outer_dim_, sum_mult, sum_result, &result);
            *scale_diff += result;
          } else {
            caffe_gpu_dot(this->outer_dim_, sum_mult, sum_result, scale_diff);
          }
        } else {
          Dtype* scale_diff = scale->mutable_gpu_diff();
          caffe_gpu_gemv(CblasTrans, this->outer_dim_, this->scale_dim_,
                         Dtype(1), sum_result, sum_mult, Dtype(scale_param),
                         scale_diff);
        }
      }
    }
  }
  if (propagate_down[0]) {
    const int count = top[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* scale_data = scale->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    ScaleForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, scale_data, this->scale_dim_, this->inner_dim_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScaleRistrettoLayer);

}  // namespace caffe
