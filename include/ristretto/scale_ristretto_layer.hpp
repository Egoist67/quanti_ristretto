#ifndef CAFFE_SCALE_RISTRETTO_LAYER_HPP_
#define CAFFE_SCALE_RISTRETTO_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/im2col.hpp"

#include "caffe/layers/scale_layer.hpp"
#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

/**
 * @brief Computes a product of two input Blobs, with the shape of the
 *        latter Blob "broadcast" to match the shape of the former.
 *        Equivalent to tiling the latter Blob, then computing the elementwise
 *        product.
 *
 * The second input may be omitted, in which case it's learned as a parameter
 * of the layer.
 */
template <typename Dtype>
class ScaleRistrettoLayer: public ScaleLayer<Dtype>, public BaseRistrettoLayer<Dtype> {
 public:
  explicit ScaleRistrettoLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ScaleRistretto"; }
  // Scale

 protected:
  /**
   * In the below shape specifications, @f$ i @f$ denotes the value of the
   * `axis` field given by `this->layer_param_.scale_param().axis()`, after
   * canonicalization (i.e., conversion from negative to positive index,
   * if applicable).
   *
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the first factor @f$ x @f$
   *   -# @f$ (d_i \times ... \times d_j) @f$
   *      the second factor @f$ y @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the product @f$ z = x y @f$ computed after "broadcasting" y.
   *      Equivalent to tiling @f$ y @f$ to have the same shape as @f$ x @f$,
   *      then computing the elementwise product.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	  
  vector<shared_ptr<Blob<Dtype> > >  scale_quantized_;	  

};


}  // namespace caffe

#endif  // CAFFE_SCALE_RISTRETTO_LAYER_HPP_