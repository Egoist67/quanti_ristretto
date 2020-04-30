#include <boost/thread.hpp>//added
#include "caffe/layer.hpp"

namespace caffe {

//added
 template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}
//added

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
