#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
// SyncedMemory::SyncedMemory()
//   : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
//     own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
// #ifndef CPU_ONLY
// #ifdef DEBUG
//   CUDA_CHECK(cudaGetDevice(&device_));
// #endif
// #endif
// }

// SyncedMemory::SyncedMemory(size_t size)
//   : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
//     own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
// #ifndef CPU_ONLY
// #ifdef DEBUG
//   CUDA_CHECK(cudaGetDevice(&device_));
// #endif
// #endif
// }//src

SyncedMemory::~SyncedMemory() {
  // check_device();//src
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    //CUDA_CHECK(cudaFree(gpu_ptr_));//src
    int initial_device;//added
    cudaGetDevice(&initial_device);//added
    if (gpu_device_ != -1) {//added
      CUDA_CHECK(cudaSetDevice(gpu_device_));//added
    }//added
    CUDA_CHECK(cudaFree(gpu_ptr_));//added
    cudaSetDevice(initial_device);//added
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  // check_device();//src
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  // check_device();//src
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaGetDevice(&gpu_device_));//added
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));//added
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  // check_device();//src
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  // check_device();//src
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
  // check_device();//src
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
  // check_device();//src
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    int initial_device;//added
    cudaGetDevice(&initial_device);//added
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }//added
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);//added
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  // check_device();//src
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  // check_device();//src
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  // check_device();//src
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaGetDevice(&gpu_device_));//added
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK(device == device_);
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    CHECK(attributes.device == device_);
  }
#endif
#endif
}

}  // namespace caffe

