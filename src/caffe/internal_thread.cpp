#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

InternalThread::~InternalThread() {
  StopInternalThread();
}

bool InternalThread::is_started() const {
  return thread_ && thread_->joinable();
}

bool InternalThread::must_stop() {
  return thread_ && thread_->interruption_requested();
}

void InternalThread::StartInternalThread() {
  CHECK(!is_started()) << "Threads should persist and not be restarted.";

  int device = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device));
#endif
  Caffe::Brew mode = Caffe::mode();
  int rand_seed = caffe_rng_rand();
  int solver_count = Caffe::solver_count();
  int solver_rank = Caffe::solver_rank();
  bool multiprocess = Caffe::multiprocess();
  bool root_solver = Caffe::root_solver();//added

  try {
    // thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
    //       rand_seed, solver_count, solver_rank, multiprocess));//src
    thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
          rand_seed, solver_count, solver_rank, multiprocess, root_solver));//added root_solver
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}


// void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
//     int solver_count, int solver_rank, bool multiprocess) {
// #ifndef CPU_ONLY
//   CUDA_CHECK(cudaSetDevice(device));
// #endif
//   Caffe::set_mode(mode);
//   Caffe::set_random_seed(rand_seed);
//   Caffe::set_solver_count(solver_count);
//   Caffe::set_solver_rank(solver_rank);
//   Caffe::set_multiprocess(multiprocess);

//   InternalThreadEntry();
// }//src

//added
void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
    int solver_count, int solver_rank, bool multiprocess, bool root_solver) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device));
#endif
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);
  Caffe::set_solver_rank(solver_rank);
  Caffe::set_multiprocess(multiprocess);
  Caffe::set_root_solver(root_solver);

  InternalThreadEntry();
}
//added root_solver

void InternalThread::StopInternalThread() {
  if (is_started()) {
    thread_->interrupt();
    try {
      thread_->join();
    } catch (boost::thread_interrupted&) {
    } catch (std::exception& e) {
      LOG(FATAL) << "Thread exception: " << e.what();
    }
  }
}

}  // namespace caffe
