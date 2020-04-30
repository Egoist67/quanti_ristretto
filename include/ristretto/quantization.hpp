#ifndef QUANTIZATION_HPP_
#define QUANTIZATION_HPP_

#include "caffe/caffe.hpp"
#include "Python.h"
#include "iostream"
#include "fstream"
using caffe::string;
using caffe::vector;
using caffe::Net;

/**
 * @brief Approximate 32-bit floating point networks.
 *
 * This is the Ristretto tool. Use it to generate file descriptions of networks
 * which use reduced word width arithmetic.
 */
class Quantization {
public:
  explicit Quantization(string model, string model_quantized, string qt_script_name,
    string PerAccuracy, string AllAccuracy, string trimming_mode,string gpus);
  void QuantizeNet();
private:
  void CheckWritePermissions(const string path);
  void SetGpu();
  /**
   * @brief Score network.
   * @param accuracy Reports the network's accuracy according to
   * accuracy_number.
   * @param do_stats: Find the maximal values in each layer.
   * @param score_number The accuracy layer that matters.
   *
   * For networks with multiple accuracy layers, set score_number to the
   * appropriate value. For example, for BVLC GoogLeNet, use score_number=7.
   */
  /**
   * @brief Quantize convolutional and fully connected layers to dynamic fixed
   * point.
   * The parameters and layer activations get quantized and the resulting
   * network will be tested.
   * Find the required number of bits required for parameters and layer
   * activations (which might differ from each other).
   */
  void Quantize2DynamicFixedPoint();
  /**
   * @brief Quantize convolutional and fully connected layers to minifloat.
   * Parameters and layer activations share the same numerical representation.
   * This simulates hardware arithmetic which uses IEEE-754 standard (with some
   * small optimizations).
   */
  void EditNetDescriptionDynamicFixedPoint(const int i, caffe::NetParameter* param,
      const string layers_2_quantize, const string network_part,
      const int fl_params);

  void EditNetDescriptionDynamicFixedPoint_inout(const int i, caffe::NetParameter* param,
      const string layers_2_quantize, const string network_part,
      const int fl_in, const int fl_out);
  /**
   * @brief Change network to minifloat.
   */

  string model_;
  string weights_;
  string model_quantized_;
  string qt_script_name_;
  string PerAccuracy_;
  string AllAccuracy_;
  int iterations_;
  string trimming_mode_;
  double error_margin_;
  string gpus_;
  float test_score_baseline_;
  // The maximal absolute values of layer inputs, parameters and
  // layer outputs.
  vector<float> max_in_, max_params_, max_out_;
  // The name of the layers that need to be quantized to dynamic fixed point.
  vector<string> layer_names_;
  // The number of bits used for dynamic fixed point layer inputs, parameters
  // and layer outputs.
  int fl_params, fl_in, fl_out;

  // The number of bits used for minifloat exponent.
  int exp_bits_;
};

#endif // QUANTIZATION_HPP_
