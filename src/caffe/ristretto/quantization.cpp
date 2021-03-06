#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "ristretto/quantization.hpp"
#include "Python.h"
#include "iostream"
#include "fstream"

using caffe::Caffe;
using caffe::Net;
using caffe::string;
using caffe::vector;
using caffe::Blob;
using caffe::LayerParameter;
using caffe::NetParameter;
using namespace std;

Quantization::Quantization(string model, string model_quantized, string qt_script_name,
  string PerAccuracy, string AllAccuracy, string trimming_mode, string gpus) {
  this->model_ = model;
  this->model_quantized_ = model_quantized;
  this->qt_script_name_ = qt_script_name;
  this->PerAccuracy_ = PerAccuracy;
  this->AllAccuracy_ = AllAccuracy;
  this->trimming_mode_ = trimming_mode;
  this->gpus_ = gpus;
  this->exp_bits_ = 8;
}

void Quantization::QuantizeNet() {
  CheckWritePermissions(model_quantized_);
  SetGpu();
  float accuracy;
  if (trimming_mode_ == "dynamic_fixed_point") {
    Quantize2DynamicFixedPoint();
  } else {
    LOG(FATAL) << "Unknown trimming mode: " << trimming_mode_;
  }
}

void Quantization::CheckWritePermissions(const string path) {
  std::ofstream probe_ofs(path.c_str());
  if (probe_ofs.good()) {
    probe_ofs.close();
    std::remove(path.c_str());
  } else {
    LOG(FATAL) << "Missing write permissions";
  }
}

void Quantization::SetGpu() {
  vector<int> gpus;
  if (gpus_ == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus.push_back(i);
    }
  } else if (gpus_.size()) {
    vector<string> strings;
    boost::split(strings, gpus_, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus.push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus.size(), 0);
  }
  // Set device id and mode
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
}

void Quantization::Quantize2DynamicFixedPoint() {
  // Score net with dynamic fixed point layer activations.
  // The rest of the net remains in high precision format.
  NetParameter param;
  float accuracy;
  string qt_script_name = qt_script_name_;
  string modelquantize = model_quantized_;
  string s1 = "execfile(\'" + qt_script_name + "\')";
  cout << s1 << endl;
  string PerAccuracy = PerAccuracy_;
  string AllAccuracy = AllAccuracy_;
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);


  for( int i=0 ; i < param.layer_size(); ++i){
    string layername = param.layer(i).name();
    string layertype = param.layer(i).type();
    LOG(INFO) << layertype;
    vector<int> FlParams;
    vector<float> params_accuracy;
    if (layertype.find("Convolution") != string::npos || layertype.find("Scale") != string::npos || layertype.find("InnerProduct") != string::npos){  
      ofstream out_results;
      out_results.open(AllAccuracy, ios::app);
      out_results<<"*****"<<layername<<"*****"<<endl; 
      out_results.close();

      for(int fl_params=0;fl_params<=7; fl_params++){
        EditNetDescriptionDynamicFixedPoint(i, &param, layertype, "Parameters", fl_params);
        param.release_state();
        WriteProtoToTextFile(param, modelquantize);
        system(qt_script_name.c_str());
        char buf[1024];
        ifstream infile;
        infile.open(PerAccuracy);
        if(infile.is_open()){
         while(infile.good() && !infile.eof()){
           memset(buf,0,1024);
           infile.getline(buf,1204);
           accuracy = atof(buf);         
           params_accuracy.push_back(accuracy);
           LOG(INFO)<<"params_layertype["<<i<<"]:"<<layertype;
           LOG(INFO)<<"params_accuracy["<<fl_params<<"]:"<<params_accuracy[fl_params];
         }
         FlParams.push_back(fl_params);
         LOG(INFO)<<"FlParams["<<fl_params<<"]:"<<FlParams[fl_params];
         infile.close();
        }
      }
  
      int fl_params_best;
      float params_accuracy_best = -1000000000;
      for(int j=0; j<=7; ++j){
        if(params_accuracy[j] > params_accuracy_best){
          LOG(INFO)<<"params_accuracy["<<j<<"]:"<<params_accuracy[j]<<"|params_accuracy_best:"<<params_accuracy_best;
          params_accuracy_best = params_accuracy[j];
          fl_params_best = FlParams[j];

        }
      }
      EditNetDescriptionDynamicFixedPoint(i,&param, layertype, "Parameters", fl_params_best);
      param.release_state();
      WriteProtoToTextFile(param, modelquantize); 
      caffe::ReadNetParamsFromTextFileOrDie(modelquantize, &param);
      param.mutable_state()->set_phase(caffe::TEST);
    } 
  }

  caffe::ReadNetParamsFromTextFileOrDie(modelquantize, &param);
  param.mutable_state()->set_phase(caffe::TEST);

  for( int i=0 ; i < param.layer_size(); ++i){
    string layername = param.layer(i).name();
    string layertype = param.layer(i).type();
    vector<int> FlIn;
    vector<int> FlOut;
    vector<float> in_accuracy;
    vector<float> out_accuracy;
    int fl_in_best;
    float in_accuracy_best = -100000000; 
    if (layername =="conv1"){
      ofstream out_results;
      out_results.open(AllAccuracy, ios::app);
      out_results<<"*****"<<layername<<"*****"<<endl; 
      out_results.close();
      for(int fl_in=0; fl_in<=7; fl_in++){
        EditNetDescriptionDynamicFixedPoint_inout(i,&param, layertype, "Activations_Input", 
                                            fl_in, -1);
        param.release_state();
        WriteProtoToTextFile(param, modelquantize);
        system(qt_script_name.c_str());

        char buf[1024];
        ifstream infile;
        infile.open(PerAccuracy);
        if(infile.is_open()){
         while(infile.good() && !infile.eof()){
           memset(buf,0,1024);
           infile.getline(buf,1204);
           accuracy = atof(buf);      
           in_accuracy.push_back(accuracy);
           LOG(INFO)<<"in_layertype["<<i<<"]:"<<layertype;
           LOG(INFO)<<"in_accuracy["<<fl_in<<"]:"<<in_accuracy[fl_in]; 
         }
         FlIn.push_back(fl_in);
         LOG(INFO)<<"FlIn["<<fl_in<<"]:"<<FlIn[fl_in];
        }
        infile.close();
      }

      //int fl_in_best;
      //float in_accuracy_best = -100000000;      
      for(int k=0; k<=7; ++k){
        if(in_accuracy[k] > in_accuracy_best){
          in_accuracy_best = in_accuracy[k];
          fl_in_best = FlIn[k];
        }
      }
      EditNetDescriptionDynamicFixedPoint_inout(i,&param, layertype, "Activations_Input", 
                                            fl_in_best, -1);
      param.release_state();
      WriteProtoToTextFile(param, modelquantize);
    }
////////////////////////////////////////////////////////////////////////////////////  
    if(layertype.find("ConvolutionRistretto") != string::npos || layertype.find("ScaleRistretto") != string::npos || layertype.find("FcRistretto") != string::npos)
    {
      if(layername != "conv1"){
        fl_in_best = -1;
      }
      ofstream out_results;
      out_results.open(AllAccuracy, ios::app);
      out_results<<"*****"<<layername<<"*****"<<endl; 
      out_results.close();
      caffe::ReadNetParamsFromTextFileOrDie(modelquantize, &param);
      param.mutable_state()->set_phase(caffe::TEST);
      for(int fl_out=0; fl_out<=7; fl_out++){
        EditNetDescriptionDynamicFixedPoint_inout(i,&param, layertype, "Activations_Output", fl_in_best, fl_out);
        param.release_state();
        WriteProtoToTextFile(param, modelquantize);
        system(qt_script_name.c_str());
        char buf[1024];
        ifstream infile;
        infile.open(PerAccuracy);
        if(infile.is_open()){
         while(infile.good() && !infile.eof()){
           memset(buf,0,1024);
           infile.getline(buf,1204);
           accuracy = atof(buf);         
           out_accuracy.push_back(accuracy);
           LOG(INFO)<<"out_layertype["<<i<<"]:"<<layertype;
           LOG(INFO)<<"out_accuracy["<<fl_out<<"]:"<<out_accuracy[fl_out];   
         }
         FlOut.push_back(fl_out);
         LOG(INFO)<<"FlOut["<<fl_out<<"]:"<<FlOut[fl_out];
        }
        infile.close();
      }

      int fl_out_best;
      float out_accuracy_best = -1000000000;      
      for(int p=0; p<=7; ++p){
        if(out_accuracy[p] > out_accuracy_best){
          out_accuracy_best = out_accuracy[p];
          fl_out_best = FlOut[p];
        }
      }
      EditNetDescriptionDynamicFixedPoint_inout(i,&param, layertype, "Activations_Output", 
                                            fl_in_best, fl_out_best);
      param.release_state();
      WriteProtoToTextFile(param, modelquantize);
      caffe::ReadNetParamsFromTextFileOrDie(modelquantize, &param);
      param.mutable_state()->set_phase(caffe::TEST);
    }
  } 
}


void Quantization::EditNetDescriptionDynamicFixedPoint(const int i,NetParameter* param,
      const string layers_2_quantize, const string net_part,  
      const int fl_params) {

  // if this is a convolutional layer which should be quantized ...
  if (layers_2_quantize.find("Convolution") != string::npos) {
    // quantize parameters
    if (net_part.find("Parameters") != string::npos) {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("ConvolutionRistretto");  
      param_layer->mutable_quantization_param()->set_fl_params(fl_params);
      param_layer->mutable_quantization_param()->set_bw_params(8);
    }
  }

  // if this is a convolutional layer which should be quantized ...
  if (layers_2_quantize.find("Scale") != string::npos) {
    // quantize parameters
    if (net_part.find("Parameters") != string::npos) {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("ScaleRistretto");  
      param_layer->mutable_quantization_param()->set_fl_params(fl_params);
      param_layer->mutable_quantization_param()->set_bw_params(8);
    }
  }     

  // if this is an inner product layer which should be quantized ...
  if (layers_2_quantize.find("InnerProduct") != string::npos) {
    // quantize parameters
    if (net_part.find("Parameters") != string::npos) {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("FcRistretto");
      param_layer->mutable_quantization_param()->set_fl_params(fl_params);
      param_layer->mutable_quantization_param()->set_bw_params(8);
    }
  }
}

void Quantization::EditNetDescriptionDynamicFixedPoint_inout(const int i,NetParameter* param,
      const string layers_2_quantize, const string net_part,  
      const int fl_in,  const int fl_out) {

  // if this is a convolutional layer which should be quantized ...
  if (layers_2_quantize.find("ConvolutionRistretto") != string::npos) {
    // quantize activations_input
    if (net_part.find("Activations_Input") != string::npos) {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->mutable_quantization_param()->set_fl_layer_in(fl_in);
      param_layer->mutable_quantization_param()->set_bw_layer_in(8);
    }
    // quantize activations_output
    if (net_part.find("Activations_Output") != string::npos) {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->mutable_quantization_param()->set_fl_layer_out(fl_out);
      param_layer->mutable_quantization_param()->set_bw_layer_out(8);
    }
  }

  // if this is a convolutional layer which should be quantized ...
  if (layers_2_quantize.find("ScaleRistretto") != string::npos) {
    // quantize activations_input
    if (net_part.find("Activations_Input") != string::npos) {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->mutable_quantization_param()->set_fl_layer_in(fl_in);
      param_layer->mutable_quantization_param()->set_bw_layer_in(8);
    }
    // quantize activations_output
    if (net_part.find("Activations_Output") != string::npos) {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->mutable_quantization_param()->set_fl_layer_out(fl_out);
      param_layer->mutable_quantization_param()->set_bw_layer_out(8);
    }
  }     

  // if this is an inner product layer which should be quantized ...
  if (layers_2_quantize.find("FcRistretto") != string::npos) {
    // quantize activations_input
    if (net_part.find("Activations_Input") != string::npos) {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->mutable_quantization_param()->set_fl_layer_in(fl_in);
      param_layer->mutable_quantization_param()->set_bw_layer_in(8);
    }
    // quantize activations_output
    if (net_part.find("Activations_Output") != string::npos) {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->mutable_quantization_param()->set_fl_layer_out(fl_out);
      param_layer->mutable_quantization_param()->set_bw_layer_out(8);
    }
  }
}