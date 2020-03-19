// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle_api.h"
#include "paddle_use_kernels.h" // NOLINT
#include "paddle_use_ops.h"     // NOLINT
#include "paddle_use_passes.h"  // NOLINT
// #include <arm_neon.h>
#include <limits>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include "string.h"

using namespace paddle::lite_api; // NOLINT

bool use_first_conv = false;

template <typename T>
void transpose(T *input_data,
               T *output_data,
               std::vector<int> input_shape,
               std::vector<int> axis) {
  int old_index = -1;
  int new_index = -1;
  int dim[4] = {0};
  std::vector<int> shape = input_shape;
  for (dim[0] = 0; dim[0] < input_shape[0]; dim[0]++) {
    for (dim[1] = 0; dim[1] < input_shape[1]; dim[1]++) {
      for (dim[2] = 0; dim[2] < input_shape[2]; dim[2]++) {
        for (dim[3] = 0; dim[3] < input_shape[3]; dim[3]++) {
          old_index = dim[0] * shape[1] * shape[2] * shape[3]
                                   + dim[1] * shape[2] * shape[3]
                                              + dim[2] * shape[3]
                                              + dim[3];
          new_index =
              dim[axis[0]] * shape[axis[1]] * shape[axis[2]] * shape[axis[3]]
                             + dim[axis[1]] * shape[axis[2]] * shape[axis[3]]
                                              + dim[axis[2]] * shape[axis[3]]
                                                               + dim[axis[3]];
          output_data[new_index] = input_data[old_index];
        }
      }
    }
  }
}

int WARMUP_COUNT = 1;
int REPEAT_COUNT = 1;
uint32_t BATCH_SIZE = 1;
const int CPU_THREAD_NUM = 2;
const PowerMode CPU_POWER_MODE =
    PowerMode::LITE_POWER_HIGH;
std::vector<float> INPUT_MEAN = {0.485f, 0.456f, 0.406f};
std::vector<float> INPUT_STD = {0.229f, 0.224f, 0.225f};



struct Object {
  std::string class_name;
  int class_id;
  float score;
};

using RESULT = std::vector<Object>;

struct ACCU{
  int top1 = 0;
  int top5 = 0;
};

inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

std::vector<std::string> load_image_pathes(const std::string &path) {
  std::ifstream file;
  std::vector<std::string> pathes;
  file.open(path);
  while (file) {
    std::string line;
    std::getline(file, line);
    std::string::size_type pos = line.find(" ");
    if (pos != std::string::npos) {
      line = line.substr(0,pos);
    }
    pathes.push_back(line);
  }
  file.clear();
  file.close();
  return pathes;
}

std::vector<int> load_labels(const std::string &path) {
  std::ifstream file;
  std::vector<int> labels;
  file.open(path);
  while (file) {
    std::string line;
    std::getline(file, line);
    std::string::size_type pos = line.find(" ");
    if (pos != std::string::npos) {
      line = line.substr(pos + 1);
    }
    labels.push_back(atoi(line.c_str()));
  }
  file.clear();
  file.close();
  return labels;
}

class Inferencer {
 public:
  Inferencer(std::shared_ptr<PaddlePredictor> p, std::vector<int64_t> input_shape): predictor_(p), i_shape_(input_shape) {
    refresh_input();
  }

  void warm_up(cv::Mat &input_image) {
    for (int i = 0; i < WARMUP_COUNT; i++) {
      for (uint32_t i = 0; i < i_shape_[0]; ++i) {
        batch(input_image);
      }
      process();
    }
    printf("warm up count: %d\n", WARMUP_COUNT);
    preprocess_time_.clear();
    prediction_time_.clear();
    postprocess_time_.clear();
  }

  void batch(const cv::Mat &input_image) {
    double preprocess_start_time = get_current_us();
  
    if (use_first_conv) {
      preprocess(input_image,
                 reinterpret_cast<uint8_t*>(input_data_) + batch_index_ * hwc_);
    } else {
      preprocess(input_image,
                 INPUT_MEAN,
                 INPUT_STD,
                 reinterpret_cast<float*>(input_data_) + batch_index_ * hwc_);
    }
  
    double preprocess_end_time = get_current_us();
    double preprocess_time = (preprocess_end_time - preprocess_start_time) / 1000.0f;

    preprocess_time_.push_back(preprocess_time);
    printf("Preprocess time: %f ms\n", preprocess_time);

    ++batch_index_;
  }

  std::vector<RESULT> process() {
    double prediction_time;
    double max_time_cost = 0.0f;
    double min_time_cost = std::numeric_limits<float>::max();
    double total_time_cost = 0.0f;
    for (int i = 0; i < REPEAT_COUNT; i++) {
      auto start = get_current_us();
      predictor_->Run();
      auto end = get_current_us();
      double cur_time_cost = (end - start) / 1000.0f;
      if (cur_time_cost > max_time_cost) {
        max_time_cost = cur_time_cost;
      }
      if (cur_time_cost < min_time_cost) {
        min_time_cost = cur_time_cost;
      }
      total_time_cost += cur_time_cost;
      prediction_time = total_time_cost / REPEAT_COUNT;
      printf("iter %d cost: %f ms\n", i, cur_time_cost);
    }
    printf("repeat: %d, average: %f ms, max: %f ms, min: %f ms\n",
            REPEAT_COUNT, prediction_time,
            max_time_cost, min_time_cost);

    prediction_time_.push_back(prediction_time / i_shape_[0]);
    printf("Prediction time: %f ms\n", prediction_time);
  
    // Get the data of output tensor and postprocess to output detected objects
    std::unique_ptr<const Tensor> output_tensor(
        std::move(predictor_->GetOutput(0)));
    const float *output_data = output_tensor->mutable_data<float>();
    auto o_shape = output_tensor->shape();
    int64_t output_size = o_shape[1] * o_shape[2] * o_shape[3];
    std::vector<RESULT> results;
    results.reserve(o_shape[0]);
    /* cv::Mat output_image = input_image.clone(); */
#if 0
    double postprocess_start_time = get_current_us();
    for (uint32_t i = 0; i < o_shape[0]; ++i) {
      results.emplace_back(postprocess(output_data + i * output_size, output_size));
      printf("batch index %u:\n", batch_index_);
      for (int j = 0; j < results[i].size(); j++) {
        printf("Top%d %s -class: %d, score %f\n", j, results[i][j].class_name.c_str(),
                results[i][j].class_id,results[i][j].score);
      }
    }
    /* std::vector<RESULT> results = */
    /*     postprocess(output_data, output_size, output_image); */
    double postprocess_end_time = get_current_us();
    double postprocess_time = (postprocess_end_time - postprocess_start_time) / 1000.0f;

    postprocess_time_.push_back(postprocess_time / i_shape_[0]);
    printf("Postprocess time: %f ms\n\n", postprocess_time);

    refresh_input();
    return results;
#endif
    refresh_input();
    return {};
  }

  float avg_preprocess_time() {
    if (preprocess_time_.size() < 2) return 0;
    double total_time = 0;
    for (uint32_t i = 0; i < preprocess_time_.size(); ++i) {
      total_time += preprocess_time_[i];
    }
    return total_time / (preprocess_time_.size() - 1);
  }

  float avg_prediction_time() {
    if (prediction_time_.size() < 2) return 0;
    double total_time = 0;
    for (uint32_t i = 0; i < prediction_time_.size(); ++i) {
      total_time += prediction_time_[i];
    }
    return total_time / (prediction_time_.size() - 1);
  }

  float avg_postprocess_time() {
    if (postprocess_time_.size() < 2) return 0;
    double total_time = 0;
    for (uint32_t i = 0; i < postprocess_time_.size(); ++i) {
      total_time += postprocess_time_[i];
    }
    return total_time / (postprocess_time_.size() - 1);
  }

 private:
  std::shared_ptr<PaddlePredictor> predictor_;
  std::vector<int64_t> i_shape_;
  std::unique_ptr<Tensor> input_tensor_;
  std::unique_ptr<Tensor> size_tensor_;
  std::vector<double> preprocess_time_;
  std::vector<double> prediction_time_;
  std::vector<double> postprocess_time_;
  void * input_data_;
  uint32_t batch_index_ = 0;
  int width_, height_;
  int hwc_;

  void refresh_input() {
    input_tensor_ = std::move(predictor_->GetInput(0));
    size_tensor_ = std::move(predictor_->GetInput(1));
    input_tensor_->Resize(i_shape_);
    size_tensor_->Resize({1, 2});
    auto size_data = size_tensor_->mutable_data<int>();
    size_data[0] = size_data[1] = 608;
    width_ = i_shape_[2];
    height_ = i_shape_[1];
    hwc_ = width_ * height_ * i_shape_[3];
    if (use_first_conv) {
      input_data_ = input_tensor_->mutable_data<int8_t>();
    } else {
      input_data_ = input_tensor_->mutable_data<float>();
    }
    batch_index_ = 0;
  }
  // for first conv
  void preprocess(const cv::Mat &input_image, uint8_t *input_data) {
  
    cv::Mat resize_image;
    cv::resize(input_image, resize_image, cv::Size(width_, height_), 0, 0);
    if (resize_image.channels() == 1) {
      cv::cvtColor(resize_image, resize_image, cv::COLOR_GRAY2RGB);
    } else {
      cv::cvtColor(resize_image, resize_image, cv::COLOR_BGRA2RGB);
    }
    cv::Mat output_image;
    resize_image.convertTo(output_image,CV_8UC3);
    memcpy(input_data, output_image.data, height_ * width_ * 3 * sizeof(uint8_t));
  }
  
  void preprocess(const cv::Mat &input_image, const std::vector<float> &input_mean,
                  const std::vector<float> &input_std, float *input_data) {
    cv::Mat rgb_img;
    if (input_image.channels() == 1) {
      cv::cvtColor(input_image, rgb_img, cv::COLOR_GRAY2RGB);
    } else {
      cv::cvtColor(input_image, rgb_img, cv::COLOR_BGR2RGB);
    }
    cv::resize(rgb_img, rgb_img, cv::Size(width_, height_));
    cv::Mat imgf;
    rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
    const float* dimg = reinterpret_cast<const float*>(imgf.data);
    /* const int size_tmp = width_ * height_; */
    for (int i = 0; i < width_ * height_; i++) {
      input_data[i * 3 + 0] =  (dimg[i * 3 + 0] - input_mean[0]) / input_std[0];
      input_data[i * 3 + 1] =  (dimg[i * 3 + 1] - input_mean[1]) / input_std[1];
      input_data[i * 3 + 2] =  (dimg[i * 3 + 2] - input_mean[2]) / input_std[2];
      /* input_data[i] =  (dimg[i * 3 + 0] - input_mean[0]) / input_std[0]; */
      /* input_data[i + size_tmp] =  (dimg[i * 3 + 1] - input_mean[1]) / input_std[1]; */
      /* input_data[i + 2 * size_tmp] =  (dimg[i * 3 + 2] - input_mean[2]) / input_std[2]; */
    }
    imgf.release();
  }
  
  /* std::vector<RESULT> postprocess(const float *output_data, int64_t output_size, */
  /*                                 cv::Mat &output_image) { */
  RESULT postprocess(const float *output_data, int64_t output_size) {
    const int TOPK = 5;
    int max_indices[TOPK];
    double max_scores[TOPK];
    for (int i = 0; i < TOPK; i++) {
      max_indices[i] = 0;
      max_scores[i] = 0;
    }
    for (int i = 0; i < output_size; i++) {
      float score = output_data[i];
      int index = i;
      for (int j = 0; j < TOPK; j++) {
        if (score > max_scores[j]) {
          index += max_indices[j];
          max_indices[j] = index - max_indices[j];
          index -= max_indices[j];
          score += max_scores[j];
          max_scores[j] = score - max_scores[j];
          score -= max_scores[j];
        }
      }
    }
    std::vector<Object> objs(TOPK);
    for (int i = 0; i < objs.size(); i++) {
      objs[i].class_id = max_indices[i];
      objs[i].score = max_scores[i];
    }
    return objs;
  }
  
};  // class Inferencer


struct ACCU get_accu(RESULT objs, int class_id)
{
  ACCU accu;
  for (size_t i = 0; i < objs.size(); i++)
  {
    if (objs[i].class_id == class_id)
    {
      accu.top5 = 1;
      if (i == 0)
      {
        accu.top1 = 1;
      }
    }
  }
  return accu;
}

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cout << "USAGE: ./" << argv[0] << " batch_size" << std::endl;
    std::cout << "e.g. ./" << argv[0] << " 8" << std::endl;
    return 1;
  } else {
    BATCH_SIZE = std::atoi(argv[1]);
    if (BATCH_SIZE < 1) {
      std::cerr << "invalid batch size" << std::endl;
      return -1;
    }
  }
  std::string model_dir = "/home/dingminghui/paddle/data/yolov3_quant";
  /* std::string model_dir = "/home/dingminghui/paddle/data/ResNet50_quant/"; */
  // std::string model_dir = "/projs/systools/zhangshijin/converted/inference_model";
  // std::string input_image_pathes = "/home/zhaoying/imagenet/val_5000.txt";
  // std::string input_image_pathes = "/home/zhaoying/imagenet/val_1000.txt";
  // std::string input_image_pathes = "/home/zhaoying/imagenet/val_100.txt";
  //std::string input_image_pathes = "/projs/systools/zhangshijin/val.txt";
  //std::string input_image_pathes = "/home/zhangmingwei/ws/filelist";
  std::string input_image_pathes = "./filelist";
  /* std::string label_path =  input_image_pathes; */
  std::cout << "model_path:  " << model_dir << std::endl;
  std::cout << "image and label path:  " << input_image_pathes  << std::endl;

  // Load Labels
  /* std::vector<int> labels = load_labels(label_path); */


  // Set MobileConfig
  CxxConfig config;
  config.set_model_dir(model_dir);
  std::vector<Place> valid_places{
    Place{TARGET(kX86), PRECISION(kFloat)},
    Place{TARGET(kMLU), PRECISION(kFloat), DATALAYOUT(kNHWC)}
    /* Place{TARGET(kMLU), PRECISION(kFP16), DATALAYOUT(kNHWC)} */
    };
  config.set_valid_places(valid_places);

  if (use_first_conv) {
    INPUT_MEAN = {124, 117, 104};
    INPUT_STD = {59, 57, 57};
    config.set_use_firstconv(use_first_conv);
    std::vector<float> mean_vec = INPUT_MEAN;
    std::vector<float> std_vec = INPUT_STD;
    config.set_mean(mean_vec);
    config.set_std(std_vec);
  }

  /* config.set_mlu_core_version(MLUCoreVersion::MLU_270); */
  /* config.set_mlu_core_number(16); */

  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<CxxConfig>(config);

  Inferencer infer(predictor, {BATCH_SIZE, 608, 608, 3});

  /* std::vector<ACCU> accus; */
  std::vector<std::string> pathes = load_image_pathes(input_image_pathes);

  // warm up
  {
    std::cout << "warm up ....." << std::endl;
    std::string image_name = pathes[0];
    cv::Mat input_image = cv::imread(image_name, -1);
    infer.warm_up(input_image);
    std::cout << "warm up end" << std::endl;
    // std::string real_path = "/home/zhaoying/imagenet/" + image_name;
    // std::string real_path = "/opt/shared/beta/models_and_data/imagenet/" + image_name;
    // cv::Mat input_image = cv::imread(real_path, 1);
    // process(input_image, predictor);
  }

  auto start = get_current_us();
  int index = 0;
  std::vector<int> batch_labels;
  batch_labels.reserve(BATCH_SIZE);
  for(int i =0; i < pathes.size() - 1; i++)
  {
    std::string image_name = pathes[i];
    std::cout << image_name << std::endl;
    // std::string real_path = "/home/zhaoying/imagenet/" + image_name;
    std::string real_path = image_name;
    cv::Mat input_image = cv::imread(real_path, -1);
    // cv::imshow("aaa", input_image);
    // cv::waitKey(0);
    printf("process %d th image",i);
    try {
      infer.batch(input_image);
      /* batch_labels.emplace_back(labels[i]); */
    } catch (cv::Exception & e) {
      continue;
    }
    if (index % BATCH_SIZE == BATCH_SIZE - 1) {
      std::vector<RESULT> results = infer.process();
      /* for (int j = 0; j < results.size(); ++j) { */
      /*   accus.push_back(get_accu(results[j], batch_labels[j])); */
      /* } */
      /* batch_labels.clear(); */
    }
    ++index;
  }
  auto end = get_current_us();
  double cur_time_cost = (end - start) / 1000.0f;
  float fps = (float)(pathes.size()  -1) / (cur_time_cost / 1000.0f);
  float mean_top1 = 0;
  float mean_top5 = 0;
  int total_top1 = 0;
  int total_top5 = 0;
  /* for (size_t i = 0; i < accus.size(); i++) */
  /* { */
  /*  total_top1 += accus[i].top1; */
  /*  total_top5 += accus[i].top5; */
  /* } */
  /* mean_top1 = (float)total_top1 / (float)accus.size(); */
  /* mean_top5 = (float)total_top5 / (float)accus.size(); */
  /* std::cout << "top1 for " << accus.size() << " images: " << mean_top1 << std::endl; */
  /* std::cout << "top5 for " << accus.size() << " images: " << mean_top5 << std::endl; */
  /* std::cout << "fps for " << " images: " << fps << std::endl; */
  std::cout << "average preprocess time :" << infer.avg_preprocess_time() << std::endl;
  std::cout << "average prediction time :" << infer.avg_prediction_time() << std::endl;
  std::cout << "average postprocess time :" << infer.avg_postprocess_time() << std::endl;
  return 0;
}
