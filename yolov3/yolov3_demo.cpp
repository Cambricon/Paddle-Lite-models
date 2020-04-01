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
  float x;
  float y;
  float w;
  float h;
};

using RESULT = std::vector<Object>;

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
  
#if 0
    // Get the data of output tensor and postprocess to output detected objects
    std::unique_ptr<const Tensor> output_tensor(
        std::move(predictor_->GetOutput(0)));
    const float *output_data = output_tensor->mutable_data<float>();
    auto o_shape = output_tensor->shape();
    std::cout << o_shape.size() << std::endl;
    for (int i = 0; i < o_shape.size(); ++i) {
      std::cout << '\t' << o_shape[i];
    }
    std::cout << std::endl;
    int64_t output_size = o_shape[0] * o_shape[1];
    std::vector<RESULT> results;
    results.reserve(o_shape[0]);
    /* cv::Mat output_image = input_image.clone(); */
    double postprocess_start_time = get_current_us();
    for (uint32_t i = 0; i < o_shape[0]; ++i) {
      results.emplace_back(postprocess(output_data + i * output_size, output_size, width_, height_));
    }
    double postprocess_end_time = get_current_us();
    double postprocess_time = (postprocess_end_time - postprocess_start_time) / 1000.0f;

    postprocess_time_.push_back(postprocess_time / i_shape_[0]);
    printf("Postprocess time: %f ms\n\n", postprocess_time);
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
    size_tensor_->Resize({i_shape_[0], 2});
    auto size_data = size_tensor_->mutable_data<int>();
    for (int i = 0; i < i_shape_[0] * 2; ++i) {
      size_data[i] = 608;
    }
    width_ = i_shape_[3];
    height_ = i_shape_[2];
    hwc_ = i_shape_[1] * i_shape_[2] * i_shape_[3];
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

#if 0
  RESULT postprocess(const float *data, size_t len, int width, int height) {
    auto range_0_1 = [](float num) { return std::max(.0f, std::min(1.0f, num)); };
    const float * net_output;
    for (int i = 0; i < 2; ++i) {
      net_output = data + i * 6;
      printf("%f, %f, %f, %f, %f, %f\n", net_output[0], net_output[1], net_output[2], net_output[3], net_output[4], net_output[5]);
    }
    for (int box_idx = 0; box_idx < box_num; ++box_idx) {
      float left = range_0_1(net_output[64 + box_idx * box_step + 3]);
      float right = range_0_1(net_output[64 + box_idx * box_step + 5]);
      float top = range_0_1(net_output[64 + box_idx * box_step + 4]);
      float bottom = range_0_1(net_output[64 + box_idx * box_step + 6]);

      // rectify
      left = (left * model_input_w - (model_input_w - scaled_w) / 2) / scaled_w;
      right = (right * model_input_w - (model_input_w - scaled_w) / 2) / scaled_w;
      top = (top * model_input_h - (model_input_h - scaled_h) / 2) / scaled_h;
      bottom = (bottom * model_input_h - (model_input_h - scaled_h) / 2) / scaled_h;
      left = std::max(0.0f, left);
      right = std::max(0.0f, right);
      top = std::max(0.0f, top);
      bottom = std::max(0.0f, bottom);

      auto obj = std::make_shared<cnstream::CNInferObject>();
      obj->id = std::to_string(static_cast<int>(net_output[64 + box_idx * box_step + 1]));
      obj->score = net_output[64 + box_idx * box_step + 2];

      obj->bbox.x = left;
      obj->bbox.y = top;
      obj->bbox.w = std::min(1.0f - obj->bbox.x, right - left);
      obj->bbox.h = std::min(1.0f - obj->bbox.y, bottom - top);

      if (obj->bbox.h <= 0 || obj->bbox.w <= 0 || (obj->score < threshold_ && threshold_ > 0)) continue;

      package->objs.push_back(obj);
    }
    return {};
  }
#endif
};  // class Inferencer

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
  std::string model_dir = "/home/dingminghui/paddle/data/yolov3_new";
  std::string input_image_pathes = "./filelist";
  std::cout << "model_path:  " << model_dir << std::endl;
  std::cout << "image path:  " << input_image_pathes  << std::endl;

  // Set MobileConfig
  CxxConfig config;
  config.set_model_dir(model_dir);
  std::vector<Place> valid_places{
    Place{TARGET(kX86), PRECISION(kFloat)},
    /* Place{TARGET(kMLU), PRECISION(kFloat), DATALAYOUT(kNHWC)} */
    /* Place{TARGET(kMLU), PRECISION(kFP16), DATALAYOUT(kNHWC)} */
    };
  config.set_valid_places(valid_places);

  config.set_mlu_use_first_conv(use_first_conv);
  if (use_first_conv) {
    INPUT_MEAN = {124, 117, 104};
    INPUT_STD = {59, 57, 57};
    std::vector<float> mean_vec = INPUT_MEAN;
    std::vector<float> std_vec = INPUT_STD;
    config.set_mlu_first_conv_mean(mean_vec);
    config.set_mlu_first_conv_std(std_vec);
  }

  config.set_mlu_core_version(MLUCoreVersion::MLU_270);
  config.set_mlu_core_number(16);
  config.set_mlu_input_layout(DATALAYOUT(kNHWC));

  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<CxxConfig>(config);

  Inferencer infer(predictor, {BATCH_SIZE, 3, 608, 608});

  std::vector<std::string> pathes = load_image_pathes(input_image_pathes);

  // warm up
  {
    std::cout << "warm up ....." << std::endl;
    std::string image_name = pathes[0];
    cv::Mat input_image = cv::imread(image_name, -1);
    infer.warm_up(input_image);
    std::cout << "warm up end" << std::endl;
  }

  auto start = get_current_us();
  int index = 0;
  for(int i =0; i < pathes.size() - 1; i++)
  {
    std::string image_name = pathes[i];
    std::cout << image_name << std::endl;
    cv::Mat input_image = cv::imread(image_name, -1);
    // cv::imshow("aaa", input_image);
    // cv::waitKey(0);
    printf("process %d th image",i);
    infer.batch(input_image);
    if (index % BATCH_SIZE == BATCH_SIZE - 1) {
      std::vector<RESULT> results = infer.process();
    }
    ++index;
  }
  auto end = get_current_us();
  double cur_time_cost = (end - start) / 1000.0f;
  /* float fps = (float)(pathes.size()  -1) / (cur_time_cost / 1000.0f); */
  /* std::cout << "fps for " << pathes.size() << " images: " << fps << std::endl; */
  std::cout << "average preprocess time :" << infer.avg_preprocess_time() << std::endl;
  std::cout << "average prediction time :" << infer.avg_prediction_time() << std::endl;
  /* std::cout << "average postprocess time :" << infer.avg_postprocess_time() << std::endl; */
  return 0;
}
