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

int WARMUP_COUNT = 0;
int REPEAT_COUNT = 1;
const int CPU_THREAD_NUM = 2;
const PowerMode CPU_POWER_MODE =
    PowerMode::LITE_POWER_HIGH;
std::vector<int64_t> INPUT_SHAPE = {1, 224, 224, 3};
std::vector<float> INPUT_MEAN = {0.485f, 0.456f, 0.406f};
std::vector<float> INPUT_STD = {0.229f, 0.224f, 0.225f};



struct RESULT {
  std::string class_name;
  int class_id;
  float score;
};

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

// for first conv
void preprocess(cv::Mat &input_image,int input_width,
                int input_height, uint8_t *input_data) {

  cv::Mat resize_image;
  cv::resize(input_image, resize_image, cv::Size(input_width, input_height), 0, 0);
  cv::cvtColor(resize_image, resize_image, CV_BGRA2RGB);
  cv::Mat output_image;
  resize_image.convertTo(output_image,CV_8UC3);
  memcpy(input_data, output_image.data, input_height * input_width * 3 * sizeof(uint8_t));
}

void preprocess(cv::Mat &input_image, const std::vector<float> &input_mean,
                const std::vector<float> &input_std, int input_width,
                int input_height, float *input_data) {
  cv::Mat rgb_img;
  cv::cvtColor(input_image, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(rgb_img, rgb_img, cv::Size(input_width, input_height), 0.f, 0.f);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
  const float* dimg = reinterpret_cast<const float*>(imgf.data);
  const int size_tmp = input_width * input_height;
  for (int i = 0; i < input_width * input_height; i++) {
    input_data[i * 3 + 0] =  (dimg[i * 3 + 0] - input_mean[0]) / input_std[0];
    input_data[i * 3 + 1] =  (dimg[i * 3 + 1] - input_mean[1]) / input_std[1];
    input_data[i * 3 + 2] =  (dimg[i * 3 + 2] - input_mean[2]) / input_std[2];
    //input_data[i] =  (dimg[i * 3 + 0] - input_mean[0]) / input_std[0];
    //input_data[i + size_tmp] =  (dimg[i * 3 + 1] - input_mean[1]) / input_std[1];
    //input_data[i + 2 * size_tmp] =  (dimg[i * 3 + 2] - input_mean[2]) / input_std[2];
  }
}

std::vector<RESULT> postprocess(const float *output_data, int64_t output_size,
                                cv::Mat &output_image) {
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
  std::vector<RESULT> results(TOPK);
  for (int i = 0; i < results.size(); i++) {
    results[i].class_id = max_indices[i];
    results[i].score = max_scores[i];
  }
  return results;
}

std::vector<RESULT> process(cv::Mat &input_image,
                std::shared_ptr<PaddlePredictor> &predictor) {
  std::unique_ptr<Tensor> input_tensor(
      std::move(predictor->GetInput(0)));
  input_tensor->Resize(INPUT_SHAPE);
  int input_width = INPUT_SHAPE[2];
  int input_height = INPUT_SHAPE[1];
  void* input_data;
  double preprocess_start_time = get_current_us();

  if (use_first_conv) {
   input_data = input_tensor->mutable_data<int8_t>();
    preprocess(input_image,
               input_width,
               input_height,
               reinterpret_cast<uint8_t*>(input_data));
  } else {
    input_data = input_tensor->mutable_data<float>();
    preprocess(input_image,
               INPUT_MEAN,
               INPUT_STD,
               input_width,
               input_height,
               reinterpret_cast<float*>(input_data));
  }

  double preprocess_end_time = get_current_us();
  double preprocess_time = (preprocess_end_time - preprocess_start_time) / 1000.0f;

  double prediction_time;
  for (int i = 0; i < WARMUP_COUNT; i++) {
    predictor->Run();
  }
  double max_time_cost = 0.0f;
  double min_time_cost = std::numeric_limits<float>::max();
  double total_time_cost = 0.0f;
  for (int i = 0; i < REPEAT_COUNT; i++) {
    auto start = get_current_us();
    predictor->Run();
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
  printf("warmup: %d repeat: %d, average: %f ms, max: %f ms, min: %f ms\n",
          WARMUP_COUNT, REPEAT_COUNT, prediction_time,
          max_time_cost, min_time_cost);

  // Get the data of output tensor and postprocess to output detected objects
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  const float *output_data = output_tensor->mutable_data<float>();
  int64_t output_size = 1;
  for (auto dim : output_tensor->shape()) {
    output_size *= dim;
  }
  cv::Mat output_image = input_image.clone();
  double postprocess_start_time = get_current_us();
  std::vector<RESULT> results =
      postprocess(output_data, output_size, output_image);
  double postprocess_end_time = get_current_us();
  double postprocess_time = (postprocess_end_time - postprocess_start_time) / 1000.0f;

  for (int i = 0; i < results.size(); i++) {
    printf("Top%d %s -class: %d, score %f\n", i, results[i].class_name.c_str(),
            results[i].class_id,results[i].score);
  }
  printf("Preprocess time: %f ms\n", preprocess_time);
  printf("Prediction time: %f ms\n", prediction_time);
  printf("Postprocess time: %f ms\n\n", postprocess_time);
  return results;
}


struct ACCU get_accu(std::vector<RESULT> results, int class_id)
{
  ACCU accu;
  for (size_t i = 0; i < results.size(); i++)
  {
    RESULT result = results[i];
    if (result.class_id == class_id)
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

  std::string model_dir = "/opt/shared/paddle-lite/models/ResNet50_quant/";
  // std::string model_dir = "/projs/systools/zhangshijin/converted/inference_model";
  // std::string input_image_pathes = "/home/zhaoying/imagenet/val_5000.txt";
  // std::string input_image_pathes = "/home/zhaoying/imagenet/val_1000.txt";
  // std::string input_image_pathes = "/home/zhaoying/imagenet/val_100.txt";
  //std::string input_image_pathes = "/projs/systools/zhangshijin/val.txt";
  //std::string input_image_pathes = "/home/zhangmingwei/ws/filelist";
  std::string input_image_pathes = "./filelist";
  std::string label_path =  input_image_pathes;
  std::cout << "model_path:  " << model_dir << std::endl;
  std::cout << "image and label path:  " << input_image_pathes  << std::endl;

  // Load Labels
  std::vector<int> labels = load_labels(label_path);


  // Set MobileConfig
  CxxConfig config;
  config.set_model_dir(model_dir);
  std::vector<Place> valid_places{
    Place{TARGET(kX86), PRECISION(kFloat)},
    Place{TARGET(kMLU), PRECISION(kFloat), DATALAYOUT(kNHWC)}
    //Place{TARGET(kMLU), PRECISION(kFP16), DATALAYOUT(kNHWC)}
    };
    // Place{TARGET(kMLU), PRECISION(kInt8)}
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

  config.set_mlu_core_version(MLUCoreVersion::MLU_270);
  config.set_mlu_core_number(16);

  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<CxxConfig>(config);

  std::vector<ACCU> accus;
  std::vector<std::string> pathes = load_image_pathes(input_image_pathes);

  // warm up
  std::cout << "warm up ....." << std::endl;
  std::string image_name = pathes[0];
  // std::string real_path = "/home/zhaoying/imagenet/" + image_name;
  // std::string real_path = "/opt/shared/beta/models_and_data/imagenet/" + image_name;
  // cv::Mat input_image = cv::imread(real_path, 1);
  // process(input_image, predictor);

  auto start = get_current_us();
  for(int i =0; i < pathes.size() - 1; i++)
  {
    std::string image_name = pathes[i];
    // std::string real_path = "/home/zhaoying/imagenet/" + image_name;
    std::string real_path = image_name;
    cv::Mat input_image = cv::imread(real_path, -1);
    // cv::imshow("aaa", input_image);
    // cv::waitKey(0);
    printf("process %d th image",i);
    std::vector<RESULT> results = process(input_image,predictor);
    ACCU accu = get_accu(results,labels[i]);
    accus.push_back(accu);
  }
  auto end = get_current_us();
  double cur_time_cost = (end - start) / 1000.0f;
  float fps = (float)(pathes.size()  -1) / (cur_time_cost / 1000.0f);
  float mean_top1 = 0;
  float mean_top5 = 0;
  int total_top1 = 0;
  int total_top5 = 0;
  for (size_t i = 0; i < accus.size(); i++)
  {
   total_top1 += accus[i].top1;
   total_top5 += accus[i].top5;
  }
  mean_top1 = (float)total_top1 / (float)accus.size();
  mean_top5 = (float)total_top5 / (float)accus.size();
  std::cout << "top1 for " << accus.size() << " images: " << mean_top1 << std::endl;
  std::cout << "top5 for " << accus.size() << " images: " << mean_top5 << std::endl;
  std::cout << "fps for " << accus.size() << " images: " << fps << std::endl;
  return 0;
}
