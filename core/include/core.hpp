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

#ifndef CORE_HPP_
#define CORE_HPP_
#include "paddle_api.h"
#include "paddle_use_kernels.h" // NOLINT
#include "paddle_use_ops.h"     // NOLINT
#include "paddle_use_passes.h"  // NOLINT
// #include <arm_neon.h>
#include "string.h"
#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

using namespace paddle::lite_api; // NOLINT

static bool use_first_conv = false;
static int BATCH_SIZE = 1;
static int WARMUP_COUNT = 1;
static int REPEAT_COUNT = 1;
static std::vector<float> INPUT_MEAN = {0.485f, 0.456f, 0.406f};
static std::vector<float> INPUT_STD = {0.229f, 0.224f, 0.225f};
static bool NCHW = false;

template <typename T>
void transpose(T *input_data, T *output_data, std::vector<int> input_shape,
               std::vector<int> axis) {
  int old_index = -1;
  int new_index = -1;
  int dim[4] = {0};
  std::vector<int> shape = input_shape;
  for (dim[0] = 0; dim[0] < input_shape[0]; dim[0]++) {
    for (dim[1] = 0; dim[1] < input_shape[1]; dim[1]++) {
      for (dim[2] = 0; dim[2] < input_shape[2]; dim[2]++) {
        for (dim[3] = 0; dim[3] < input_shape[3]; dim[3]++) {
          old_index = dim[0] * shape[1] * shape[2] * shape[3] +
                      dim[1] * shape[2] * shape[3] + dim[2] * shape[3] + dim[3];
          new_index =
              dim[axis[0]] * shape[axis[1]] * shape[axis[2]] * shape[axis[3]] +
              dim[axis[1]] * shape[axis[2]] * shape[axis[3]] +
              dim[axis[2]] * shape[axis[3]] + dim[axis[3]];
          output_data[new_index] = input_data[old_index];
        }
      }
    }
  }
}

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

struct ACCU {
  int top1 = 0;
  int top5 = 0;
};

inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

static std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
  case CV_8U:
    r = "8U";
    break;
  case CV_8S:
    r = "8S";
    break;
  case CV_16U:
    r = "16U";
    break;
  case CV_16S:
    r = "16S";
    break;
  case CV_32S:
    r = "32S";
    break;
  case CV_32F:
    r = "32F";
    break;
  case CV_64F:
    r = "64F";
    break;
  default:
    r = "User";
    break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

static std::vector<std::string> load_image_pathes(const std::string &path) {
  std::ifstream file;
  std::vector<std::string> pathes;
  file.open(path);
  while (file) {
    std::string line;
    std::getline(file, line);
    std::string::size_type pos = line.find(" ");
    if (pos != std::string::npos) {
      line = line.substr(0, pos);
    }
    pathes.push_back(line);
  }
  file.clear();
  file.close();
  return pathes;
}

static std::vector<int> load_labels(const std::string &path) {
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
  Inferencer(std::shared_ptr<PaddlePredictor> p,
             std::vector<int64_t> input_shape)
      : predictor_(p), i_shape_(input_shape) {}

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

  int64_t get_i_shape_0() { return i_shape_[0]; }

  void batch(const cv::Mat &input_image) {
    double preprocess_start_time = get_current_us();

    if (use_first_conv) {
      preprocess(input_image, reinterpret_cast<uint8_t *>(input_data_) +
                                  batch_index_ * hwc_);
    } else {
      preprocess(input_image, INPUT_MEAN, INPUT_STD,
                 reinterpret_cast<float *>(input_data_) + batch_index_ * hwc_);
    }

    double preprocess_end_time = get_current_us();
    double preprocess_time =
        (preprocess_end_time - preprocess_start_time) / 1000.0f;

    preprocess_time_.push_back(preprocess_time);
    printf("Preprocess time: %f ms\n", preprocess_time);

    ++batch_index_;
  }

  virtual std::vector<RESULT> process() = 0;

  float avg_preprocess_time() {
    if (preprocess_time_.size() < 2)
      return 0;
    double total_time = 0;
    for (uint32_t i = 0; i < preprocess_time_.size(); ++i) {
      total_time += preprocess_time_[i];
    }
    return total_time / (preprocess_time_.size() - 1);
  }

  float avg_prediction_time() {
    if (prediction_time_.size() < 2)
      return 0;
    double total_time = 0;
    for (uint32_t i = 0; i < prediction_time_.size(); ++i) {
      total_time += prediction_time_[i];
    }
    return total_time / (prediction_time_.size() - 1);
  }

  float avg_postprocess_time() {
    if (postprocess_time_.size() < 2)
      return 0;
    double total_time = 0;
    for (uint32_t i = 0; i < postprocess_time_.size(); ++i) {
      total_time += postprocess_time_[i];
    }
    return total_time / (postprocess_time_.size() - 1);
  }

protected:
  std::shared_ptr<PaddlePredictor> predictor_;
  std::vector<int64_t> i_shape_;
  std::unique_ptr<Tensor> input_tensor_;
  std::vector<double> preprocess_time_;
  std::vector<double> prediction_time_;
  std::vector<double> postprocess_time_;
  void *input_data_;
  uint32_t batch_index_ = 0;
  int width_, height_;
  int hwc_;
  // virtual void refresh_input() = 0;

private:
  // for first conv
  virtual void preprocess(const cv::Mat &input_image, uint8_t *input_data) {

    cv::Mat resize_image;
    cv::resize(input_image, resize_image, cv::Size(width_, height_), 0, 0);
    if (resize_image.channels() == 1) {
      cv::cvtColor(resize_image, resize_image, cv::COLOR_GRAY2RGB);
    } else {
      cv::cvtColor(resize_image, resize_image, cv::COLOR_BGRA2RGB);
    }
    cv::Mat output_image;
    resize_image.convertTo(output_image, CV_8UC3);
    memcpy(input_data, output_image.data,
           height_ * width_ * 3 * sizeof(uint8_t));
  }

  virtual void preprocess(const cv::Mat &input_image,
                          const std::vector<float> &input_mean,
                          const std::vector<float> &input_std,
                          float *input_data) {
    cv::Mat rgb_img;
    if (input_image.channels() == 1) {
      cv::cvtColor(input_image, rgb_img, cv::COLOR_GRAY2RGB);
    } else {
      cv::cvtColor(input_image, rgb_img, cv::COLOR_BGR2RGB);
    }
    cv::resize(rgb_img, rgb_img, cv::Size(width_, height_));
    cv::Mat imgf;
    rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
    const float *dimg = reinterpret_cast<const float *>(imgf.data);
    const int size_tmp = width_ * height_;
    for (int i = 0; i < width_ * height_; i++) {
      if (NCHW) {
        input_data[i] = (dimg[i * 3 + 0] - input_mean[0]) / input_std[0];
        input_data[i + size_tmp] =
            (dimg[i * 3 + 1] - input_mean[1]) / input_std[1];
        input_data[i + 2 * size_tmp] =
            (dimg[i * 3 + 2] - input_mean[2]) / input_std[2];
      } else {
        input_data[i * 3 + 0] =
            (dimg[i * 3 + 0] - input_mean[0]) / input_std[0];
        input_data[i * 3 + 1] =
            (dimg[i * 3 + 1] - input_mean[1]) / input_std[1];
        input_data[i * 3 + 2] =
            (dimg[i * 3 + 2] - input_mean[2]) / input_std[2];
      }
    }
    imgf.release();
  }

  /* std::vector<RESULT> postprocess(const float *output_data, int64_t
   * output_size, */
  /*                                 cv::Mat &output_image) { */
protected:
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

}; // class Inferencer

// class InputShapeChangeableChecker {
//  public:
//   InputShapeChangeableChecker() {
//
//   }
// }

static struct ACCU get_accu(RESULT objs, int class_id) {
  ACCU accu;
  for (size_t i = 0; i < objs.size(); i++) {
    if (objs[i].class_id == class_id) {
      accu.top5 = 1;
      if (i == 0) {
        accu.top1 = 1;
      }
    }
  }
  return accu;
}
#endif
