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

#include <taskflow/taskflow.hpp>  // Cpp-Taskflow is header-only and  must be in first line
#include "paddle_api.h"
#include "paddle_use_kernels.h"  // NOLINT
#include "paddle_use_ops.h"      // NOLINT
#include "paddle_use_passes.h"   // NOLINT
// #include <arm_neon.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <vector>

#include "string.h"

using namespace paddle::lite_api;  // NOLINT

bool use_first_conv = false;

int WARMUP_COUNT = 10;
int REPEAT_COUNT = 1;
uint32_t BATCH_SIZE = 1;
const int CPU_THREAD_NUM = 2;
const PowerMode CPU_POWER_MODE = PowerMode::LITE_POWER_HIGH;
std::vector<float> INPUT_MEAN = {0.485f, 0.456f, 0.406f};
std::vector<float> INPUT_STD = {0.229f, 0.224f, 0.225f};

struct Object {
  std::string class_name;
  int class_id;
  float score;
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

std::vector<std::string> load_image_pathes(const std::string &path) {
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
#define DEFINE_INFERENCER(_INFERENCER__)                                      \
  class _INFERENCER__ {                                                       \
   public:                                                                    \
    _INFERENCER__(std::shared_ptr<PaddlePredictor> p,                         \
                  std::vector<int64_t> input_shape)                           \
        : predictor_(p), i_shape_(input_shape) {                              \
      refresh_input();                                                        \
    }                                                                         \
                                                                              \
    void warm_up(cv::Mat &input_image) {                                      \
      for (int i = 0; i < WARMUP_COUNT; i++) {                                \
        for (uint32_t i = 0; i < i_shape_[0]; ++i) {                          \
          batch(input_image);                                                 \
        }                                                                     \
        process();                                                            \
      }                                                                       \
      printf("warm up count: %d\n", WARMUP_COUNT);                            \
      preprocess_time_.clear();                                               \
      prediction_time_.clear();                                               \
      postprocess_time_.clear();                                              \
    }                                                                         \
                                                                              \
    void batch(const cv::Mat &input_image) {                                  \
      double preprocess_start_time = get_current_us();                        \
                                                                              \
      if (use_first_conv) {                                                   \
        preprocess(                                                           \
            input_image,                                                      \
            reinterpret_cast<uint8_t *>(input_data_) + batch_index_ * hwc_);  \
      } else {                                                                \
        preprocess(                                                           \
            input_image,                                                      \
            INPUT_MEAN,                                                       \
            INPUT_STD,                                                        \
            reinterpret_cast<float *>(input_data_) + batch_index_ * hwc_);    \
      }                                                                       \
                                                                              \
      double preprocess_end_time = get_current_us();                          \
      double preprocess_time =                                                \
          (preprocess_end_time - preprocess_start_time) / 1000.0f;            \
                                                                              \
      preprocess_time_.push_back(preprocess_time);                            \
      printf("Preprocess time: %f ms\n", preprocess_time);                    \
                                                                              \
      ++batch_index_;                                                         \
    }                                                                         \
                                                                              \
    std::vector<RESULT> process() {                                           \
      double prediction_time;                                                 \
      double max_time_cost = 0.0f;                                            \
      double min_time_cost = std::numeric_limits<float>::max();               \
      double total_time_cost = 0.0f;                                          \
      for (int i = 0; i < REPEAT_COUNT; i++) {                                \
        auto start = get_current_us();                                        \
        predictor_->Run();                                                    \
        auto end = get_current_us();                                          \
        double cur_time_cost = (end - start) / 1000.0f;                       \
        if (cur_time_cost > max_time_cost) {                                  \
          max_time_cost = cur_time_cost;                                      \
        }                                                                     \
        if (cur_time_cost < min_time_cost) {                                  \
          min_time_cost = cur_time_cost;                                      \
        }                                                                     \
        total_time_cost += cur_time_cost;                                     \
        printf("iter %d cost: %f ms\n", i, cur_time_cost);                    \
      }                                                                       \
      prediction_time = total_time_cost / REPEAT_COUNT;                       \
      printf("repeat: %d, average: %f ms, max: %f ms, min: %f ms\n",          \
             REPEAT_COUNT,                                                    \
             prediction_time,                                                 \
             max_time_cost,                                                   \
             min_time_cost);                                                  \
                                                                              \
      std::unique_ptr<const Tensor> output_tensor(                            \
          std::move(predictor_->GetOutput(0)));                               \
      const float *output_data = output_tensor->mutable_data<float>();        \
      auto o_shape = output_tensor->shape();                                  \
      std::cout << o_shape.size() << std::endl;                               \
      int64_t output_size = o_shape[1];                                       \
      std::vector<RESULT> results;                                            \
      results.reserve(o_shape[0]);                                            \
      /* cv::Mat output_image = input_image.clone(); */                       \
      double postprocess_start_time = get_current_us();                       \
      for (uint32_t i = 0; i < o_shape[0]; ++i) {                             \
        results.emplace_back(                                                 \
            postprocess(output_data + i * output_size, output_size));         \
        printf("batch index %u:\n", i);                                       \
        for (int j = 0; j < results[i].size(); j++) {                         \
          printf("Top%d %s -class: %d, score %f\n",                           \
                 j,                                                           \
                 results[i][j].class_name.c_str(),                            \
                 results[i][j].class_id,                                      \
                 results[i][j].score);                                        \
        }                                                                     \
      }                                                                       \
      double postprocess_end_time = get_current_us();                         \
      double postprocess_time =                                               \
          (postprocess_end_time - postprocess_start_time) / 1000.0f;          \
                                                                              \
      postprocess_time_.push_back(postprocess_time / i_shape_[0]);            \
      prediction_time_.push_back(prediction_time / i_shape_[0]);              \
      printf("Prediction time: %f ms\n", prediction_time);                    \
      printf("Postprocess time: %f ms\n\n", postprocess_time);                \
                                                                              \
      refresh_input();                                                        \
      return results;                                                         \
    }                                                                         \
                                                                              \
    float avg_preprocess_time() {                                             \
      double total_time = 0;                                                  \
      for (uint32_t i = 0; i < preprocess_time_.size(); ++i) {                \
        total_time += preprocess_time_[i];                                    \
      }                                                                       \
      return total_time / (preprocess_time_.size());                          \
    }                                                                         \
                                                                              \
    float avg_prediction_time() {                                             \
      double total_time = 0;                                                  \
      for (uint32_t i = 0; i < prediction_time_.size(); ++i) {                \
        total_time += prediction_time_[i];                                    \
      }                                                                       \
      return total_time / (prediction_time_.size());                          \
    }                                                                         \
                                                                              \
    float avg_postprocess_time() {                                            \
      double total_time = 0;                                                  \
      for (uint32_t i = 0; i < postprocess_time_.size(); ++i) {               \
        total_time += postprocess_time_[i];                                   \
      }                                                                       \
      return total_time / (postprocess_time_.size());                         \
    }                                                                         \
                                                                              \
   private:                                                                   \
    std::shared_ptr<PaddlePredictor> predictor_;                              \
    std::vector<int64_t> i_shape_;                                            \
    std::unique_ptr<Tensor> input_tensor_;                                    \
    std::vector<double> preprocess_time_;                                     \
    std::vector<double> prediction_time_;                                     \
    std::vector<double> postprocess_time_;                                    \
    void *input_data_;                                                        \
    uint32_t batch_index_ = 0;                                                \
    int width_, height_;                                                      \
    int hwc_;                                                                 \
                                                                              \
    void refresh_input() {                                                    \
      input_tensor_ = std::move(predictor_->GetInput(0));                     \
      input_tensor_->Resize(i_shape_);                                        \
      width_ = i_shape_[3];                                                   \
      height_ = i_shape_[2];                                                  \
      hwc_ = i_shape_[1] * i_shape_[2] * i_shape_[3];                         \
      if (use_first_conv) {                                                   \
        input_data_ = input_tensor_->mutable_data<int8_t>();                  \
      } else {                                                                \
        input_data_ = input_tensor_->mutable_data<float>();                   \
      }                                                                       \
      batch_index_ = 0;                                                       \
    }                                                                         \
    void preprocess(const cv::Mat &input_image, uint8_t *input_data) {        \
      cv::Mat resize_image;                                                   \
      cv::resize(input_image, resize_image, cv::Size(width_, height_), 0, 0); \
      if (resize_image.channels() == 1) {                                     \
        cv::cvtColor(resize_image, resize_image, cv::COLOR_GRAY2RGB);         \
      } else {                                                                \
        cv::cvtColor(resize_image, resize_image, cv::COLOR_BGRA2RGB);         \
      }                                                                       \
      cv::Mat output_image;                                                   \
      resize_image.convertTo(output_image, CV_8UC3);                          \
      memcpy(input_data,                                                      \
             output_image.data,                                               \
             height_ *width_ * 3 * sizeof(uint8_t));                          \
    }                                                                         \
                                                                              \
    void preprocess(const cv::Mat &input_image,                               \
                    const std::vector<float> &input_mean,                     \
                    const std::vector<float> &input_std,                      \
                    float *input_data) {                                      \
      cv::Mat rgb_img;                                                        \
      if (input_image.channels() == 1) {                                      \
        cv::cvtColor(input_image, rgb_img, cv::COLOR_GRAY2RGB);               \
      } else {                                                                \
        cv::cvtColor(input_image, rgb_img, cv::COLOR_BGR2RGB);                \
      }                                                                       \
      cv::resize(rgb_img, rgb_img, cv::Size(width_, height_));                \
      cv::Mat imgf;                                                           \
      rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);                           \
      const float *dimg = reinterpret_cast<const float *>(imgf.data);         \
      /* const int size_tmp = input_width * input_height; */                  \
      for (int i = 0; i < width_ * height_; i++) {                            \
        input_data[i * 3 + 0] =                                               \
            (dimg[i * 3 + 0] - input_mean[0]) / input_std[0];                 \
        input_data[i * 3 + 1] =                                               \
            (dimg[i * 3 + 1] - input_mean[1]) / input_std[1];                 \
        input_data[i * 3 + 2] =                                               \
            (dimg[i * 3 + 2] - input_mean[2]) / input_std[2];                 \
      }                                                                       \
      imgf.release();                                                         \
    }                                                                         \
                                                                              \
    RESULT postprocess(const float *output_data, int64_t output_size) {       \
      const int TOPK = 5;                                                     \
      int max_indices[TOPK];                                                  \
      double max_scores[TOPK];                                                \
      for (int i = 0; i < TOPK; i++) {                                        \
        max_indices[i] = 0;                                                   \
        max_scores[i] = 0;                                                    \
      }                                                                       \
      for (int i = 0; i < output_size; i++) {                                 \
        float score = output_data[i];                                         \
        int index = i;                                                        \
        for (int j = 0; j < TOPK; j++) {                                      \
          if (score > max_scores[j]) {                                        \
            index += max_indices[j];                                          \
            max_indices[j] = index - max_indices[j];                          \
            index -= max_indices[j];                                          \
            score += max_scores[j];                                           \
            max_scores[j] = score - max_scores[j];                            \
            score -= max_scores[j];                                           \
          }                                                                   \
        }                                                                     \
      }                                                                       \
      std::vector<Object> objs(TOPK);                                         \
      for (int i = 0; i < objs.size(); i++) {                                 \
        objs[i].class_id = max_indices[i];                                    \
        objs[i].score = max_scores[i];                                        \
      }                                                                       \
      return objs;                                                            \
    }                                                                         \
  };

struct ACCU get_accu(RESULT objs, int class_id) {
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
// CRREATE_TASK(predictor3, Inferencer3, infer3, 3, task_3)
#define CRREATE_TASK(                                                      \
    PREDICTOR, INFERENCER, INFERENCER_NAME, TASK_ID, TASK_NAME)            \
  std::shared_ptr<PaddlePredictor> PREDICTOR =                             \
      CreatePaddlePredictor<CxxConfig>(config);                            \
  INFERENCER INFERENCER_NAME(PREDICTOR, {BATCH_SIZE, 3, 224, 224});        \
  std::cout << "warm up " << TASK_ID << "....." << std::endl;              \
  INFERENCER_NAME.warm_up(input_images[TASK_ID][0]);                       \
  std::cout << "warm up " << TASK_ID << "3 end" << std::endl;              \
  tf::Task TASK_NAME = taskflow.emplace([&]() {                            \
    int index = 0;                                                         \
    std::vector<int> batch_labels;                                         \
    batch_labels.reserve(BATCH_SIZE);                                      \
    for (int i = 0; i < task_image_num; i++) {                             \
      try {                                                                \
        INFERENCER_NAME.batch(input_images[TASK_ID][i]);                   \
        batch_labels.emplace_back(labels[task_image_num * TASK_ID + i]);   \
      } catch (cv::Exception & e) {                                        \
        continue;                                                          \
      }                                                                    \
      if (index % BATCH_SIZE == BATCH_SIZE - 1) {                          \
        std::vector<RESULT> results = INFERENCER_NAME.process();           \
        for (int j = 0; j < results.size(); ++j) {                         \
          accus[TASK_ID].push_back(get_accu(results[j], batch_labels[j])); \
        }                                                                  \
        batch_labels.clear();                                              \
      }                                                                    \
      ++index;                                                             \
    }                                                                      \
  });

#define ADD_TIME(INFER_NAME)                               \
  avg_preprocess_time += INFER_NAME.avg_preprocess_time(); \
  avg_prediction_time += INFER_NAME.avg_prediction_time(); \
  avg_postprocess_time += INFER_NAME.avg_postprocess_time();

#define SAVE_OUTPUT_CSV(OUTPUT_DEVICE)                                     \
  OUTPUT_DEVICE                                                            \
      << "place type, batch size, thread num, core num, top1, top5, fps, " \
         "average preprocess time, average prediction time, average "      \
         "postprocess time, average latency time, image_num"               \
      << std::endl;                                                        \
  OUTPUT_DEVICE << strs[type];                                             \
  OUTPUT_DEVICE << " , " << BATCH_SIZE;                                    \
  OUTPUT_DEVICE << " , " << task_num;                                      \
  OUTPUT_DEVICE << " , " << core_num;                                      \
  OUTPUT_DEVICE << " , " << mean_top1;                                     \
  OUTPUT_DEVICE << " , " << mean_top5;                                     \
  OUTPUT_DEVICE << " , " << fps;                                           \
  OUTPUT_DEVICE << " , " << avg_preprocess_time;                           \
  OUTPUT_DEVICE << " , " << avg_prediction_time;                           \
  OUTPUT_DEVICE << " , " << avg_postprocess_time;                          \
  OUTPUT_DEVICE << " , " << total_latency;                                 \
  OUTPUT_DEVICE << " , " << image_num << std::endl;

#define SAVE_OUTPUT(OUTPUT_DEVICE)                                             \
  OUTPUT_DEVICE << "place type " << strs[type] << std::endl;                   \
  OUTPUT_DEVICE << "batch size " << BATCH_SIZE << std::endl;                   \
  OUTPUT_DEVICE << "thread num " << task_num << std::endl;                     \
  OUTPUT_DEVICE << "core num " << core_num << std::endl;                       \
  OUTPUT_DEVICE << "top1 for " << image_num << " images: " << mean_top1        \
                << std::endl;                                                  \
  OUTPUT_DEVICE << "top5 for " << image_num << " images: " << mean_top5        \
                << std::endl;                                                  \
  OUTPUT_DEVICE << "fps for " << image_num << " images: " << fps << std::endl; \
  OUTPUT_DEVICE << "average preprocess time :" << avg_preprocess_time          \
                << std::endl;                                                  \
  OUTPUT_DEVICE << "average prediction time :" << avg_prediction_time          \
                << std::endl;                                                  \
  OUTPUT_DEVICE << "average postprocess time :" << avg_postprocess_time        \
                << std::endl;                                                  \
  OUTPUT_DEVICE << "average latency time :" << total_latency << std::endl;

#define LINK_TASK(TASK_NAME)    \
  task_time.precede(TASK_NAME); \
  TASK_NAME.precede(task_final);

// modify here when add a task
#ifdef TASK_1
DEFINE_INFERENCER(Inferencer0);
#endif
#ifdef TASK_2
DEFINE_INFERENCER(Inferencer1);
#endif
#ifdef TASK_3
DEFINE_INFERENCER(Inferencer2);
#endif
#ifdef TASK_4
DEFINE_INFERENCER(Inferencer3);
#endif
#ifdef TASK_5
DEFINE_INFERENCER(Inferencer4);
#endif
#ifdef TASK_6
DEFINE_INFERENCER(Inferencer5);
#endif
#ifdef TASK_7
DEFINE_INFERENCER(Inferencer6);
#endif
#ifdef TASK_8
DEFINE_INFERENCER(Inferencer7);
#endif
#ifdef TASK_9
DEFINE_INFERENCER(Inferencer8);
#endif
#ifdef TASK_10
DEFINE_INFERENCER(Inferencer9);
#endif
#ifdef TASK_11
DEFINE_INFERENCER(Inferencer10);
#endif
#ifdef TASK_12
DEFINE_INFERENCER(Inferencer11);
#endif
#ifdef TASK_13
DEFINE_INFERENCER(Inferencer12);
#endif
#ifdef TASK_14
DEFINE_INFERENCER(Inferencer13);
#endif
#ifdef TASK_15
DEFINE_INFERENCER(Inferencer14);
#endif
#ifdef TASK_16
DEFINE_INFERENCER(Inferencer15);
#endif
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
  int core_num = std::atoi(argv[2]);
  int task_num = std::atoi(argv[3]);
  // 0 int8 1:fp16 2:fp32
  int type = std::atoi(argv[4]);
  std::string model_dir = argv[5];
  std::string input_image_pathes = argv[6];
  std::string mlu_type = "MLU270";
  mlu_type = argv[7];
  std::string label_path = input_image_pathes;
  // Load Labels
  std::vector<int> labels = load_labels(label_path);
  // Set MobileConfig
  CxxConfig config;
  config.set_model_dir(model_dir);
  std::vector<Place> valid_places;
  valid_places.push_back(Place{TARGET(kX86), PRECISION(kFloat)});
  if (type == 0) {
    valid_places.push_back(
        Place{TARGET(kMLU), PRECISION(kFP16), DATALAYOUT(kNHWC)});
    valid_places.push_back(
        Place{TARGET(kMLU), PRECISION(kInt8), DATALAYOUT(kNHWC)});
    use_first_conv = true;
  } else if (type == 1) {
    valid_places.push_back(
        Place{TARGET(kMLU), PRECISION(kFP16), DATALAYOUT(kNHWC)});
  } else if (type == 2) {
    valid_places.push_back(
        Place{TARGET(kMLU), PRECISION(kFloat), DATALAYOUT(kNHWC)});
  }
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
  if(mlu_type=="MLU270")
     config.set_mlu_core_version(MLUCoreVersion::MLU_270);
  else
     config.set_mlu_core_version(MLUCoreVersion::MLU_220);
  config.set_mlu_core_number(core_num);
  config.set_mlu_input_layout(DATALAYOUT(kNHWC));
  std::vector<std::vector<ACCU>> accus(task_num);
  std::vector<std::string> pathes = load_image_pathes(input_image_pathes);
  int img_file_num = (pathes.size() - 1);
  int image_num = img_file_num - (img_file_num) % (task_num * BATCH_SIZE);
  int task_image_num = image_num / task_num;
  std::vector<std::vector<cv::Mat>> input_images(task_num);
  // read images
  for (int i = 0, j = 0; i < task_num; i++) {
    for (; j < image_num; j++) {
      cv::Mat input_image = cv::imread(pathes[j], -1);
      input_images[i].push_back(input_image);
      if ((j + 1) % task_image_num == 0) {
        j++;
        break;
      }
    }
  }
  int64_t start = 0;
  tf::Executor executor;
  tf::Taskflow taskflow;
  tf::Task task_time = taskflow.emplace([&]() { start = get_current_us(); });

// modify here when add a task
#ifdef TASK_1
  CRREATE_TASK(predictor0, Inferencer0, infer0, 0, task_0);
#endif
#ifdef TASK_2
  CRREATE_TASK(predictor1, Inferencer1, infer1, 1, task_1);
#endif
#ifdef TASK_3
  CRREATE_TASK(predictor2, Inferencer2, infer2, 2, task_2);
#endif
#ifdef TASK_4
  CRREATE_TASK(predictor3, Inferencer3, infer3, 3, task_3);
#endif
#ifdef TASK_5
  CRREATE_TASK(predictor4, Inferencer4, infer4, 4, task_4);
#endif
#ifdef TASK_6
  CRREATE_TASK(predictor5, Inferencer5, infer5, 5, task_5);
#endif
#ifdef TASK_7
  CRREATE_TASK(predictor6, Inferencer6, infer6, 6, task_6);
#endif
#ifdef TASK_8
  CRREATE_TASK(predictor7, Inferencer7, infer7, 7, task_7);
#endif
#ifdef TASK_9
  CRREATE_TASK(predictor8, Inferencer8, infer8, 8, task_8);
#endif
#ifdef TASK_10
  CRREATE_TASK(predictor9, Inferencer9, infer9, 9, task_9);
#endif
#ifdef TASK_11
  CRREATE_TASK(predictor10, Inferencer10, infer10, 10, task_10);
#endif
#ifdef TASK_12
  CRREATE_TASK(predictor11, Inferencer11, infer11, 11, task_11);
#endif
#ifdef TASK_13
  CRREATE_TASK(predictor12, Inferencer12, infer12, 12, task_12);
#endif
#ifdef TASK_14
  CRREATE_TASK(predictor13, Inferencer13, infer13, 13, task_13);
#endif
#ifdef TASK_15
  CRREATE_TASK(predictor14, Inferencer14, infer14, 14, task_14);
#endif
#ifdef TASK_16
  CRREATE_TASK(predictor15, Inferencer15, infer15, 15, task_15);
#endif
  tf::Task task_final = taskflow.emplace([&]() {
    auto end = get_current_us();
    double cur_time_cost = (end - start) / 1000.0f;
    float fps = (float)(image_num) / (cur_time_cost / 1000.0f);
    float mean_top1 = 0;
    float mean_top5 = 0;
    int total_top1 = 0;
    int total_top5 = 0;
    for (auto items : accus) {
      for (auto ac : items) {
        total_top1 += ac.top1;
        total_top5 += ac.top5;
      }
    }
    mean_top1 = (float)total_top1 / (float)(image_num);
    mean_top5 = (float)total_top5 / (float)(image_num);
    float avg_preprocess_time = 0;
    float avg_prediction_time = 0;
    float avg_postprocess_time = 0;
    float total_latency = 0;
    std::vector<std::string> strs{"INT8", "FP16", "FP32"};
    if (type > strs.size()) {
      type = 0;
    }
// modify here when add a task
#ifdef TASK_1
    ADD_TIME(infer0);
#endif
#ifdef TASK_2
    ADD_TIME(infer1);
#endif
#ifdef TASK_3
    ADD_TIME(infer2);
#endif
#ifdef TASK_4
    ADD_TIME(infer3);
#endif
#ifdef TASK_5
    ADD_TIME(infer4);
#endif
#ifdef TASK_6
    ADD_TIME(infer5);
#endif
#ifdef TASK_7
    ADD_TIME(infer6);
#endif
#ifdef TASK_8
    ADD_TIME(infer7);
#endif
#ifdef TASK_9
    ADD_TIME(infer8);
#endif
#ifdef TASK_10
    ADD_TIME(infer9);
#endif
#ifdef TASK_11
    ADD_TIME(infer10);
#endif
#ifdef TASK_12
    ADD_TIME(infer11);
#endif
#ifdef TASK_13
    ADD_TIME(infer12);
#endif
#ifdef TASK_14
    ADD_TIME(infer13);
#endif
#ifdef TASK_15
    ADD_TIME(infer14);
#endif
#ifdef TASK_16
    ADD_TIME(infer15);
#endif

    avg_preprocess_time /= task_num;
    avg_prediction_time /= task_num;
    avg_postprocess_time /= task_num;
    total_latency += avg_preprocess_time;
    total_latency += avg_prediction_time;
    total_latency += avg_postprocess_time;
    SAVE_OUTPUT(std::cout);
    std::ofstream ofs("Resnets50_Betchmark.txt", std::ios::app);
    if (!ofs.is_open()) {
      std::cout << "open result file failed";
    }
    SAVE_OUTPUT(ofs);
    ofs.close();
    std::ofstream ofs_csv("Resnets50_Betchmark.csv", std::ios::app);
    if (!ofs_csv.is_open()) {
      std::cout << "open result file failed";
    }
    SAVE_OUTPUT_CSV(ofs_csv);
    ofs_csv.close();
  });

// modify here when add a task
#ifdef TASK_1
  LINK_TASK(task_0);
#endif
#ifdef TASK_2
  LINK_TASK(task_1);
#endif
#ifdef TASK_3
  LINK_TASK(task_2);
#endif
#ifdef TASK_4
  LINK_TASK(task_3);
#endif
#ifdef TASK_5
  LINK_TASK(task_4);
#endif
#ifdef TASK_6
  LINK_TASK(task_5);
#endif
#ifdef TASK_7
  LINK_TASK(task_6);
#endif
#ifdef TASK_8
  LINK_TASK(task_7);
#endif
#ifdef TASK_9
  LINK_TASK(task_8);
#endif
#ifdef TASK_10
  LINK_TASK(task_9);
#endif
#ifdef TASK_11
  LINK_TASK(task_10);
#endif
#ifdef TASK_12
  LINK_TASK(task_11);
#endif
#ifdef TASK_13
  LINK_TASK(task_12);
#endif
#ifdef TASK_14
  LINK_TASK(task_13);
#endif
#ifdef TASK_15
  LINK_TASK(task_14);
#endif
#ifdef TASK_16
  LINK_TASK(task_15);
#endif
  executor.run(taskflow).wait();
  return 0;
}
