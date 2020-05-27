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

/*
#include "paddle_api.h"
#include "paddle_use_kernels.h" // NOLINT
#include "paddle_use_ops.h"     // NOLINT
#include "paddle_use_passes.h"  // NOLINT
#include <gtest/gtest.h>
// #include <arm_neon.h>
#include "string.h"
#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
*/
#include "core.hpp"
#include <gtest/gtest.h>
#include <stdlib.h>

class Inferencer_classification : public Inferencer {
public:
  Inferencer_classification(std::shared_ptr<PaddlePredictor> p)
      : Inferencer(p) {}

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
    printf("repeat: %d, average: %f ms, max: %f ms, min: %f ms\n", REPEAT_COUNT,
           prediction_time, max_time_cost, min_time_cost);

    // Get the data of output tensor and postprocess to output detected objects
    std::unique_ptr<const Tensor> output_tensor(
        std::move(predictor_->GetOutput(0)));
    const float *output_data = output_tensor->mutable_data<float>();
    auto o_shape = output_tensor->shape();
    std::cout << o_shape.size() << std::endl;
    int64_t output_size = o_shape[1];
    std::vector<RESULT> results;
    results.reserve(o_shape[0]);
    /* cv::Mat output_image = input_image.clone(); */
    double postprocess_start_time = get_current_us();
    for (uint32_t i = 0; i < o_shape[0]; ++i) {
      results.emplace_back(
          postprocess(output_data + i * output_size, output_size));
      printf("batch index %u:\n", i);
      for (int j = 0; j < results[i].size(); j++) {
        printf("Top%d %s -class: %d, score %f\n", j,
               results[i][j].class_name.c_str(), results[i][j].class_id,
               results[i][j].score);
      }
    }
    /* std::vector<RESULT> results = */
    /*     postprocess(output_data, output_size, output_image); */
    double postprocess_end_time = get_current_us();
    double postprocess_time =
        (postprocess_end_time - postprocess_start_time) / 1000.0f;

    postprocess_time_.push_back(postprocess_time / i_shape_[0]);
    printf("Postprocess time: %f ms\n\n", postprocess_time);
    prediction_time_.push_back(prediction_time / i_shape_[0]);
    printf("Prediction time: %f ms\n", prediction_time);
    // refresh_input(i_shape_);

    return results;
  }
  void refresh_input(std::vector<int64_t> shape) {
    input_tensor_ = std::move(predictor_->GetInput(0));
    i_shape_ = shape;
    input_tensor_->Resize(i_shape_);
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
};

class classification_test : public testing::Test {
public:
  void test() {
    int compile_times = 1;
    std::vector<ACCU> accus;
    pathes_ = load_image_pathes(data_file_);
    labels_ = load_labels(data_file_);

    if (shape_changed_ != "no_changed") {
      changed_shape_ = {{8, 3, 224, 224}, {4, 3, 448, 448},   {6, 3, 224, 224},
                        {6, 3, 448, 448}, {1, 3, 1096, 1096}, {9, 3, 666, 666},
                        {4, 3, 224, 224}, {2, 3, 448, 224},   {6, 3, 448, 448}};
      infer_->refresh_input(changed_shape_[shape_i_++ % 9]);
    } else {
      infer_->refresh_input({BATCH_SIZE, 3, 224, 224});
    }

    // warm up
    {
      std::cout << "warm up ....." << std::endl;
      std::string image_name = pathes_[0];
      cv::Mat input_image = cv::imread(image_name, -1);
      infer_->warm_up(input_image);
      if (shape_changed_ != "no_changed") {
        infer_->refresh_input(changed_shape_[shape_i_++ % 9]);
      } else {
        infer_->refresh_input({BATCH_SIZE, 3, 224, 224});
      }
      std::cout << "warm up end" << std::endl;
    }

    auto start = get_current_us();
    int index = 0;
    std::vector<int> batch_labels;
    batch_labels.reserve(BATCH_SIZE);
    for (int i = 0; i < pathes_.size() - 1; i++) {
      std::string image_name = pathes_[i];
      std::cout << image_name << std::endl;
      std::string real_path = image_name;
      cv::Mat input_image = cv::imread(real_path, -1);
      printf("process %d th image", i);
      try {
        infer_->batch(input_image);
        batch_labels.emplace_back(labels_[i]);
      } catch (cv::Exception &e) {
        continue;
      }
      if (shape_changed_ != "no_changed") {
        if (index % infer_->get_i_shape_0() == infer_->get_i_shape_0() - 1) {
          std::vector<RESULT> results = infer_->process();
          for (int j = 0; j < results.size(); ++j) {
            accus.push_back(get_accu(results[j], batch_labels[j]));
          }
          batch_labels.clear();
          index = -1;
          infer_->refresh_input(changed_shape_[shape_i_++ % 9]);
        }
      } else {
        if (index % BATCH_SIZE == BATCH_SIZE - 1) {
          std::vector<RESULT> results = infer_->process();
          for (int j = 0; j < results.size(); ++j) {
            accus.push_back(get_accu(results[j], batch_labels[j]));
          }
          batch_labels.clear();
          infer_->refresh_input({BATCH_SIZE, 3, 224, 224});
        }
      }
      ++index;
    }
    auto end = get_current_us();
    double cur_time_cost = (end - start) / 1000.0f;
    float fps = (float)(pathes_.size() - 1) / (cur_time_cost / 1000.0f);
    float mean_top1 = 0;
    float mean_top5 = 0;
    int total_top1 = 0;
    int total_top5 = 0;
    for (size_t i = 0; i < accus.size(); i++) {
      total_top1 += accus[i].top1;
      total_top5 += accus[i].top5;
    }
    mean_top1 = (float)total_top1 / (float)accus.size();
    mean_top5 = (float)total_top5 / (float)accus.size();
    std::cout << "top1 for " << accus.size() << " images: " << mean_top1
              << std::endl;
    std::cout << "top5 for " << accus.size() << " images: " << mean_top5
              << std::endl;
    std::cout << "fps for " << accus.size() << " images: " << fps << std::endl;
    std::cout << "average preprocess time :" << infer_->avg_preprocess_time()
              << std::endl;
    std::cout << "average prediction time :" << infer_->avg_prediction_time()
              << std::endl;
    std::cout << "average postprocess time :" << infer_->avg_postprocess_time()
              << std::endl;
    if (!use_first_conv) {
      EXPECT_GE(mean_top1, min_top1_);
      EXPECT_GE(mean_top5, min_top5_);
    }
    if (shape_changed_ == "shape_changed") {
      if (shape_i_ > changed_shape_.size()) {
        compile_times += changed_shape_.size();
      } else {
        compile_times += (shape_i_ - 1);
      }
    }
    std::cout << "compile_times: " << compile_times << std::endl;
  }

protected:
  std::unique_ptr<Inferencer_classification> infer_;
  CxxConfig config_;
  std::shared_ptr<PaddlePredictor> predictor_;
  std::vector<std::string> pathes_;
  std::vector<int> labels_;
  std::vector<Place> valid_places_;
  std::vector<std::vector<int64_t>> changed_shape_;
  float min_top1_;
  float min_top5_;
  std::string data_file_;
  int shape_i_;
  std::string shape_changed_;
  virtual void SetUp() {
    shape_i_ = 0;
    shape_changed_ = "no_changed";
    valid_places_ = {Place{TARGET(kX86), PRECISION(kFloat)},
                     Place{TARGET(kX86), PRECISION(kFP16)},
                     Place{TARGET(kMLU), PRECISION(kFP16), DATALAYOUT(kNHWC)}};
    config_.set_valid_places(valid_places_);
    config_.set_mlu_core_version(MLUCoreVersion::MLU_270);
    config_.set_mlu_core_number(16);
    config_.set_mlu_input_layout(DATALAYOUT(kNHWC));
  }
  virtual void TearDown() {}
};

TEST_F(classification_test, resnet50) {
  std::vector<std::string> shape_changed_choices = {"no_changed",
                                                    "shape_changed"};
  use_first_conv = false;
  // The following parameters are variable
  BATCH_SIZE = 1;
  data_file_ = "./filelist";
  config_.set_model_dir("/opt/share/paddle_model/ResNet50_quant/");

  for (auto choice : shape_changed_choices) {
    shape_changed_ = choice;
    config_.set_mlu_use_first_conv(use_first_conv);
    predictor_ = CreatePaddlePredictor<CxxConfig>(config_);
    infer_.reset(new Inferencer_classification(predictor_));
    if (shape_changed_ == "shape_changed") {
      min_top1_ = 0.65;
      min_top5_ = 0.85;
    } else {
      min_top1_ = 0.7;
      min_top5_ = 0.9;
    }
    test();
  }
}

TEST_F(classification_test, resnet101) {
  NCHW = false;
  // The following parameters are variable
  BATCH_SIZE = 1;
  config_.set_model_dir("/opt/share/paddle_model/resnet101_KL_quant/");
  data_file_ = "./filelist";

  predictor_ = CreatePaddlePredictor<CxxConfig>(config_);
  infer_.reset(new Inferencer_classification(predictor_));
  min_top1_ = 0.7;
  min_top5_ = 0.9;
  test();
}
TEST_F(classification_test, mobilenetv2_KL) {
  NCHW = false;
  // The following parameters are variable
  BATCH_SIZE = 1;
  data_file_ = "./filelist";
  config_.set_model_dir("/opt/share/paddle_model/mobilenetv2_KL_quant/");

  predictor_ = CreatePaddlePredictor<CxxConfig>(config_);
  infer_.reset(new Inferencer_classification(predictor_));
  min_top1_ = 0.65;
  min_top5_ = 0.85;
  test();
}
TEST_F(classification_test, googlenet_KL) {
  NCHW = false;
  // The following parameters are variable
  BATCH_SIZE = 1;
  config_.set_model_dir("/opt/share/paddle_model/googlenet_KL_quant/");
  data_file_ = "./filelist";

  predictor_ = CreatePaddlePredictor<CxxConfig>(config_);
  infer_.reset(new Inferencer_classification(predictor_));
  min_top1_ = 0.65;
  min_top5_ = 0.85;
  test();
}
TEST_F(classification_test, MobileNetV1) {
  NCHW = false;
  // The following parameters are variable
  BATCH_SIZE = 1;
  config_.set_model_dir("/opt/share/paddle_model/MobileNetV1_quant/");
  data_file_ = "./filelist";

  predictor_ = CreatePaddlePredictor<CxxConfig>(config_);
  infer_.reset(new Inferencer_classification(predictor_));
  min_top1_ = 0.65;
  min_top5_ = 0.85;
  test();
}

TEST_F(classification_test, resnet50_extra) {
  NCHW = false;
  use_first_conv = false;
  std::vector<std::vector<Place>> places = {
      {Place{TARGET(kX86), PRECISION(kFloat)},
       Place{TARGET(kX86), PRECISION(kFP16)},
       Place{TARGET(kMLU), PRECISION(kInt8), DATALAYOUT(kNHWC)},
       Place{TARGET(kMLU), PRECISION(kFloat), DATALAYOUT(kNHWC)}},
      {Place{TARGET(kX86), PRECISION(kFloat)},
       Place{TARGET(kX86), PRECISION(kFP16)},
       Place{TARGET(kMLU), PRECISION(kInt8), DATALAYOUT(kNHWC)},
       Place{TARGET(kMLU), PRECISION(kFP16), DATALAYOUT(kNHWC)}}};

  // The following parameters are variable
  BATCH_SIZE = 1;
  data_file_ = "./filelist";
  config_.set_model_dir("/opt/share/paddle_model/ResNet50_quant/");

  for (auto first_conv : {false, true}) {
    for (auto layout : {false, true}) {
      for (auto v_places : places) {
        use_first_conv = first_conv;
        NCHW = layout;
        if (NCHW) {
          config_.set_mlu_input_layout(DATALAYOUT(kNCHW));
        } else {
          config_.set_mlu_input_layout(DATALAYOUT(kNHWC));
        }
        config_.set_valid_places(v_places);
        config_.set_mlu_use_first_conv(use_first_conv);
        if (use_first_conv) {
          INPUT_MEAN = {124, 117, 104};
          INPUT_STD = {59, 57, 57};
          std::vector<float> mean_vec = INPUT_MEAN;
          std::vector<float> std_vec = INPUT_STD;
          config_.set_mlu_first_conv_mean(mean_vec);
          config_.set_mlu_first_conv_std(std_vec);
        }

        predictor_ = CreatePaddlePredictor<CxxConfig>(config_);
        infer_.reset(new Inferencer_classification(predictor_));
        min_top1_ = 0.7;
        min_top5_ = 0.9;
        test();
      }
    }
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
