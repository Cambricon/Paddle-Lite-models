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

class Inferencer_classification : public Inferencer {
public:
  Inferencer_classification(std::shared_ptr<PaddlePredictor> p,
                            std::vector<int64_t> input_shape)
      : Inferencer(p, input_shape) {
    refresh_input();
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

    refresh_input();
    return results;
  }
  void refresh_input() {
    input_tensor_ = std::move(predictor_->GetInput(0));
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

// int main(int argc, char **argv) {
// TEST(paddle, classification) {
// std::string input_image_pathes = "/home/zhaoying/imagenet/val_5000.txt";
// std::string input_image_pathes = "/home/zhaoying/imagenet/val_1000.txt";
// std::string input_image_pathes = "/home/zhaoying/imagenet/val_100.txt";
// std::string input_image_pathes = "/projs/systools/zhangshijin/val.txt";
// std::string input_image_pathes = "/home/zhangmingwei/ws/filelist";
void test_classification(std::string model_dir) {
  // std::string model_dir = "/home/dingminghui/paddle/data/ResNet50_quant/";
  std::string input_image_pathes = "./filelist";
  std::string label_path = input_image_pathes;
  std::cout << "model_path:  " << model_dir << std::endl;
  std::cout << "image and label path:  " << input_image_pathes << std::endl;

  // Load Labels
  std::vector<int> labels = load_labels(label_path);

  // Set MobileConfig
  CxxConfig config;
  config.set_model_dir(model_dir);
  std::vector<Place> valid_places{
      Place{TARGET(kX86), PRECISION(kFloat)},
      Place{TARGET(kMLU), PRECISION(kFP16), DATALAYOUT(kNHWC)}
      /* Place{TARGET(kMLU), PRECISION(kFloat), DATALAYOUT(kNHWC)} */
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

  Inferencer_classification infer(predictor, {BATCH_SIZE, 3, 224, 224});

  std::vector<ACCU> accus;
  std::vector<std::string> pathes = load_image_pathes(input_image_pathes);

  // warm up
  {
    std::cout << "warm up ....." << std::endl;
    std::string image_name = pathes[0];
    cv::Mat input_image = cv::imread(image_name, -1);
    infer.warm_up(input_image);
    std::cout << "warm up end" << std::endl;
    // std::string real_path = "/home/zhaoying/imagenet/" + image_name;
    // std::string real_path = "/opt/shared/beta/models_and_data/imagenet/" +
    // image_name; cv::Mat input_image = cv::imread(real_path, 1);
    // process(input_image, predictor);
  }

  auto start = get_current_us();
  int index = 0;
  std::vector<int> batch_labels;
  batch_labels.reserve(BATCH_SIZE);
  for (int i = 0; i < pathes.size() - 1; i++) {
    std::string image_name = pathes[i];
    std::cout << image_name << std::endl;
    // std::string real_path = "/home/zhaoying/imagenet/" + image_name;
    std::string real_path = image_name;
    cv::Mat input_image = cv::imread(real_path, -1);
    // cv::imshow("aaa", input_image);
    // cv::waitKey(0);
    printf("process %d th image", i);
    try {
      infer.batch(input_image);
      batch_labels.emplace_back(labels[i]);
    } catch (cv::Exception &e) {
      continue;
    }
    if (index % BATCH_SIZE == BATCH_SIZE - 1) {
      std::vector<RESULT> results = infer.process();
      for (int j = 0; j < results.size(); ++j) {
        accus.push_back(get_accu(results[j], batch_labels[j]));
      }
      batch_labels.clear();
    }
    ++index;
  }
  auto end = get_current_us();
  double cur_time_cost = (end - start) / 1000.0f;
  float fps = (float)(pathes.size() - 1) / (cur_time_cost / 1000.0f);
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
  std::cout << "average preprocess time :" << infer.avg_preprocess_time()
            << std::endl;
  std::cout << "average prediction time :" << infer.avg_prediction_time()
            << std::endl;
  std::cout << "average postprocess time :" << infer.avg_postprocess_time()
            << std::endl;
  EXPECT_GT(mean_top1, 0.7);
  EXPECT_GT(mean_top5, 0.9);
}

TEST(paddle, classification_resnet50) {
  test_classification("/home/dingminghui/paddle/data/ResNet50_quant/");
}
TEST(paddle, classification_resnet101) {
  test_classification("/home/jiaopu/model_0515/resnet101_KL_quant/");
}
TEST(paddle, classification_mobilenetV2_KL) {
  test_classification("/home/jiaopu/model_0515/mobilenetv2_KL_quant/");
}
TEST(paddle, classification_MobileNetV1) {
  test_classification("/home/jiaopu/model_0515/MobileNetV1_quant/");
}
TEST(paddle, classification_googlenet_KL) {
  test_classification("/home/jiaopu/model_0515/googlenet_KL_quant/");
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
