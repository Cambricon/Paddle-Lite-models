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
#include <gtest/gtest.h>
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
*/
#include <gtest/gtest.h>
#include "core.hpp"


const int CPU_THREAD_NUM = 2;
const PowerMode CPU_POWER_MODE =
    PowerMode::LITE_POWER_HIGH;

// int main(int argc, char **argv) {
TEST(paddle, classification) {
  // std::string input_image_pathes = "/home/zhaoying/imagenet/val_5000.txt";
  // std::string input_image_pathes = "/home/zhaoying/imagenet/val_1000.txt";
  // std::string input_image_pathes = "/home/zhaoying/imagenet/val_100.txt";
  //std::string input_image_pathes = "/projs/systools/zhangshijin/val.txt";
  //std::string input_image_pathes = "/home/zhangmingwei/ws/filelist";
  std::string model_dir = "/home/dingminghui/paddle/data/ResNet50_quant/";
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

  Inferencer infer(predictor, {BATCH_SIZE, 3, 224, 224});

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
      batch_labels.emplace_back(labels[i]);
    } catch (cv::Exception & e) {
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
  std::cout << "average preprocess time :" << infer.avg_preprocess_time() << std::endl;
  std::cout << "average prediction time :" << infer.avg_prediction_time() << std::endl;
  std::cout << "average postprocess time :" << infer.avg_postprocess_time() << std::endl;
  EXPECT_GT(mean_top1, 0.7);
  EXPECT_GT(mean_top5, 0.9);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
