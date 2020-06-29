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

#include "core.hpp"
#include <gtest/gtest.h>

class Inferencer_detection : public Inferencer {
public:
  Inferencer_detection(std::shared_ptr<PaddlePredictor> p) : Inferencer(p) {}

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

    prediction_time_.push_back(prediction_time / i_shape_[0]);
    printf("Prediction time: %f ms\n", prediction_time);

    // refresh_input();
    return {};
  }

  void refresh_input(std::vector<int64_t> shape) {
    input_tensor_ = std::move(predictor_->GetInput(0));
    size_tensor_ = std::move(predictor_->GetInput(1));
    i_shape_ = shape;
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

private:
  std::unique_ptr<Tensor> size_tensor_;
};

class detection_test : public testing::Test {
public:
  void test() {

    pathes_ = load_image_pathes(data_file_);

    infer_->refresh_input({BATCH_SIZE, 3, 608, 608});
    // warm up
    {
      std::cout << "warm up ....." << std::endl;
      std::string image_name = pathes_[0];
      cv::Mat input_image = cv::imread(image_name, -1);
      infer_->warm_up(input_image);
      infer_->refresh_input({BATCH_SIZE, 3, 608, 608});
      std::cout << "warm up end" << std::endl;
    }

    auto start = get_current_us();
    int index = 0;
    for (int i = 0; i < pathes_.size() - 1; i++) {
      std::string image_name = pathes_[i];
      std::cout << image_name << std::endl;
      cv::Mat input_image = cv::imread(image_name, -1);
      // cv::imshow("aaa", input_image);
      // cv::waitKey(0);
      printf("process %d th image", i);
      infer_->batch(input_image);
      if (index % BATCH_SIZE == BATCH_SIZE - 1) {
        std::vector<RESULT> results = infer_->process();
        infer_->refresh_input({BATCH_SIZE, 3, 608, 608});
      }
      ++index;
    }
    auto end = get_current_us();
    double cur_time_cost = (end - start) / 1000.0f;
    /* float fps = (float)(pathes.size()  -1) / (cur_time_cost / 1000.0f); */
    /* std::cout << "fps for " << pathes.size() << " images: " << fps <<
     * std::endl; */
    std::cout << "average preprocess time :" << infer_->avg_preprocess_time()
              << std::endl;
    std::cout << "average prediction time :" << infer_->avg_prediction_time()
              << std::endl;
    /* std::cout << "average postprocess time :" << infer.avg_postprocess_time()
     * << std::endl; */
  }

protected:
  std::string data_file_;
  std::unique_ptr<Inferencer_detection> infer_;
  CxxConfig config_;
  std::shared_ptr<PaddlePredictor> predictor_;
  std::vector<std::string> pathes_;
  std::vector<Place> valid_places_;
  virtual void SetUp() {
    valid_places_ = {
        Place{TARGET(kX86), PRECISION(kFloat)},
        /* Place{TARGET(kMLU), PRECISION(kFloat), DATALAYOUT(kNHWC)} */
        /* Place{TARGET(kMLU), PRECISION(kFP16), DATALAYOUT(kNHWC)} */
    };
    config_.set_valid_places(valid_places_);
    config_.set_mlu_core_version(MLUCoreVersion::MLU_270);
    config_.set_mlu_core_number(16);
    config_.set_mlu_input_layout(DATALAYOUT(kNHWC));
  }
  virtual void TearDown() {}
};

TEST_F(detection_test, yolov3) {
  use_first_conv = false;
  data_file_ = "./filelist";
  config_.set_model_dir("/opt/share/paddle_model/yolov3_quant/");
  if (use_first_conv) {
    INPUT_MEAN = {124, 117, 104};
    INPUT_STD = {59, 57, 57};
    std::vector<float> mean_vec = INPUT_MEAN;
    std::vector<float> std_vec = INPUT_STD;
    config_.set_mlu_firstconv_param(mean_vec, std_vec);
  }
  predictor_ = CreatePaddlePredictor<CxxConfig>(config_);
  infer_.reset(new Inferencer_detection(predictor_));
  test();
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
