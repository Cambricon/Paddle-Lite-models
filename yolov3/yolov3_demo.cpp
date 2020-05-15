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

class Inferencer_detection : public Inferencer {
 public:
  Inferencer_detection(std::shared_ptr<PaddlePredictor> p, std::vector<int64_t> input_shape):Inferencer(p, input_shape){
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
    printf("repeat: %d, average: %f ms, max: %f ms, min: %f ms\n",
            REPEAT_COUNT, prediction_time,
            max_time_cost, min_time_cost);
  
    prediction_time_.push_back(prediction_time / i_shape_[0]);
    printf("Prediction time: %f ms\n", prediction_time);

    refresh_input();
    return {};
  }

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

 private:
  std::unique_ptr<Tensor> size_tensor_;
};

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
  std::string model_dir = "/home/jiaopu/new/data/yolov3_quant/";
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

  Inferencer_detection infer(predictor, {BATCH_SIZE, 3, 608, 608});

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
