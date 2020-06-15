#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>
#include "cnrt.h"
#include <sys/time.h>

clock_t GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}
bool use_first_conv = false;
std::vector<float> INPUT_MEAN = {0.485f, 0.456f, 0.406f};
std::vector<float> INPUT_STD = {0.229f, 0.224f, 0.225f};

#define CHECK_CNRT_RET(func, msg)                                \
  do {                                                           \
    int ret = (func);                                            \
    if (0 != ret) {                                              \
      std::cout << (msg " error code : " + std::to_string(ret)); \
    }                                                            \
  } while (0)
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
template <typename dtype>
void transpose(dtype *input_data,
               dtype *output_data,
               std::vector<int> input_shape,
               std::vector<int> axis) {
  int old_index = -1;
  int new_index = -1;
  std::vector<int> shape;
  std::vector<int> expand_axis;
  if (input_shape.size() < 5u) {
    for (size_t i = 0; i < 5 - input_shape.size(); i++) {
      shape.push_back(1);
      expand_axis.push_back(i);
    }
    for (size_t i = 0; i < input_shape.size(); i++) {
      shape.push_back(input_shape[i]);
      expand_axis.push_back(axis[i] + 5 - input_shape.size());
    }
  } else {
    shape = input_shape;
    expand_axis = axis;
  }
  int dim[5] = {0};
  for (dim[0] = 0; dim[0] < shape[0]; dim[0]++) {
    for (dim[1] = 0; dim[1] < shape[1]; dim[1]++) {
     for (dim[2] = 0; dim[2] < shape[2]; dim[2]++) {
        for (dim[3] = 0; dim[3] < shape[3]; dim[3]++) {
          for (dim[4] = 0; dim[4] < shape[4]; dim[4]++) {
            old_index = dim[0] * shape[1] * shape[2] * shape[3] * shape[4] +
                        dim[1] * shape[2] * shape[3] * shape[4] +
                        dim[2] * shape[3] * shape[4] + dim[3] * shape[4] +
                        dim[4];
            new_index = dim[expand_axis[0]] * shape[expand_axis[1]] *
                            shape[expand_axis[2]] * shape[expand_axis[3]] *
                            shape[expand_axis[4]] +
                        dim[expand_axis[1]] * shape[expand_axis[2]] *
                            shape[expand_axis[3]] * shape[expand_axis[4]] +
                        dim[expand_axis[2]] * shape[expand_axis[3]] *
                            shape[expand_axis[4]] +
                        dim[expand_axis[3]] * shape[expand_axis[4]] +
                        dim[expand_axis[4]];
            output_data[new_index] = input_data[old_index];
          }
        }
      }
    }
  }
}

/**
 * @brief Inference helper class
 */
class EasyInfer {
 public:
  /**
   * @brief Construct a new Easy Infer object
   */
  EasyInfer() {}

  /**
   * @brief Destroy the Easy Infer object
   */
  ~EasyInfer() {
    if (tmp) {
      free(tmp);
    }
    for (int i = 0; i < input_num; i++) {
      if (inputCpuPtrS[i]) {
        free(inputCpuPtrS[i]);
      }
      if (inputMluPtrS[i]) {
        cnrtFree(inputMluPtrS[i]);
      }
    }
    for (int i = 0; i < output_num; i++) {
      if (outputCpuPtrS[i]) free(outputCpuPtrS[i]);
      if (outputMluPtrS[i]) cnrtFree(outputMluPtrS[i]);
    }
    if (input_dim_values) free(input_dim_values);
    if (output_dim_values) free(output_dim_values);
    if (inputCpuPtrS) free(inputCpuPtrS);
    if (outputCpuPtrS) free(outputCpuPtrS);
    if (param) free(param);
    if (input_params) cnrtDestroyParamDescArray(input_params, 1);
    if (output_params) cnrtDestroyParamDescArray(output_params, 1);
    if (queue) cnrtDestroyQueue(queue);
    if (ctx) cnrtDestroyRuntimeContext(ctx);
    if (function) cnrtDestroyFunction(function);
    if (model) cnrtUnloadModel(model);
    cnrtDestroy();
  }

  void print_data_type(const cnrtDataType &type) {
    switch (type) {
      case CNRT_UINT8:
        input_type = "UINT8";
        break;
      case CNRT_FLOAT32:
        input_type = "FLOAT32";
        break;
      case CNRT_FLOAT16:
        input_type = "FLOAT16";
        break;
      case CNRT_INT16:
        input_type = "INT16";
        break;
      case CNRT_INT32:
        input_type = "INT32";
        break;
      default:
        input_type = "unknown";
    }
  }
  void init(const std::string &fname, const std::string &function_name) {
    CHECK_CNRT_RET(cnrtInit(0), "Init failed.");
    CHECK_CNRT_RET(cnrtLoadModel(&model, fname.c_str()), "Load Model failed.");
    CHECK_CNRT_RET(cnrtGetDeviceHandle(&dev, 0), "Get Device Handle failed.");
    CHECK_CNRT_RET(cnrtSetCurrentDevice(dev), "Set Current Device failed.");
    CHECK_CNRT_RET(cnrtCreateFunction(&function), "Create Function failed.");
    CHECK_CNRT_RET(cnrtExtractFunction(&function, model, function_name.c_str()),
                   "Extra Function failed.");
    CHECK_CNRT_RET(cnrtCreateRuntimeContext(&ctx, function, nullptr),
                   "Create Runtime Context failed.");
    CHECK_CNRT_RET(cnrtSetRuntimeContextDeviceId(ctx, 0),
                   "Set Runtime Context Device Id failed.");
    CHECK_CNRT_RET(cnrtInitRuntimeContext(ctx, nullptr),
                   "Init Runtime Context failed.");
    CHECK_CNRT_RET(cnrtRuntimeContextCreateQueue(ctx, &queue),
                   "Runtime Context Create Queue failed.");
    CHECK_CNRT_RET(cnrtGetRuntimeContextInfo(ctx,CNRT_RT_CTX_CORE_NUMBER,reinterpret_cast<void **> (&core_number)), "Get Runtime Context Info failed.");
    CHECK_CNRT_RET(cnrtGetInputDataSize(&input_sizes, &input_num, function),
                   "Get Input Data Size failed.");
    CHECK_CNRT_RET(cnrtGetInputDataType(&input_dtypes, &input_num, function),
                   "Get Input Data Type failed.");
    if (input_dtypes[0] == CNRT_UINT8) {
      use_first_conv = true;
    }
    CHECK_CNRT_RET(cnrtGetOutputDataSize(&output_sizes, &output_num, function),
                   "Get Output Data Size failed.");
    CHECK_CNRT_RET(
        cnrtGetInputDataShape(&input_dim_values, &input_dim_num, 0, function),
        "Get Input Data Shape failed.");
    batchsize = input_dim_values[0];
    image_size =
        input_dim_values[1] * input_dim_values[2] * input_dim_values[3];
    width_ = input_dim_values[2];
    height_ = input_dim_values[1];

    CHECK_CNRT_RET(cnrtGetOutputDataShape(
                       &output_dim_values, &output_dim_num, 0, function),
                   "Get Output Data Shape failed.");
    inputCpuPtrS = (void **)malloc(input_num * sizeof(void *));
    outputCpuPtrS = (void **)malloc(output_num * sizeof(void *));
    param = (void **)malloc(sizeof(void *) * (input_num + output_num));

    inputMluPtrS = (void **)malloc(input_num * sizeof(void *));
    outputMluPtrS = (void **)malloc(output_num * sizeof(void *));
    CHECK_CNRT_RET(cnrtCreateParamDescArray(&input_params, 1),
                   "Create ParamDesc Array failed.");
    CHECK_CNRT_RET(cnrtCreateParamDescArray(&output_params, 1),
                   "Create ParamDesc Array failed.");

    CHECK_CNRT_RET(
        cnrtSetShapeToParamDesc(*input_params, input_dim_values, input_dim_num),
        "Set Shape To Param Desc failed.");
    CHECK_CNRT_RET(
        cnrtInferFunctionOutputShape(
            function, input_num, input_params, output_num, output_params),
        "Infer Function Output Shape failed.");
    param_descs[0] = input_params[0];
    param_descs[1] = output_params[0];
    for (int i = 0; i < output_num; i++) {
      outputCpuPtrS[i] = malloc(output_sizes[i]);
      CHECK_CNRT_RET(cnrtMalloc(&(outputMluPtrS[i]), output_sizes[i]),
                     "Malloc failed.");
    }

    for (int i = 0; i < input_num; i++) {
      inputCpuPtrS[i] = malloc(input_sizes[i]);
      CHECK_CNRT_RET(cnrtMalloc(&(inputMluPtrS[i]), input_sizes[i]),
                     "Malloc failed.");
      tmp = malloc(input_sizes[i]);
    }

    for (int i = 0; i < input_num; i++) {
      param[i] = inputMluPtrS[i];
    }
    for (int i = 0; i < output_num; i++) {
      param[input_num + i] = outputMluPtrS[i];
    }
  }
  void preprocess(const cv::Mat &input_image, uint8_t *input_data) {
    cv::Mat resize_image;

    cv::resize(input_image, resize_image, cv::Size(width_, height_), 0, 0);
    if (resize_image.channels() == 1) {
      cv::cvtColor(resize_image, resize_image, cv::COLOR_GRAY2RGB);
    } else {
      cv::cvtColor(resize_image, resize_image, cv::COLOR_BGRA2RGB);
    }
    cv::Mat output_image;
    resize_image.convertTo(output_image, CV_8UC3);
    memcpy(
        input_data, output_image.data, height_ * width_ * 3 * sizeof(uint8_t));
  }
  void preprocess(const cv::Mat &input_image,
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
    for (int i = 0; i < width_ * height_; i++) {
      input_data[i * 3 + 0] = (dimg[i * 3 + 0] - input_mean[0]) / input_std[0];
      input_data[i * 3 + 1] = (dimg[i * 3 + 1] - input_mean[1]) / input_std[1];
      input_data[i * 3 + 2] = (dimg[i * 3 + 2] - input_mean[2]) / input_std[2];
    }
    imgf.release();
  }

  void batch(const cv::Mat &input_image, int batch_index) {
    if (use_first_conv) {
      preprocess(input_image,
                 reinterpret_cast<uint8_t *>(inputCpuPtrS[0]) +
                     batch_index * image_size);
    } else {
      preprocess(input_image,
                 INPUT_MEAN,
                 INPUT_STD,
                 reinterpret_cast<float *>(inputCpuPtrS[0]) +
                     batch_index * image_size);
    }
  }
  void run_end2end(const std::string &fname) {
    std::vector<int> labels = load_labels(fname);
    std::vector<std::string> pathes = load_image_pathes(fname);

    int total_imgs = (pathes.size() - 1);
    int imgs_num = total_imgs - total_imgs % batchsize;
    float top1_num = 0;
    float top5_num = 0;
    // auto fps_start = GetCurrentUS();
    int batch_index = 0;
    for (int j = 0; j < imgs_num;) {
      int label[batchsize];
      int image_index = 0;
      while (j < imgs_num) {
        cv::Mat input_image = cv::imread(pathes[j], -1);
        try {
          batch(input_image, image_index);
          label[image_index] = labels[j];
        } catch (cv::Exception &e) {
          continue;
        }
        image_index++;
        j++;
        if (image_index == batchsize) {
          break;
        }
      }
      cnrtMemcpy(inputMluPtrS[0],
                 inputCpuPtrS[0],
                 input_sizes[0],
                 CNRT_MEM_TRANS_DIR_HOST2DEV);
      auto start = GetCurrentUS();
      cnrtInvokeRuntimeContext_V2(ctx, param_descs, param, queue, NULL);
      cnrtSyncQueue(queue);
      auto end = GetCurrentUS();
      perf_vet.push_back((end-start)/1000.0);
      cnrtMemcpy(outputCpuPtrS[0],
                 outputMluPtrS[0],
                 output_sizes[0],
                 CNRT_MEM_TRANS_DIR_DEV2HOST);
      // std::cout << output_sizes[0] << std::endl;
      float *out = reinterpret_cast<float *>(outputCpuPtrS[0]);
      int out_num = output_dim_values[0] * output_dim_values[1];
      const int TOPK = 5;
      int max_indices[TOPK];
      double max_scores[TOPK];
      int label_index = 0;
      for (int i = 0; i < out_num;) {
        for (int j = 0; j < TOPK; j++) {
          max_indices[j] = 0;
          max_scores[j] = 0;
        }
        while (i < out_num) {
          float score = *out;
          int index = i % output_dim_values[1];
          // std::cout << score << std::endl;
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
          i++;
          out++;
          if (i % output_dim_values[1] == output_dim_values[1] - 1) {
            break;
          }
        }
        if (label_index == batchsize) {
          break;
        }
        std::cout << std::setprecision(6);
        std::cout << "batch index " << batch_index << " image index "
                  << label_index << std::endl;
        std::cout << "lable " << label[label_index] << std::endl;

        if (label[label_index] == max_indices[0]) {
          top1_num++;
        }
        for (int n = 0; n < TOPK; n++) {
          std::cout << "top " << n << " index " << max_indices[n]
                    << " score: " << max_scores[n] << std::endl;
          if (label[label_index] == max_indices[n]) {
            top5_num++;
          }
        }
        label_index++;
      }
      batch_index++;
    }
    // auto fps_end = GetCurrentUS();
    // float fps = 0; 
    // fps = (float)imgs_num / ((fps_end - fps_start) /1000.0/1000.0);
    print_data_type(input_dtypes[0]);
    std::sort(perf_vet.begin(), perf_vet.end());
    min_res = perf_vet.front() / batchsize;
    max_res = perf_vet.back() / batchsize;
    float total_res = accumulate(perf_vet.begin(), perf_vet.end(), 0.0);
    avg_res = total_res / imgs_num ;
    top1 = top1_num / (float) imgs_num;
    top5 = top5_num / (float) imgs_num;
    // std::cout << "fps for  " << imgs_num << " images: " << fps << std::endl;
  }

  void run_binary(const std::string &data_path) {
    int total_imgs = 500;
    int imgs_num = total_imgs - total_imgs % batchsize;
    float top1_num = 0;
    float top5_num = 0;

    int batch_index = 0;
    for (int j = 0; j < imgs_num;) {
      int label[batchsize];
      float *dimg = reinterpret_cast<float *>(tmp);
      // float *after = reinterpret_cast<float *>(inputCpuPtrS[0]);
      auto input_data_tmp = dimg;
      int image_index = 0;
      while (j < imgs_num) {
        std::string filename = data_path + "/" + std::to_string(j) + "11.data";
        std::ifstream fs(filename, std::ifstream::binary);
        if (!fs.is_open()) {
          std::cout << "open input file fail.";
        }
        for (int k = 0; k < image_size; ++k) {
          fs.read(reinterpret_cast<char *>(input_data_tmp),
                  sizeof(*input_data_tmp));
          input_data_tmp++;
        }
        fs.read(reinterpret_cast<char *>(&label[image_index]), sizeof(label));
        fs.close();
        image_index++;
        j++;
        if (image_index == batchsize) {
          break;
        }
      }
      transpose<float>(dimg,
                       reinterpret_cast<float *>(inputCpuPtrS[0]),
                       {static_cast<int>(input_dim_values[0]),
                        static_cast<int>(input_dim_values[3]),
                        static_cast<int>(input_dim_values[1]),
                        static_cast<int>(input_dim_values[2])},
                       {0, 2, 3, 1});
      cnrtMemcpy(inputMluPtrS[0],
                 inputCpuPtrS[0],
                 input_sizes[0],
                 CNRT_MEM_TRANS_DIR_HOST2DEV);

      cnrtInvokeRuntimeContext_V2(ctx, param_descs, param, queue, NULL);
      cnrtSyncQueue(queue);

      cnrtMemcpy(outputCpuPtrS[0],
                 outputMluPtrS[0],
                 output_sizes[0],
                 CNRT_MEM_TRANS_DIR_DEV2HOST);
      // std::cout << output_sizes[0] << std::endl;
      float *out = reinterpret_cast<float *>(outputCpuPtrS[0]);
      int out_num = output_dim_values[0] * output_dim_values[1];
      const int TOPK = 5;
      int max_indices[TOPK];
      double max_scores[TOPK];
      int label_index = 0;
      for (int i = 0; i < out_num;) {
        for (int j = 0; j < TOPK; j++) {
          max_indices[j] = 0;
          max_scores[j] = 0;
        }
        while (i < out_num) {
          float score = *out;
          int index = i % output_dim_values[1];
          // std::cout << score << std::endl;
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
          i++;
          out++;
          if (i % output_dim_values[1] == output_dim_values[1] - 1) {
            break;
          }
        }
        if (label_index == batchsize) {
          break;
        }
        std::cout << std::setprecision(6);
        std::cout << "batch index " << batch_index << " image index "
                  << label_index << std::endl;
        std::cout << "lable " << label[label_index] << std::endl;

        if (label[label_index] == max_indices[0]) {
          top1_num++;
        }
        for (int n = 0; n < TOPK; n++) {
          std::cout << "top " << n << " index " << max_indices[n]
                    << " score: " << max_scores[n] << std::endl;
          if (label[label_index] == max_indices[n]) {
            top5_num++;
          }
        }
        label_index++;
      }
      batch_index++;
    }
    print_data_type(input_dtypes[0]);
    top1 = top1_num / (float) imgs_num;
    top5 = top5_num / (float) imgs_num;
  }
 public:
   std::vector<float> perf_vet;
   int core_number[2];
   int batchsize = 0;
   std::string input_type;
   float top1 = 0;
   float top5 = 0;
   float min_res = 0;
   float max_res = 0;
   float avg_res = 0;

      
 private:
  int image_size = 0;
  int width_, height_;
  int input_num = 0;
  int output_num = 0;
  int64_t *input_sizes;
  int64_t *output_sizes;
  int *input_dim_values = nullptr;  // NHWC
  int input_dim_num = 0;
  int *output_dim_values = nullptr;
  int output_dim_num = 0;
  void **inputCpuPtrS = nullptr;
  void **outputCpuPtrS = nullptr;
  void **param = nullptr;
  void **inputMluPtrS = nullptr;
  void **outputMluPtrS = nullptr;
  cnrtDataType_t *input_dtypes = nullptr;
  cnrtParamDescArray_t input_params = nullptr;
  void *tmp = nullptr;
  cnrtParamDescArray_t output_params = nullptr;
  cnrtParamDesc_t param_descs[2];
  cnrtModel_t model;
  cnrtDev_t dev;
  cnrtFunction_t function;
  cnrtRuntimeContext_t ctx;
  cnrtQueue_t queue;
  EasyInfer(const EasyInfer &) = delete;
  EasyInfer &operator=(const EasyInfer &) = delete;
};  // class EasyInfer


#define SAVE_OUTPUT_CSV(OUTPUT_DEVICE)                                     \
  OUTPUT_DEVICE                                                            \
      << "offline model name, input data type, batch size, core num, top1, top5," \
         "max preprocess time per image (ms),min prediction time per image (ms)," \
         "average prediction time per image (ms) "      \
      << std::endl;                                                        \
  OUTPUT_DEVICE << fname;                                             \
  OUTPUT_DEVICE << " , " << infer.batchsize;                                    \
  OUTPUT_DEVICE << " , " << infer.input_type;                                      \
  OUTPUT_DEVICE << " , " << infer.core_number[0];                                      \
  OUTPUT_DEVICE << " , " << infer.top1;                                     \
  OUTPUT_DEVICE << " , " << infer.top5;                                     \
  OUTPUT_DEVICE << " , " << infer.max_res;                           \
  OUTPUT_DEVICE << " , " << infer.min_res;                           \
  OUTPUT_DEVICE << " , " << infer.avg_res;     \
  OUTPUT_DEVICE << std::endl;	

#define SAVE_OUTPUT(OUTPUT_DEVICE)                                             \
  OUTPUT_DEVICE << "offline model name: " << fname << std::endl;\
  OUTPUT_DEVICE << "batchsize: " <<infer.batchsize << std::endl;\
  OUTPUT_DEVICE << "input type: " <<infer.input_type << std::endl;\
  OUTPUT_DEVICE << "core number: " << infer.core_number[0] << std::endl;\
  OUTPUT_DEVICE << "top1 acc: " << infer.top1 << std::endl;\
  OUTPUT_DEVICE << "top5 acc: " << infer.top5 << std::endl;\
  OUTPUT_DEVICE << "max prediction time per image: " << infer.max_res << " ms" << std::endl;\
  OUTPUT_DEVICE << "min prediction time per image: " << infer.min_res << " ms" << std::endl;\
  OUTPUT_DEVICE << "average prediction time per image: " << infer.avg_res << " ms" << std::endl;
int main(int argc, char **argv) {
  std::string fname = argv[1];
  std::string data_path = argv[2];
  std::string file_type = "image";
  if (argc == 4) file_type = "binary";
  const std::string &funtion_name = "subnet0";
  EasyInfer infer;
  infer.init(fname, funtion_name);
  if (file_type == "binary") {
    infer.run_binary(data_path);
  } else
    infer.run_end2end(data_path);
  SAVE_OUTPUT(std::cout);
  std::ofstream ofs("output.txt", std::ios::app);
  if(!ofs.is_open()) {
    std::cout<< "open result file failed";
  }
  SAVE_OUTPUT(ofs);
  std::ofstream ofs_csv("output.csv", std::ios::app);
  if (!ofs_csv.is_open()) {
    std::cout << "open result file failed";
  }
  SAVE_OUTPUT_CSV(ofs_csv);
  ofs_csv.close();
//  std::cout << "offline model name: " << fname << std::endl;
//  std::cout << "batchsize: " <<infer.batchsize << std::endl;
//  std::cout << "input type: " <<infer.input_type << std::endl;
//  std::cout << "core number: " << infer.core_number[0] << std::endl;
//  std::cout << "top1 acc: " << infer.top1 << std::endl;
//  std::cout << "top5 acc: " << infer.top5 << std::endl;
//  std::cout << "max prediction time per image: " << infer.max_res << " ms" << std::endl;
//  std::cout << "min prediction time per image: " << infer.min_res << " ms" << std::endl;
//  std::cout << "average prediction time per image: " << infer.avg_res << " ms" << std::endl;
}
