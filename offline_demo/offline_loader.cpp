#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include "cnrt.h"

void transpose(float *input_data,
               float *output_data,
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

int offline_test(const char *fname, const char *function_name) {
  int input_size = 4;
  int input_dim[input_size] = {1, 3, 224, 224};
  cnrtInit(0);
  cnrtModel_t model;
  cnrtLoadModel(&model, fname);
  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, 0);
  cnrtSetCurrentDevice(dev);

  cnrtFunction_t function;
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model, function_name);

  int inputNum, outputNum;
  int64_t *inputSizeS, *outputSizeS;
  cnrtGetInputDataSize(&inputSizeS, &inputNum, function);
  cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);
  void **inputCpuPtrS = (void **)malloc(inputNum * sizeof(void *));
  void **outputCpuPtrS = (void **)malloc(outputNum * sizeof(void *));

  void **inputMluPtrS = (void **)malloc(inputNum * sizeof(void *));
  void **outputMluPtrS = (void **)malloc(outputNum * sizeof(void *));

  for (int i = 0; i < inputNum; i++) {
    inputCpuPtrS[i] = malloc(inputSizeS[i] * 4);

    void *tmp = malloc(inputSizeS[i] * 4);
    std::string filename = "9911.data";
    std::ifstream fs(filename, std::ifstream::binary);
    if (!fs.is_open()) {
      std::cout << "open input file fail.";
    }
    int input_num = 3 * 224 * 224;

    // memset(inputCpuPtrS[i], 1, inputSizeS[i]);
    float *dimg = reinterpret_cast<float *>(tmp);
    auto input_data_tmp = dimg;
    for (int i = 0; i < input_num; ++i) {
      fs.read(reinterpret_cast<char *>(input_data_tmp),
              sizeof(*input_data_tmp));
      input_data_tmp++;
    }
    std::cout << "before traspose: ";
    std::cout << dimg[0] << " " << dimg[10] << " " << dimg[input_num - 1]
              << std::endl;
    int label = 0;
    fs.read(reinterpret_cast<char *>(&label), sizeof(label));
    std::cout << "lable " << label << std::endl;
    fs.close();
    transpose(dimg,
              reinterpret_cast<float *>(inputCpuPtrS[i]),
              {static_cast<int>(input_dim[0]),
               static_cast<int>(input_dim[1]),
               static_cast<int>(input_dim[2]),
               static_cast<int>(input_dim[3])},
              {0, 2, 3, 1});
    float *after = reinterpret_cast<float *>(inputCpuPtrS[i]);
    std::cout << "after traspose: ";
    std::cout << after[0] << " " << after[10] << " " << after[input_num - 1]
              << std::endl;
    cnrtMalloc(&(inputMluPtrS[i]), inputSizeS[i]);
    cnrtMemcpy(inputMluPtrS[i],
               inputCpuPtrS[i],
               inputSizeS[i],
               CNRT_MEM_TRANS_DIR_HOST2DEV);
    free(tmp);
  }
  for (int i = 0; i < outputNum; i++) {
    outputCpuPtrS[i] = malloc(outputSizeS[i]);
    cnrtMalloc(&(outputMluPtrS[i]), outputSizeS[i]);
  }

  void **param = (void **)malloc(sizeof(void *) * (inputNum + outputNum));
  for (int i = 0; i < inputNum; i++) {
    param[i] = inputMluPtrS[i];
  }
  for (int i = 0; i < outputNum; i++) {
    param[inputNum + i] = outputMluPtrS[i];
  }

  cnrtRuntimeContext_t ctx;
  cnrtCreateRuntimeContext(&ctx, function, NULL);

  cnrtSetRuntimeContextDeviceId(ctx, 0);
  cnrtInitRuntimeContext(ctx, NULL);

  cnrtQueue_t queue;
  cnrtRuntimeContextCreateQueue(ctx, &queue);

  cnrtParamDescArray_t input_params = NULL;
  cnrtParamDescArray_t output_params = NULL;

  cnrtCreateParamDescArray(&input_params, 1);
  cnrtCreateParamDescArray(&output_params, 1);

  cnrtSetShapeToParamDesc(*input_params, input_dim, input_size);
  cnrtInferFunctionOutputShape(
      function, inputNum, input_params, outputNum, output_params);

  cnrtParamDesc_t param_descs[2];
  param_descs[0] = input_params[0];
  param_descs[1] = output_params[0];

  cnrtInvokeRuntimeContext_V2(ctx, param_descs, param, queue, NULL);
  cnrtSyncQueue(queue);

  for (int i = 0; i < outputNum; i++) {
    cnrtMemcpy(outputCpuPtrS[i],
               outputMluPtrS[i],
               outputSizeS[i],
               CNRT_MEM_TRANS_DIR_DEV2HOST);
    std::cout << outputSizeS[i] << std::endl;
    float *out = reinterpret_cast<float *>(outputCpuPtrS[i]);
    int out_num = outputSizeS[i] / sizeof(float);
    const int TOPK = 5;
    int max_indices[TOPK];
    double max_scores[TOPK];
    for (int i = 0; i < TOPK; i++) {
      max_indices[i] = 0;
      max_scores[i] = 0;
    }
    for (int i = 0; i < out_num; i++) {
      float score = *out;
      int index = i;
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
      out++;
    }
    std::cout << std::setprecision(6);
    for (int i = 0; i < TOPK; i++) {
      std::cout << "top " << i << " index " << max_indices[i]
                << " score: " << max_scores[i] << std::endl;
    }
  }

  for (int i = 0; i < inputNum; i++) {
    free(inputCpuPtrS[i]);
    cnrtFree(inputMluPtrS[i]);
  }

  for (int i = 0; i < outputNum; i++) {
    free(outputCpuPtrS[i]);
    cnrtFree(outputMluPtrS[i]);
  }

  free(inputCpuPtrS);
  free(outputCpuPtrS);
  free(param);
  cnrtDestroyParamDescArray(input_params, 1);
  cnrtDestroyParamDescArray(output_params, 1);
  cnrtDestroyQueue(queue);
  cnrtDestroyRuntimeContext(ctx);
  cnrtDestroyFunction(function);
  cnrtUnloadModel(model);
  cnrtDestroy();

  return 0;
}

int main(int argc, char **argv) { offline_test(argv[1], "subnet0"); }