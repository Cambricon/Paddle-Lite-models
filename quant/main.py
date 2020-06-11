import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
import func

exe, train_program, val_program, inputs, outputs = \
    func.func([1, 28, 28], 10, use_gpu=False)

import paddle.dataset.mnist as reader

train_reader = paddle.fluid.io.batch(
        reader.train(), batch_size=128, drop_last=True)
test_reader = paddle.fluid.io.batch(
        reader.train(), batch_size=128, drop_last=True)
train_feeder = fluid.DataFeeder(inputs, fluid.CPUPlace())

quant_program = slim.quant.quant_aware(train_program, exe.place, for_test=False)
val_quant_program = slim.quant.quant_aware(val_program, exe.place, for_test=True)

float_prog, int8_prog = slim.quant.convert(val_quant_program, exe.place, save_int8=True)

fluid.io.save_inference_model(dirname='./inference_model/float',
        feeded_var_names=['image'],
        target_vars=[outputs],
        executor=exe,
        main_program=float_prog)
        #target_vars=target_vars,
        #feeded_var_names=[var.name for var in inputs],
fluid.io.save_inference_model(dirname='./inference_model/int8',
        feeded_var_names=['image'],
        target_vars=[outputs],
        executor=exe,
        main_program=int8_prog)
        #target_vars=target_vars,
        #feeded_var_names=[var.name for var in inputs],
