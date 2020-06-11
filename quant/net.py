import paddle
import paddle.fluid as fluid
import numpy as np
import random

def net(input):
    fc1 = fluid.layers.fc(input=input, size=20, act='softmax')
    predict = fluid.layers.fc(input=fc1, size=20,act='softmax')

    #return predict
    return predict
