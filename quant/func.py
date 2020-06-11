import paddle.fluid as fluid
import net

def func(image_shape, class_num, use_gpu=False):
    train_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(train_program, startup_program):
        image = fluid.layers.data(
            name='image', shape=image_shape, dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        out = net.net(input=image)
        val_program = fluid.default_main_program().clone(for_test=True)

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
    return exe, train_program, val_program, (image, label), out
