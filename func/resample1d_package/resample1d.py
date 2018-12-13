import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
import resample1d_cuda


class Resample1dFunction(Function):
    def __init__(self, kernel_size=1):
        super(Resample1dFunction, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)

        assert (input1.is_contiguous() == True)
        assert (input2.is_contiguous() == True)

        with torch.cuda.device_of(input1):
            _, d, _, _ = input1.size()
            b, _, h, w = input2.size()
            output = input1.new().resize_(b, d, h, w).zero_()

            resample1d_cuda.forward(input1, input2, output, self.kernel_size)

        return output

    def backward(self, gradOutput):
        input1, input2 = self.saved_tensors

        assert (gradOutput.is_contiguous() == True)

        with torch.cuda.device_of(input1):
            b, c, h, w = input1.size()
            gradInput1 = input1.new().resize_(b, c, h, w).zero_()

            b, c, h, w = input2.size()
            gradInput2 = input2.new().resize_(b, c, h, w).zero_()

            resample1d_cuda.backward(input1, input2, gradOutput, gradInput1, gradInput2, self.kernel_size)

        return gradInput1, gradInput2


class Resample1d(Module):
    def __init__(self, kernel_size=1):
        super(Resample1d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        input1_c = input1.contiguous()

        result = Resample1dFunction(self.kernel_size)(input1_c, input2)

        return result
