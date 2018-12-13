#include <torch/torch.h>
#include <ATen/ATen.h>
#include <stdio.h>

#include "resample1d_cuda_kernel.cuh"


int resample1d_cuda_forward(at::Tensor& input1, at::Tensor& input2, at::Tensor& output, int kernel_size) {
    resample1d_kernel_forward(input1, input2, output, kernel_size);
    return 1;
}

int resample1d_cuda_backward(at::Tensor& input1, at::Tensor& input2, at::Tensor& gradOutput, at::Tensor& gradInput1, at::Tensor& gradInput2, int kernel_size) {
    resample1d_kernel_backward(input1, input2, gradOutput, gradInput1, gradInput2, kernel_size);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &resample1d_cuda_forward, "Resample1D forward (CUDA)");
  m.def("backward", &resample1d_cuda_backward, "Resample1D backward (CUDA)");
}
