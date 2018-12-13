#include <torch/torch.h>
#include <ATen/ATen.h>
#include <stdio.h>

#include "correlation1d_cuda_kernel.cuh"

#define real float

int correlation1d_forward_cuda(at::Tensor& input1, at::Tensor& input2, at::Tensor& rInput1, at::Tensor& rInput2, at::Tensor& output,
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply)
{

  int batchSize = input1.size(0);
  int nInputChannels = input1.size(1);
  int inputHeight = input1.size(2);
  int inputWidth = input1.size(3);

  int kernel_radius = (kernel_size - 1) / 2;
  int border_radius = kernel_radius + max_displacement;

  int paddedInputHeight = inputHeight + 0;
  int paddedInputWidth = inputWidth + 2 * pad_size;

  // TODO: single direction or not
  int nOutputChannels = ((max_displacement/stride2) + 1); // * ((max_displacement/stride2)*2 + 1);

  int outputHeight = ceil((float)(paddedInputHeight - 2 * kernel_radius) / (float)stride1);
  int outputwidth = ceil((float)(paddedInputWidth - 2 * border_radius) / (float)stride1);

  rInput1.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
  rInput2.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
  output.resize_({batchSize, nOutputChannels, outputHeight, outputwidth});

  rInput1.fill_(0);
  rInput2.fill_(0);
  output.fill_(0);

  int success = 0;
  success = correlation1d_forward_cuda_kernel(output.data<float>(),
                                            output.size(0),
                                            output.size(1),
                                            output.size(2),
                                            output.size(3),
                                            output.stride(0),
                                            output.stride(1),
                                            output.stride(2),
                                            output.stride(3),

                                            input1.data<float>(),
                                            input1.size(1),
                                            input1.size(2),
                                            input1.size(3),
                                            input1.stride(0),
                                            input1.stride(1),
                                            input1.stride(2),
                                            input1.stride(3),

                                            input2.data<float>(),
                                            input2.size(1),
                                            input2.stride(0),
                                            input2.stride(1),
                                            input2.stride(2),
                                            input2.stride(3),

                                            rInput1.data<float>(),
                                            rInput2.data<float>(),
                                            pad_size,
                                            kernel_size,
                                            max_displacement,
                                            stride1,
                                            stride2,
                                            corr_type_multiply,

                                            at::globalContext().getCurrentCUDAStream());

  if (!success) {
    AT_ERROR("CUDA call failed");
  }
  return 1;
}

int correlation1d_backward_cuda(at::Tensor& input1, at::Tensor& input2, at::Tensor& rInput1, at::Tensor& rInput2, at::Tensor& gradOutput,
                       at::Tensor& gradInput1, at::Tensor& gradInput2,
                       int pad_size,
                       int kernel_size,
                       int max_displacement,
                       int stride1,
                       int stride2,
                       int corr_type_multiply)
{

  int batchSize = input1.size(0);
  int nInputChannels = input1.size(1);
  int paddedInputHeight = input1.size(2)+ 0;
  int paddedInputWidth = input1.size(3)+ 2 * pad_size;

  int height = input1.size(2);
  int width = input1.size(3);

  rInput1.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
  rInput2.resize_({batchSize, paddedInputHeight, paddedInputWidth, nInputChannels});
  gradInput1.resize_({batchSize, nInputChannels, height, width});
  gradInput2.resize_({batchSize, nInputChannels, height, width});
  
  rInput1.fill_(0);
  rInput2.fill_(0);
  gradInput1.fill_(0);
  gradInput2.fill_(0);

  int success = 0;
  success = correlation1d_backward_cuda_kernel(
                                            gradOutput.data<float>(),
                                            gradOutput.size(0),
                                            gradOutput.size(1),
                                            gradOutput.size(2),
                                            gradOutput.size(3),
                                            gradOutput.stride(0),
                                            gradOutput.stride(1),
                                            gradOutput.stride(2),
                                            gradOutput.stride(3),
                                            input1.data<float>(),
                                            input1.size(1),
                                            input1.size(2),
                                            input1.size(3),
                                            input1.stride(0),
                                            input1.stride(1),
                                            input1.stride(2),
                                            input1.stride(3),
                                            input2.data<float>(),
                                            input2.stride(0),
                                            input2.stride(1),
                                            input2.stride(2),
                                            input2.stride(3),
                                            gradInput1.data<float>(),
                                            gradInput1.stride(0),
                                            gradInput1.stride(1),
                                            gradInput1.stride(2),
                                            gradInput1.stride(3),
                                            gradInput2.data<float>(),
                                            gradInput2.size(1),
                                            gradInput2.stride(0),
                                            gradInput2.stride(1),
                                            gradInput2.stride(2),
                                            gradInput2.stride(3),

                                            rInput1.data<float>(),
                                            rInput2.data<float>(),
                                            pad_size,
                                            kernel_size,
                                            max_displacement,
                                            stride1,
                                            stride2,
                                            corr_type_multiply,
                                            at::globalContext().getCurrentCUDAStream());

  if (!success) {
    AT_ERROR("CUDA call failed");
  }

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &correlation1d_forward_cuda, "Correlation1d forward (CUDA)");
  m.def("backward", &correlation1d_backward_cuda, "Correlation1d backward (CUDA)");
}
