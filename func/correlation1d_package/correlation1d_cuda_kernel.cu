#include <stdio.h>

#include "correlation1d_cuda_kernel.cuh"

#define real float

#define CUDA_NUM_THREADS 128  //  1024
#define THREADS_PER_BLOCK 64  // 32

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

__global__ void channels_first(float* input, float* rinput, int channels, int height, int width, int pad_size)
{
    // n (batch size), c (num of channels), y (height), x (width)
    int n = blockIdx.x;
    int y = blockIdx.y;
    int x = blockIdx.z;

    int ch_off = threadIdx.x;
    float value;

    int dimcyx = channels * height * width;
    int dimyx = height * width;

    int p_dimx = (width + 2 * pad_size);
    int p_dimy = (height + 0);
    int p_dimyxc = channels * p_dimy * p_dimx;
    int p_dimxc = p_dimx * channels;

    for (int c = ch_off; c < channels; c += THREADS_PER_BLOCK) {
      value = input[n * dimcyx + c * dimyx + y * width + x];
      rinput[n * p_dimyxc + (y) * p_dimxc + (x + pad_size) * channels + c] = value;
    }
}



__global__ void pad_left(float* input, float* rinput, int channels, int height, int width, int pad_size)
{
    // n (batch size), c (num of channels), y (height), x (width)
    int n = blockIdx.x;
    int c = blockIdx.y;
    int y = blockIdx.z;

    int x_off = threadIdx.x;
    float value;

    int dimcyx = channels * height * width;
    int dimyx = height * width;
    int dimx = width;

    int p_dimx = width + 2 * pad_size;
    int p_dimyx = p_dimx * height;
    int p_dimcyx = p_dimyx * channels;

//    int p_dimx = (width + 2 * pad_size);
//    int p_dimy = (height + 0);
//    int p_dimyxc = channels * p_dimy * p_dimx;
//    int p_dimxc = p_dimx * channels;

    for (int x = x_off; x < width; x += THREADS_PER_BLOCK) {
      value = input[n * dimcyx + c * dimyx + y * dimx + x];
      rinput[n * p_dimcyx + c * p_dimyx + y * p_dimx + (x + pad_size)] = value;
    }
}


__global__ void Correlation1d_forward( float *output, int nOutputChannels, int outputHeight, int outputWidth,
                                     float *rInput1, int nInputChannels, int inputHeight, int inputWidth, 
                                     float *rInput2,
                                     int pad_size,
                                     int kernel_size,
                                     int max_displacement,
                                     int stride1,
                                     int stride2)
{
    stride1 = stride2 = kernel_size = 1;
    // n (batch size), c (num of channels), y (height), x (width)
    
    int pInputWidth = inputWidth + 2 * pad_size;
    int pInputHeight = inputHeight + 0;

    int kernel_rad = (kernel_size - 1) / 2;
    int displacement_rad = max_displacement / stride2;
//    int displacement_size = 2 * displacement_rad + 1;

    int n  = blockIdx.x;
    int y1 = blockIdx.y * stride1 + kernel_rad;  // TODO: not sure ?
    int x1 = blockIdx.z * stride1 + max_displacement + kernel_rad;  // TODO: need to plus extra kernel_rad ?
    int c = threadIdx.x;

    int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
    int pdimxc = pInputWidth * nInputChannels;
    int pdimc = nInputChannels;

    int tdimcyx = nOutputChannels * outputHeight * outputWidth;
    int tdimyx = outputHeight * outputWidth;
    int tdimx = outputWidth;

    float nelems = kernel_size * kernel_size * pdimc;

    __shared__ float prod_sum[THREADS_PER_BLOCK];

    // no significant speed-up in using chip memory for input1 sub-data, 
    // not enough chip memory size to accomodate memory per block for input2 sub-data
    // instead i've used device memory for both 

    // element-wise product along channel axis
    int x_offset = -displacement_rad;  // -displacement_rad
    int x_end = 0;  // displacement_rad
    for (int ti = x_offset; ti <= x_end; ++ti ) {
      prod_sum[c] = 0;
      int x2 = x1 + ti*stride2;
      int y2 = y1 + 0;

      for (int j = -kernel_rad; j <= kernel_rad; ++j) {
        for (int i = -kernel_rad; i <= kernel_rad; ++i) {
          for (int ch = c; ch < pdimc; ch += THREADS_PER_BLOCK) {
            int indx1 = n * pdimyxc + (y1+j) * pdimxc + (x1 + i) * pdimc + ch;
            int indx2 = n * pdimyxc + (y2+j) * pdimxc + (x2 + i) * pdimc + ch;

            prod_sum[c] += rInput1[indx1] * rInput2[indx2];
          }
        }
      }

      // accumulate
      __syncthreads();
      if (c == 0) {
        float reduce_sum = 0;
        for (int index = 0; index < THREADS_PER_BLOCK; ++index) {
          reduce_sum += prod_sum[index];
        }
        int tc = (ti + displacement_rad);  // totally displacement_rad + 1 channels
        const int tindx = n * tdimcyx + tc * tdimyx + blockIdx.y * tdimx + blockIdx.z;
        output[tindx] = reduce_sum / nelems;
      }
      __syncthreads();

    }

}

__global__ void Correlation1d_backward_input1(int item_xx, float *gradInput1, int nInputChannels, int inputHeight, int inputWidth,
                                            float *gradOutput, int nOutputChannels, int outputHeight, int outputWidth, 
                                            float *rInput2, 
                                            int pad_size,
                                            int kernel_size,
                                            int max_displacement,
                                            int stride1,
                                            int stride2)
  {
    stride1 = stride2 = kernel_size = 1;
    // n (batch size), c (num of channels), y (height), x (width)

//    int n = item;  ok
//    int y = blockIdx.x * stride1;  ok
//    int x = blockIdx.y * stride1 + pad_size;  // spread to threadIdx
//    int c = blockIdx.z;  ok
//    int tch_off = threadIdx.x;
    int n = blockIdx.x;
    int c = blockIdx.y;
    int y = blockIdx.z * stride1;

    int kernel_rad = (kernel_size - 1) / 2;
    int displacement_rad = max_displacement / stride2;
//    int displacement_size = 2 * displacement_rad + 1;

    int ymin = (y - kernel_rad - 0) / stride1;
    int ymax = (y + kernel_rad - 0) / stride1;

    ymin = max(0,ymin);
    ymax = min(outputHeight-1,ymax);

    int pInputWidth = inputWidth + 2 * pad_size;
    int pInputHeight = inputHeight + 0;

    int pdimcyx = nInputChannels * pInputHeight * pInputWidth;
    int pdimyx = pInputHeight * pInputWidth;
    int pdimx = pInputWidth;

    int tdimcyx = nOutputChannels * outputHeight * outputWidth;
    int tdimyx = outputHeight * outputWidth;
    int tdimx = outputWidth;

    int odimcyx = nInputChannels * inputHeight* inputWidth;
    int odimyx = inputHeight * inputWidth;
    int odimx = inputWidth;

    float nelems = kernel_size * kernel_size * nInputChannels;

    for(int xid=threadIdx.x; xid < inputWidth; xid+=CUDA_NUM_THREADS){


        int x = xid * stride1 + pad_size;

        int xmin = (x - kernel_rad - max_displacement) / stride1;
        int xmax = (x + kernel_rad - max_displacement) / stride1;

        if (xmax < 0 || ymax < 0 || xmin >= outputWidth || ymin >= outputHeight || xmin > xmax || ymin > ymax) {
            // assumes gradInput1 is pre-allocated and zero filled
            continue;
        }

        xmin = max(0,xmin);
        xmax = min(outputWidth-1,xmax);

    //    __shared__ float prod_sum[CUDA_NUM_THREADS];
        float cur_prod_sum = 0;

        for (int tc = 0; tc < nOutputChannels; tc += 1) {

          int i2 = (tc - displacement_rad) * stride2;
          int j2 = 0;

          int indx2 = n * pdimcyx + c * pdimyx + (y + j2)* pdimx + (x + i2);

          float val2 = rInput2[indx2];

          for (int j = ymin; j <= ymax; ++j) {
            for (int i = xmin; i <= xmax; ++i) {
              int tindx = n * tdimcyx + tc * tdimyx + j * tdimx + i;
              cur_prod_sum += gradOutput[tindx] * val2;
            }
          }
        }
        const int indx1 = n * odimcyx + c * odimyx + y * odimx + (x - pad_size);
        gradInput1[indx1] = cur_prod_sum / nelems;
    }
}

__global__ void Correlation1d_backward_input2(int item_xx, float *gradInput2, int nInputChannels, int inputHeight, int inputWidth,
                                            float *gradOutput, int nOutputChannels, int outputHeight, int outputWidth,
                                            float *rInput1,
                                            int pad_size,
                                            int kernel_size,
                                            int max_displacement,
                                            int stride1,
                                            int stride2)
{
    stride1 = stride2 = kernel_size = 1;
    // n (batch size), c (num of channels), y (height), x (width)

//    int n = item;
//    int y = blockIdx.x * stride1;
//    int x = blockIdx.y * stride1 + pad_size;
//    int c = blockIdx.z;
//    int tch_off = threadIdx.x;
    int n = blockIdx.x;
    int c = blockIdx.y;
    int y = blockIdx.z * stride1;

    int kernel_rad = (kernel_size - 1) / 2;
    int displacement_rad = max_displacement / stride2;
//    int displacement_size = 2 * displacement_rad + 1;

    int pInputWidth = inputWidth + 2 * pad_size;
    int pInputHeight = inputHeight + 0;

//    int pdimyxc = pInputHeight * pInputWidth * nInputChannels;
//    int pdimxc = pInputWidth * nInputChannels;
//    int pdimc = nInputChannels;
    int pdimcyx = nInputChannels * pInputHeight * pInputWidth;
    int pdimyx = pInputHeight * pInputWidth;
    int pdimx = pInputWidth;

    int tdimcyx = nOutputChannels * outputHeight * outputWidth;
    int tdimyx = outputHeight * outputWidth;
    int tdimx = outputWidth;

    int odimcyx = nInputChannels * inputHeight* inputWidth;
    int odimyx = inputHeight * inputWidth;
    int odimx = inputWidth;

    float nelems = kernel_size * kernel_size * nInputChannels;

    for(int xid=threadIdx.x; xid<inputWidth; xid+=CUDA_NUM_THREADS){
        int x = xid * stride1 + pad_size;

//        __shared__ float prod_sum[CUDA_NUM_THREADS];
//        prod_sum[tch_off] = 0;

        float cur_prod_sum = 0;

        for (int tc = 0; tc < nOutputChannels; tc += 1) {
          int i2 = (tc - displacement_rad) * stride2;
          int j2 = 0;

          int xmin = (x - kernel_rad - max_displacement - i2) / stride1;
          int ymin = (y - kernel_rad - 0 - j2) / stride1;

          int xmax = (x + kernel_rad - max_displacement - i2) / stride1;
          int ymax = (y + kernel_rad - 0 - j2) / stride1;

          if (xmax < 0 || ymax < 0 || xmin >= outputWidth || ymin >= outputHeight || xmin > xmax || ymin > ymax) {
              // assumes gradInput2 is pre-allocated and zero filled
              continue;
          }

          xmin = max(0,xmin);
          xmax = min(outputWidth-1,xmax);

          ymin = max(0,ymin);
          ymax = min(outputHeight-1,ymax);

          int indx1 = n * pdimcyx + c * pdimyx + (y - j2)* pdimx + (x - i2);
          float val1 = rInput1[indx1];

          for (int j = ymin; j <= ymax; ++j) {
            for (int i = xmin; i <= xmax; ++i) {
              int tindx = n * tdimcyx + tc * tdimyx + j * tdimx + i;
              cur_prod_sum += gradOutput[tindx] * val1;
            }
          }
        }
        const int indx2 = n * odimcyx + c * odimyx + y * odimx + (x - pad_size);
        gradInput2[indx2] = cur_prod_sum / nelems;

    }
}

int correlation1d_forward_cuda_kernel(/*THCudaTensor_data(state, output)*/ float *output,
                                    /*THCudaTensor_size(state, output, 0)*/ int ob,
                                    /*THCudaTensor_size(state, output, 1)*/ int oc,
                                    /*THCudaTensor_size(state, output, 2)*/ int oh,
                                    /*THCudaTensor_size(state, output, 3)*/ int ow,
                                    /*THCudaTensor_stride(state, output, 0)*/ int osb,
                                    /*THCudaTensor_stride(state, output, 1)*/ int osc,
                                    /*THCudaTensor_stride(state, output, 2)*/ int osh,
                                    /*THCudaTensor_stride(state, output, 3)*/ int osw,

                                    /*THCudaTensor_data(state, input1)*/ float *input1,
                                    /*THCudaTensor_size(state, input1, 1)*/ int ic,
                                    /*THCudaTensor_size(state, input1, 2)*/ int ih,
                                    /*THCudaTensor_size(state, input1, 3)*/ int iw,
                                    /*THCudaTensor_stride(state, input1, 0)*/ int isb,
                                    /*THCudaTensor_stride(state, input1, 1)*/ int isc,
                                    /*THCudaTensor_stride(state, input1, 2)*/ int ish,
                                    /*THCudaTensor_stride(state, input1, 3)*/ int isw,

                                    /*THCudaTensor_data(state, input2)*/ float *input2,
                                    /*THCudaTensor_size(state, input2, 1)*/ int gc,
                                    /*THCudaTensor_stride(state, input2, 0)*/ int gsb,
                                    /*THCudaTensor_stride(state, input2, 1)*/ int gsc,
                                    /*THCudaTensor_stride(state, input2, 2)*/ int gsh,
                                    /*THCudaTensor_stride(state, input2, 3)*/ int gsw,

                                    /*THCudaTensor_data(state, rInput1)*/ float *rInput1,
                                    /*THCudaTensor_data(state, rInput2)*/ float *rInput2,
                                    int pad_size,
                                    int kernel_size,
                                    int max_displacement,
                                    int stride1,
                                    int stride2,
                                    int corr_type_multiply,
                                    /*THCState_getCurrentStream(state)*/ cudaStream_t stream)
{
   int batchSize = ob;

   int nInputChannels = ic;
   int inputWidth = iw;
   int inputHeight = ih;

   int nOutputChannels = oc;
   int outputWidth = ow;
   int outputHeight = oh;

   dim3 blocks_grid(batchSize, inputHeight, inputWidth);
   dim3 threads_block(THREADS_PER_BLOCK);

  channels_first<<<blocks_grid,threads_block, 0, stream>>> (input1,rInput1, nInputChannels, inputHeight, inputWidth,pad_size);
  channels_first<<<blocks_grid,threads_block, 0, stream>>> (input2,rInput2, nInputChannels, inputHeight, inputWidth, pad_size);

   dim3 threadsPerBlock(THREADS_PER_BLOCK);
   dim3 totalBlocksCorr(batchSize, outputHeight, outputWidth);

   Correlation1d_forward <<< totalBlocksCorr, threadsPerBlock, 0, stream >>>
                        (output, nOutputChannels, outputHeight, outputWidth,
                         rInput1, nInputChannels, inputHeight, inputWidth,
                         rInput2,
                         pad_size,
                         kernel_size,
                         max_displacement,
                         stride1,
                         stride2);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in Correlation1d_forward_cuda_kernel: %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}

int correlation1d_backward_cuda_kernel(
                                    /*THCudaTensor_data(state, gradOutput)*/    float *gradOutput,
                                    /*THCudaTensor_size(state, gradOutput, 0)*/ int gob,
                                    /*THCudaTensor_size(state, gradOutput, 1)*/ int goc,
                                    /*THCudaTensor_size(state, gradOutput, 2)*/ int goh,
                                    /*THCudaTensor_size(state, gradOutput, 3)*/ int gow,
                                    /*THCudaTensor_stride(state, gradOutput, 0)*/ int gosb,
                                    /*THCudaTensor_stride(state, gradOutput, 1)*/ int gosc,
                                    /*THCudaTensor_stride(state, gradOutput, 2)*/ int gosh,
                                    /*THCudaTensor_stride(state, gradOutput, 3)*/ int gosw,

                                    /*THCudaTensor_data(state, input1)*/        float* input1,
                                    /*THCudaTensor_size(state, input1, 1)*/     int ic,
                                    /*THCudaTensor_size(state, input1, 2)*/     int ih,
                                    /*THCudaTensor_size(state, input1, 3)*/     int iw,
                                    /*THCudaTensor_stride(state, input1, 0)*/   int isb,
                                    /*THCudaTensor_stride(state, input1, 1)*/   int isc,
                                    /*THCudaTensor_stride(state, input1, 2)*/   int ish,
                                    /*THCudaTensor_stride(state, input1, 3)*/   int isw,

                                    /*THCudaTensor_data(state, input2)*/        float *input2,
                                    /*THCudaTensor_stride(state, input2, 0)*/   int gsb,
                                    /*THCudaTensor_stride(state, input2, 1)*/   int gsc,
                                    /*THCudaTensor_stride(state, input2, 2)*/   int gsh,
                                    /*THCudaTensor_stride(state, input2, 3)*/   int gsw,

                                    /*THCudaTensor_data(state, gradInput1)*/    float *gradInput1,
                                    /*THCudaTensor_stride(state, gradInput1, 0)*/ int gisb,
                                    /*THCudaTensor_stride(state, gradInput1, 1)*/ int gisc,
                                    /*THCudaTensor_stride(state, gradInput1, 2)*/ int gish,
                                    /*THCudaTensor_stride(state, gradInput1, 3)*/ int gisw,

                                    /*THCudaTensor_data(state, gradInput2)*/      float *gradInput2,
                                    /*THCudaTensor_size(state, gradInput2, 1)*/   int ggc,
                                    /*THCudaTensor_stride(state, gradInput2, 0)*/ int ggsb,
                                    /*THCudaTensor_stride(state, gradInput2, 1)*/ int ggsc,
                                    /*THCudaTensor_stride(state, gradInput2, 2)*/ int ggsh,
                                    /*THCudaTensor_stride(state, gradInput2, 3)*/ int ggsw,

                                    /*THCudaTensor_data(state, rInput1)*/             float *rInput1,
                                    /*THCudaTensor_data(state, rInput2)*/             float *rInput2,
                                    int pad_size,
                                    int kernel_size,
                                    int max_displacement,
                                    int stride1,
                                    int stride2,
                                    int corr_type_multiply,
                                    /*THCState_getCurrentStream(state)*/cudaStream_t stream)
{

    int batchSize = gob;
    int num = batchSize;

    int nInputChannels = ic;
    int inputWidth = iw;
    int inputHeight = ih;

    int nOutputChannels = goc;
    int outputWidth = gow;
    int outputHeight = goh;

    dim3 blocks_grid(batchSize, nInputChannels, inputHeight);
    dim3 threads_block(THREADS_PER_BLOCK);

    pad_left<<<blocks_grid,threads_block, 0, stream>>> (input1, rInput1, nInputChannels,inputHeight, inputWidth, pad_size);
    pad_left<<<blocks_grid,threads_block, 0, stream>>> (input2, rInput2, nInputChannels, inputHeight, inputWidth, pad_size);

    dim3 threadsPerBlock(CUDA_NUM_THREADS);
    dim3 totalBlocksCorr(num, nInputChannels, inputHeight);

//    for (int n = 0; n < num; ++n) {
        Correlation1d_backward_input1 << <totalBlocksCorr, threadsPerBlock, 0, stream >> > (
            0, gradInput1, nInputChannels, inputHeight, inputWidth,
            gradOutput, nOutputChannels, outputHeight, outputWidth,
            rInput2,
            pad_size,
            kernel_size,
            max_displacement,
            stride1,
            stride2);
//    }
//
//    for(int n = 0; n < batchSize; n++) {
        Correlation1d_backward_input2<<<totalBlocksCorr, threadsPerBlock, 0, stream>>>(
            0, gradInput2, nInputChannels, inputHeight, inputWidth,
            gradOutput, nOutputChannels, outputHeight, outputWidth,
            rInput1,
            pad_size,
            kernel_size,
            max_displacement,
            stride1,
            stride2);
//    }

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in Correlation1d_backward_cuda_kernel: %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}