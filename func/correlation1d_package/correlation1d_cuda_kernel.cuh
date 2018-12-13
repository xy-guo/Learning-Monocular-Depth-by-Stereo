#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <cuda_runtime.h>

int correlation1d_forward_cuda_kernel(float* output,
    int ob,
    int oc,
    int oh,
    int ow,
    int osb,
    int osc,
    int osh,
    int osw,

    float* input1,
    int ic,
    int ih,
    int iw,
    int isb,
    int isc,
    int ish,
    int isw,

    float*  input2,
    int gc,
    int gsb,
    int gsc,
    int gsh,
    int gsw,

    float* rInput1,
    float* rInput2,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply,
    cudaStream_t stream);


int correlation1d_backward_cuda_kernel(
    float* gradOutput,
    int gob,
    int goc,
    int goh,
    int gow,
    int gosb,
    int gosc,
    int gosh,
    int gosw,

    float*  input1,
    int ic,
    int ih,
    int iw,
    int isb,
    int isc,
    int ish,
    int isw,

    float*  input2,
    int gsb,
    int gsc,
    int gsh,
    int gsw,

    float*  gradInput1,
    int gisb,
    int gisc,
    int gish,
    int gisw,

    float*  gradInput2,
    int ggc,
    int ggsb,
    int ggsc,
    int ggsh,
    int ggsw,

    float*  rInput1,
    float*  rInput2,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply,
    cudaStream_t stream);