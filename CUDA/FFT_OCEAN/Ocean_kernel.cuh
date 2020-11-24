#include<gl\glew.h>
#include<gl\GL.h>
// CUDA standard includes
#include<cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include<cufft.h>
#include"vmath.h"

// wrapper functions

extern "C" void cuda_GenerateSpectrumKernel(float2 *d_h0, float2 *d_ht, unsigned int in_width, unsigned int out_width, unsigned int out_height, float animeTime, float patchsize);


extern "C" void cuda_UpdateHeightMapKernel(float *d_heightMap,float2 *d_ht,unsigned int width, unsigned int height,bool autoTest);


extern "C" void cuda_CalculateSlopKernel(float *hptr,float2 *slopeOut,unsigned int width,unsigned int height);