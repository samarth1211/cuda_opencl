#pragma once


#ifndef __SIMPLE_GL__
#define __SIMPLE_GL__

#define _USE_MATH_DEFINES 1

//#include<stdio.h>
//
//#include<gl\glew.h>
//#include<gl\GL.h>

#include"CommonHeader.h"

#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cuda_gl_interop.h>
#include<vector_types.h>

#include<math.h>

#define MESH_WIDTH	128
#define MESH_HEIGHT	128

__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time);
void RunCUDA(cudaGraphicsResource ** vbo_resource, unsigned int mesh_width, unsigned int mesh_height, float time);
void LaunchKernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);
void CleanUPCuda(cudaGraphicsResource ** vbo_resource);

//extern struct cudaGraphicsResource *g_cuda_vbo_resource = 0;

#endif // __SIMPLE_GL__
