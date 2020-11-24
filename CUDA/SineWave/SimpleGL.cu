
#include "SimpleGL.cuh"

__global__ void simple_vbo_kernel(float4 * pos, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = (float)x / (float)width;
	float v = (float)y / (float)height;

	u = u*2.0f - 1.0f;
	v = v*2.0f - 1.0f;

	float freq = 5.0f;
	float w = cosf(u*freq + time) * sinf(v*freq + time) * 0.5f;

	pos[y*width + x] = make_float4(u, w, v, 1.0f);
}

void RunCUDA(cudaGraphicsResource ** vbo_resource, unsigned int mesh_width, unsigned int mesh_height, float time)
{
	void LaunchKernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);
	// Map OpenGL buffer with CUDA
	
	cudaError status;

	float4 *dptr;
	status = cudaGraphicsMapResources(1, vbo_resource, 0);
	if (status != cudaSuccess)
	{
		//fprintf_s(g_pFile, "cudaGraphicsMapResources Failed. \n %s:%s \n", cudaGetErrorName(status), cudaGetErrorString(status));
	}

	size_t num_bytes;
	status = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);
	if (status != cudaSuccess)
	{
		//fprintf_s(g_pFile, "cudaGraphicsMapResources Failed. \n");
	}

	// Execute The kernel
	LaunchKernel(dptr, mesh_width, mesh_height, time);

	// Unmap OpenGL buffer with CUDA
	status = cudaGraphicsUnmapResources(1, vbo_resource, 0);
	if (status != cudaSuccess)
	{
		//fprintf_s(g_pFile, "cudaGraphicsMapResources Failed. \n");
	}
}

void LaunchKernel(float4 * pos, unsigned int mesh_width, unsigned int mesh_height, float time)
{
	// Execute The kernel
	dim3 block(32, 32, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	simple_vbo_kernel <<<grid, block >>>(pos, mesh_width, mesh_height, time);
}


void CleanUPCuda(cudaGraphicsResource ** vbo_resource)
{
	cudaError status;

	status = cudaGraphicsUnmapResources(1, vbo_resource, 0);
	if (status != cudaSuccess)
	{
		//fprintf_s(g_pFile, "cudaGraphicsMapResources Failed. \n");
	}

	status = cudaGraphicsUnregisterResource(*vbo_resource);
	if (status != cudaSuccess)
	{
		//fprintf_s(g_pFile, "cudaGraphicsUnregisterResource Failed. \n");
	}
	cudaDeviceReset();
}

