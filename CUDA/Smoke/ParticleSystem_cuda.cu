

#include<gl\glew.h>
#include<gl\GL.h>

#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cuda_gl_interop.h>
#include<vector_types.h>

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

#include"Particles_kernel_device.cuh"
#include"ParticleSystem.cuh"



extern "C"
{

	cudaArray *noiseArray;

	void setParameters(SimParams * hostParams)
	{
		cudaError status = cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams));
		if (status != cudaSuccess)
		{
			fprintf(stdout, "Failed to cudaMemcpyToSymbol\n");
			exit(EXIT_FAILURE);
		}
	}
	
	// Round a/b to nearest higher integer value
	int iDivUp(int a, int b)
	{
		return (a%b != 0) ? (a / b + 1) : (a / b);
	}

	// Compute grid and thread block size for a given number of elements
	void ComputeGridSize(int n, int blockSize, int &numBlocks, int &numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = iDivUp(n, numThreads);
	}

	inline float frand()
	{
		return rand() / (float)RAND_MAX;
	}

	// Create 3D texture having random values
	void createNoiseTexture(int w, int h, int d)
	{
		cudaError status;
		cudaExtent size = make_cudaExtent(w,h,d);
		size_t elements = size.width * size.height * size.depth;

		float *volumeData = (float*)calloc(elements,sizeof(float)*4);
		float *ptr = volumeData;

		for (size_t i = 0; i < elements; i++)
		{
			*ptr++ = frand()* 2.0f - 1.0f;
			*ptr++ = frand()* 2.0f - 1.0f;
			*ptr++ = frand()* 2.0f - 1.0f;
			*ptr++ = frand()* 2.0f - 1.0f;
		}

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		status = cudaMalloc3DArray(&noiseArray, &channelDesc, size);
		if (status != cudaSuccess)
		{
			fprintf(stdout, "Failed to cudaMalloc3DArray\n");
			exit(EXIT_FAILURE);
		}

		cudaMemcpy3DParms copyParams = {0};
		copyParams.srcPtr = make_cudaPitchedPtr((void*)volumeData,size.width*sizeof(float4),size.width,size.height);
		copyParams.dstArray = noiseArray;
		copyParams.extent = size;
		copyParams.kind = cudaMemcpyHostToDevice;

		status = cudaMemcpy3D(&copyParams);
		if (status != cudaSuccess)
		{
			fprintf(stdout, "Failed to cudaMemcpy3D\n");
			exit(EXIT_FAILURE);
		}

		free(volumeData);
		volumeData = NULL;

		// set texture parameters
		noiseTex.normalized = true;	// access with normalized texture coordinates
		noiseTex.filterMode = cudaFilterModeLinear; // linear interpolation
		noiseTex.addressMode[0] = cudaAddressModeWrap; // wrap texture coordinates
		noiseTex.addressMode[1] = cudaAddressModeWrap;
		noiseTex.addressMode[2] = cudaAddressModeWrap;

		// bind Array to 3D texture
		status = cudaBindTextureToArray(noiseTex, noiseArray, channelDesc);
		if (status != cudaSuccess)
		{
			fprintf(stdout, "Failed to cudaBindTextureToArray\n");
			exit(EXIT_FAILURE);
		}

	}

	void integrateSystem(float4 * oldPos, float4 * newPos, float4 * oldVel, float4 * newVel, float deltaTime, int numParticles)
	{
		thrust::device_ptr<float4> d_newPos(newPos);
		thrust::device_ptr<float4> d_oldPos(oldPos);
		thrust::device_ptr<float4> d_newVel(newVel);
		thrust::device_ptr<float4> d_oldVel(oldVel);
		
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(d_newPos, d_newVel, d_oldPos, d_oldVel)),
		thrust::make_zip_iterator(thrust::make_tuple(d_newPos + numParticles, d_newVel + numParticles, d_oldPos + numParticles, d_oldVel + numParticles)),
			intergate_functor(deltaTime));
	}

	void calcDepth(float4 *pos, float *keys, unsigned int *indices, float3 sortVector, int numParticles)
	{
		thrust::device_ptr<float4> d_pos(pos);
		thrust::device_ptr<float> d_keys(keys);
		thrust::device_ptr<unsigned int> d_indices(indices);

		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(d_pos, d_keys)),
			thrust::make_zip_iterator(thrust::make_tuple(d_pos + numParticles, d_keys + numParticles)),
			calcDepth_functor(sortVector));


		thrust::sequence(d_indices, d_indices + numParticles);
		
	}

	void sortParticles(float *sortKeys, unsigned int *indices, unsigned int numParticles)
	{
		thrust::sort_by_key(thrust::device_ptr<float>(sortKeys), thrust::device_ptr<float>(sortKeys + numParticles), thrust::device_ptr<unsigned int>(indices));
	}

}
