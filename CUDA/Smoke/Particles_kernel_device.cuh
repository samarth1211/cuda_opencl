#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "particles_kernel.cuh"
#include <vector_types.h>
#include "helper_math.h"

texture<float4, 3, cudaReadModeElementType> noiseTex;

// Simulation Parameters
__constant__ SimParams params;

// look up in 3D noise texture
__device__ float3 noise3D(float3 p)
{
	float4 n = tex3D(noiseTex,p.x,p.y,p.z);
	return make_float3(n.x,n.y,n.z);
}

// integrate particle attribbutes
struct intergate_functor
{
	float deltaTime;

	__host__ __device__ intergate_functor(float delta_time) :deltaTime(delta_time) {}

	template <typename Tuple> __device__ void operator()(Tuple t)
	{
		volatile float4 posData = thrust::get<2>(t);
		volatile float4 velData = thrust::get<3>(t);

		float3 pos = make_float3(posData.x, posData.y, posData.z);
		float3 vel = make_float3(velData.x, velData.y, velData.z);

		// update particle age
		float age = posData.w;
		float lifetime = velData.w;

		if (age < lifetime)
		{
			age += deltaTime;
		}
		else
		{
			age = lifetime;
		}

		// apply accelerations
		vel += params.f3Gravity*deltaTime;

		// apply procedural noise
		float3 noise = noise3D(pos*params.fNoiseFreq + params.fTime * params.f3NoiseSpeed);
		vel += noise * params.fNoiseAmp;

		// new position = old position + velocity * deltaTime
		pos += vel * deltaTime;
		vel *= params.fGlobalDamping;

		// store new position and velocity
		thrust::get<0>(t) = make_float4(pos.x, pos.y, pos.z,age);
		thrust::get<1>(t) = make_float4(vel.x, vel.y, vel.z, velData.w);

	}

};

struct calcDepth_functor
{
	float3 sortVector;

	__host__ __device__ calcDepth_functor(float3 sort_vector) :sortVector(sort_vector) {}

	template<typename Tuple> __host__ __device__ void operator()(Tuple t)
	{
		volatile float4 p = thrust::get<0>(t);
		// project onto sort vector
		float key = -dot(make_float3(p.x,p.y,p.z),sortVector); 
		thrust::get<1>(t) = key;
	}
};

#endif