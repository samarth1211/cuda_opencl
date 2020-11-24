#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#include <vector_types.h>

#ifdef USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

struct SimParams
{
	float3 f3Gravity;
	float  fGlobalDamping;
	float  fNoiseFreq;
	float  fNoiseAmp;
	float3 f3CursorPos;

	float  fTime;
	float3 f3NoiseSpeed;
};

#endif
