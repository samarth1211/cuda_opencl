#ifndef __STABLE_FLUIDS_KERNELS_CUH__
#define __STABLE_FLUIDS_KERNELS_CUH__

#include "Defines.h"

typedef float2 cData;

void setupTexture(int x, int y);
void bindTexture();
void unbindTexture();
void updateTexture(cData *data, size_t w, size_t h, size_t pitch);
void deleteTexture();

/*
this method adds constants force vectors to the velociity field
stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
*/
__global__ void adForces_K(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch);

/*
This method performs the velocity aadvection step, where we trace
velocity vectors back in time to update each grid cell.
That is, v(x,t+1) = v(p(x,-dt),t), Here we perform bilinear inerpolation
in the velocity space
*/
__global__ void advectVelocity_K(cData *v, float *vx, float *vy, int dx, int pdx, int dy, float dt, int lb);


/*
this method performs velocity diffusion and forces mass conservation in the
frequency domain. THe inouts 'vx' and 'vy' arfe complex-valued arrays
holding the fourier coefficients of the velicity field in X and Y.
Diffusion in spcae takes a simple form described as :
v(k,t) = v(k,t) / (1 + visc * dt * k^2)

visc  => viscosity
k	  => wave number
The projection step forces the Fourier velocity vectors to be original to
the weve vector for each wave number:
v(k,t) = v(k,t) - ((k dot v(k,t))* k)/k^2
*/
__global__ void diffuseProject_K(cData *vx, cData *vy, int dx, int dy, float dt, float visc, int lb);


/*
This method updates the velocity field 'v' using the two complex arrays
from the previous step: 'vx' and 'vy'. Here we scale the real components
by 1/(dx*dy) to account for an unnormalized FFT.
*/
__global__ void updateVelocity_K(cData *v, float *vx, float *vy, int dx, int pdx, int dy, int lb, size_t pitch);


/*
This method updates the particles by moving particle positions according to
the velocity field and time step. That is for each particle
p(t+1) = p(t) + dt * v(p(t)).
*/
__global__ void advectParticles_K(cData *part, cData *v, int dx, int dy, float dt, int lb, size_t pitch);

// External funtion calls necessay for launching fluid simulation

extern "C" void addForces(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r);
extern "C" void addvectVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy, float dt);
extern "C" void diffuseProject(cData *vx, cData *vy, int dx, int dy, float dt, float visc);
extern "C" void updateVelocity(cData *v, float *vx, float *vy, int dx, int pdx, int dy);
extern "C" void advectParticles(GLuint vbo, cData *v, int dx, int dy, float dt);


#endif // !__STABLE_FLUIDS_KERNELS_CUH__
