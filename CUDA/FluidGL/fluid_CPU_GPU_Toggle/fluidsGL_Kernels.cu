#include <stdio.h>
#include<stdlib.h>

// helper GL
#include<gl\glew.h>
#include<gl\GL.h>

// CUDA standard includes
#include<cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
#include <cufft.h>

#include "FluidsGL_Kernels.cuh"

texture<float2, 2> texref;
static cudaArray *refArray = NULL;

// particle data
//extern GLuint g_iVertexBufferObject_GPU;		//opengl vertyex buffer object
extern struct cudaGraphicsResource *cuda_vbo_resource; // opengl -cuda interop

// Texture pitch
extern size_t tPitch;
extern cufftHandle planr2c; // cufftHandle => int
extern cufftHandle planc2r; 
extern FILE *g_pFile;
cData_float *vxField = NULL;
cData_float *vyField = NULL;

void setupTexture(int x, int y)
{
	void bindTextureWithDesc(cudaChannelFormatDesc *desc);

	cudaError_t err ;
	// wraap mode set ot defaault
	texref.filterMode = cudaFilterModeLinear;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

	err = cudaMallocArray(&refArray, &desc, y, x);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile,"In setupTexture : cudaMallocArray failed with %s\n", cudaGetErrorString(cudaGetLastError()));
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	//bindTextureWithDesc(&desc);
}

void bindTexture()
{
	cudaError_t err;

	texref.filterMode = cudaFilterModeLinear;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

	err = cudaBindTextureToArray(texref, refArray);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "In bindTexture : cudaBindTextureToArray failed with %s\n", cudaGetErrorString(cudaGetLastError()));
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

/*
void bindTextureWithDesc(cudaChannelFormatDesc *desc)
{
	cudaError_t err;

	err = cudaBindTextureToArray(texref, refArray, desc);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "In bindTextureWithDesc : cudaMallocArray failed with %s\n", cudaGetErrorString(cudaGetLastError()));
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}*/

void unbindTexture()
{
	cudaError_t err;

	err = cudaUnbindTexture(texref);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "In unbindTexture : cudaUnbindTexture failed with %s\n", cudaGetErrorString(cudaGetLastError()));
		//cudaDeviceReset();
		//exit(EXIT_FAILURE);
	}

}

void updateTexture(cData_float * data, size_t w, size_t h, size_t pitch)
{
	cudaError_t err;

	err = cudaMemcpy2DToArray(refArray,0,0,data,pitch,w,h,cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "In updateTexture : cudaMemcpy2DToArray failed with %s\n", cudaGetErrorString(cudaGetLastError()));
		//cudaDeviceReset();
		//exit(EXIT_FAILURE);
	}
}

void deleteTexture()
{
	cudaError_t err;
	err = cudaFreeArray(refArray);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "In deleteTexture : cudaFreeArray failed with %s\n", cudaGetErrorString(cudaGetLastError()));
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}


/*
This method adds constant force vectors to the velocity field
stored in 'v' according to v(x,t+1) = v(x,t) + dt * f.
*/
__global__ void adForces_K(cData_float *v, int dx, int dy, int spx, int spy, float fx, float fy, int r, size_t pitch)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	cData_float *fj = (cData_float*)((char*)v + (ty+spy)*pitch)+tx + spx;

	cData_float vterm = *fj;
	tx -= r;
	ty -= r;

	float s = 1.0f / (1.0f + tx*tx*tx*tx + ty*ty*ty*ty);
	vterm.x += s*fx;
	vterm.y += s*fy;
	*fj = vterm;
}

/*
velocity advection step: 
trace velocity vector back in time to update each grid cell.
v(x,t+1) = v(p(x,-dt),t);
bi-linear interpolation is calculated in valocity space.
*/
__global__ void advectVelocity_K(cData_float *v, float *vx,float *vy, int dx,int pdx, int dy, float dt, int lb)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb* blockDim.y) + threadIdx.y * lb;

	cData_float vterm, ploc;
	float vxterm, vyterm;

	//gtidx is the domain location in x for this thread
	if (gtidx < dx)
	{
		for (int p = 0; p < lb; p++)
		{
			int fi = gtidy + p; // fi domain location in y for this thread
			if (fi < dy)
			{
				int fj = fi * pdx + gtidx;
				vterm = tex2D(texref,(float)gtidx,(float)fi);
				ploc.x = (gtidx + 0.5f) - (dt*vterm.x*dx);
				ploc.y = (fi + 0.5f) - (dt * vterm.y * dy);
				vterm = tex2D(texref,ploc.x,ploc.y);
				vxterm = vterm.x;
				vyterm = vterm.y;
				vx[fj] = vxterm;
				vy[fj] = vyterm;
			}
		}
	}
}

/*
This method performs velocity diffusion and forces mass consevation
in the frequency domain, the inputs 'vx' and 'vy' are complex valued
arrays holding the Fourier coefficients of the velocity field in x and y.
Diffusion in this space takes a simple from described as:
v(k,t) = v(k,t) / (1 + visc * dt * k^2)
v => viscocity
k = > wave number
the projection step forces the Fourier Velocity vectors to be orthogonal
to the to ther vectors for each wave number as:
v(k,t) = v(k,t) - ((k dot v(k,t))* k) / k^2;
*/
__global__ void diffuseProject_K(cData_float *vx, cData_float *vy, int dx, int dy, float dt, float visc, int lb)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb * blockDim.y) + threadIdx.y * lb;
	int p;
	cData_float xterm, yterm;

	// gtidx is the domain location in x for this thread
	if (gtidx < dx)
	{
		for (p = 0; p < lb;p++)
		{
			// fi is the domain location in y for this thread
			int fi = gtidy + p;
			if (fi < dy)
			{
				int fj = fi * dx + gtidx;
				xterm = vx[fj];
				yterm = vy[fj];

				// compute the index of the wavenumber based on the
				// data order produced bt a statement NN FFT.
				int iix = gtidx;
				int iiy = (fi > dy / 2) ? (fi - (dy)) : fi;

				// velocity diffusion
				float kk = (float)(iix*iix + iiy*iiy); // k^2
				float diff = 1.0f / (1.0f + visc * dt * kk);
				xterm.x *= diff;
				xterm.y *= diff;
				yterm.x *= diff;
				yterm.y *= diff;

				if (kk > 0.0f)
				{
					float rkk = 1.0f / kk;
					// real portion of velocity projrction
					float rkp = (iix * xterm.x + iiy * yterm.x);
					// Imaginary portion of velocity projrction
					float ikp = (iix * xterm.y + iiy * yterm.y);
					xterm.x -= rkk * rkp * iix;
					xterm.y -= rkk * ikp * iix;
					yterm.x -= rkk * rkp * iiy;
					yterm.y -= rkk * ikp * iiy;
				}

				vx[fj] = xterm;
				vy[fj] = yterm;
			}
		}
	}

}

/*
Updates the velocity field 'v' using the two complex arrays from previous 
step: 'vx' and 'vy'.
here we scale the real components by 1/(dx*dy) to account for an 
unnormalized FFT
*/
__global__ void updateVelocity_K(cData_float *v, float *vx, float *vy, int dx, int pdx, int dy, int lb, size_t pitch)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y * (lb* blockDim.y) + threadIdx.y * lb;
	int p;

	float vxterm, vyterm;
	cData_float nvterm;

	// gtidx is the domain location in x for this thread
	if (gtidx < dx)
	{
		for (p = 0; p < lb; p++)
		{
			// fi iss the domain location in y for  this thread
			int fi = gtidy + p;

			if (fi < dy)
			{
				int fjr = fi * pdx + gtidx;
				vxterm = vx[fjr];
				vyterm = vy[fjr];

				// Normalize the result of the inverse FFT
				float scale = 1.0f / (dx*dy);
				nvterm.x = vxterm*scale;
				nvterm.y = vyterm*scale;

				cData_float *fj = (cData_float*)((char*)v + fi * pitch) + gtidx;
				*fj = nvterm;
			}
		} // if this thread is inside the domain in Y
	} // if this thread in inside the domain in X
}

__global__ void advectParticles_K(cData_float * part, cData_float * v, int dx, int dy, float dt, int lb, size_t pitch)
{
	int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gtidy = blockIdx.y *(lb * blockDim.y) + threadIdx.y * lb;
	int p;

	// gtidx is the domain location in x for this thread
	cData_float pterm, vterm;

	if (gtidx < dx)
	{
		for ( p = 0; p < lb; p++)
		{
			// fi is domain location in y for this thread
			int fi = gtidy + p;

			if (fi < dy)
			{
				int fj = fi * dx + gtidx;
				pterm = part[fj];

				int xvi = ((int)(pterm.x * dx));
				int yvi = ((int)(pterm.y * dy));
				vterm = *((cData_float*)((char*)v + yvi * pitch) + xvi);

				pterm.x += dt * vterm.x;
				pterm.x = pterm.x - (int)pterm.x;
				pterm.x += 1.0f;
				pterm.x = pterm.x - (int)pterm.x;
				pterm.y += dt * vterm.y;
				pterm.y = pterm.y - (int)pterm.y;
				pterm.y += 1.0f;
				pterm.y = pterm.y - (int)pterm.y;

				part[fj] = pterm;
			}
		} // if this thread is inside the dom,ain in Y
	} // if this thread is inside the domain in X

}



void addForces(cData_float * v, int dx, int dy, int spx, int spy, float fx, float fy, int r)
{
	cudaError_t err;

	dim3 tids(2*r+1, 2 * r + 1);
	adForces_K <<<1,tids>>>(v,dx,dy,spx,spy,fx,fy,r,tPitch);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int addForces, kernel failed %s \n", cudaGetErrorString(err));
	}
}

void addvectVelocity(cData_float * v, float * vx, float * vy, int dx, int pdx, int dy, float dt)
{
	cudaError_t err;
	dim3 grid( (dx/TILEX)+(!(dx%TILEX)?0:1), 
				(dy / TILEY) + (!(dy%TILEY) ? 0 : 1));
	dim3 tids(TIDSX,TIDSY);

	updateTexture(v, DIM*sizeof(cData_float), DIM, tPitch);
	advectVelocity_K << <grid, tids >> >(v,vx,vy,dx,pdx,dy,dt,TILEY/TIDSY);

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int addvectVelocity, kernel failed %s \n", cudaGetErrorString(err));
	}
}

void diffuseProject(cData_float * vx, cData_float * vy, int dx, int dy, float dt, float visc)
{
	
	cudaError_t err;
	int cufftErrs;
	// Forwarfd FFT
	cufftErrs = cufftExecR2C(planr2c,(cufftReal*)vx,(cufftComplex*)vx);
	if (cufftErrs != (int)CUFFT_SUCCESS)
	{
		fprintf_s(g_pFile, "int diffuseProject, cufftExecR2C 1 failed %d \n", cufftErrs);
	}
	cufftErrs = cufftExecR2C(planr2c, (cufftReal*)vy, (cufftComplex*)vy);
	if (cufftErrs != (int)CUFFT_SUCCESS)
	{
		fprintf_s(g_pFile, "int diffuseProject, cufftExecR2C 2 failed %d \n", cufftErrs);
	}

	uint3 grid = make_uint3((dx / TILEX) + (!(dx%TILEX) ? 0 : 1),
							(dy / TILEY) + (!(dy%TILEY) ? 0 : 1),1);
	uint3 tids = make_uint3(TIDSX,TIDSY,1);

	diffuseProject_K<<<grid, tids>>>(vx,vy,dx,dy,dt,visc,TILEY/TIDSY);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int diffuseProject, kernel failed %s \n", cudaGetErrorString(err));
	}

	cufftErrs = cufftExecC2R(planc2r, (cufftComplex*)vx, (cufftReal*)vx);
	if (cufftErrs != (int)CUFFT_SUCCESS)
	{
		fprintf_s(g_pFile, "int diffuseProject, cufftExecC2R 1 failed %d \n", cufftErrs);
	}
	
	cufftErrs = cufftExecC2R(planc2r, (cufftComplex*)vy, (cufftReal*)vy);
	if (cufftErrs != (int)CUFFT_SUCCESS)
	{
		fprintf_s(g_pFile, "int diffuseProject, cufftExecC2R 2 failed %d \n", cufftErrs);
	}
}

void updateVelocity(cData_float * v, float * vx, float * vy, int dx, int pdx, int dy)
{
	cudaError_t err;

	dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1),(dy/TILEY)+(!(dy%TILEY)?0:1));
	dim3 tids(TIDSX,TIDSY);

	updateVelocity_K<<<grid, tids>>>(v,vx,vy,dx,pdx,dy,TILEY/TIDSY,tPitch);
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int updateVelocity_K, kernel failed %s \n", cudaGetErrorString(err));
	}
}

void advectParticles(GLuint vbo, cData_float * v, int dx, int dy, float dt)
{
	cudaError_t err;

	dim3 grid((dx/TILEX)+(!(dx%TILEX)?0:1),(dy/TILEY)+(!(dy%TILEY)?0:1));
	dim3 tids(TIDSX,TIDSY);

	cData_float *p;

	err = cudaGraphicsMapResources(1, &cuda_vbo_resource,0);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int advectParticles, cudaGraphicsMapResources failed %s \n", cudaGetErrorString(err));
	}

	size_t num_bytes;
	err = cudaGraphicsResourceGetMappedPointer((void**)&p,&num_bytes,cuda_vbo_resource);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int advectParticles, cudaGraphicsResourceGetMappedPointer failed %s \n", cudaGetErrorString(err));
	}

	advectParticles_K <<<grid, tids >>>(p,v,dx,dy,dt,TILEY/TIDSY,tPitch);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int advectParticles, kernel failed %s \n", cudaGetErrorString(err));
	}

	err = cudaGraphicsUnmapResources(1,&cuda_vbo_resource,0);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int advectParticles, cudaGraphicsUnmapResources failed %s \n", cudaGetErrorString(err));
	}

}

