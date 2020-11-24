//#pragma once

#include <stdio.h>

#include<gl\glew.h>
#include<gl\GL.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
#include <cufft.h>

#ifndef DEFINES_H
#define DEFINES_H



#define		DIM			2048		//	Square size of solver DOMAIN
#define		DS			(DIM*DIM)	//	Total Domain Size
#define		CPADW		(DIM/2+1)	//	Padded width for real->complex in-place FFT
#define		RPADW		(2*(DIM/2+1))// Paded width for real->complex in-place FFT
#define		PDS			(DIM*CPADW)	 //	Padded total domain size

#define		DT			0.09f		// Delta T for interactive solver
#define		VIS			0.0025f		// Viscocity Constant
#define		FORCE		(5.8f*DIM)	// Force scale factor
#define		FR			4			// Force Update radius


#define		TILEX		64			// Tile Width
#define		TILEY		64			// Tile height
#define		TIDSX		64			// Tids in X
#define		TIDSY		4			// Tids in Y
#define		MAX(a,b)	((a>b)?a:b)
#endif // !DEFINES_H
