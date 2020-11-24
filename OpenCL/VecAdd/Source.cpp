
#include<stdio.h>
#include<conio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<strsafe.h>

#include<CL/OpenCL.h>

#include"helper_timer.h"

#pragma comment(lib,"OpenCL.lib")

#define ARRAY_ELEMENTS 11444777


cl_int ret_ocl;
cl_platform_id oclPlatformID;
cl_platform_id oclPlatformIDs[5];
cl_device_id oclComputeDeviceID;
cl_context oclContext;
cl_command_queue oclCommandQueue;
cl_program oclProgram;
cl_kernel oclKernel;

char *chOCLSourceCode = NULL;
size_t sizeKernelCodeLength;

int iNumberOfArrayElements = ARRAY_ELEMENTS;
size_t localWorkSize = 1024;
size_t globalWorkSize;

float *hostInput1 = NULL;
float *hostInput2 = NULL;
float *hostOutput = NULL;
float *gold = NULL;

cl_mem deviceInput1 = NULL;
cl_mem deviceInput2 = NULL;
cl_mem deviceOutput = NULL;

float timeOnCPU;
float timeOnGPU;

cl_uint iNumber = 0;

void main(void)
{
	void fillFloatArrayaWithRandomNumbers(float *, int);
	size_t roundGlobalSizetoNearestMultipleofLocalSize(int, unsigned int);
	void vedAddHost(const float*, const float*,float*,int);
	char* loadOclProgramSource(const char *, const char *, size_t*);
	void CleanUp(void);


	// allocate host memory
	hostInput1 = (float*)calloc(iNumberOfArrayElements, sizeof(float));
	if (hostInput1 == NULL)
	{
		printf("CPU MEMORY Fatal Error = Can Not Allocate Enough Memory for Host hostInput1 \nInput Array 1\nExitting....!!!");
		exit(EXIT_FAILURE);
	}

	hostInput2 = (float*)calloc(iNumberOfArrayElements, sizeof(float));
	if (hostInput1 == NULL)
	{
		printf("CPU MEMORY Fatal Error = Can Not Allocate Enough Memory for Host hostInput2\nInput Array 2\nExitting....!!!");
		exit(EXIT_FAILURE);
	}

	hostOutput = (float*)calloc(iNumberOfArrayElements, sizeof(float));
	if (hostInput1 == NULL)
	{
		printf("CPU MEMORY Fatal Error = Can Not Allocate Enough Memory for Host hostOutput\nOutput Array \nExitting....!!!");
		exit(EXIT_FAILURE);
	}

	gold = (float*)calloc(iNumberOfArrayElements, sizeof(float));
	if (hostInput1 == NULL)
	{
		printf("CPU MEMORY Fatal Error = Can Not Allocate Enough Memory for Host Gold\nOutput Array \nExitting....!!!");
		exit(EXIT_FAILURE);
	}

	fillFloatArrayaWithRandomNumbers(hostInput1, iNumberOfArrayElements);
	fillFloatArrayaWithRandomNumbers(hostInput2, iNumberOfArrayElements);

	// get OPenCl suppourting platforms

	// ret_ocl = clGetPlatformIDs(0, NULL, &iNumber);
	// if (ret_ocl != CL_SUCCESS)
	// {
	// 	printf("OpenCL Error : clGetPlatformIDs Failed : %d. \nExitting Now..\n", ret_ocl);
	// 	CleanUp();
	// 	exit(EXIT_FAILURE);
	// }

	ret_ocl = clGetPlatformIDs(1,&oclPlatformID,NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clGetPlatformIDs Failed : %d. \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	// get OpenCL suppourting GPU device's id
	ret_ocl = clGetDeviceIDs(oclPlatformID,CL_DEVICE_TYPE_GPU,1,&oclComputeDeviceID,NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clGetDeviceIDs Failed : %d. \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	char gpu_name[255];
	clGetDeviceInfo(oclComputeDeviceID,CL_DEVICE_NAME,sizeof(gpu_name),&gpu_name,NULL);
	printf("SAM : GPU Name %s\n",gpu_name);

	oclContext = clCreateContext(NULL,1,&oclComputeDeviceID,NULL,NULL,&ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clCreateContext Failed : %d. \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	// create command queue
	oclCommandQueue = clCreateCommandQueue(oclContext,oclComputeDeviceID,0,&ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clCreateCommandQueue Failed : %d. \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}
	
	// create opncl code file from given file
	chOCLSourceCode = loadOclProgramSource("VecAdd.cl","",&sizeKernelCodeLength);

	cl_int status = 0;
	oclProgram = clCreateProgramWithSource(oclContext,1,(const char **)&chOCLSourceCode,&sizeKernelCodeLength,&ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clCreateProgramWithSource Failed : %d. \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	// Build OpenCL Program
	ret_ocl = clBuildProgram(oclProgram,0,NULL, NULL, NULL, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clBuildProgram Failed : %d. \nExitting Now..\n", ret_ocl);

		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(oclProgram, oclComputeDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("OpenCL Program Build log : %s \n", buffer);

		CleanUp();
		exit(EXIT_FAILURE);
	}

	// Craete OpenCl kernel function
	oclKernel = clCreateKernel(oclProgram,"VecAdd",&ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clCreateKernel Failed : %d. \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	int iSize = iNumberOfArrayElements * sizeof(cl_float);

	deviceInput1 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, iSize, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clCreateBuffer Failed : %d. \nFor Input Buffer 1 \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	deviceInput2 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, iSize, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clCreateBuffer Failed : %d. \nFor Input Buffer 2 \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	// Output of OpenCL
	deviceOutput = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, iSize, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clCreateBuffer Failed : %d. \nFor Input deviceOutput \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	// Set kernel arguments
	ret_ocl = clSetKernelArg(oclKernel,0,sizeof(cl_mem),(void*)&deviceInput1);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clSetKernelArg Failed : %d. \nFor Device Input Buffer 1 \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	ret_ocl = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void*)&deviceInput2);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clSetKernelArg Failed : %d. \nFor Device Input Buffer 2 \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	ret_ocl = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void*)&deviceOutput);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clSetKernelArg Failed : %d. \nFor Device Output Buffer 1 \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	ret_ocl = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void*)&iNumberOfArrayElements);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clSetKernelArg Failed : %d. \nFor Length \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	ret_ocl = clEnqueueWriteBuffer(oclCommandQueue,deviceInput1,CL_FALSE,0,iSize,hostInput1,0,NULL,NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clEnqueueWriteBuffer Failed : %d. \nFor Device Input Buffer 1 \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}


	ret_ocl = clEnqueueWriteBuffer(oclCommandQueue, deviceInput2, CL_FALSE, 0, iSize, hostInput2, 0, NULL, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clEnqueueWriteBuffer Failed : %d. \nFor Device Input Buffer 2 \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	globalWorkSize = roundGlobalSizetoNearestMultipleofLocalSize((int)localWorkSize,iNumberOfArrayElements);

	IStopWatchTimer *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	ret_ocl = clEnqueueNDRangeKernel(oclCommandQueue,oclKernel,1,NULL,&globalWorkSize,&localWorkSize,0,NULL,NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		printf("OpenCL Error : clEnqueueNDRangeKernel Failed : %d. \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	clFinish(oclCommandQueue);

	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	// read back results from the device (i.e from deviceOutput) into CPU variable

	ret_ocl = clEnqueueReadBuffer(oclCommandQueue,deviceOutput,CL_TRUE,0,iSize,hostOutput,0,NULL,NULL);
	if (ret_ocl!=CL_SUCCESS)
	{
		printf("OpenCL Error : clEnqueueReadBuffer Failed : %d. \nExitting Now..\n", ret_ocl);
		CleanUp();
		exit(EXIT_FAILURE);
	}

	vedAddHost(hostInput1,hostInput2,gold,iNumberOfArrayElements);


	// comaper results for golden-host
	const float epsillon = 0.000001f;
	bool bAccuracy = true;
	int breakValue = 0;
	int i;
	for ( i = 0; i < iNumberOfArrayElements; i++)
	{
		float val1 = gold[i];
		float val2 = hostOutput[i];
		if (fabs(val1-val2) > epsillon)
		{
			bAccuracy = false;
			breakValue = i;
			break;
		}
	}

	if (bAccuracy==false)
	{
		printf("Break Value = %d\n",breakValue);
	}

	char str[125];
	if (bAccuracy == true)
		sprintf_s(str, "%s", "Comparison Of Output Arrays On CPU And GPU Are Accurate Within The Limit Of 0.000001");
	else
		sprintf_s(str, "%s", "Not All Comparison Of Output Arrays On CPU And GPU Are Accurate Within The Limit Of 0.000001");

	printf("1st Array Is From 0th Element %.6f To %dth Element %.6f\n", hostInput1[0], iNumberOfArrayElements - 1, hostInput1[iNumberOfArrayElements - 1]);
	printf("2nd Array Is From 0th Element %.6f To %dth Element %.6f\n", hostInput2[0], iNumberOfArrayElements - 1, hostInput2[iNumberOfArrayElements - 1]);
	printf("Global Work Size = %u And Local Work Size Size = %u\n", (unsigned int)globalWorkSize, (unsigned int)localWorkSize);
	printf("Sum Of Each Element From Above 2 Arrays Creates 3rd Array As :\n");
	printf("3rd Array Is From 0th Element %.6f To %dth Element %.6f\n", hostOutput[0], iNumberOfArrayElements - 1, hostOutput[iNumberOfArrayElements - 1]);
	printf("The Time Taken To Do Above Addition On CPU = %.6f (ms)\n", timeOnCPU);
	printf("The Time Taken To Do Above Addition On GPU = %.6f (ms)\n", timeOnGPU);
	printf("%s\n", str);

}



void fillFloatArrayaWithRandomNumbers(float *pFloatArray, int iSize)
{
	int i;
	const float fScale = 1.0f / (float)RAND_MAX;
	for ( i = 0; i < iSize; i++)
	{
		pFloatArray[i] = fScale * rand();
	}

}

size_t roundGlobalSizetoNearestMultipleofLocalSize(int local_size, unsigned int global_size)
{
	unsigned int r = global_size % local_size;
	if (r==0)
	{
		return(global_size);
	}
	else
	{
		return(global_size + local_size - r);
	}
}

void vedAddHost(const float *in1, const float *in2, float *out, int iSize)
{
	int i;

	IStopWatchTimer *timer = NULL;
	sdkCreateTimer(&timer);
	
	sdkStartTimer(&timer);

	for (i = 0; i < iSize; i++)
	{
		out[i] = in1[i] + in2[i];
	}

	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
}

char* loadOclProgramSource(const char *filename, const char *preamble, size_t *iSize)
{
	FILE *pFile = NULL;
	size_t sizeSourceLength;
	size_t sizePreambleLength = (size_t)strlen(preamble);

	if(fopen_s(&pFile, filename, "rb") !=0)
	{
		return NULL;
	}

	fseek(pFile,0,SEEK_END);
	sizeSourceLength = ftell(pFile);
	fseek(pFile,0,SEEK_SET);

	char *sourceString = (char*)calloc(sizeSourceLength+ sizePreambleLength+1,sizeof(char));
	memcpy(sourceString, preamble, sizePreambleLength);//push preabmble

	if (fread( (sourceString)+sizePreambleLength, sizeSourceLength,1,pFile) !=1)
	{
		fclose(pFile);
		free(sourceString);
		return 0;
	}


	fclose(pFile);
	if (iSize !=0)
	{
		*iSize = sizeSourceLength + sizePreambleLength + 1;
	}

	sourceString[sizeSourceLength + sizePreambleLength] = '\0';

	return sourceString;
}

void CleanUp(void)
{

	if (chOCLSourceCode)
	{
		free((void*)chOCLSourceCode);
		chOCLSourceCode = NULL;
	}

	if (oclKernel)
	{
		clReleaseKernel(oclKernel);
		oclKernel = NULL;
	}

	if (oclProgram)
	{
		clReleaseProgram(oclProgram);
		oclProgram = NULL;
	}

	if (oclCommandQueue)
	{
		clReleaseCommandQueue(oclCommandQueue);
		oclCommandQueue = NULL;
	}

	if (oclContext)
	{
		clReleaseContext(oclContext);
		oclContext = NULL;
	}

	if (deviceInput1)
	{
		clReleaseMemObject(deviceInput1);
		deviceInput1 = NULL;
	}

	if (deviceInput2)
	{
		clReleaseMemObject(deviceInput2);
		deviceInput2 = NULL;
	}

	if (deviceOutput)
	{
		clReleaseMemObject(deviceOutput);
		deviceOutput = NULL;
	}

	// Free all CPU memory
	if (hostInput1)
	{
		free(hostInput1);
		hostInput1 = NULL;
	}

	if (hostInput2)
	{
		free(hostInput2);
		hostInput2 = NULL;
	}

	if (hostOutput)
	{
		free(hostOutput);
		hostOutput = NULL;
	}

	_getch();
}

