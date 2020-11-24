/* 
    Understand "clGetPlatformIDs" 
    clGetPlatformIDs( 1, 2,3 );
    1 => How many entries are we expecting
            0 means to get the count in last parameter
    2 => in-out param of "cl_platform_id" to get refrences of platform we have fiven in 1st param
            if 1st param is 0, then this must be NULL.
    3 => 
    Program to obtain all Platforms and Devices on our machine
*/

#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <strsafe.h>

#include <CL/opencl.h>

#pragma comment(lib, "OpenCL.lib")

int main (int argc,char **argv, char **envp)
{
    // Platforms
    cl_uint iPlatformCount=0;
    cl_platform_id *pPlatforms = NULL;

    // Repective Devices
    cl_uint iDeviceCount=0;
    cl_device_id *pDevices=NULL;

    // Max Compute Units
    cl_uint iMaxComputeUnits=0;

    // Check return values of OpenCL
    cl_int ret_ocl=-1;

    char *value=NULL;
    size_t iValueSize;

    system("cls");

    ret_ocl = clGetPlatformIDs(0, NULL, &iPlatformCount);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error : Could not get Count of OpenCL Platforms..\n");
        printf("OpenCL Error : clGetPlatformIDs Failed : %d. \nExitting Now..\n", ret_ocl);

        goto LAST_SA;
    }

    printf("OpenCL INFO : Count of OpenCL Platforms => %d\n", iPlatformCount);
    
    pPlatforms = (cl_platform_id *)calloc(iPlatformCount, sizeof(cl_platform_id));
    if (pPlatforms==NULL)
    {
        printf("System Error : Could not allocate Count of OpenCL Platforms..\n");
    }

    // obtain all Platform IDs
    ret_ocl = clGetPlatformIDs(iPlatformCount, pPlatforms,NULL);
    if (ret_ocl != CL_SUCCESS)
    {
        printf("OpenCL Error : Could not get all of OpenCL Platforms..\n");
        printf("OpenCL Error : clGetPlatformIDs Failed : %d. \nExitting Now..\n", ret_ocl);

        goto LAST_SA;
    }

    // Iterate for all platforms and get respective device
    for (cl_uint i = 0; i < iPlatformCount; i++)
    {
        // Get all devices for each platform
        ret_ocl = clGetDeviceIDs(pPlatforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &iDeviceCount);
        if (ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error : Could not get all of OpenCL Devices..\n");
            printf("OpenCL Error : clGetDeviceIDs Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        pDevices = (cl_device_id*)calloc(iDeviceCount, sizeof(cl_device_id));
        if (pDevices==NULL)
        {
            printf("System Error : Could not allocate Count of OpenCL Devices..\n");
        }

        ret_ocl = clGetDeviceIDs(pPlatforms[i], CL_DEVICE_TYPE_ALL, iDeviceCount, pDevices, NULL);
        if (ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error : Could not get specific of OpenCL Devices..\n");
            printf("OpenCL Error : clGetDeviceIDs Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        printf("\nOpenCL INFO : In OpenCL Platform %d and Devices present => %d\n", i+1, iDeviceCount);
        printf("++++++++++++++++ Platform Info ++++++++++++++++++++++\n");

        // name
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_NAME, 0, NULL, &iValueSize);
        if (ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error : Could not get Name count of platform..\n");
            printf("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        value = (char *)calloc(iValueSize, sizeof(char));
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_NAME, iValueSize, value, NULL);
        if (ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error : Could not get Name of platform..\n");
            printf("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }
        printf("Platform Name = %s\n", value);
        free(value);
        value = NULL;

        // Profile
        ret_ocl = clGetPlatformInfo(pPlatforms[i],CL_PLATFORM_PROFILE,0,NULL,&iValueSize);
        if (ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error : Could not get Profile count of platform..\n");
            printf("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        value = (char*)calloc(iValueSize,sizeof(char));
        ret_ocl = clGetPlatformInfo(pPlatforms[i],CL_PLATFORM_PROFILE,iValueSize, value,NULL);
        if (ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error : Could not get Profile of platform..\n");
            printf("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }
        printf("Platform Profile = %s\n", value);
        free(value);
        value=NULL;

        // version
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_VERSION, 0, NULL, &iValueSize);
        if (ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error : Could not get Version count of platform..\n");
            printf("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        value = (char *)calloc(iValueSize, sizeof(char));
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_VERSION, iValueSize, value, NULL);
        if (ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error : Could not get Version of platform..\n");
            printf("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }
        printf("Platform Version = %s\n", value);
        free(value);
        value = NULL;

        // Vendor
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_VENDOR, 0, NULL, &iValueSize);
        if (ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error : Could not get Vendor count of platform..\n");
            printf("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        value = (char *)calloc(iValueSize, sizeof(char));
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_VENDOR, iValueSize, value, NULL);
        if (ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error : Could not get Vendor of platform..\n");
            printf("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }
        printf("Platform Vendor = %s\n", value);
        free(value);
        value = NULL;
        // Extentions
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &iValueSize);
        if (ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error : Could not get Extension count of platform..\n");
            printf("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        value = (char *)calloc(iValueSize, sizeof(char));
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_EXTENSIONS, iValueSize, value, NULL);
        if (ret_ocl != CL_SUCCESS)
        {
            printf("OpenCL Error : Could not get Extension of platform..\n");
            printf("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }
        printf("Platform Extension = %s\n", value);
        free(value);
        value = NULL;

        // Platform LOOP 
        for (cl_uint j = 0; j < iDeviceCount; j++)
        {

            printf("\n");
            printf("************* DEVICE GENERAL INFORMATION ***********\n");
            printf("=====================================================\n");
            // Print Device Type
            cl_device_type devType;
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_TYPE, sizeof(devType), (void *)&devType, NULL);
            
           switch (devType)
            {
                case CL_DEVICE_TYPE_CPU:
                    printf("%d. Device Type : %s \n", j + 1, "CL_DEVICE_TYPE_CPU ");
                    break;
                    
                case CL_DEVICE_TYPE_GPU:
                    printf("%d. Device Type : %s \n", j + 1, "CL_DEVICE_TYPE_GPU ");
                    break;

                case CL_DEVICE_TYPE_ACCELERATOR:
                    printf("%d. Device Type : %s \n", j + 1, "CL_DEVICE_TYPE_ACCELERATOR");
                    break;

                case CL_DEVICE_TYPE_DEFAULT:
                    printf("%d. Device Type : %s \n", j + 1, "CL_DEVICE_TYPE_CPU ");
                    break;
        }

            free(value);
            value = NULL;

            // print Device Name
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_NAME, 0, NULL, &iValueSize);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get Name of OpenCL Device %d..\n",j+1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            value = (char *)calloc(iValueSize, sizeof(char));
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_NAME, iValueSize, value, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get Name of OpenCL Device %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            printf("%d. Device Name : %s \n",j+1,value);
            free(value);
            value=NULL;

            // Device vendor
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_VENDOR, 0, NULL, &iValueSize);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get Vendor of OpenCL Device %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            value = (char *)calloc(iValueSize, sizeof(char));
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_VENDOR, iValueSize, value, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get Vendor of OpenCL Device %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            printf("%d. Device Vendor : %s \n", j + 1, value);
            free(value);
            value = NULL;

            // print Hardware Device Version
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_VERSION, 0, NULL, &iValueSize);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get Hardware Device Version of OpenCL Device %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            value = (char *)calloc(iValueSize, sizeof(char));
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_VERSION, iValueSize, value, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get Hardware Device Version of OpenCL Device %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            printf("%d. Device Version: %s \n", j + 1, value);
            free(value);
            value = NULL;

            // print Software Driver Version
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DRIVER_VERSION, 0, NULL, &iValueSize);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get Software Driver Version of OpenCL Device %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            value = (char *)calloc(iValueSize, sizeof(char));
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DRIVER_VERSION, iValueSize, value, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get Software Driver Version of OpenCL Device %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            printf("%d. Device Driver Version: %s \n", j + 1, value);
            free(value);
            value = NULL;

            // print max clock frequency
            cl_uint clock_frequency =0;

            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get Software Driver Version of OpenCL Device %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            printf("%d. Device Max Clock Frequency: %u Hz\n", j + 1, clock_frequency);
            free(value);
            value = NULL;

            // print C Version suppourted by compiler for Device
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &iValueSize);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get C Version of OpenCL Device %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            value = (char *)calloc(iValueSize, sizeof(char));
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_OPENCL_C_VERSION, iValueSize, value, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get C Version of OpenCL Device %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            printf("%d. Device OpenCL C Version : %s \n", j + 1, value);
            free(value);
            value = NULL;

            printf("\n");
            printf("*************** DEVICE MEMORY INFORMATION **********\n");
            printf("====================================================\n");
            // Device Global Memory
            cl_ulong mem_size;
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), (void*)&mem_size, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get Device Global Memory %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            printf("%d. Device Global Memory  : %llu Bytes\n",j+1, mem_size);

            // Device Local Memory
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), (void *)&mem_size, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get Device Local Memory %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            printf("%d. Device Local Memory  : %llu Bytes\n",j+1, mem_size);

            // Max constant buffer size
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), (void *)&mem_size, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                printf("OpenCL Error : Could not get Device Constant Memory size %d..\n", j + 1);
                printf("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            printf("%d. Device Constant Memory size  : %llu Bytes\n", j + 1, mem_size);

            cl_ulong max_mem_alloc_size;
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), (void *)&max_mem_alloc_size, NULL);

            printf("%d. Device Max Memory Alloc %llu\n",j+1, max_mem_alloc_size);

            printf("\n");
            printf("***************** DEVICE COMPUTE INFORMATION **************\n");
            printf("============================================================\n");
            // print Parallel compute units
            clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(iMaxComputeUnits), &iMaxComputeUnits, NULL);
            printf("%d. Parallel compute units: %d\n", j + 1, iMaxComputeUnits);

            size_t workgroup_size;
            clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), (void *)&workgroup_size, NULL);
            printf("%d. Device Max Work Group Size : %zd\n", j + 1, workgroup_size);

            size_t workIten_dims;
            clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workIten_dims), (void *)&workIten_dims, NULL);
            printf("%d. Device Max Work Item Dimensions : %zd\n", j + 1, workIten_dims);

            size_t workItem_size[3];
            clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workItem_size), (void *)&workItem_size, NULL);
            printf("%d. Device Max Work Item Sizes : %u,%u,%u\n", j + 1, (unsigned int)workItem_size[0], (unsigned int)workItem_size[1], (unsigned int)workItem_size[2]);

            printf("\n");
            printf("**************** DEVICE IMAGE SUPPORT **********\n");
            printf("============================================================\n");

            size_t szMaxDims[5];
            clGetDeviceInfo(pDevices[j], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), (void *)&szMaxDims[0], NULL);
            clGetDeviceInfo(pDevices[j], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), (void *)&szMaxDims[1], NULL);
            printf("%d. Device Supported 2D image W X H  : %zu X %zu\n", j + 1, szMaxDims[0], szMaxDims[1]);

            clGetDeviceInfo(pDevices[j], CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), (void *)&szMaxDims[2], NULL);
            clGetDeviceInfo(pDevices[j], CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), (void *)&szMaxDims[3], NULL);
            clGetDeviceInfo(pDevices[j], CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), (void *)&szMaxDims[4], NULL);
            printf("%d. Device Supported 2D image W X H X D : %zu X %zu X %zu\n", j + 1, szMaxDims[2], szMaxDims[3], szMaxDims[4]);
        }

        printf("############################################################\n");

        if (pDevices)
        {
            free(pDevices);
            pDevices=NULL;
        }
        }
    
    LAST_SA :
    // All unitialize calls after this
    if (value)
    {
        free(value);
        value=NULL;
    }
    
    if (pDevices)
    {
        free(pDevices);
        pDevices=NULL;
    }
    
    if (pPlatforms)
    {
        free(pPlatforms);
        pPlatforms=NULL;
    }
        
    printf("Press any key to exit . . .!!");
    _getch();
    return EXIT_SUCCESS;
}
