#include <jni.h>
#include <string>
#include <CL/cl.h>
#include<android/log.h>


#define LOG_TAG "OCL-INFO"
#define PRINT_LOG(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)

// Platforms
cl_uint  iPlatformCount=0;
cl_platform_id *pPlatforms = NULL;

// Respective Devices
cl_uint iDeviceCount=0;
cl_device_id *pDevices=NULL;

// Max Compute Units
cl_uint iMaxComputeUnits=0;

// Check return values of OpenCL
cl_int ret_ocl=-1;

char *value=NULL;
size_t iValueSize;

extern "C" JNIEXPORT jstring JNICALL
Java_com_astromedicomp_Fibonachi_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject thisObj )
{
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

static jlong fib(jlong n)
{
    return n <= 0? 0: n==1? 1 :fib(n-1) + fib(n-2);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_astromedicomp_Fibonachi_FibLib_fibNR(JNIEnv *env,jobject thisObj,jlong n)
{
    return fib(n);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_astromedicomp_Fibonachi_FibLib_fibNI(JNIEnv *env,jobject thisObj,jlong n)
{
    jlong previous = -1;
    jlong result = 1;
    jlong i;
    for ( i =0; i<n; i++)
    {
        jlong sum = result + previous;
        previous=result;
        result =sum;
    }
    return result;
}

extern "C" JNIEXPORT void JNICALL
        Java_com_astromedicomp_Fibonachi_MainActivity_jniIntializeOpencl(JNIEnv *env,jobject thisObj)
{
    ret_ocl = clGetPlatformIDs(0, NULL, &iPlatformCount);
    if (ret_ocl != CL_SUCCESS)
    {
        PRINT_LOG("OpenCL Error : Could not get Count of OpenCL Platforms..\n");
        goto LAST_SA;
    }

    PRINT_LOG("OpenCL INFO : Count of OpenCL Platforms => %d\n", iPlatformCount);

    pPlatforms = (cl_platform_id *)calloc(iPlatformCount, sizeof(cl_platform_id));
    if (pPlatforms==NULL)
    {
        PRINT_LOG("System Error : Could not allocate Count of OpenCL Platforms..\n");
    }

    // obtain all Platform IDs
    ret_ocl = clGetPlatformIDs(iPlatformCount, pPlatforms,NULL);
    if (ret_ocl != CL_SUCCESS)
    {
        PRINT_LOG("OpenCL Error : Could not get all of OpenCL Platforms..\n");
        PRINT_LOG("OpenCL Error : clGetPlatformIDs Failed : %d. \nExitting Now..\n", ret_ocl);

        goto LAST_SA;
    }

    // Iterate for all platforms and get respective device
    for (cl_uint i = 0; i < iPlatformCount; i++)
    {
        // Get all devices for each platform
        ret_ocl = clGetDeviceIDs(pPlatforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &iDeviceCount);
        if (ret_ocl != CL_SUCCESS)
        {
            PRINT_LOG("OpenCL Error : Could not get all of OpenCL Devices..\n");
            PRINT_LOG("OpenCL Error : clGetDeviceIDs Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        pDevices = (cl_device_id*)calloc(iDeviceCount, sizeof(cl_device_id));
        if (pDevices==NULL)
        {
            PRINT_LOG("System Error : Could not allocate Count of OpenCL Devices..\n");
        }

        ret_ocl = clGetDeviceIDs(pPlatforms[i], CL_DEVICE_TYPE_ALL, iDeviceCount, pDevices, NULL);
        if (ret_ocl != CL_SUCCESS)
        {
            PRINT_LOG("OpenCL Error : Could not get specific of OpenCL Devices..\n");
            PRINT_LOG("OpenCL Error : clGetDeviceIDs Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        PRINT_LOG("\nOpenCL INFO : In OpenCL Platform %d and Devices present => %d\n", i+1, iDeviceCount);
        PRINT_LOG("++++++++++++++++ Platform Info ++++++++++++++++++++++\n");

        // name
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_NAME, 0, NULL, &iValueSize);
        if (ret_ocl != CL_SUCCESS)
        {
            PRINT_LOG("OpenCL Error : Could not get Name count of platform..\n");
            PRINT_LOG("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        value = (char *)calloc(iValueSize, sizeof(char));
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_NAME, iValueSize, value, NULL);
        if (ret_ocl != CL_SUCCESS)
        {
            PRINT_LOG("OpenCL Error : Could not get Name of platform..\n");
            PRINT_LOG("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }
        PRINT_LOG("Platform Name = %s\n", value);
        free(value);
        value = NULL;

        // Profile
        ret_ocl = clGetPlatformInfo(pPlatforms[i],CL_PLATFORM_PROFILE,0,NULL,&iValueSize);
        if (ret_ocl != CL_SUCCESS)
        {
            PRINT_LOG("OpenCL Error : Could not get Profile count of platform..\n");
            PRINT_LOG("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        value = (char*)calloc(iValueSize,sizeof(char));
        ret_ocl = clGetPlatformInfo(pPlatforms[i],CL_PLATFORM_PROFILE,iValueSize, value,NULL);
        if (ret_ocl != CL_SUCCESS)
        {
            PRINT_LOG("OpenCL Error : Could not get Profile of platform..\n");
            PRINT_LOG("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }
        PRINT_LOG("Platform Profile = %s\n", value);
        free(value);
        value=NULL;

        // version
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_VERSION, 0, NULL, &iValueSize);
        if (ret_ocl != CL_SUCCESS)
        {
            PRINT_LOG("OpenCL Error : Could not get Version count of platform..\n");
            PRINT_LOG("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        value = (char *)calloc(iValueSize, sizeof(char));
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_VERSION, iValueSize, value, NULL);
        if (ret_ocl != CL_SUCCESS)
        {
            PRINT_LOG("OpenCL Error : Could not get Version of platform..\n");
            PRINT_LOG("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }
        PRINT_LOG("Platform Version = %s\n", value);
        free(value);
        value = NULL;

        // Vendor
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_VENDOR, 0, NULL, &iValueSize);
        if (ret_ocl != CL_SUCCESS)
        {
            PRINT_LOG("OpenCL Error : Could not get Vendor count of platform..\n");
            PRINT_LOG("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        value = (char *)calloc(iValueSize, sizeof(char));
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_VENDOR, iValueSize, value, NULL);
        if (ret_ocl != CL_SUCCESS)
        {
            PRINT_LOG("OpenCL Error : Could not get Vendor of platform..\n");
            PRINT_LOG("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }
        PRINT_LOG("Platform Vendor = %s\n", value);
        free(value);
        value = NULL;

        // Extentions
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &iValueSize);
        if (ret_ocl != CL_SUCCESS)
        {
            PRINT_LOG("OpenCL Error : Could not get Extension count of platform..\n");
            PRINT_LOG("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }

        value = (char *)calloc(iValueSize, sizeof(char));
        ret_ocl = clGetPlatformInfo(pPlatforms[i], CL_PLATFORM_EXTENSIONS, iValueSize, value, NULL);
        if (ret_ocl != CL_SUCCESS)
        {
            PRINT_LOG("OpenCL Error : Could not get Extension of platform..\n");
            PRINT_LOG("OpenCL Error : clGetPlatformInfo Failed : %d. \nExitting Now..\n", ret_ocl);

            goto LAST_SA;
        }
        PRINT_LOG("Platform Extension = %s\n", value);
        free(value);
        value = NULL;

        for (cl_uint j = 0; j < iDeviceCount; j++)
        {

            PRINT_LOG("\n");
            PRINT_LOG("************* DEVICE GENERAL INFORMATION ***********\n");
            PRINT_LOG("=====================================================\n");
            // Print Device Type
            cl_device_type devType;
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_TYPE, sizeof(devType), (void *)&devType, NULL);

            switch (devType)
            {
                case CL_DEVICE_TYPE_CPU:
                    PRINT_LOG("%d. Device Type : %s \n", j + 1, "CL_DEVICE_TYPE_CPU ");
                    break;

                case CL_DEVICE_TYPE_GPU:
                    PRINT_LOG("%d. Device Type : %s \n", j + 1, "CL_DEVICE_TYPE_GPU ");
                    break;

                case CL_DEVICE_TYPE_ACCELERATOR:
                    PRINT_LOG("%d. Device Type : %s \n", j + 1, "CL_DEVICE_TYPE_ACCELERATOR");
                    break;

                case CL_DEVICE_TYPE_DEFAULT:
                    PRINT_LOG("%d. Device Type : %s \n", j + 1, "CL_DEVICE_TYPE_CPU ");
                    break;
            }

            free(value);
            value = NULL;

            // print Device Name
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_NAME, 0, NULL, &iValueSize);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get Name of OpenCL Device %d..\n",j+1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            value = (char *)calloc(iValueSize, sizeof(char));
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_NAME, iValueSize, value, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get Name of OpenCL Device %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            PRINT_LOG("%d. Device Name : %s \n",j+1,value);
            free(value);
            value=NULL;

            // Device vendor
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_VENDOR, 0, NULL, &iValueSize);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get Vendor of OpenCL Device %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            value = (char *)calloc(iValueSize, sizeof(char));
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_VENDOR, iValueSize, value, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get Vendor of OpenCL Device %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            PRINT_LOG("%d. Device Vendor : %s \n", j + 1, value);
            free(value);
            value = NULL;

            // print Hardware Device Version
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_VERSION, 0, NULL, &iValueSize);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get Hardware Device Version of OpenCL Device %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            value = (char *)calloc(iValueSize, sizeof(char));
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_VERSION, iValueSize, value, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get Hardware Device Version of OpenCL Device %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            PRINT_LOG("%d. Device Version: %s \n", j + 1, value);
            free(value);
            value = NULL;

            // print Software Driver Version
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DRIVER_VERSION, 0, NULL, &iValueSize);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get Software Driver Version of OpenCL Device %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            value = (char *)calloc(iValueSize, sizeof(char));
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DRIVER_VERSION, iValueSize, value, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get Software Driver Version of OpenCL Device %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            PRINT_LOG("%d. Device Driver Version: %s \n", j + 1, value);
            free(value);
            value = NULL;

            // print max clock frequency
            cl_uint clock_frequency =0;

            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get Software Driver Version of OpenCL Device %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            PRINT_LOG("%d. Device Max Clock Frequency: %u Hz\n", j + 1, clock_frequency);
            free(value);
            value = NULL;

            // print C Version suppourted by compiler for Device
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &iValueSize);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get C Version of OpenCL Device %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            value = (char *)calloc(iValueSize, sizeof(char));
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_OPENCL_C_VERSION, iValueSize, value, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get C Version of OpenCL Device %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            PRINT_LOG("%d. Device OpenCL C Version : %s \n", j + 1, value);
            free(value);
            value = NULL;

            PRINT_LOG("\n");
            PRINT_LOG("*************** DEVICE MEMORY INFORMATION **********\n");
            PRINT_LOG("====================================================\n");
            // Device Global Memory
            cl_ulong mem_size;
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), (void*)&mem_size, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get Device Global Memory %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            PRINT_LOG("%d. Device Global Memory  : %llu Bytes\n",j+1, mem_size);

            // Device Local Memory
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), (void *)&mem_size, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get Device Local Memory %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            PRINT_LOG("%d. Device Local Memory  : %llu Bytes\n",j+1, mem_size);

            // Max constant buffer size
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), (void *)&mem_size, NULL);
            if (ret_ocl != CL_SUCCESS)
            {
                PRINT_LOG("OpenCL Error : Could not get Device Constant Memory size %d..\n", j + 1);
                PRINT_LOG("OpenCL Error : clGetDeviceInfo Failed : %d. \nExitting Now..\n", ret_ocl);
                goto LAST_SA;
            }
            PRINT_LOG("%d. Device Constant Memory size  : %llu Bytes\n", j + 1, mem_size);

            cl_ulong max_mem_alloc_size;
            ret_ocl = clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), (void *)&max_mem_alloc_size, NULL);

            PRINT_LOG("%d. Device Max Memory Alloc %llu\n",j+1, max_mem_alloc_size);

            PRINT_LOG("\n");
            PRINT_LOG("***************** DEVICE COMPUTE INFORMATION **************\n");
            PRINT_LOG("============================================================\n");
            // print Parallel compute units
            clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(iMaxComputeUnits), &iMaxComputeUnits, NULL);
            PRINT_LOG("%d. Parallel compute units: %d\n", j + 1, iMaxComputeUnits);

            size_t workgroup_size;
            clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), (void *)&workgroup_size, NULL);
            PRINT_LOG("%d. Device Max Work Group Size : %zd\n", j + 1, workgroup_size);

            size_t workIten_dims;
            clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workIten_dims), (void *)&workIten_dims, NULL);
            PRINT_LOG("%d. Device Max Work Item Dimensions : %zd\n", j + 1, workIten_dims);

            size_t workItem_size[3];
            clGetDeviceInfo(pDevices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workItem_size), (void *)&workItem_size, NULL);
            PRINT_LOG("%d. Device Max Work Item Sizes : %u,%u,%u\n", j + 1, (unsigned int)workItem_size[0], (unsigned int)workItem_size[1], (unsigned int)workItem_size[2]);

            PRINT_LOG("\n");
            PRINT_LOG("**************** DEVICE IMAGE SUPPORT **********\n");
            PRINT_LOG("============================================================\n");

            size_t szMaxDims[5];
            clGetDeviceInfo(pDevices[j], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), (void *)&szMaxDims[0], NULL);
            clGetDeviceInfo(pDevices[j], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), (void *)&szMaxDims[1], NULL);
            PRINT_LOG("%d. Device Supported 2D image W X H  : %zu X %zu\n", j + 1, szMaxDims[0], szMaxDims[1]);

            clGetDeviceInfo(pDevices[j], CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), (void *)&szMaxDims[2], NULL);
            clGetDeviceInfo(pDevices[j], CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), (void *)&szMaxDims[3], NULL);
            clGetDeviceInfo(pDevices[j], CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), (void *)&szMaxDims[4], NULL);
            PRINT_LOG("%d. Device Supported 2D image W X H X D : %zu X %zu X %zu\n", j + 1, szMaxDims[2], szMaxDims[3], szMaxDims[4]);

        }

        PRINT_LOG("############################################################\n");

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
}