#include<stdio.h>
#include<conio.h>
#include<stdlib.h>

#define N 66000


__global__ void vecAddKernel(int *in_A,int *in_B,int *out_C)
{
    int tid = blockIdx.x;

    if(tid < N)
    {
        out_C[tid] = in_A[tid] + in_B[tid];
    }
}

int main()
{
    void vecAdd(int*,int*,int*);

    int *H_a=NULL,*H_b=NULL,*H_c=NULL,*H_Gold=NULL;
    int *D_a=NULL,*D_b=NULL,*D_c=NULL;

    cudaError status;
    srand(rand());

    // Allocate memory on Host
    H_a = (int*)calloc(N,sizeof(int));
    if(H_a==NULL)
    {
        printf_s("Falied to calloc \nLeaving.\n");
        goto LAST_SA;
    }

     H_b = (int*)calloc(N,sizeof(int));
    if(H_b==NULL)
    {
        printf_s("Falied to calloc \nLeaving.\n");
        goto LAST_SA;
    }

     H_c = (int*)calloc(N,sizeof(int));
    if(H_c==NULL)
    {
        printf_s("Falied to calloc \nLeaving.\n");
        goto LAST_SA;
    }

     H_Gold = (int*)calloc(N,sizeof(int));
    if(H_Gold==NULL)
    {
        printf_s("Falied to calloc \nLeaving.\n");
        goto LAST_SA;
    }
    // fill input
    printf_s("Fill Host inputs\n");
    for(int i = 0; i < N; i++)
    {
        H_a[i] = rand();
        H_b[i] = rand();
    }

    // Allocate memory on device
    printf_s("Allocate memory on device\n");
    status = cudaMalloc((void**)&D_a,sizeof(int)*N);
    if(status!=cudaSuccess)
    {
        printf_s("Falied to cudaMalloc \nLeaving.\n");
        goto LAST_SA;
    }

    status = cudaMalloc((void**)&D_b,sizeof(int)*N);
    if(status!=cudaSuccess)
    {
        printf_s("Falied to cudaMalloc \nLeaving.\n");
        goto LAST_SA;
    }

    status = cudaMalloc((void**)&D_c,sizeof(int)*N);
    if(status!=cudaSuccess)
    {
        printf_s("Falied to cudaMalloc \nLeaving.\n");
        goto LAST_SA;
    }

    // fill memory on device
    printf_s("Fill memory on device\n");
    status = cudaMemcpy(D_a,H_a,sizeof(int)*N,cudaMemcpyHostToDevice);
    if(status!=cudaSuccess)
    {
        printf_s("Falied to cudacudaMemcpyMalloc \nLeaving.\n");
        goto LAST_SA;
    }

    status = cudaMemcpy(D_b,H_b,sizeof(int)*N,cudaMemcpyHostToDevice);
    if(status!=cudaSuccess)
    {
        printf_s("Falied to cudacudaMemcpyMalloc \nLeaving.\n");
        goto LAST_SA;
    }

    // Do Operation
    printf_s("Do Host Operation\n");
    vecAdd(H_a,H_b,H_c);

    // Do GPU call
    printf_s("Do DEVICE Operation\n");
    vecAddKernel<<<N,1>>>(D_a,D_b,D_c);

    //H_Gold
     status = cudaMemcpy(H_Gold,D_c,sizeof(int)*N,cudaMemcpyDeviceToHost);
    if(status!=cudaSuccess)
    {
        printf_s("Falied to cudacudaMemcpyMalloc \nLeaving.\n");
        goto LAST_SA;
    }

    // Display Output array
    printf_s("Print Result\n");
    for(int i = 0; i < N; i++)
    {
        printf_s("%8d = %8d",H_c[i],H_Gold[i]),
        printf_s( ((i % 5)==4)? "|\n": "|\t" );
    }
    
    LAST_SA:


    if(H_a==NULL)
    {
        free(H_a);
        H_a=NULL;
    }

    if(H_b==NULL)
    {
        free(H_b);
        H_b=NULL;
    }

    if(H_c==NULL)
    {
        free(H_c);
        H_c=NULL;
    }

    if(H_Gold==NULL)
    {
        free(H_Gold);
        H_Gold=NULL;
    }

    if(D_a)
    {
        cudaFree(D_a);
        D_a=NULL;
    }

    if(D_b)
    {
        cudaFree(D_b);
        D_b=NULL;
    }

    if(D_c)
    {
        cudaFree(D_c);
        D_c=NULL;
    }

    printf_s("Press Any Key to Exit. . . . \n");
    _getch();
    return EXIT_SUCCESS;
}

void vecAdd(int *in_A, int *in_B, int *out_C)
{
    int tid = 0;
    while(tid < N)
    {
        out_C[tid] = in_A[tid] + in_B[tid];
        tid = tid + 1;
    }
    
}

