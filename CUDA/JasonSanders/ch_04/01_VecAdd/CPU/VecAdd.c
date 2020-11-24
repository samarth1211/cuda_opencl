#include<stdio.h>
#include<conio.h>
#include<stdlib.h>

#define N 20


int main (void)
{
    void vecAdd(int*,int*,int*);

    int a[N],b[N],c[N];

    srand(rand());
    // fill input
    for(int i = 0; i < N; i++)
    {
        a[i] = rand();
        b[i] = rand();
    }
    
    // Do Operation
    vecAdd(a,b,c);

    // Display Output array
    for(int i = 0; i < N; i++)
    {
        printf_s("%10d",c[i]),
        printf_s( ((i % 5)==4)? "\n": "\t" );
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
