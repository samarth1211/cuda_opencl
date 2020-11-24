cls
del *.exe *.obj *.txt *.lib *.exp
nvcc -c -o SimpleGL.obj SimpleGL.cu
cl /c /EHsc Source.cpp /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include"
link SimpleGL.obj Source.obj /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64" /OUT:SimpleGL_CUDA.exe
SimpleGL_CUDA.exe

