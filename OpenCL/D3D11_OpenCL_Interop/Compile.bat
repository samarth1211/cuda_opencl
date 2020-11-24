del *.exe *.obj  *.txt

cls

cl /c /EHsc Source.cpp /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include"

link Source.obj /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64" /OUT:D3D11_OpenCL_Interop.exe

del *.obj 

D3D11_OpenCL_Interop.exe
