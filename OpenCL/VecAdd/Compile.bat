del *.exe *.obj  *.txt

cl /c /EHsc Source.cpp /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include"

link Source.obj /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64" /OUT:VectorAddition.exe

del *.obj 

cls

VectorAddition.exe
