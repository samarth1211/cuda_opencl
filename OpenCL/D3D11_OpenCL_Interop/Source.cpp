#include<Windows.h>
#include<stdio.h>

#include<d3d11.h>
#include<d3dcompiler.h> // for compiling the shaders

#include<CL\opencl.h>
#include<CL\cl_d3d11.h>
#include<CL\cl_d3d11_ext.h>

#pragma warning(disable:4838)
#include"XNAMath\xnamath.h" // Calculations here are Row Major. This also Includes 5 (*).inl files

#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib,"OpenCL.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

#define MESH_WIDTH	1024
#define MESH_HEIGHT	1024

// Global Function Callback
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);


enum OCLInitErrorCodes
{
	OCL_FAILURE_ENQ_ACQUIRE_D3D_OBJ = -25,
	OCL_FAILURE_CREATE_CL_BUFFER_FROM_D3D = -24,
	OCL_FAILURE_OBTAIN_PLATFORM_PROC = -23,
	OCL_FAILURE_CL_FINISH = -22,
	OCL_FAILURE_ENQ_RELEASE_GL_OBJ = -21,
	OCL_FAILURE_ENQ_KERNEL = -20,
	OCL_FAILURE_ENQ_ACQUIRE_GL_OBJ = -19,
	OCL_FAILURE_SET_KERNEL_ARG = -18,
	OCL_FAILURE_CREATE_KERNEL = -17,
	OCL_FAILURE_BUILD_PROG = -16,
	OCL_FAILURE_CREATE_PROG_WITH_SRC = -15,
	OCL_FAILURE_CREATE_CL_BUFFER_FROM_GL = -14,
	OCL_FAILURE_CREATE_COMMAND_QUEUE = -13,
	OCL_FAILURE_OBTAIN_DEVICE = -12,
	OCL_FAILURE_CREATE_CONTEXT = -11,
	OCL_FAILURE_OBTAIN_PLATFORM = -10,

	INIT_ALL_OK=0,
};


FILE *g_pFile = NULL;
char g_szLogFileName[] = "D3D_11_OCL_Interop.txt";

HWND g_hwnd = NULL;

DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(wpPrev) };

bool g_bActiveWindow = false;
bool g_bEscapePressed = false;
bool g_bFullscreen = false;

float g_fClearColor[4]; // to fill background with this color

						// Configure D3D
IDXGISwapChain *g_pIDXGISwapChain = NULL;
ID3D11Device *g_pID3D11Device = NULL;
ID3D11DeviceContext *g_pID3D11DeviceContext = NULL;
ID3D11RenderTargetView *g_pID3D11RenderTargetView = NULL;

// Pointers For Shader Objects
ID3D11VertexShader *g_pID3D11VertexShader = NULL; // Vertex Shader Object
ID3D11PixelShader *g_pID3D11PixelShader = NULL; // Fragment Shader Object
ID3D11Buffer *g_pID3D11Buffer_VertexBuffer_Position = NULL; // Vertex_Buffer_Object_Position
ID3D11InputLayout *g_pID3D11InputLayout_Position = NULL; // To map location if data in Shader akin to "layout=0" in OpenGl Shader



ID3D11Buffer *g_pID3D11Buffer_ConstantBuffer = NULL;// To map  Constant Buffer in to our buffer in Shader.

// This is akin to Structure used in Uniform Buffer Objects(to be mapped with layouts) in OpenGL/
// This Struct is to Map with Costant Buffer in Shaders
struct CBUFFER
{
	XMMATRIX WorldViewProjectionMatrix;// ModelViewProjectionMatrix
};

XMMATRIX g_PerspectiveProjectionMatrix;



// OpenCL Variables

cl_platform_id firstPlatformID = 0;
cl_device_id fastestDeviceID = 0;

cl_context g_context = NULL;
cl_command_queue g_commandQueue = NULL;
cl_device_id g_device = 0;
cl_program g_program = 0;
cl_kernel g_kernel = 0;
cl_mem cl_vbo_resource = NULL;

char *chOCLSourceCode = NULL;
size_t sizeKernelCodeLength;

int meshWidth = MESH_WIDTH;
int meshHeight = MESH_HEIGHT;

clGetDeviceIDsFromD3D11NV_fn		pfn_clGetDeviceIDsFromD3D11			= NULL;
clCreateFromD3D11BufferNV_fn		pfn_clCreateFromD3D11Buffer			= NULL;
clEnqueueAcquireD3D11ObjectsNV_fn	pfn_clEnqueueAcquireD3D11ObjectsNV	= NULL;
clEnqueueReleaseD3D11ObjectsNV_fn	pfn_clEnqueueReleaseD3D11ObjectsNV	= NULL;

float g_fAnime = 0.0f;

// Entry-Point Function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevHInsatnce, LPSTR szCmdLine, int iCmdShow)
{
	HRESULT Initialize(void);
	void UnInitialize(void);
	void Display(void);
	int Update(void);

	WNDCLASSEX wndclass;
	HWND hwnd = NULL;
	TCHAR szClassName[] = TEXT("SamD3D11");
	MSG msg;
	bool bDone = false;
	HRESULT hr = NULL;

	cl_int retStatus;

	if (fopen_s(&g_pFile, g_szLogFileName, "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File Can Not Be Created \nLeaving Now...!!\n "), TEXT("ERROR...!!"), MB_OK | MB_TOPMOST | MB_ICONSTOP);
		exit(EXIT_FAILURE);
	}
	else
	{
		fprintf_s(g_pFile, "Log File Is Created  \n");
		fclose(g_pFile);
	}

	wndclass.cbSize = sizeof(wndclass);
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(GRAY_BRUSH);
	wndclass.hIcon = LoadIcon(hInstance, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(hInstance, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(hInstance, IDC_ARROW);

	if (!RegisterClassEx(&wndclass))
	{
		MessageBox(NULL, TEXT("Could Not RegisterClassEx"), TEXT("Error....!!"), MB_OK);
		exit(EXIT_FAILURE);
	}

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName, TEXT("D3D11 Colored Triangle"), WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, WIN_WIDTH, WIN_HEIGHT, (HWND)NULL, (HMENU)NULL,
		hInstance, (LPVOID)NULL);

	if (hwnd == NULL)
	{
		MessageBox(NULL, TEXT("Could Not CreateWindow"), TEXT("Error....!!"), MB_OK);
		exit(EXIT_FAILURE);
	}

	g_hwnd = hwnd;

	ShowWindow(hwnd, SW_NORMAL);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	hr = Initialize();
	if (FAILED(hr))
	{
		switch (hr)
		{
		case OCL_FAILURE_CREATE_CL_BUFFER_FROM_D3D:
			break;
		case OCL_FAILURE_CL_FINISH:
			break;
		case OCL_FAILURE_ENQ_RELEASE_GL_OBJ:
			break;
		case OCL_FAILURE_ENQ_KERNEL:
			break;
		case OCL_FAILURE_ENQ_ACQUIRE_GL_OBJ:
			break;
		case OCL_FAILURE_SET_KERNEL_ARG:
			break;
		case OCL_FAILURE_CREATE_KERNEL:
			break;
		case OCL_FAILURE_BUILD_PROG:
			break;
		case OCL_FAILURE_CREATE_PROG_WITH_SRC:
			break;
		case OCL_FAILURE_CREATE_CL_BUFFER_FROM_GL:
			break;
		case OCL_FAILURE_CREATE_COMMAND_QUEUE:
			break;
		case OCL_FAILURE_OBTAIN_DEVICE:
			break;
		case OCL_FAILURE_CREATE_CONTEXT:
			break;
		case OCL_FAILURE_OBTAIN_PLATFORM:
			break;
		
		}

		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "Initialize Failed.\nLeaving Now...!!!\n");
		fclose(g_pFile);
		DestroyWindow(hwnd);
		hwnd = NULL;
	}
	else
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "Initialize Completed.\n");
		fclose(g_pFile);
	}

	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				bDone = true;
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			Display();
			retStatus = Update();

			switch (retStatus)
			{
			case OCL_FAILURE_ENQ_ACQUIRE_GL_OBJ:
				fprintf_s(g_pFile, "Failed OCL_FAILURE_ENQ_ACQUIRE_GL_OBJ \n");
				DestroyWindow(hwnd);
				break;

			case OCL_FAILURE_SET_KERNEL_ARG:
				fprintf_s(g_pFile, "Failed OCL_FAILURE_SET_KERNEL_ARG \n");
				DestroyWindow(hwnd);
				break;

			case OCL_FAILURE_ENQ_KERNEL:
				fprintf_s(g_pFile, "Failed OCL_FAILURE_ENQ_KERNEL \n");
				DestroyWindow(hwnd);
				break;

			case OCL_FAILURE_ENQ_RELEASE_GL_OBJ:
				fprintf_s(g_pFile, "Failed OCL_FAILURE_ENQ_RELEASE_GL_OBJ \n");
				DestroyWindow(hwnd);
				break;

			case OCL_FAILURE_CL_FINISH:
				fprintf_s(g_pFile, "Failed OCL_FAILURE_CL_FINISH \n");
				DestroyWindow(hwnd);
				break;
			case INIT_ALL_OK:
				break;
			}

			if (g_bActiveWindow == true)
			{
				if (g_bEscapePressed)
					bDone = true;

				retStatus = Update();

				switch (retStatus)
				{
				case OCL_FAILURE_ENQ_ACQUIRE_GL_OBJ:
					fprintf_s(g_pFile, "Failed OCL_FAILURE_ENQ_ACQUIRE_GL_OBJ \n");
					DestroyWindow(hwnd);
					break;

				case OCL_FAILURE_SET_KERNEL_ARG:
					fprintf_s(g_pFile, "Failed OCL_FAILURE_SET_KERNEL_ARG \n");
					DestroyWindow(hwnd);
					break;

				case OCL_FAILURE_ENQ_KERNEL:
					fprintf_s(g_pFile, "Failed OCL_FAILURE_ENQ_KERNEL \n");
					DestroyWindow(hwnd);
					break;

				case OCL_FAILURE_ENQ_RELEASE_GL_OBJ:
					fprintf_s(g_pFile, "Failed OCL_FAILURE_ENQ_RELEASE_GL_OBJ \n");
					DestroyWindow(hwnd);
					break;

				case OCL_FAILURE_CL_FINISH:
					fprintf_s(g_pFile, "Failed OCL_FAILURE_CL_FINISH \n");
					DestroyWindow(hwnd);
					break;
				case INIT_ALL_OK:
					break;
				}
			}
		}
	}

	UnInitialize();

	return (int)msg.wParam;
}


LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	void UnInitialize(void);
	HRESULT Resize(int iWidth, int iHeight);
	void ToggleFullScreen(void);

	HRESULT hr = 0;

	switch (iMsg)
	{
	case WM_CREATE:
		PostMessage(hwnd, (UINT)WM_KEYDOWN, (WPARAM)0x46,(LPARAM)NULL);
		break;
	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			g_bActiveWindow = true;
		else
			g_bActiveWindow = false;
		break;
	case WM_SIZE:
		if (g_pID3D11DeviceContext)
		{
			hr = Resize(LOWORD(lParam), HIWORD(lParam));
			if (FAILED(hr))
			{
				fopen_s(&g_pFile, g_szLogFileName, "a+");
				fprintf_s(g_pFile, "Resize Failed .\n");
				fclose(g_pFile);
				return hr;
			}
			else
			{
				fopen_s(&g_pFile, g_szLogFileName, "a+");
				fprintf_s(g_pFile, "Resize Completed .\n");
				fclose(g_pFile);
			}
		}
		break;

	case WM_KEYDOWN:
		switch (LOWORD(wParam))
		{

		case VK_ESCAPE:
			if (g_bEscapePressed == false)
				g_bEscapePressed = true;
			break;

		case 0x46: //f of F
			if (g_bFullscreen == false)
			{
				ToggleFullScreen();
				g_bFullscreen = true;
			}
			else
			{
				ToggleFullScreen();
				g_bFullscreen = false;
			}
			break;
		}
		break;
	case WM_ERASEBKGND:
		return(0);
		break;
	case WM_CLOSE:
		UnInitialize();
		break;
	case WM_DESTROY:
		UnInitialize();
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return (DefWindowProc(hwnd, iMsg, wParam, lParam));
}


void ToggleFullScreen(void)
{

	MONITORINFO mi = { sizeof(mi) };

	if (g_bFullscreen == false)
	{
		dwStyle = GetWindowLong(g_hwnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(g_hwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(g_hwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(g_hwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(g_hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, (mi.rcMonitor.right - mi.rcMonitor.left), (mi.rcMonitor.bottom - mi.rcMonitor.top), SWP_NOZORDER | SWP_FRAMECHANGED);
			}
			ShowCursor(FALSE);
		}
	}
	else
	{
		SetWindowLong(g_hwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hwnd, &wpPrev);
		SetWindowPos(g_hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED | SWP_NOZORDER | SWP_NOOWNERZORDER);
		ShowCursor(TRUE);
	}

}

HRESULT Initialize(void)
{
	void UnInitialize(void);
	char* load_programSource(const char *filename, const char *preamble, size_t *iSize);

	cl_int ret_ocl;
	cl_uint numDevices;

	HRESULT Resize(int iWidth, int iHeight);

	HRESULT hr = NULL;

	D3D_DRIVER_TYPE d3dDriverType;
	D3D_DRIVER_TYPE d3dDriverTypes[] = { D3D_DRIVER_TYPE_HARDWARE,D3D_DRIVER_TYPE_WARP,D3D_DRIVER_TYPE_REFERENCE };

	D3D_FEATURE_LEVEL d3dFeatureLevelRequired = D3D_FEATURE_LEVEL_11_0;
	D3D_FEATURE_LEVEL d3dFeatureLevelAccquired = D3D_FEATURE_LEVEL_10_0;

	UINT creativeDeviceFlags = 0;
	UINT numDriverTypes = 0;
	UINT numFeatureLevels = 1;


	numDriverTypes = sizeof(d3dDriverTypes) / sizeof(d3dDriverTypes[0]);

	DXGI_SWAP_CHAIN_DESC dxgiSwapChainDesc;
	ZeroMemory((void*)&dxgiSwapChainDesc, sizeof(dxgiSwapChainDesc));

	dxgiSwapChainDesc.BufferCount = 1;
	dxgiSwapChainDesc.BufferDesc.Width = WIN_WIDTH;
	dxgiSwapChainDesc.BufferDesc.Height = WIN_HEIGHT;
	dxgiSwapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	dxgiSwapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
	dxgiSwapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
	dxgiSwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	dxgiSwapChainDesc.OutputWindow = g_hwnd;
	dxgiSwapChainDesc.SampleDesc.Count = 1;
	dxgiSwapChainDesc.SampleDesc.Quality = 0; // Defined by format
	dxgiSwapChainDesc.Windowed = TRUE;

	for (UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++)
	{
		d3dDriverType = d3dDriverTypes[driverTypeIndex];
		hr = D3D11CreateDeviceAndSwapChain(NULL, d3dDriverType,
			NULL, creativeDeviceFlags,
			&d3dFeatureLevelRequired, numFeatureLevels,
			D3D11_SDK_VERSION, &dxgiSwapChainDesc, &g_pIDXGISwapChain, &g_pID3D11Device,
			&d3dFeatureLevelAccquired, &g_pID3D11DeviceContext);
		if (SUCCEEDED(hr))
			break;
	}

	if (FAILED(hr))
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "D3D11CreateDeviceAndSwapChain Failed .\n");
		fclose(g_pFile);
		return hr;
	}
	else
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "D3D11CreateDeviceAndSwapChain Completed .\n");
		fprintf_s(g_pFile, "The Chosen Driver is of Type : ");
		switch (d3dDriverType)
		{
		case D3D_DRIVER_TYPE_UNKNOWN:
			fprintf_s(g_pFile, "Unknown Type. \n");
			break;
		case D3D_DRIVER_TYPE_HARDWARE:
			fprintf_s(g_pFile, "Hardware Type. \n");
			break;
		case D3D_DRIVER_TYPE_REFERENCE:
			fprintf_s(g_pFile, "Refrence Type. \n");
			break;
		case D3D_DRIVER_TYPE_NULL:
			fprintf_s(g_pFile, "Unknown/NULL Type. \n");
			break;
		case D3D_DRIVER_TYPE_SOFTWARE:
			fprintf_s(g_pFile, "Software Type. \n");
			break;
		case D3D_DRIVER_TYPE_WARP:
			fprintf_s(g_pFile, "Warp Type. \n");
			break;
		default:
			break;
		}

		fprintf_s(g_pFile, "The Feature Level Acquired is : ");
		switch (d3dFeatureLevelAccquired)
		{
		case D3D_FEATURE_LEVEL_10_0:
			fprintf_s(g_pFile, "10.0. \n");
			break;
		case D3D_FEATURE_LEVEL_10_1:
			fprintf_s(g_pFile, "10.1. \n");
			break;
		case D3D_FEATURE_LEVEL_11_0:
			fprintf_s(g_pFile, "11.0. \n");
			break;
		case D3D_FEATURE_LEVEL_11_1:
			fprintf_s(g_pFile, "11.1. \n");
			break;
		default:
			fprintf_s(g_pFile, "Unknown. \n");
			break;
		}

		fclose(g_pFile);
	}

	/*		Initalize OpenCL Start		*/
	ret_ocl = clGetPlatformIDs(1, &firstPlatformID, NULL); // Going for descrete device directly
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : Could not get all of OpenCL Platforms..\n");
		fprintf(g_pFile, "OpenCL Error : clGetPlatformIDs Failed : %d. \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);
		return OCL_FAILURE_OBTAIN_PLATFORM;
	}

	// Setup All required Function Pointers Start

	pfn_clGetDeviceIDsFromD3D11 = (clGetDeviceIDsFromD3D11NV_fn)clGetExtensionFunctionAddressForPlatform(firstPlatformID, "clGetDeviceIDsFromD3D11NV");
	if (pfn_clGetDeviceIDsFromD3D11 == NULL)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : Could not get Platforms Function Pointer 'clGetDeviceIDsFromD3D11NV'..\n");
		fprintf(g_pFile, "OpenCL Error : clGetExtensionFunctionAddressForPlatform() Failed . \nExitting Now..\n");
		fclose(g_pFile);
		return OCL_FAILURE_OBTAIN_PLATFORM_PROC;
	}

	pfn_clCreateFromD3D11Buffer = (clCreateFromD3D11BufferNV_fn)clGetExtensionFunctionAddressForPlatform(firstPlatformID, "clCreateFromD3D11BufferNV");
	if (pfn_clCreateFromD3D11Buffer == NULL)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : Could not get Platforms Function Pointer 'clCreateFromD3D11BufferNV'..\n");
		fprintf(g_pFile, "OpenCL Error : clGetExtensionFunctionAddressForPlatform() Failed. \nExitting Now..\n");
		fclose(g_pFile);
		return OCL_FAILURE_OBTAIN_PLATFORM_PROC;
	}

	pfn_clEnqueueAcquireD3D11ObjectsNV = (clEnqueueAcquireD3D11ObjectsNV_fn)clGetExtensionFunctionAddressForPlatform(firstPlatformID, "clEnqueueAcquireD3D11ObjectsNV");
	if (pfn_clEnqueueAcquireD3D11ObjectsNV == NULL)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : Could not get Platforms Function Pointer 'clCreateFromD3D11BufferNV'..\n");
		fprintf(g_pFile, "OpenCL Error : clGetExtensionFunctionAddressForPlatform() Failed. \nExitting Now..\n");
		fclose(g_pFile);
		return OCL_FAILURE_OBTAIN_PLATFORM_PROC;
	}

	pfn_clEnqueueReleaseD3D11ObjectsNV = (clEnqueueReleaseD3D11ObjectsNV_fn)clGetExtensionFunctionAddressForPlatform(firstPlatformID, "clEnqueueReleaseD3D11ObjectsNV");
	if (pfn_clEnqueueReleaseD3D11ObjectsNV == NULL)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : Could not get Platforms Function Pointer 'clCreateFromD3D11BufferNV'..\n");
		fprintf(g_pFile, "OpenCL Error : clGetExtensionFunctionAddressForPlatform() Failed. \nExitting Now..\n");
		fclose(g_pFile);
		return OCL_FAILURE_OBTAIN_PLATFORM_PROC;
	}

	// Setup All required Function Pointers End

	ret_ocl = pfn_clGetDeviceIDsFromD3D11(firstPlatformID, CL_D3D11_DEVICE_NV, g_pID3D11Device, CL_PREFERRED_DEVICES_FOR_D3D11_NV,1, &g_device,(cl_uint*)&numDevices);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : Could not get D3D compatible Device..\n");
		fprintf(g_pFile, "OpenCL Error : clGetDeviceIDsFromD3D11NV() Failed : %d. \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);
		return OCL_FAILURE_OBTAIN_DEVICE;
	}


	cl_context_properties properties[] =
	{
		CL_CONTEXT_D3D11_DEVICE_NV, (cl_context_properties)g_pID3D11Device,
		CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformID,
		//CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE,
		0
	};

	g_context = clCreateContext(properties, 1, &g_device, NULL, NULL, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : Could not Create D3D co,patible Context..\n");
		fprintf(g_pFile, "OpenCL Error : clCreateContext() Failed : %d. \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);
		return OCL_FAILURE_CREATE_CONTEXT;
	}

	g_commandQueue = clCreateCommandQueue(g_context, g_device, 0, &ret_ocl);
	if ((g_commandQueue == NULL) || (ret_ocl != CL_SUCCESS))
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "Could Not Create CommandQueue...!! \n");
		fprintf(g_pFile, "OpenCL Error : clCreateCommandQueue() Failed : %d. \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);
		return OCL_FAILURE_CREATE_COMMAND_QUEUE;
	}

	/*		Initalize OpenCL Stop 		*/

	// Vertex Shader Start
	const char *vertexShaderSourceCode =
		"cbuffer ConstantBuffer" \
		"{" \
		"	float4x4 worldViewProjectionMatrix;" \
		"}" \
		"struct vertex_output" \
		"{" \
		"	float4 position:SV_POSITION;" \
		"};" \
		"vertex_output main( float4 pos : POSITION) " \
		"{" \
		"	vertex_output vert_op;"	\
		"	vert_op.position = mul(worldViewProjectionMatrix,pos);" \
		"	return(vert_op);" \
		"}";

	ID3DBlob *pID3DBlob_VertexShaderCode = NULL;
	ID3DBlob *pID3DBlob_Error = NULL;
	// D3DCompile Calls PInvoke to get desired output from fxc.exe 
	hr = D3DCompile(
		vertexShaderSourceCode, // Src Code
		lstrlenA(vertexShaderSourceCode) + 1, // to accomodate last null pointer in Shader String
		"VS", // goes to fxc.exe to prepare and Compile Vertex Shader from given src code above
		NULL, // D3D_SHADER_MACRO => needed when we have #defines
		D3D_COMPILE_STANDARD_FILE_INCLUDE, // ID3DInclude  if we have #includes in our shader
		"main", // EntriPoint
		"vs_5_0", // Shader Level
		0, // Shader Compiler Constant ==> for Debugging and Optimization 
		0, // Effects Constant
		&pID3DBlob_VertexShaderCode,  // Shader Byte Code
		&pID3DBlob_Error // Erros String in Sahder
		);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&g_pFile, g_szLogFileName, "a+");
			fprintf_s(g_pFile, "D3DCompile() failed for Vertex Sahder : %s \n", (char*)pID3DBlob_Error->GetBufferPointer());
			fclose(g_pFile);
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
			return(hr);
		}
		else
		{
			// Check Com Errors
			fopen_s(&g_pFile, g_szLogFileName, "a+");
			fprintf_s(g_pFile, "\nD3DCompile() failed for Vertex Sahder(COM Specific Issue) : \n");
			fclose(g_pFile);
		}
	}
	else
	{
		// Success Message
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "D3DCompile() Succeeded for Vertex Sahder \n");
		fclose(g_pFile);
	}

	hr = g_pID3D11Device->CreateVertexShader(pID3DBlob_VertexShaderCode->GetBufferPointer(), pID3DBlob_VertexShaderCode->GetBufferSize(), NULL, &g_pID3D11VertexShader);

	if (FAILED(hr))
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "ID3D11Device->CreateVertexShader() Failed \n");
		fclose(g_pFile);
		return hr;
	}
	else
	{
		// Success Message
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "ID3D11Device->CreateVertexShader() Succeeded \n");
		fclose(g_pFile);
	}

	g_pID3D11DeviceContext->VSSetShader(g_pID3D11VertexShader, 0, 0);
	// Vertex Shader End


	// Pixel Shader Start
	const char *pixelShaderSourceCode =
		"float4 main(float4 pos : SV_POSITION) : SV_TARGET"	\
		"{"	\
		"	return (float4(1.0,0.0,0.0,1.0));"	\
		"}";

	ID3DBlob *pID3DBlob_PixelShaderSource = NULL;
	pID3DBlob_Error = NULL;
	hr = D3DCompile(pixelShaderSourceCode, lstrlenA(pixelShaderSourceCode) + 1, "PS", NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "ps_5_0", 0, 0, &pID3DBlob_PixelShaderSource, &pID3DBlob_Error);

	if (FAILED(hr))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&g_pFile, g_szLogFileName, "a+");
			fprintf_s(g_pFile, "D3DCompile() failed for Pixel Shader : %s \n", (char*)pID3DBlob_Error->GetBufferPointer());
			fclose(g_pFile);
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
			return(hr);
		}
		else
		{
			// Check Com Errors
			fopen_s(&g_pFile, g_szLogFileName, "a+");
			fprintf_s(g_pFile, "\nD3DCompile() failed for Pixel Shader(COM Specific Issue) : \n");
			fclose(g_pFile);
		}
	}
	else
	{
		// Success Message
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "D3DCompile() Succeeded forPixel Shader \n");
		fclose(g_pFile);
	}

	hr = g_pID3D11Device->CreatePixelShader(pID3DBlob_PixelShaderSource->GetBufferPointer(), pID3DBlob_PixelShaderSource->GetBufferSize(), NULL, &g_pID3D11PixelShader);
	if (FAILED(hr))
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "ID3D11Device->CreatePixelShader() Failed \n");
		fclose(g_pFile);
		pID3DBlob_VertexShaderCode->Release();
		pID3DBlob_VertexShaderCode = NULL;
		pID3DBlob_PixelShaderSource->Release();
		pID3DBlob_PixelShaderSource = NULL;
		return hr;
	}
	else
	{
		// Success Message
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "ID3D11Device->CreatePixelShader() Succeeded \n");
		fclose(g_pFile);
	}

	g_pID3D11DeviceContext->PSSetShader(g_pID3D11PixelShader, 0, 0);

	// Pixel Shader End

	// Create and set Input Layout
	D3D11_INPUT_ELEMENT_DESC inputElementDesc[1];
	ZeroMemory(inputElementDesc, sizeof(inputElementDesc));
	inputElementDesc[0].SemanticName = "POSITION";
	inputElementDesc[0].SemanticIndex = 0;
	inputElementDesc[0].Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	inputElementDesc[0].InputSlot = 0;
	inputElementDesc[0].AlignedByteOffset = 0;
	inputElementDesc[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc[0].InstanceDataStepRate = 0;

	hr = g_pID3D11Device->CreateInputLayout(inputElementDesc, 1, pID3DBlob_VertexShaderCode->GetBufferPointer(), pID3DBlob_VertexShaderCode->GetBufferSize(), &g_pID3D11InputLayout_Position);

	if (FAILED(hr))
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "ID3D11Device->CreateInputLayout() Position Failed. \n");
		fclose(g_pFile);
		return(hr);
	}
	else
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "ID3D11Device->CreateInputLayout() Position Succeded. \n");
		fclose(g_pFile);
	}
	g_pID3D11DeviceContext->IASetInputLayout(g_pID3D11InputLayout_Position);


	
	pID3DBlob_VertexShaderCode->Release();
	pID3DBlob_VertexShaderCode = NULL;
	pID3DBlob_PixelShaderSource->Release();
	pID3DBlob_PixelShaderSource = NULL;

	float vertices[] =
	{
		0.0f,1.0f,0.0f,
		1.0f,-1.0f,0.0f,
		-1.0f,-1.0f,0.0f
	};

	// Create Vertex Buffer
	D3D11_BUFFER_DESC bufferDesc;
	ZeroMemory(&bufferDesc, sizeof(bufferDesc));
	bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	bufferDesc.ByteWidth = sizeof(float)*meshHeight*meshWidth*4;// _ARRAYSIZE for Visual Studio 17 only.
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	hr = g_pID3D11Device->CreateBuffer(&bufferDesc, NULL, &g_pID3D11Buffer_VertexBuffer_Position);
	if (FAILED(hr))
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "ID3D11Device->CreateBuffer() Failed. \n");
		fclose(g_pFile);
		return hr;
	}
	else
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "ID3D11Device->CreateBuffer() Succeded. \n");
		fclose(g_pFile);
	}

	cl_vbo_resource = pfn_clCreateFromD3D11Buffer(g_context, CL_MEM_WRITE_ONLY, g_pID3D11Buffer_VertexBuffer_Position,&ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "Could Not CL Buffer from GL buffer...!! \n");
		fprintf(g_pFile, "OpenCL Error : clCreateFromGLBuffer() Failed : %d. \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);
		return OCL_FAILURE_CREATE_CL_BUFFER_FROM_D3D;
	}

	// Copy Data into GPU
	// Copy Vertices into Above Buffer
	// This Can Be mapped to glMapBuffer() in OpenGL
	/*D3D11_MAPPED_SUBRESOURCE mappedSubresource;
	ZeroMemory(&mappedSubresource, sizeof(mappedSubresource));
	g_pID3D11DeviceContext->Map(g_pID3D11Buffer_VertexBuffer_Position, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSubresource);
	memcpy(mappedSubresource.pData, vertices, sizeof(vertices));
	g_pID3D11DeviceContext->Unmap(g_pID3D11Buffer_VertexBuffer_Position, NULL);*/

	// define and Set Constant Buffer
	D3D11_BUFFER_DESC bufferDesc_ConstantBuffer;
	ZeroMemory(&bufferDesc_ConstantBuffer, sizeof(bufferDesc_ConstantBuffer));
	bufferDesc_ConstantBuffer.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc_ConstantBuffer.ByteWidth = sizeof(CBUFFER);
	bufferDesc_ConstantBuffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

	hr = g_pID3D11Device->CreateBuffer(&bufferDesc_ConstantBuffer, nullptr, &g_pID3D11Buffer_ConstantBuffer);
	if (FAILED(hr))
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "ID3D11Device->CreateBuffer() Constant Buffer Failed. \n");
		fclose(g_pFile);
		return hr;
	}
	else
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "ID3D11Device->CreateBuffer() Constant Buffer Succeded. \n");
		fclose(g_pFile);
	}

	g_pID3D11DeviceContext->VSSetConstantBuffers(0, 1, &g_pID3D11Buffer_ConstantBuffer);

	// glClearColor();
	g_fClearColor[0] = 0.0f;
	g_fClearColor[1] = 0.0f;
	g_fClearColor[2] = 0.0f;
	g_fClearColor[3] = 1.0f;

	g_PerspectiveProjectionMatrix = XMMatrixIdentity();

	/*			OCL Program	Start			*/
	// create opncl code file from given file
	chOCLSourceCode = load_programSource("kernel.cl", "", &sizeKernelCodeLength);

	cl_int status = 0;
	g_program = clCreateProgramWithSource(g_context, 1, (const char **)&chOCLSourceCode, &sizeKernelCodeLength, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : clCreateProgramWithSource Failed : %d. \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);
		return OCL_FAILURE_CREATE_PROG_WITH_SRC;
	}

	// Build OpenCL Program
	ret_ocl = clBuildProgram(g_program, 0, NULL, NULL, NULL, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : clBuildProgram Failed : %d. \nExitting Now..\n", ret_ocl);

		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		fprintf(g_pFile, "OpenCL Program Build log : %s \n", buffer);
		fclose(g_pFile);

		return OCL_FAILURE_BUILD_PROG;
	}

	// Craete OpenCl kernel function
	g_kernel = clCreateKernel(g_program, "sine_wave", &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : clCreateKernel Failed : %d. \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);

		return OCL_FAILURE_CREATE_KERNEL;
	}

	// Set kernel arguments
	ret_ocl = clSetKernelArg(g_kernel, 0, sizeof(cl_mem), (void*)&cl_vbo_resource);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : clSetKernelArg Failed : %d. \nFor Parameter 0 \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);

		return OCL_FAILURE_SET_KERNEL_ARG;
	}

	ret_ocl = clSetKernelArg(g_kernel, 1, sizeof(cl_int), (void*)&meshWidth);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : clSetKernelArg Failed : %d. \nFor Parameter 1 \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);
		return OCL_FAILURE_SET_KERNEL_ARG;
	}

	ret_ocl = clSetKernelArg(g_kernel, 2, sizeof(cl_int), (void*)&meshHeight);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : clSetKernelArg Failed : %d. \nFor Parameter 2 \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);
		return OCL_FAILURE_SET_KERNEL_ARG;
	}

	ret_ocl = clSetKernelArg(g_kernel, 3, sizeof(cl_float), (void*)&g_fAnime);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : clSetKernelArg Failed : %d. \nFor Parameter 3 \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);
		return OCL_FAILURE_SET_KERNEL_ARG;
	}

	/*			OCL Program	Stop 			*/

	// Warm Up Call
	hr = Resize(WIN_WIDTH, WIN_HEIGHT);
	if (FAILED(hr))
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "Resize Failed .\n");
		fclose(g_pFile);
		return hr;
	}
	else
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "Resize Completed .\n");
		fclose(g_pFile);
	}


	return(S_OK);

}


char* load_programSource(const char *filename, const char *preamble, size_t *iSize)
{
	FILE *pFile = NULL;
	size_t sizeSourceLength;
	size_t sizePreambleLength = (size_t)strlen(preamble);

	if (fopen_s(&pFile, filename, "rb") != 0)
	{
		return NULL;
	}

	fseek(pFile, 0, SEEK_END);
	sizeSourceLength = ftell(pFile);
	fseek(pFile, 0, SEEK_SET);

	char *sourceString = (char*)calloc(sizeSourceLength + sizePreambleLength + 1, sizeof(char));
	memcpy(sourceString, preamble, sizePreambleLength);//push preabmble

	if (fread((sourceString)+sizePreambleLength, sizeSourceLength, 1, pFile) != 1)
	{
		fclose(pFile);
		free(sourceString);
		return 0;
	}


	fclose(pFile);
	if (iSize != 0)
	{
		*iSize = sizeSourceLength + sizePreambleLength + 1;
	}

	sourceString[sizeSourceLength + sizePreambleLength] = '\0';

	return sourceString;
}


int Update(void)
{
	cl_int ret_ocl = !CL_SUCCESS;
	size_t globalWorkSize[3] = { meshWidth, meshHeight,1 };
	size_t localWorkSize[3] = { 32,32 ,1 };

	// Acquire OpenGL Object
	ret_ocl = pfn_clEnqueueAcquireD3D11ObjectsNV(g_commandQueue,1, &cl_vbo_resource, 0, 0, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : IN UPdate : clEnqueueAcquireGLObjects Failed : %d. \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);

		return OCL_FAILURE_ENQ_ACQUIRE_D3D_OBJ;
	}

	// Update Animate parameter
	ret_ocl = clSetKernelArg(g_kernel, 3, sizeof(float), (void*)&g_fAnime);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : IN Update : clSetKernelArg Failed : %d. \nFor Parameter 3 \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);

		return OCL_FAILURE_SET_KERNEL_ARG;
	}

	ret_ocl = clEnqueueNDRangeKernel(g_commandQueue, g_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, 0, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : clEnqueueNDRangeKernel Failed : %d. \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);

		return OCL_FAILURE_ENQ_KERNEL;
	}

	// Release OpenGL Object
	ret_ocl = pfn_clEnqueueReleaseD3D11ObjectsNV(g_commandQueue, 1, &cl_vbo_resource, 0, 0, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : IN UPdate : clEnqueueReleaseGLObjects Failed : %d. \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);
		return OCL_FAILURE_ENQ_RELEASE_GL_OBJ;
	}

	ret_ocl = clFinish(g_commandQueue);
	if (ret_ocl != CL_SUCCESS)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf(g_pFile, "OpenCL Error : IN UPdate : clFinish Failed : %d. \nExitting Now..\n", ret_ocl);
		fclose(g_pFile);
		return OCL_FAILURE_CL_FINISH;
	}


	g_fAnime = g_fAnime + 0.0009f;


	return INIT_ALL_OK;
}

HRESULT Resize(int iWidth, int iHeight)
{
	HRESULT hr = S_OK;

	if (g_pID3D11RenderTargetView)
	{
		g_pID3D11RenderTargetView->Release();
		g_pID3D11RenderTargetView = NULL;
	}

	g_pIDXGISwapChain->ResizeBuffers(1, iWidth, iHeight, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

	ID3D11Texture2D *pID3D11Texture2D_BackBuffer;
	g_pIDXGISwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&pID3D11Texture2D_BackBuffer);

	hr = g_pID3D11Device->CreateRenderTargetView(pID3D11Texture2D_BackBuffer, NULL, &g_pID3D11RenderTargetView);

	if (FAILED(hr))
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "ID3D11Device::CreateRenderTargetView Failed .\n");
		fclose(g_pFile);
		return hr;
	}
	else
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "ID3D11Device::CreateRenderTargetView Completed .\n");
		fclose(g_pFile);
	}

	pID3D11Texture2D_BackBuffer->Release();
	pID3D11Texture2D_BackBuffer = NULL;

	g_pID3D11DeviceContext->OMSetRenderTargets(1, &g_pID3D11RenderTargetView, NULL);

	D3D11_VIEWPORT d3dViewPort;

	d3dViewPort.TopLeftX = 0;
	d3dViewPort.TopLeftY = 0;
	d3dViewPort.Width = (float)iWidth;
	d3dViewPort.Height = (float)iHeight;
	d3dViewPort.MinDepth = 0.0f;
	d3dViewPort.MaxDepth = 1.0f;
	g_pID3D11DeviceContext->RSSetViewports(1, &d3dViewPort);

	g_PerspectiveProjectionMatrix = XMMatrixPerspectiveFovLH(XMConvertToRadians(45.0f), (float)iWidth / (float)iHeight, 0.1f, 100.0f);


	return hr;

}

void Display()
{
	g_pID3D11DeviceContext->ClearRenderTargetView(g_pID3D11RenderTargetView, g_fClearColor);

	UINT stride = sizeof(float) * 4;
	UINT offset = 0;

	g_pID3D11DeviceContext->IASetVertexBuffers(0, 1, &g_pID3D11Buffer_VertexBuffer_Position, &stride, &offset);
	

	//g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	g_pID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
	
	

	XMMATRIX worldMatrix = XMMatrixIdentity();
	XMMATRIX viewMatrix = XMMatrixIdentity();

	worldMatrix = XMMatrixTranslation(0.0f, 0.0f, 3.0f);
	XMMATRIX wvpMatrix = worldMatrix*viewMatrix*g_PerspectiveProjectionMatrix;

	CBUFFER constantBuffer;
	constantBuffer.WorldViewProjectionMatrix = wvpMatrix;
	g_pID3D11DeviceContext->UpdateSubresource(g_pID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer, 0, 0);

	g_pID3D11DeviceContext->Draw(MESH_WIDTH * MESH_WIDTH, 0);


	g_pIDXGISwapChain->Present(0, 0);
}

void UnInitialize(void)
{

	/*		OpenCL UnIntialize Start		*/	

	if (pfn_clGetDeviceIDsFromD3D11)
	{
		
	}

	if (g_commandQueue != 0)
	{
		clReleaseCommandQueue(g_commandQueue);
		g_commandQueue = 0;
	}

	if (g_kernel != 0)
	{
		clReleaseKernel(g_kernel);
		g_kernel = 0;
	}


	if (g_program != 0)
	{
		clReleaseProgram(g_program);
		g_program = 0;
	}

	if (g_context != 0)
	{
		clReleaseContext(g_context);
		g_context = NULL;
	}

	/*		OpenCL UnIntialize End 		*/

	if (g_pID3D11Buffer_ConstantBuffer)
	{
		g_pID3D11Buffer_ConstantBuffer->Release();
		g_pID3D11Buffer_ConstantBuffer = NULL;
	}

	if (g_pID3D11InputLayout_Position)
	{
		g_pID3D11InputLayout_Position->Release();
		g_pID3D11InputLayout_Position = NULL;
	}

	if (g_pID3D11Buffer_VertexBuffer_Position)
	{
		g_pID3D11Buffer_VertexBuffer_Position->Release();
		g_pID3D11Buffer_VertexBuffer_Position = NULL;
	}

	if (g_pID3D11PixelShader)
	{
		g_pID3D11PixelShader->Release();
		g_pID3D11PixelShader = NULL;
	}


	if (g_pID3D11VertexShader)
	{
		g_pID3D11VertexShader->Release();
		g_pID3D11VertexShader = NULL;
	}

	if (g_pID3D11RenderTargetView)
	{
		g_pID3D11RenderTargetView->Release();
		g_pID3D11RenderTargetView = NULL;
	}

	if (g_pIDXGISwapChain)
	{
		g_pIDXGISwapChain->Release();
		g_pIDXGISwapChain = NULL;
	}

	if (g_pID3D11DeviceContext)
	{
		g_pID3D11DeviceContext->Release();
		g_pID3D11DeviceContext = NULL;
	}

	if (g_pID3D11Device)
	{
		g_pID3D11Device->Release();
		g_pID3D11Device = NULL;
	}

	if (g_pFile)
	{
		fopen_s(&g_pFile, g_szLogFileName, "a+");
		fprintf_s(g_pFile, "UnInitialize Completed.\n");
		fprintf_s(g_pFile, "Log File Closed.\n");
		fclose(g_pFile);
		g_pFile = NULL;
	}

}
