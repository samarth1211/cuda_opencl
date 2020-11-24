#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>

#include<gl\glew.h>
#include<gl\GL.h>

#include"vmath.h"
#include <CL\opencl.h>
#include <CL\cl_gl.h>
#include <CL\cl_gl_ext.h>



#define WIN_WIDTH	800
#define WIN_HEIGHT	600

#define MESH_WIDTH	1024
#define MESH_HEIGHT	1024

#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"OpenCL.lib")

//using namespace std;

enum InitErrorCodes
{
	/*OCL Error Codes*/
	OCL_FAILURE_CL_FINISH = -22,
	OCL_FAILURE_ENQ_RELEASE_GL_OBJ = -21,
	OCL_FAILURE_ENQ_KERNEL = -20,
	OCL_FAILURE_ENQ_ACQUIRE_GL_OBJ = -19,
	OCL_FAILURE_SET_KERNEL_ARG = -18,
	OCL_FAILURE_CREATE_KERNEL = -17,
	OCL_FAILURE_BUILD_PROG = -16,
	OCL_FAILURE_CREATE_PROG_WITH_SRC = -15,
	OCL_FAILURE_CREATE_CL_BUFFER_FROM_GL =-14,
	OCL_FAILURE_CREATE_COMMAND_QUEUE = -13,
	OCL_FAILURE_OBTAIN_DEVICE = -12,
	OCL_FAILURE_CREATE_CONTEXT = -11,
	OCL_FAILURE_OBTAIN_PLATFORM = -10,
	/*OGL Error Codes*/
	INIT_VERTEX_SHADER_COMPILATION_FAILED = -9,
	INIT_FRAGMENT_SHADER_COMPILATION_FAILED,
	INIT_LINK_SHADER_PROGRAM_FAILED,
	INIT_FAIL_GLEW_INIT ,
	INIT_FAIL_BRIDGE_CONTEX_SET,
	INIT_FAIL_BRIDGE_CONTEX_CREATION,
	INIT_FAIL_SET_PIXEL_FORMAT,
	INIT_FAIL_NO_PIXEL_FORMAT,
	INIT_FAIL_NO_HDC,
	INIT_ALL_OK,
};


enum attributeBindLocations
{
	SAM_ATTRIBUTE_POSITION = 0,
	SAM_ATTRIBUTE_COLOR,
	SAM_ATTRIBUTE_NORNAL,
	SAM_ATTRIBUTE_TEXTURE0,
};


LRESULT CALLBACK MainWndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

bool g_bWindowActive = false;
HWND g_hwnd = NULL;
HDC  g_hdc = NULL;
HGLRC g_hrc = NULL;

WINDOWPLACEMENT wpPrev;
DWORD dwStyle;
bool g_bFullScreen = false;

FILE *g_pFile = NULL;

// Shaders
//GLuint iVertexShaderObject = 0;
//GLuint iFragmentShaderObject = 0;
GLuint g_ShaderProgramObject = 0;

// All Vertex Buffers
GLuint g_VertexArrayObject_cl = 0;
GLuint g_VertexBufferObject_Position_cl = 0;
GLuint g_VertexBufferObject_Color_cl = 0;

// Uniforms
GLuint g_Uniform_Model_Matrix = 0;
GLuint g_Uniform_View_Matrix = 0;
GLuint g_Uniform_Projection_Matrix = 0;
// sampler
GLuint g_uniform_TextureSampler;


// Projection
vmath::mat4 g_PersPectiveProjectionMatrix;
float g_fAnime = 0.0f;

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

size_t meshWidth = MESH_WIDTH;
size_t meshHeight = MESH_HEIGHT;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR szCmdLine, int iCmdShow)
{
	//int UnInitialize(void);
	int Initialize(void);
	int Update(void);
	void Render(void);

	// Windowing Elelments
	WNDCLASSEX wndclass;
	MSG msg;
	HWND hwnd = NULL;
	TCHAR szClassName[] = TEXT("Sam_OGL");
	RECT windowRect;

	// Game Loop Control
	bool bDone = false;

	// Initialization Status
	int iInitRet = 0;


	SecureZeroMemory((void*)&wndclass, sizeof(wndclass));
	wndclass.cbSize = sizeof(wndclass);
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.lpfnWndProc = MainWndProc;
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.hInstance = hInstance;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(GRAY_BRUSH);
	wndclass.hIcon = LoadIcon(hInstance, IDI_APPLICATION);
	wndclass.hIconSm = LoadIcon(hInstance, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(hInstance, IDC_ARROW);

	if (!RegisterClassEx(&wndclass))
	{
		MessageBox(NULL, TEXT("Issue...!!!"), TEXT("Could Not RegisterClass() "), MB_OK | MB_ICONERROR);
		exit(EXIT_FAILURE);
	}

	if ((fopen_s(&g_pFile, "SamLogFile.txt", "w+")) == 0)
	{
		fprintf_s(g_pFile, "File Opened Successfully. \n");
	}
	else
	{
		MessageBox(NULL, TEXT("Issue...!!!"), TEXT("Could not open File"), MB_OK | MB_ICONERROR);
		exit(EXIT_FAILURE);
	}

	SecureZeroMemory((void*)&windowRect, sizeof(windowRect));
	windowRect.left = 0;
	windowRect.top = 0;
	windowRect.bottom = WIN_HEIGHT;
	windowRect.right = WIN_WIDTH;
	AdjustWindowRectEx(&windowRect, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_APPWINDOW);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW, szClassName,
		TEXT("First_OpenGL_Window"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		CW_USEDEFAULT, CW_USEDEFAULT,
		windowRect.right - windowRect.left,
		windowRect.bottom - windowRect.top,
		NULL, NULL, hInstance, NULL);

	if (hwnd == NULL)
	{
		MessageBox(NULL, TEXT("Issue...!!!"), TEXT("Could Not CreateWindow() "), MB_OK | MB_ICONERROR);
		exit(EXIT_FAILURE);
	}

	g_hwnd = hwnd;

	iInitRet = Initialize();
	switch (iInitRet)
	{
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
	case INIT_VERTEX_SHADER_COMPILATION_FAILED:
		fprintf_s(g_pFile, "Failed to Compile vertex Shader \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FRAGMENT_SHADER_COMPILATION_FAILED:
		fprintf_s(g_pFile, "Failed to Compile fragment Shader \n");
		DestroyWindow(hwnd);
		break;
	case INIT_LINK_SHADER_PROGRAM_FAILED:
		fprintf_s(g_pFile, "Failed to Link Shader Program Object \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FAIL_GLEW_INIT:
		fprintf_s(g_pFile, "Failed to glewInit \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FAIL_BRIDGE_CONTEX_SET:
		fprintf_s(g_pFile, "Failed to wglMakeCurrent \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FAIL_BRIDGE_CONTEX_CREATION:
		fprintf_s(g_pFile, "Failed to wglCreateContext \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FAIL_SET_PIXEL_FORMAT:
		fprintf_s(g_pFile, "Failed to set Pixel Format \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FAIL_NO_PIXEL_FORMAT:
		fprintf_s(g_pFile, "Failed to get PixelFormat \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FAIL_NO_HDC:
		fprintf_s(g_pFile, "Failed to Get HDC \n");
		DestroyWindow(hwnd);
		break;
	case INIT_ALL_OK:
		fprintf_s(g_pFile, "Initialize Complete \n");
		break;
	default:
		fprintf_s(g_pFile, "Failed UnKnown Reasons \n");
		DestroyWindow(hwnd);
		break;
	}

	ShowWindow(hwnd, SW_SHOWNORMAL);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	cl_int retStatus;

	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				bDone = true;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}

		}
		else
		{
			if (g_bWindowActive)
			{
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
			// Show all Animations
			Render();

		}
	}


	//UnInitialize();

	return ((int)msg.wParam);
}


LRESULT CALLBACK MainWndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	int UnInitialize(void);
	void FullScreen(void);
	bool Resize(int, int);
	switch (iMsg)
	{
	case WM_CREATE:
		PostMessage(hwnd, WM_KEYDOWN, (WPARAM)0x46, (LPARAM)NULL);
		break;

	case WM_SETFOCUS:
		g_bWindowActive = true;
		break;

	case WM_KILLFOCUS:
		g_bWindowActive = false;
		break;

	case WM_KEYDOWN:

		switch (LOWORD(wParam))
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		case 0x46: // 'f' or 'F'
			//MessageBox(hwnd, TEXT("F is pressed"), TEXT("Status"), MB_OK);
			FullScreen();
			break;

		default:
			break;
		}
		break;

	case WM_SIZE:
		Resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_ERASEBKGND:
		return(0);
		//break;
	case WM_CLOSE:
		DestroyWindow(hwnd);
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

int Initialize(void)
{


	bool Resize(int, int);
	char* load_programSource(const char *filename, const char *preamble, size_t *iSize);



	int iPixelIndex = 0;
	
	cl_int ret_ocl = 0;
	// Shader Programs
	GLuint iVertexShaderObject = 0;
	GLuint iFragmentShaderObject = 0;
	
	// Initialize
	PIXELFORMATDESCRIPTOR pfd;

	

	GLenum err = NULL; // GLEW Error codes

	SecureZeroMemory(&pfd, sizeof(pfd));
	pfd.nSize = sizeof(pfd);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;

	g_hdc = GetDC(g_hwnd);
	if (g_hdc == NULL)
	{
		return INIT_FAIL_NO_HDC;
	}

	iPixelIndex = ChoosePixelFormat(g_hdc, &pfd);
	if (iPixelIndex == 0)
	{
		return INIT_FAIL_NO_PIXEL_FORMAT;
	}

	if (SetPixelFormat(g_hdc, iPixelIndex, &pfd) == FALSE)
	{
		return INIT_FAIL_SET_PIXEL_FORMAT;
	}

	g_hrc = wglCreateContext(g_hdc);
	if (g_hrc == NULL)
	{
		return INIT_FAIL_BRIDGE_CONTEX_CREATION;
	}

	if (wglMakeCurrent(g_hdc, g_hrc) == FALSE)
	{
		return INIT_FAIL_BRIDGE_CONTEX_SET;
	}

	// Enables Feature Required for Programable Pipeline
	err = glewInit();
	if (err != GLEW_OK)
	{
		return INIT_FAIL_GLEW_INIT;
	}

	// GL information Start
	fprintf_s(g_pFile, "SHADER_INFO : Vendor is : %s\n", glGetString(GL_VENDOR));
	fprintf_s(g_pFile, "SHADER_INFO : Renderer is : %s\n", glGetString(GL_RENDER));
	fprintf_s(g_pFile, "SHADER_INFO : OpenGL Version is : %s\n", glGetString(GL_VERSION));
	fprintf_s(g_pFile, "SHADER_INFO : GLSL Version is : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	//fprintf_s(g_pFile, "SHADER_INFO : Extention is : %s \n", glGetString(GL_EXTENSIONS));
	// GL information End


	/*		OpenCL	Initialize	Basic Start		*/
	// Get Platform
	ret_ocl = clGetPlatformIDs(1, &firstPlatformID, NULL); // Going for descrete device directly
	if (ret_ocl!=CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : Could not get all of OpenCL Platforms..\n");
		fprintf(g_pFile, "OpenCL Error : clGetPlatformIDs Failed : %d. \nExitting Now..\n", ret_ocl);

		return OCL_FAILURE_OBTAIN_PLATFORM;
	}

	// Get Device
	// To Do

	// Create OCL Context from required device
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformID,//PlatformID
		CL_GL_CONTEXT_KHR,(cl_context_properties)wglGetCurrentContext(),//HGLRC of Window
		CL_WGL_HDC_KHR,(cl_context_properties)wglGetCurrentDC(),// HDC of Window
		0, 0
	};

	g_context = clCreateContextFromType(contextProperties,CL_DEVICE_TYPE_GPU,NULL,NULL,&ret_ocl);
	if ((g_context==NULL) || (ret_ocl != CL_SUCCESS))
	{
		fprintf( g_pFile, "OpenCL Error : clCreateContextFromType Failed : %d. \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_CREATE_CONTEXT;
	}

	// Make Commad Queue....!!!
	/// Get Device for Command Queue

	ret_ocl = clGetDeviceIDs(firstPlatformID, CL_DEVICE_TYPE_GPU, 1, &g_device, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf( g_pFile, "Could Not Get Device...!! \n");
		fprintf(g_pFile, "OpenCL Error : clGetDeviceIDs() Failed : %d. \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_OBTAIN_DEVICE;
	}

	g_commandQueue = clCreateCommandQueue(g_context,g_device,0,&ret_ocl);
	if ( (g_commandQueue==NULL) || (ret_ocl != CL_SUCCESS))
	{
		fprintf(g_pFile, "Could Not Create CommandQueue...!! \n");
		fprintf(g_pFile, "OpenCL Error : clCreateCommandQueue() Failed : %d. \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_CREATE_COMMAND_QUEUE;
	}

	/*		OpenCL	Initialize	Basic Stop		*/
	/// Sam : all Shader Code Start

	/*Vertex Shader Start*/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
	const GLchar *vertexShaderSourceCode = "#version 450 core"	\
		"\n" \
		"layout (location = 0)in vec4 vPosition;" \
		"layout (location = 1)in vec4 vColor;" \
		"layout (location = 0)out vec4 out_Color;" \
		"uniform mat4 u_model_matrix,u_view_matrix,u_projection_matrix;" \
		"void main(void)" \
		"{" \
		"	gl_Position =  u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \
		"	out_Color = vColor;"	\
		"}";

	glShaderSource(iVertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);

	// Compile it
	glCompileShader(iVertexShaderObject);
	GLint iInfoLogLength = 0;
	GLint iShaderCompileStatus = 0;
	GLchar *szInfoLog = NULL;
	glGetShaderiv(iVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus==GL_FALSE)
	{
		glGetShaderiv(iVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog!=NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(iVertexShaderObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf_s(g_pFile,"ERROR : Vertex Shader Compilation Log : %s \n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				return INIT_VERTEX_SHADER_COMPILATION_FAILED;
				//DestroyWindow(g_hwnd);
				//exit(EXIT_FAILURE);
			}
		}
	}

	/*Vertex Shader End*/

	/*Fragment Shader Start*/
	iFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
	const GLchar *fragmentShaderSourceCode = "#version 450 core"	\
		"\n"	\
		"layout (location = 0)in vec4 out_Color;"	\
		"layout (location = 0)out vec4 FragColor;"	\
		"void main(void)"	\
		"{"	\
		"	FragColor = out_Color;"	\
		"}";

	glShaderSource(iFragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);
	glCompileShader(iFragmentShaderObject);
	iInfoLogLength = 0;
	iShaderCompileStatus = 0;
	szInfoLog = NULL;
	glGetShaderiv(iFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus==GL_FALSE)
	{
		glGetShaderiv(iFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog!=NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(iFragmentShaderObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf(g_pFile,"ERROR: Fragment Shader Compilation Log : %s \n",szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				return INIT_FRAGMENT_SHADER_COMPILATION_FAILED;
				//DestroyWindow(g_hwnd);
				//exit(EXIT_FAILURE);
			}
		}
	}
	/*Fragment Shader End*/

	/* Shader Program Start */
	g_ShaderProgramObject = glCreateProgram();
	glAttachShader(g_ShaderProgramObject, iVertexShaderObject);
	glAttachShader(g_ShaderProgramObject, iFragmentShaderObject);
	glBindAttribLocation(g_ShaderProgramObject, SAM_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(g_ShaderProgramObject, SAM_ATTRIBUTE_COLOR, "vColor");
	glLinkProgram(g_ShaderProgramObject);

	GLint iShaderLinkStatus = 0;
	iInfoLogLength = 0;
	glGetProgramiv(g_ShaderProgramObject, GL_LINK_STATUS, &iShaderLinkStatus);
	if (iShaderLinkStatus==GL_FALSE)
	{
		glGetProgramiv(g_ShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog!=NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(g_ShaderProgramObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf_s(g_pFile, "ERROR : Linking Shader Program Objects Failed %s \n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				return INIT_LINK_SHADER_PROGRAM_FAILED;
				//DestroyWindow(g_hwnd);
				//exit(EXIT_FAILURE);
			}
		}
	}
	/* Shader Program End */

	/*Setup Uniforms Start*/
	g_Uniform_Model_Matrix = glGetUniformLocation(g_ShaderProgramObject,"u_model_matrix");
	g_Uniform_Projection_Matrix = glGetUniformLocation(g_ShaderProgramObject, "u_projection_matrix");
	g_Uniform_View_Matrix = glGetUniformLocation(g_ShaderProgramObject, "u_view_matrix");
	/*Setup Uniforms End*/

	/* Fill Buffers Start*/

	// vbo_cl
	
	unsigned int bufferSize = MESH_HEIGHT * MESH_HEIGHT * 4 * sizeof(float);
	glGenVertexArrays(1, &g_VertexArrayObject_cl);//VAO
	glBindVertexArray(g_VertexArrayObject_cl);

	glGenBuffers(1, &g_VertexBufferObject_Position_cl);// vbo position
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_Position_cl);
	glBufferData(GL_ARRAY_BUFFER, bufferSize, NULL, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION);

	cl_vbo_resource = clCreateFromGLBuffer(g_context, CL_MEM_WRITE_ONLY, g_VertexBufferObject_Position_cl, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf(g_pFile,  "Could Not CL Buffer from GL buffer...!! \n");
		fprintf(g_pFile, "OpenCL Error : clCreateFromGLBuffer() Failed : %d. \nExitting Now..\n", ret_ocl);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		return OCL_FAILURE_CREATE_CL_BUFFER_FROM_GL;
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glVertexAttrib3f(SAM_ATTRIBUTE_COLOR, 1.0f, 0.0f, 0.0f);

	glBindVertexArray(0);
	/* Fill Buffers End*/
	/// Sam : all Shader Code End

	/*			OCL Program	Start			*/
	// create opncl code file from given file
	chOCLSourceCode = load_programSource("kernel.cl", "", &sizeKernelCodeLength);

	cl_int status = 0;
	g_program = clCreateProgramWithSource(g_context, 1, (const char **)&chOCLSourceCode, &sizeKernelCodeLength, &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : clCreateProgramWithSource Failed : %d. \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_CREATE_PROG_WITH_SRC;
	}

	// Build OpenCL Program
	ret_ocl = clBuildProgram(g_program, 0, NULL, NULL, NULL, NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : clBuildProgram Failed : %d. \nExitting Now..\n", ret_ocl);

		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		fprintf(g_pFile, "OpenCL Program Build log : %s \n", buffer);

		return OCL_FAILURE_BUILD_PROG;
	}

	// Craete OpenCl kernel function
	g_kernel = clCreateKernel(g_program, "sine_wave", &ret_ocl);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : clCreateKernel Failed : %d. \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_CREATE_KERNEL;
	}

	// Set kernel arguments
	ret_ocl = clSetKernelArg(g_kernel,0,sizeof(cl_mem), (void*)&cl_vbo_resource);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : clSetKernelArg Failed : %d. \nFor Parameter 0 \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_SET_KERNEL_ARG;
	}

	ret_ocl = clSetKernelArg(g_kernel, 1, sizeof(cl_int), (void*)&meshWidth);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : clSetKernelArg Failed : %d. \nFor Parameter 1 \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_SET_KERNEL_ARG;
	}

	ret_ocl = clSetKernelArg(g_kernel, 2, sizeof(cl_int), (void*)&meshHeight);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : clSetKernelArg Failed : %d. \nFor Parameter 2 \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_SET_KERNEL_ARG;
	}

	ret_ocl = clSetKernelArg(g_kernel, 3, sizeof(cl_float), (void*)&g_fAnime);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : clSetKernelArg Failed : %d. \nFor Parameter 3 \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_SET_KERNEL_ARG;
	}

	/*			OCL Program	Stop 			*/


	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	//glEnable(GL_CULL_FACE);

	glClearColor(0.125f, 0.125f, 0.125f, 1.0f);

	g_PersPectiveProjectionMatrix = vmath::mat4::identity();

	Resize(WIN_WIDTH, WIN_HEIGHT);

	return INIT_ALL_OK;
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
	size_t localWorkSize[3] = { 32,32 ,1};

	// Acquire OpenGL Object
	ret_ocl = clEnqueueAcquireGLObjects(g_commandQueue,1,&cl_vbo_resource,0,0,NULL);
	if (ret_ocl!= CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : IN UPdate : clEnqueueAcquireGLObjects Failed : %d. \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_ENQ_ACQUIRE_GL_OBJ;
	}

	// Update Animate parameter
	ret_ocl = clSetKernelArg(g_kernel,3,sizeof(float),(void*)&g_fAnime);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : IN Update : clSetKernelArg Failed : %d. \nFor Parameter 3 \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_SET_KERNEL_ARG;
	}

	ret_ocl = clEnqueueNDRangeKernel(g_commandQueue,g_kernel,3,NULL,globalWorkSize,localWorkSize,0,0,NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : clEnqueueNDRangeKernel Failed : %d. \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_ENQ_KERNEL;
	}

	// Release OpenGL Object
	ret_ocl = clEnqueueReleaseGLObjects(g_commandQueue,1,&cl_vbo_resource,0,0,NULL);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : IN UPdate : clEnqueueReleaseGLObjects Failed : %d. \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_ENQ_RELEASE_GL_OBJ;
	}

	ret_ocl = clFinish(g_commandQueue);
	if (ret_ocl != CL_SUCCESS)
	{
		fprintf(g_pFile, "OpenCL Error : IN UPdate : clFinish Failed : %d. \nExitting Now..\n", ret_ocl);
		return OCL_FAILURE_CL_FINISH;
	}

	// Update floating values
	g_fAnime = g_fAnime + 0.01f;

	/*		Execute The Kernel		*/


	return INIT_ALL_OK;
}

void Render(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	vmath::mat4 modelMatrix = vmath::mat4::identity();
	vmath::mat4 viewMatrix = vmath::mat4::identity();

	/*glUseProgram(g_ShaderProgramObject);
	
	modelMatrix = vmath::translate(0.0f, 0.0f, -3.0f);

	glUniformMatrix4fv(g_Uniform_Model_Matrix, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(g_Uniform_View_Matrix, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(g_Uniform_Projection_Matrix, 1, GL_FALSE, g_PersPectiveProjectionMatrix);
	
	glBindVertexArray(g_VertexArrayObject);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindVertexArray(0);

	glUseProgram(0);*/


	glUseProgram(g_ShaderProgramObject);

	modelMatrix = vmath::translate(0.0f, 0.0f, -3.0f);

	glUniformMatrix4fv(g_Uniform_Model_Matrix, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(g_Uniform_View_Matrix, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(g_Uniform_Projection_Matrix, 1, GL_FALSE, g_PersPectiveProjectionMatrix);

	glBindVertexArray(g_VertexArrayObject_cl);
	glDrawArrays(GL_POINTS, 0, MESH_WIDTH*MESH_HEIGHT);
	glBindVertexArray(0);

	glUseProgram(0);

	SwapBuffers(g_hdc);
}

void FullScreen(void)
{
	MONITORINFO mi = { sizeof(mi) };
	dwStyle = GetWindowLong(g_hwnd, GWL_STYLE);
	if (g_bFullScreen == false)
	{
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(g_hwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(g_hwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(g_hwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(g_hwnd, HWND_TOP,
					mi.rcMonitor.left, mi.rcMonitor.top,
					mi.rcMonitor.right - mi.rcMonitor.left,
					mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
		g_bFullScreen = true;
	}
	else
	{
		SetWindowLong(g_hwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hwnd, &wpPrev);
		SetWindowPos(g_hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE);
		ShowCursor(TRUE);
		g_bFullScreen = false;
	}
}

bool Resize(int iWidth, int iHeight)
{
	if (iHeight <= 0)
	{
		iHeight = 1;
	}

	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);

	g_PersPectiveProjectionMatrix = vmath::perspective(45.0f, (float)iWidth / (float)iHeight, 0.1f, 100.0f);

	return true;
}

int UnInitialize(void)
{
	if (g_bFullScreen == true)
	{
		SetWindowLong(g_hwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hwnd, &wpPrev);
		SetWindowPos(g_hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE);
		ShowCursor(TRUE);
		g_bFullScreen = false;
	}

	/*		OpenCL UnInitialize	Start	*/	

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
		
	/*		OpenCL UnInitialize	Stop 	*/

	glUseProgram(0);

	if (g_ShaderProgramObject)
	{
		GLsizei iShaderCount;
		GLsizei iShaderNumber;
		glUseProgram(g_ShaderProgramObject);
		glGetProgramiv(g_ShaderProgramObject,GL_ATTACHED_SHADERS,&iShaderCount);
		GLuint *pShaders = (GLuint*) calloc(iShaderCount,sizeof(GLuint));
		if (pShaders)
		{
			glGetAttachedShaders(g_ShaderProgramObject, iShaderCount,&iShaderCount, pShaders);
			for (iShaderNumber = 0; iShaderNumber < iShaderCount; iShaderNumber++)
			{
				glDetachShader(g_ShaderProgramObject,pShaders[iShaderNumber]);
				glDeleteShader(pShaders[iShaderNumber]);
				pShaders[iShaderNumber] = 0;
			}
			free(pShaders);
			pShaders = NULL;
		}
		glUseProgram(0);
		glDeleteProgram(g_ShaderProgramObject);
		g_ShaderProgramObject = NULL;
	}


	if (wglGetCurrentContext() == g_hrc)
	{
		wglMakeCurrent(NULL, NULL);
	}

	if (g_hrc)
	{
		wglDeleteContext(g_hrc);
		g_hrc = NULL;
	}

	if (g_hdc)
	{
		ReleaseDC(g_hwnd, g_hdc);
		g_hdc = NULL;
	}


	if (g_pFile)
	{
		fprintf_s(g_pFile, "Closing File \n");
		fclose(g_pFile);
		g_pFile = NULL;
	}
	return 0;
}
