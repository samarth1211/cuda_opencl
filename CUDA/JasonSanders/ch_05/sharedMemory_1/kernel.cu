#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include<gl\glew.h>
#include<gl\GL.h>

#include<cuda.h>
#include<cuda_gl_interop.h>


#include"vmath.h"


#define		DIM				1024
#define		PI				3.1415926535897932f
#define WIN_WIDTH	800
#define WIN_HEIGHT	600

#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"cudart.lib")

//using namespace std;
enum CUDAInitErrorCodes
{
	/* min no -10 */
	INIT_CUDA_SETGLDEVICE_FAILED = -21,
	CUDA_STREAM_SYNC_FAILED,
	CUDA_INIT_DESTROY_SURFACE_OBJ_FAILED,
	INIT_CUDA_REGISTER_IMAGE_FAILED,
	INIT_CUDA_REGISTER_BUFFER_FAILED,
	CUDA_INIT_GRAPHICS_MAPPED_ARRAY_FAILED,
	CUDA_INIT_GRAPHICS_MAPPED_RES_FAILED,
	CUDA_INIT_GRAPHICS_MAPPED_RES_POINTER_FAILED,
	CUDA_INIT_GRAPHICS_UNMAP_RES_FAILED,
	CUDA_INIT_GRAPHICS_ERR_1,
	CUDA_INIT_GRAPHICS_ERR_3,
	INIT_CUDA_CHOOSEDEVICE_FAILED = -10,
	CUDA_INIT_ALL_OK = 0,
};


enum InitErrorCodes
{
	INIT_VERTEX_SHADER_COMPILATION_FAILED = -9,
	INIT_FRAGMENT_SHADER_COMPILATION_FAILED,
	INIT_LINK_SHADER_PROGRAM_FAILED,
	INIT_FAIL_GLEW_INIT,
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
GLuint g_VertexArrayObject = 0;
GLuint g_VertexBufferObject_Position = 0;
GLuint g_VertexBufferObject_TexCoords = 0;

// Uniforms
GLuint g_Uniform_Model_Matrix = 0;
GLuint g_Uniform_View_Matrix = 0;
GLuint g_Uniform_Projection_Matrix = 0;
// sampler
GLuint g_uniform_TextureSampler;

GLuint g_cuda_texture;


// Projection
vmath::mat4 g_PersPectiveProjectionMatrix;

//CUDA Res
cudaGraphicsResource *resource = NULL;

float g_fanimate = 0.0f;
bool animation_flag = false;

// cuda kernel
__global__ void normal_kernel(cudaSurfaceObject_t target, dim3 texDim, float time)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (x < texDim.x && y < texDim.y)
	{
		// now calculate the value at the position
		float fx = x / (float)DIM - 0.5f;
		float fy = y / (float)DIM - 0.5f;
		float  green = 0.5f + 0.5f * sinf(fabsf(fx * 100 * time) - fabsf(fy * 100 * time));
		float4 data = make_float4(0.0f, green, 0.0f, 1.0f);
		surf2Dwrite(data, target, x * sizeof(float4), y, cudaBoundaryModeTrap);

	}
}

__global__ void sharedMem_kernel(cudaSurfaceObject_t target, dim3 texDim)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	__shared__ float shared[16][16];
	const float period = 128;

	if (x < texDim.x && y < texDim.y)
	{
		shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x * 2.0f * PI /period)+1.0f) * (sinf(y * 2.0f * PI /period)+1.0f) / 4.0f;
		uchar4 data = make_uchar4(0.0f, shared[15-threadIdx.x][15-threadIdx.y], 0.0f,255);
		surf2Dwrite(data, target, x * sizeof(uchar4), y, cudaBoundaryModeTrap);
	}
}

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
	case INIT_ALL_OK:
		fprintf_s(g_pFile, "Initialize Complete \n");
		break;
	case INIT_FAIL_NO_HDC:
		fprintf_s(g_pFile, "Failed to Get HDC \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FAIL_NO_PIXEL_FORMAT:
		fprintf_s(g_pFile, "Failed to get PixelFormat \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FAIL_SET_PIXEL_FORMAT:
		fprintf_s(g_pFile, "Failed to set Pixel Format \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FAIL_BRIDGE_CONTEX_CREATION:
		fprintf_s(g_pFile, "Failed to wglCreateContext \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FAIL_BRIDGE_CONTEX_SET:
		fprintf_s(g_pFile, "Failed to wglMakeCurrent \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FAIL_GLEW_INIT:
		fprintf_s(g_pFile, "Failed to glewInit \n");
		DestroyWindow(hwnd);
		break;
	case INIT_LINK_SHADER_PROGRAM_FAILED:
		fprintf_s(g_pFile, "Failed to Link Shader Program Object \n");
		DestroyWindow(hwnd);
		break;
	case INIT_VERTEX_SHADER_COMPILATION_FAILED:
		fprintf_s(g_pFile, "Failed to Compile vertex Shader \n");
		DestroyWindow(hwnd);
		break;
	case INIT_FRAGMENT_SHADER_COMPILATION_FAILED:
		fprintf_s(g_pFile, "Failed to Compile fragment Shader \n");
		DestroyWindow(hwnd);
		break;
	default:
		/*fprintf_s(g_pFile, "Failed UnKnown Reasons \n");
		DestroyWindow(hwnd);*/
		switch (iInitRet)
		{
		case INIT_CUDA_CHOOSEDEVICE_FAILED:
			fprintf_s(g_pFile, "cudaChooseDevice Failed  \n");
			DestroyWindow(hwnd);
			break;
			/*default:
			fprintf_s(g_pFile, "CUDA Failed UnKnown Reasons \n");
			DestroyWindow(hwnd);
			break;*/
		}

		// General Failure
		fprintf_s(g_pFile, "Failed UnKnown Reasons \n");
		DestroyWindow(hwnd);
		break;
	}

	ShowWindow(hwnd, SW_SHOWNORMAL);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);


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
				Update();
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
	int iPixelIndex = 0;
	PIXELFORMATDESCRIPTOR pfd;
	cudaError cuErr;

	// Shader Programs
	GLuint iVertexShaderObject = 0;
	GLuint iFragmentShaderObject = 0;

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

	// CUDA Initalization

	cuErr = cudaGLSetGLDevice(0); // Default device 0 will share resources with OpenGL
	if (cuErr != cudaSuccess)
	{
		return INIT_CUDA_SETGLDEVICE_FAILED;
	}

	// GL information Start
	fprintf_s(g_pFile, "SHADER_INFO : Vendor is : %s\n", glGetString(GL_VENDOR));
	fprintf_s(g_pFile, "SHADER_INFO : Renderer is : %s\n", glGetString(GL_RENDER));
	fprintf_s(g_pFile, "SHADER_INFO : OpenGL Version is : %s\n", glGetString(GL_VERSION));
	fprintf_s(g_pFile, "SHADER_INFO : GLSL Version is : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	int maxAttachments = 0;
	glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxAttachments);
	fprintf_s(g_pFile, "SHADER_INFO : GL_MAX_COLOR_ATTACHMENTS is : %d\n", maxAttachments);
	//fprintf_s(g_pFile, "SHADER_INFO : Extention is : %s \n", glGetString(GL_EXTENSIONS));
	// GL information End

	/// Sam : all Shader Code Start
	/*gl_Position =  u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;*/
	/*Vertex Shader Start*/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
	const GLchar *vertexShaderSourceCode = "#version 450 core"	\
		"\n" \
		"layout (location = 0)in vec4 vPosition;" \
		"layout (location = 3)in vec2 vTexture0_Coord;" \
		"layout (location = 0)out vec2 out_Texture0_Coord;" \
		"uniform mat4 u_model_matrix,u_view_matrix,u_projection_matrix;" \
		"void main(void)" \
		"{" \
		"	gl_Position =  vPosition;" \
		"	out_Texture0_Coord = vTexture0_Coord;"	\
		"}";

	glShaderSource(iVertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);

	// Compile it
	glCompileShader(iVertexShaderObject);
	GLint iInfoLogLength = 0;
	GLint iShaderCompileStatus = 0;
	GLchar *szInfoLog = NULL;
	glGetShaderiv(iVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(iVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(iVertexShaderObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf_s(g_pFile, "ERROR : Vertex Shader Compilation Log : %s \n", szInfoLog);
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
		"layout (location = 0)in vec2 out_Texture0_Coord;"	\
		"layout (location = 0)out vec4 FragColor;"	\
		"uniform sampler2D u_texture0_sampler;"	\
		"void main(void)"	\
		"{"	\
		"	FragColor = texture(u_texture0_sampler,out_Texture0_Coord);"	\
		"}";

	glShaderSource(iFragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);
	glCompileShader(iFragmentShaderObject);
	iInfoLogLength = 0;
	iShaderCompileStatus = 0;
	szInfoLog = NULL;
	glGetShaderiv(iFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(iFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(iFragmentShaderObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf(g_pFile, "ERROR: Fragment Shader Compilation Log : %s \n", szInfoLog);
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
	glBindAttribLocation(g_ShaderProgramObject, SAM_ATTRIBUTE_TEXTURE0, "vTexture0_Coord");
	glLinkProgram(g_ShaderProgramObject);

	GLint iShaderLinkStatus = 0;
	iInfoLogLength = 0;
	glGetProgramiv(g_ShaderProgramObject, GL_LINK_STATUS, &iShaderLinkStatus);
	if (iShaderLinkStatus == GL_FALSE)
	{
		glGetProgramiv(g_ShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
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
	g_Uniform_Model_Matrix = glGetUniformLocation(g_ShaderProgramObject, "u_model_matrix");
	g_Uniform_Projection_Matrix = glGetUniformLocation(g_ShaderProgramObject, "u_projection_matrix");
	g_Uniform_View_Matrix = glGetUniformLocation(g_ShaderProgramObject, "u_view_matrix");
	//g_uniform_TextureSampler = glGetUniformLocation(g_ShaderProgramObject, "u_texture0_sampler");
	/*Setup Uniforms End*/

	/* Fill Buffers Start*/
	
	//// Cube Section Start
	const GLfloat squareVertices[] = {
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f
	};

	const GLfloat squareTexCords[] =
	{
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f,0.0f,
		1.0f,1.0f
	};

	glGenVertexArrays(1, &g_VertexArrayObject);//VAO
	glBindVertexArray(g_VertexArrayObject);

	glGenBuffers(1, &g_VertexBufferObject_Position);// vbo position
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_Position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &g_VertexBufferObject_TexCoords); // vbo texcoords
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_TexCoords);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareTexCords), squareTexCords, GL_STATIC_DRAW);

	glVertexAttribPointer(SAM_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
	/* Fill Buffers End*/

	// Generate texture for working with cuda
	glGenTextures(1, &g_cuda_texture);
	glBindTexture(GL_TEXTURE_2D, g_cuda_texture);

	// Texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// give texture some storage
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA, DIM, DIM,0,GL_RGBA,GL_UNSIGNED_INT,NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	/// Sam : all Shader Code End

	glEnable(GL_TEXTURE_2D);

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	//glEnable(GL_CULL_FACE);

	glClearColor(0.125f, 0.125f, 0.125f, 1.0f);

	
	g_PersPectiveProjectionMatrix = vmath::mat4::identity();

	/// Register With CUDA  Start
	// last param as "cudaGraphicsRegisterFlagsSurfaceLoadStore"
	cuErr = cudaGraphicsGLRegisterImage(&resource, g_cuda_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	if (cuErr!=cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA ERROR : cudaGraphicsGLRegisterImage failed at line %d\n",__LINE__);
		return INIT_CUDA_REGISTER_IMAGE_FAILED;
	}

	/// Register With CUDA  Start
	Resize(WIN_WIDTH, WIN_HEIGHT);

	return INIT_ALL_OK;
}

int Update(void)
{

	if (animation_flag)
	{
		g_fanimate = g_fanimate + 0.005f;
		if ((g_fanimate >1.0f))
		{
			animation_flag = false;
		}
	}
	else
	{
		g_fanimate = g_fanimate - 0.005f;
		if ((g_fanimate <0.0f))
		{
			animation_flag = true;
		}
	}


	//uchar4 *devPtr = NULL;
	size_t size;
	cudaError status;
	cudaArray_t cudaWriteArray;

	status = cudaGraphicsMapResources(1, &resource, 0);
	if (status != cudaSuccess)
	{
		fprintf_s(g_pFile,"IN Update()  cudaGraphicsMapResources failed...!! \n");
		return CUDA_INIT_GRAPHICS_MAPPED_RES_FAILED;
	}

	status = cudaGraphicsSubResourceGetMappedArray(&cudaWriteArray,resource,0,0);
	if (status != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN Update()  cudaGraphicsSubResourceGetMappedArray failed...!! \n");
		return CUDA_INIT_GRAPHICS_MAPPED_ARRAY_FAILED;
	}

	// Prepare a Surface object for cuda
	cudaResourceDesc writeDescriptor;
	ZeroMemory((void**)&writeDescriptor,sizeof(writeDescriptor));
	writeDescriptor.resType = cudaResourceTypeArray;
	writeDescriptor.res.array.array = cudaWriteArray;

	cudaSurfaceObject_t writeSurface;
	status = cudaCreateSurfaceObject(&writeSurface, &writeDescriptor);
	if (status != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN Update()  cudaCreateSurfaceObject failed...!! \n");
		return CUDA_INIT_GRAPHICS_MAPPED_ARRAY_FAILED;
	}

	
	// After successfully creating surface object write to the texture using kernel
	// dim3 thread(32,32);
	// dim3 block(DIM/ thread.x, DIM / thread.y);
	//normal_kernel << <block, thread >> >(writeSurface, dim3(DIM, DIM), g_fanimate);
	dim3 thread(16,16);
	dim3 block(DIM/ thread.x, DIM / thread.y);
	sharedMem_kernel<< <block, thread >> >(writeSurface, dim3(DIM, DIM));

	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN Update() Kernel failed : %s \n", cudaGetErrorString(status));
	}

	/*status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN Update() cudaDeviceSynchronize failed...!! \n");
		return CUDA_INIT_DESTROY_SURFACE_OBJ_FAILED;
	}*/

	status = cudaDestroySurfaceObject(writeSurface);
	if (status != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN Update() cudaDestroySurfaceObject failed...!! \n");
		return CUDA_INIT_DESTROY_SURFACE_OBJ_FAILED;
	}

	status = cudaGraphicsUnmapResources(1, &resource, 0);
	if (status != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN Update() cudaGraphicsUnmapResources failed...!! \n");
		return CUDA_INIT_GRAPHICS_UNMAP_RES_FAILED;
	}

	status = cudaStreamSynchronize(0);
	if (status != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN Update() cudaStreamSynchronize failed...!! \n");
		return CUDA_INIT_GRAPHICS_UNMAP_RES_FAILED;
	}

	return INIT_ALL_OK;
}

void Render(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	vmath::mat4 modelMatrix = vmath::mat4::identity();
	vmath::mat4 viewMatrix = vmath::mat4::identity();

	glUseProgram(g_ShaderProgramObject);

	modelMatrix = vmath::translate(0.0f, 0.0f, -3.0f);

	glUniformMatrix4fv(g_Uniform_Model_Matrix, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(g_Uniform_View_Matrix, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(g_Uniform_Projection_Matrix, 1, GL_FALSE, g_PersPectiveProjectionMatrix);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g_cuda_texture);
	//glUniform1i(g_uniform_TextureSampler, 0);

	glBindVertexArray(g_VertexArrayObject);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
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
	cudaError status;

	if (g_bFullScreen == true)
	{
		SetWindowLong(g_hwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hwnd, &wpPrev);
		SetWindowPos(g_hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE);
		ShowCursor(TRUE);
		g_bFullScreen = false;
	}

	// Uninitalize CUDA objects
	status = cudaGraphicsUnmapResources(1, &resource, 0);
	if (status != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN UnInitialize() cudaGraphicsUnmapResources failed...!! \n");
		cudaDeviceReset();
		return CUDA_INIT_GRAPHICS_UNMAP_RES_FAILED;
	}




	if (g_VertexBufferObject_TexCoords)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_TexCoords);
		g_VertexBufferObject_TexCoords = NULL;
	}

	if (g_VertexBufferObject_Position)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_Position);
		g_VertexBufferObject_Position = NULL;
	}

	if (g_VertexArrayObject)
	{
		glDeleteVertexArrays(1, &g_VertexArrayObject);
		g_VertexArrayObject = NULL;
	}

	glUseProgram(0);
	/*
	glDetachShader(g_ShaderProgramObject, iVertexShaderObject);
	glDetachShader(g_ShaderProgramObject, iFragmentShaderObject);

	if (iFragmentShaderObject)
	{
	glDeleteShader(iFragmentShaderObject);
	iFragmentShaderObject = 0;
	}

	if (iVertexShaderObject)
	{
	glDeleteShader(iVertexShaderObject);
	iVertexShaderObject = 0;
	}

	if (g_ShaderProgramObject)
	{
	glDeleteProgram(g_ShaderProgramObject);
	g_ShaderProgramObject = NULL;
	}*/

	if (g_ShaderProgramObject)
	{
		GLsizei iShaderCount;
		GLsizei iShaderNumber;


		glUseProgram(g_ShaderProgramObject);
		glGetProgramiv(g_ShaderProgramObject, GL_ATTACHED_SHADERS, &iShaderCount);
		GLuint *pShaders = (GLuint*)calloc(iShaderCount, sizeof(GLuint));

		if (pShaders)
		{
			glGetAttachedShaders(g_ShaderProgramObject, iShaderCount, &iShaderCount, pShaders);
			for (iShaderNumber = 0; iShaderNumber < iShaderCount; iShaderNumber++)
			{
				glDetachShader(g_ShaderProgramObject, pShaders[iShaderNumber]);
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

	if (g_cuda_texture)
	{
		glDeleteTextures(1,&g_cuda_texture);
		g_cuda_texture = 0;
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
