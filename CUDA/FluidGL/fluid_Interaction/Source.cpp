
#include<windows.h>
#include<gl\glew.h>
#include<gl\GL.h>


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <map>
#include <string>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
#include <cufft.h>

#include"my_helper_timer.h"
#include"Defines.h"
#include"FluidsGL_Kernels.h"
#include"FluidsGL_Kernels.cuh"

#include"vmath.h"


#define WIN_WIDTH 800
#define WIN_HEIGHT 600
#define MAX_EPSILON_ERROR 1.0f

#pragma comment (lib,"glew32.lib")
#pragma comment (lib,"opengl32.lib")
#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib,"cudart.lib")
#pragma comment(lib,"cufft.lib")
#pragma comment(lib,"cufftw.lib")

//using namespace std;

enum InitErrorCodes
{
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
bool g_bUpdateFlag = false;

FILE *g_pFile = NULL;

// Shaders
//GLuint iVertexShaderObject = 0;
//GLuint iFragmentShaderObject = 0;
GLuint g_ShaderProgramObject = 0;

// All Vertex Buffers
GLuint g_VertexArrayObject = 0;
GLuint g_VertexBufferObject_Position = 0;
GLuint g_VertexBufferObject_Color = 0;

// Uniforms
GLuint g_Uniform_Model_Matrix = 0;
GLuint g_Uniform_View_Matrix = 0;
GLuint g_Uniform_Projection_Matrix = 0;
// sampler
GLuint g_uniform_TextureSampler;
GLuint g_uniform_color;

// Projection
vmath::mat4 g_PersPectiveProjectionMatrix;

/*		Simulation Variables Start		*/
cufftHandle planr2c;
cufftHandle planc2r;
static cData *xvField = NULL;
static cData *yvField = NULL;

cData *hvField = NULL;  // Host Vector
cData *dvField = NULL;  // Device Vector

static int wWidth = MAX(512, DIM);
static int wHeight = MAX(512, DIM);

/*static int wWidth = MAX(1024, DIM);
static int wHeight = MAX(1024, DIM);*/

static int clicked = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
IStopWatchTimer *g_Timer = NULL;

/*		Particle Data Start		*/
GLuint g_iVertexBufferObject = 0;
GLuint g_iVertexArrayObject_fluid = 0;
struct cudaGraphicsResource *cuda_vbo_resource;
static cData *particles = NULL; // particle position in host memory
								// Texture pitch
size_t tPitch = 0;
/*		Particle Data Stop		*/



static int lastx = 0, lasty = 0;
static int currentX = 0, currentY = 0;
/*		Simulation Variables Stop		*/

GLfloat particleColor[3] = { 0.86f, 0.08f, 0.24f };


GLfloat bkColor[4] = { 0.125f, 0.125f, 0.125f, 1.0f };
const GLfloat depthClear = 1.0f;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR szCmdLine, int iCmdShow)
{
	//int UnInitialize(void);
	int Initialize(void);
	void Update(void);
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
	void motion(int x, int y);


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
			if (g_bFullScreen == false)
			{
				FullScreen();
				g_bFullScreen = true;
			}
			else
			{
				FullScreen();
				g_bFullScreen = false;
			}
			break;

			case 0x55: //U or u => Update
				g_bUpdateFlag = true;
				break;

			case 0x52: // reset

				memset(hvField,0,sizeof(cData)*DS);
				cudaMemcpy(dvField, hvField,sizeof(cData)*DS,cudaMemcpyHostToDevice);
				// Unregister the buffer
				cudaGraphicsUnregisterResource(cuda_vbo_resource);
				glBindBuffer(GL_ARRAY_BUFFER, g_iVertexBufferObject);
				glBufferData(GL_ARRAY_BUFFER, sizeof(cData)*DS,particles, GL_DYNAMIC_DRAW_ARB);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				// re-register the buffer
				cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, g_iVertexBufferObject, cudaGraphicsMapFlagsNone);

				g_bUpdateFlag = false;
				break;

			case 0x31://1
				//  => crimson
				particleColor[0] = 0.86f;
				particleColor[1] = 0.08f;
				particleColor[2] = 0.24f;

				bkColor[0] = 0.125f;
				bkColor[1] = 0.125f;
				bkColor[2] = 0.125f;
				break;

			case 0x32: //2
				//  => green 
				particleColor[0] = 0.13f;
				particleColor[1] = 0.55f;
				particleColor[2] = 0.13f;

				bkColor[0] = 0.125f;
				bkColor[1] = 0.125f;
				bkColor[2] = 0.125f;
				break;

			case 0x33: //3
				//  => corn flower blue

				particleColor[0] = 0.39f;
				particleColor[1] = 0.58f;
				particleColor[2] = 0.93f;

				bkColor[0] = 0.125f;
				bkColor[1] = 0.125f;
				bkColor[2] = 0.125f;
				break;

			case 0x34: //4
			//  => Cyna   0.00, 1.00, 1.0

				particleColor[0] = 0.0f;
				particleColor[1] = 1.0f;
				particleColor[2] = 1.0f;

				bkColor[0] = 0.125f;
				bkColor[1] = 0.125f;
				bkColor[2] = 0.125f;
				break;

			case 0x35: //5
		   //  => Yellow  

				particleColor[0] = 1.0f;
				particleColor[1] = 0.84f;
				particleColor[2] = 0.0f;

				bkColor[0] = 0.125f;
				bkColor[1] = 0.125f;
				bkColor[2] = 0.125f;
				break;

			case 0x36: //6
			//  => Magents

				particleColor[0] = 1.0f;
				particleColor[1] = 0.0f;
				particleColor[2] = 1.0f;

				bkColor[0] = 0.125f;
				bkColor[1] = 0.125f;
				bkColor[2] = 0.125f;
				break;

			case 0x30: //0 
				// black and white

				particleColor[0] = 0.0f;
				particleColor[1] = 0.0f;
				particleColor[2] = 0.0f;

				bkColor[0] = 1.0f;
				bkColor[1] = 1.0f;
				bkColor[2] = 1.0f;
				break;

		default:
			break;
		}
		break;

	case WM_SIZE:
		Resize(LOWORD(lParam), HIWORD(lParam));
		wWidth = LOWORD(lParam);
		wHeight = HIWORD(lParam);
		break;
	case WM_ERASEBKGND:
		return(0);
		//break;
	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_LBUTTONUP:
	case WM_LBUTTONDOWN:
		// click logic
		lastx = LOWORD(lParam);
		lasty = HIWORD(lParam);
		clicked = !clicked;
		break;

	case WM_MOUSEMOVE:
		// logic for motion()
		//LOWORD(lParam); // x
		//HIWORD(lParam);//y
		currentX = LOWORD(lParam), currentY = HIWORD(lParam);
		
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


void motion(int x, int y)
{
	// convert motion coordinates to domain
	float fx = (lastx / (float)wWidth);
	float fy = (lasty / (float)wHeight);
	int nx = (int)(fx*DIM);
	int ny = (int)(fy*DIM);

	if (clicked && (nx  < DIM - FR) && (nx>FR - 1) && (ny < DIM - FR) && (ny > FR - 1))
	{
		int ddx = x - lastx;
		int ddy = y - lasty;
		fx = ddx / (float)wWidth;
		fy = ddy / (float)wHeight;
		int spy = ny - FR;
		int spx = nx - FR;
		addForces(dvField, DIM, DIM, spx, spy, FORCE*DT*fx, FORCE*DT*fy, FR);

		lastx = x;
		lasty = y;
	}
}

int Initialize(void)
{
	bool Resize(int, int);
	void initParticles(cData *p, int dx, int dy);


	int iPixelIndex = 0;
	PIXELFORMATDESCRIPTOR pfd;

	// Shader Programs
	GLuint iVertexShaderObject = 0;
	GLuint iFragmentShaderObject = 0;

	GLenum glewErr = NULL; // GLEW Error codes
	cudaError err;  //CUDA error
	int cufftErrs;  // CUDA FFT Error

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
	glewErr = glewInit();
	if (glewErr != GLEW_OK)
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

	/// Sam : all Shader Code Start

	/*Vertex Shader Start*/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
	const GLchar *vertexShaderSourceCode = "#version 450 core"	\
		"\n" \
		"layout (location = 0)in vec2 vPosition;\n" \
		"uniform mat4 u_model_matrix,u_view_matrix,u_projection_matrix;\n" \
		"void main(void)" \
		"{\n" \
		"	gl_Position =  u_projection_matrix * u_view_matrix * u_model_matrix * vec4(vPosition.x,vPosition.y,0.0,1.0);\n" \
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
		"layout (location = 0)out vec4 FragColor;\n"	\
		"uniform vec3 u_color;"	\
		"void main(void)"	\
		"{\n"	\
		"	FragColor = vec4(u_color, 0.5);\n"	\
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
	//glBindAttribLocation(g_ShaderProgramObject, SAM_ATTRIBUTE_COLOR, "vColor");
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
	g_uniform_color = glGetUniformLocation(g_ShaderProgramObject, "u_color");
	/*Setup Uniforms End*/

	/* Fill Buffers Start
	const GLfloat triangleVertices[] = { 0.0f,1.0f,0.0f,
		-1.0f,-1.0f,0.0f,
		1.0f ,-1.0f ,0.0f };
	const GLfloat triangleColors[] = { 1.0f,0.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,0.0f,1.0f };
	glGenVertexArrays(1, &g_VertexArrayObject);//VAO
	glBindVertexArray(g_VertexArrayObject);

	glGenBuffers(1, &g_VertexBufferObject_Position);// vbo position
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_Position);
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW);
	
	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &g_VertexBufferObject_Color); // vbo color
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_Color);
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangleColors), triangleColors, GL_STATIC_DRAW);

	glVertexAttribPointer(SAM_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_COLOR);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	glBindVertexArray(0);
	/* Fill Buffers End*/
	/// Sam : all Shader Code End

	/* Simmulation stuff start  */
	//GLint bsize;
	
	sdkCreateTimer(&g_Timer);
	sdkResetTimer(&g_Timer);

	hvField = (cData *)malloc(DS*sizeof(cData));
	memset(hvField, 0, sizeof(cData)*DS);
	if (hvField == NULL)
	{
		fprintf_s(g_pFile, "Memory allocation Failed...!!\n");
	}

	err = cudaMallocPitch((void**)&dvField, &tPitch, sizeof(cData)*DIM, DIM);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMallocPitch()\n");
	}

	err = cudaMemcpy(dvField, hvField, sizeof(cData)*DS, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMemcpy()\n");
	}

	// Temporary complex velocity field data
	err = cudaMalloc((void**)&xvField, sizeof(cData)*PDS);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMalloc()\n");
	}

	err = cudaMalloc((void**)&yvField, sizeof(cData)*PDS);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMalloc()\n");
	}

	setupTexture(DIM, DIM);
	bindTexture();

	// Create Particle Array
	particles = (cData*)malloc(DS*sizeof(cData));
	memset(particles, 0, sizeof(cData)*DS);
	if (particles == NULL)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMalloc()\n");
	}

	initParticles(particles, DIM, DIM);
	/*for (int i = 0; i < DIM; i++)
	{
	for (int j = 0; j < DIM; j++)
	{
	particles[i*DIM + j].x = (j+0.5f+(rand()/RAND_MAX - 0.5f)) / DIM;
	particles[i*DIM + j].y = (i + 0.5f + (rand() / RAND_MAX - 0.5f)) / DIM;
	}
	}*/

	cufftErrs = cufftPlan2d(&planr2c, DIM, DIM, CUFFT_R2C);
	if (cufftErrs != (int)CUFFT_SUCCESS)
	{
		fprintf_s(g_pFile, "int diffuseProject, cufftPlan2d 1 failed %d \n", cufftErrs);
	}

	cufftErrs = cufftPlan2d(&planc2r, DIM, DIM, CUFFT_C2R);
	if (cufftErrs != (int)CUFFT_SUCCESS)
	{
		fprintf_s(g_pFile, "int diffuseProject, cufftPlan2d 2 failed %d \n", cufftErrs);
	}

	glGenVertexArrays(1, &g_iVertexArrayObject_fluid);
	glBindVertexArray(g_iVertexArrayObject_fluid);

	glGenBuffers(1, &g_iVertexBufferObject);
	glBindBuffer(GL_ARRAY_BUFFER, g_iVertexBufferObject);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cData)*DS, particles, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	err = cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, g_iVertexBufferObject, cudaGraphicsMapFlagsNone);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaGraphicsGLRegisterBuffer()\n");
	}

	/* Simmulation stuff stop   */

	glEnable(GL_POINT_SMOOTH);
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


void initParticles(cData *p, int dx, int dy)
{
	int i, j;

	for (i = 0; i < dy; i++)
	{
		for (j = 0; j < dx; j++)
		{
			p[i*dx + j].x = (j + 0.5f + ((rand() / (float)RAND_MAX) - 0.5f)) / dx;
			p[i*dx + j].y = (i + 0.5f + ((rand() / (float)RAND_MAX) - 0.5f)) / dy;
		}
	}
}

void Update(void)
{
	void motion(int x, int y);
	void SimulateFluids();

	motion(currentX, currentY);
	SimulateFluids();
}

void Render(void)
{
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearBufferfv(GL_COLOR, 0, bkColor);
	glClearBufferfv(GL_DEPTH, 0, &depthClear);

	vmath::mat4 modelMatrix = vmath::mat4::identity();
	vmath::mat4 viewMatrix = vmath::mat4::identity();

	glUseProgram(g_ShaderProgramObject);
	
	glUniformMatrix4fv(g_Uniform_Model_Matrix, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(g_Uniform_View_Matrix, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(g_Uniform_Projection_Matrix, 1, GL_FALSE, g_PersPectiveProjectionMatrix);
	
	glUniform3f(g_uniform_color, particleColor[0], particleColor[1], particleColor[2]);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	glBindVertexArray(g_iVertexArrayObject_fluid);
	glDrawArrays(GL_POINTS, 0, DS);
	glBindVertexArray(0);

	glDisable(GL_BLEND);
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
		//ShowCursor(FALSE);
		g_bFullScreen = true;
	}
	else
	{
		SetWindowLong(g_hwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hwnd, &wpPrev);
		SetWindowPos(g_hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE);
		//ShowCursor(TRUE);
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

	//g_PersPectiveProjectionMatrix = vmath::perspective(45.0f, (float)iWidth / (float)iHeight, 0.1f, 100.0f);

	g_PersPectiveProjectionMatrix = vmath::ortho(0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f);

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

	cudaGraphicsUnregisterResource(cuda_vbo_resource);

	unbindTexture();
	deleteTexture();

	// Free all host and device resources
	if (hvField)
	{
		free(hvField);
		hvField = NULL;
	}
	
	if (particles)
	{
		free(particles);
		particles = NULL;
	}
	
	if (dvField)
	{
		cudaFree(dvField);
		dvField = NULL;
	}
	
	if (xvField)
	{
		cudaFree(xvField);
		xvField = NULL;
	}
	
	if (yvField)
	{
		cudaFree(yvField);
		yvField = NULL;
	}
	
	if (planr2c)
	{
		cufftDestroy(planr2c);
		planr2c = NULL;
	}
	
	if (planc2r)
	{
		cufftDestroy(planc2r);
		planc2r = NULL;
	}
	

	cudaDeviceReset();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &g_iVertexBufferObject);

	sdkDeleteTimer(&g_Timer);

	if (g_VertexBufferObject_Color)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_Color);
		g_VertexBufferObject_Color = NULL;
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

void SimulateFluids()
{

	fprintf_s(g_pFile, "\nStart Simmulation\n");
	fprintf_s(g_pFile, "++++++++++++++++++++++++++++++++++++\n");

	addvectVelocity(dvField, (float*)xvField, (float*)yvField, DIM, RPADW, DIM, DT);
	diffuseProject(xvField, yvField, CPADW, DIM, DT, VIS);
	if (g_bUpdateFlag)
	{
		updateVelocity(dvField, (float*)xvField, (float*)yvField, DIM, RPADW, DIM);
	}
	advectParticles(g_iVertexBufferObject, dvField, DIM, DIM, DT);

	fprintf_s(g_pFile, "++++++++++++++++++++++++++++++++++++\n");
	fprintf_s(g_pFile, "Stop  Simmulation\n");
}

