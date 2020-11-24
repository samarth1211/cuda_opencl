#include<Windows.h>
#include<stdio.h>

#include<gl\glew.h>
#include<gl\GL.h>


// CUDA standard includes
#include<cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include"my_helper_timer.h"
#include"Ocean_kernel.cuh"
#include"Camera.h"

#include"vmath.h" // vermillion math library
#include"resource.h"

#pragma comment (lib,"user32.lib")
#pragma comment (lib,"gdi32.lib")
#pragma comment(lib,"cudart.lib")
#pragma comment(lib,"cufft.lib")
#pragma comment(lib,"cufftw.lib")
#pragma comment (lib,"glew32.lib")
#pragma comment (lib,"opengl32.lib")


#define WIN_WIDTH	800
#define WIN_HEIGHT	600
#define ONE_BY_ROOT_TWO 0.707106781f
#define DELTA			0.0166666666666667f // camera


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
	SAM_ATTRIBUTE_NORMAL,
	SAM_ATTRIBUTE_TEXTURE0,
	SAM_ATTRIBUTE_HEIGHT,
	SAM_ATTRIBUTE_SLOPE,
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


// Projection
vmath::mat4 g_PersPectiveProjectionMatrix;

GLuint g_uniform_TextureSampler;

GLuint g_Texture_Smilie;

/* Ocean Variables Start*/

struct cudaGraphicsResource *heightMap_resource = NULL, *slope_resource = NULL;

// shader program
GLuint g_ShaderProgramObject_ocean = 0;

// Buffers
GLuint g_VertexArrayObject_ocean = 0;
GLuint g_VertexBufferObject_ocean_Position = 0;
GLuint g_VertexBufferObject_ocean_Normal = 0;
GLuint g_VertexBufferObject_ocean_TexCoord = 0;
GLuint g_VertexBufferObject_ocean_HeightMap = 0;
GLuint g_VertexBufferObject_ocean_slope = 0;
GLuint g_ElementArray_ocean = 0;

// uniforms
GLuint g_Uniform_Model_Matrix_ocean = 0;
GLuint g_Uniform_View_Matrix_ocean = 0;
GLuint g_Uniform_Projection_Matrix_ocean = 0;
GLuint g_Uniform_HeightScale_ocean = 0;
GLuint g_Uniform_Choppiness = 0;
GLuint g_Uniform_Size = 0;
GLuint g_Uniform_DeepColor = 0;
GLuint g_Uniform_ShalowColor = 0;
GLuint g_Uniform_SkyColor = 0;
GLuint g_Uniform_LightDir = 0;

//  constants
const unsigned int meshSize = 256;
const unsigned int spectrumW = meshSize + 4;
const unsigned int spectrumH = meshSize + 1;

const int frameCompare=4;

// FFT Data
cufftHandle fftPlan;
float2 *d_h0 = NULL; // height field at time 0
float2 *h_h0 = NULL; // height field at time 0
float2 *d_ht = NULL; // height field at time t
float2 *d_slope = NULL;


// pointers to device object
float *g_hptr = NULL;
float2 *g_sptr = NULL;

// simulation parameters
const float g = 9.81f;
const float A = 1e-7f;
const float patchSize = 100;
float windSpeed = 100.0f;
float windDir = (float)M_PI / 3.0f;
float dirDepend = 0.07f;

IStopWatchTimer *timer = 0;
float animTime = 0;
float prevTime = 0;
float animationRate = -0.001f;
/* Ocean Variables Stop */


// Camera Keys
Camera camera(vmath::vec3(0.0f, 0.0f, -1.5f));
GLfloat g_fLastX = WIN_WIDTH / 2;
GLfloat g_fLastY = WIN_HEIGHT / 2;
float zDirn = -10.0f;

GLfloat g_DeltaTime = 0.0f;
GLboolean g_bFirstMouse = true;
GLfloat g_fCurrrentWidth;
GLfloat g_fCurrrentHeight;

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
		case 0x41:// A is pressed
			camera.ProcessKeyBoard(E_LEFT, DELTA);
			break;
		case 0x44:// D is pressed
			camera.ProcessKeyBoard(E_RIGHT, DELTA);
			break;
		case 0x57:// W is pressed
			camera.ProcessKeyBoard(E_FORWARD, DELTA);
			break;
		case 0x53:// S is pressed
			camera.ProcessKeyBoard(E_BACKARD, DELTA);
			break;

			// Arraow Keys
		case VK_UP:
			zDirn = zDirn - DELTA;
			break;
		case VK_DOWN:
			zDirn = zDirn + DELTA;
			break;
		case VK_LEFT:
			break;
		case VK_RIGHT:
			break;
		default:
			break;
		}
		break;

	case WM_MOUSEMOVE: // g_fLastX  g_fLastY
	{
		GLfloat xPos = LOWORD(lParam);
		GLfloat yPos = HIWORD(lParam);

		if (g_bFirstMouse)
		{
			g_fLastX = xPos;
			g_fLastY = yPos;

			g_bFirstMouse = false;
		}

		GLfloat xOffset = xPos - g_fLastX;
		GLfloat yOffset = g_fLastY - yPos;

		g_fLastX = xPos;
		g_fLastY = yPos;

		/*g_fLastX = g_fCurrrentWidth / 2;
		g_fLastY = g_fCurrrentHeight / 2;*/

		camera.ProcessMouseMovements(xOffset, yOffset);
	}
	break;

	case WM_MOUSEWHEEL:
	{
		GLfloat xPos = LOWORD(lParam);
		GLfloat yPos = HIWORD(lParam);
		camera.ProcessMouseScrool(xPos, yPos);
	}
	break;

	case WM_SIZE:
		g_fCurrrentWidth = LOWORD(lParam);
		g_fCurrrentHeight = HIWORD(lParam);
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
	void Generate_h0(float2 *h_h0);
	int LoadGLTextures(GLuint *texture, TCHAR imageResourceId[]);

	cudaError err;
	int cufftErrs;

	int iPixelIndex = 0;
	PIXELFORMATDESCRIPTOR pfd;

	// Shader Programs
	GLuint iVertexShaderObject = 0;
	GLuint iFragmentShaderObject = 0;

	GLuint iVertexShaderObject_ocean = 0;
	GLuint iFragmentShaderObject_ocean = 0;

	GLenum glew_error = NULL; // GLEW Error codes

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
	glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		wglDeleteContext(g_hrc);
		g_hrc = NULL;
		ReleaseDC(g_hwnd, g_hdc);
		g_hdc = NULL;
		return INIT_FAIL_GLEW_INIT;
	}

	// GL information Start
	fprintf_s(g_pFile, "SHADER_INFO : Vendor is : %s\n", glGetString(GL_VENDOR));
	fprintf_s(g_pFile, "SHADER_INFO : Renderer is : %s\n", glGetString(GL_RENDER));
	fprintf_s(g_pFile, "SHADER_INFO : OpenGL Version is : %s\n", glGetString(GL_VERSION));
	fprintf_s(g_pFile, "SHADER_INFO : GLSL Version is : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	//fprintf_s(g_pFile, "SHADER_INFO : Extention is : %s \n", glGetString(GL_EXTENSIONS));
	// GL information End

	// Init CUDA
	err = cudaSetDevice(0);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaSetDevice %s\n",cudaGetErrorString(err));
	}
	else
	{
		fprintf_s(g_pFile, "CUDA Info : CUDA Initalized \n");
	}

	/// Sam : all Shader Code Start
	/** Ocean start **/
	/*Vertex Shader Start*/
	iVertexShaderObject_ocean = glCreateShader(GL_VERTEX_SHADER);
	const GLchar *vertexShaderSourceCode_ocean = "#version 450 core"	\
		"\n" \
		"layout (location = 0)in vec4  vPosition;" \
		"layout (location = 2)in vec3  vNormal;" \
		"layout (location = 3)in vec2  vTexCoord;" \
		"layout (location = 4)in float vHeight;" \
		"layout (location = 5)in vec2  vSlope;" \
		"layout (location = 0)out vec3 out_EyeSpacePos;" \
		"layout (location = 1)out vec3 out_WorldSpaceNormal;" \
		"layout (location = 2)out vec3 out_EyeSpaceNormal;" \
		"layout (location = 3)out vec3 out_color;" \
		"layout (location = 4)out vec2 out_texCoord;" \
		"uniform mat4  u_model_matrix,u_view_matrix,u_projection_matrix;" \
		"uniform float u_chopiness;"	\
		"uniform float u_HeightScale;"	\
		"uniform vec2  u_Size;"	\
		"void main(void)" \
		"{" \
		"	mat4 modelViewMatrix = u_view_matrix * u_model_matrix;"	\
		"	mat3 normalMatrix=inverse(transpose(mat3(modelViewMatrix)));"	\
		"	vec4 pos = vec4(vPosition.x,vHeight*u_HeightScale,vPosition.z,1.0);"	\
		"	vec3 normal = normalize(cross(vec3(0.0,vSlope.y*u_HeightScale,2.0/u_Size.x),vec3(2.0/u_Size.y,vSlope.x*u_HeightScale,0.0)));"	\
		"	out_WorldSpaceNormal = normal;"	\
		"	out_EyeSpacePos = (modelViewMatrix * pos).xyz;"	\
		"	out_EyeSpaceNormal = (normalMatrix*normal).xyz;"	\
		"	gl_Position =  u_projection_matrix * modelViewMatrix * pos;" \
		"}";

	glShaderSource(iVertexShaderObject_ocean, 1, (const GLchar**)&vertexShaderSourceCode_ocean, NULL);

	// Compile it
	glCompileShader(iVertexShaderObject_ocean);
	GLint iInfoLogLength = 0;
	GLint iShaderCompileStatus = 0;
	GLchar *szInfoLog = NULL;
	glGetShaderiv(iVertexShaderObject_ocean, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(iVertexShaderObject_ocean, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(iVertexShaderObject_ocean, GL_INFO_LOG_LENGTH, &written, szInfoLog);
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
	iFragmentShaderObject_ocean = glCreateShader(GL_FRAGMENT_SHADER);
	const GLchar *fragmentShaderSourceCode_ocean = "#version 450 core"	\
		"\n"	\
		"layout (location = 0)in vec3 out_EyeSpacePos;" \
		"layout (location = 1)in vec3 out_WorldSpaceNormal;" \
		"layout (location = 2)in vec3 out_EyeSpaceNormal;" \
		"layout (location = 3)in vec3 out_color;" \
		"layout (location = 4)in vec2 out_texCoord;" \
		"layout (location = 0)out vec4 FragColor;"	\
		"uniform vec4 u_DeepColor;"	\
		"uniform vec4 u_ShalowColor;"	\
		"uniform vec4 u_SkyColor;"	\
		"uniform vec3 u_LightDir;"	\
		"uniform sampler2D u_texture0_sampler;"	\
		"void main(void)"	\
		"{"	\
		"	vec3 eyeVector = normalize(out_EyeSpacePos);"	\
		"	vec3 eyeSpaceNormalVector = normalize(out_EyeSpaceNormal);"	\
		"	vec3 worldSpaceNormalVector=normalize(out_WorldSpaceNormal);"	\
		"	float facing=max(0.0,dot(eyeSpaceNormalVector,-eyeVector));"	\
		"	float fresnel = pow(1.0 - facing,5.0);"	\
		"	float diffuse = max(0.0,dot(worldSpaceNormalVector,u_LightDir));"	\
		"	vec4 waterColor = mix(u_ShalowColor,u_DeepColor,facing);"	\
		"	FragColor = waterColor * diffuse + u_SkyColor * fresnel;"	\
		"}";

	//"	FragColor = texture(u_texture0_sampler,out_texCoord);"	
	glShaderSource(iFragmentShaderObject_ocean, 1, (const GLchar**)&fragmentShaderSourceCode_ocean, NULL);
	glCompileShader(iFragmentShaderObject_ocean);
	iInfoLogLength = 0;
	iShaderCompileStatus = 0;
	szInfoLog = NULL;
	glGetShaderiv(iFragmentShaderObject_ocean, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(iFragmentShaderObject_ocean, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(iFragmentShaderObject_ocean, GL_INFO_LOG_LENGTH, &written, szInfoLog);
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
	g_ShaderProgramObject_ocean = glCreateProgram();
	glAttachShader(g_ShaderProgramObject_ocean, iVertexShaderObject_ocean);
	glAttachShader(g_ShaderProgramObject_ocean, iFragmentShaderObject_ocean);
	glBindAttribLocation(g_ShaderProgramObject_ocean, SAM_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(g_ShaderProgramObject_ocean, SAM_ATTRIBUTE_NORMAL, "vNormal");
	glBindAttribLocation(g_ShaderProgramObject_ocean, SAM_ATTRIBUTE_TEXTURE0, "vTexCoord");
	glBindAttribLocation(g_ShaderProgramObject_ocean, SAM_ATTRIBUTE_HEIGHT, "vHeight");
	glBindAttribLocation(g_ShaderProgramObject_ocean, SAM_ATTRIBUTE_SLOPE, "vSlope");
	glLinkProgram(g_ShaderProgramObject_ocean);

	GLint iShaderLinkStatus = 0;
	iInfoLogLength = 0;
	glGetProgramiv(g_ShaderProgramObject_ocean, GL_LINK_STATUS, &iShaderLinkStatus);
	if (iShaderLinkStatus == GL_FALSE)
	{
		glGetProgramiv(g_ShaderProgramObject_ocean, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(g_ShaderProgramObject_ocean, GL_INFO_LOG_LENGTH, &written, szInfoLog);
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
	g_Uniform_Model_Matrix_ocean = glGetUniformLocation(g_ShaderProgramObject_ocean, "u_model_matrix");
	g_Uniform_Projection_Matrix_ocean = glGetUniformLocation(g_ShaderProgramObject_ocean, "u_projection_matrix");
	g_Uniform_View_Matrix_ocean = glGetUniformLocation(g_ShaderProgramObject_ocean, "u_view_matrix");
	g_Uniform_Choppiness = glGetUniformLocation(g_ShaderProgramObject_ocean, "u_chopiness");
	g_Uniform_HeightScale_ocean = glGetUniformLocation(g_ShaderProgramObject_ocean, "u_HeightScale");
	g_Uniform_Size = glGetUniformLocation(g_ShaderProgramObject_ocean, "u_Size");

	g_Uniform_DeepColor = glGetUniformLocation(g_ShaderProgramObject_ocean, "u_DeepColor");
	g_Uniform_ShalowColor = glGetUniformLocation(g_ShaderProgramObject_ocean, "u_ShalowColor");
	g_Uniform_SkyColor = glGetUniformLocation(g_ShaderProgramObject_ocean, "u_SkyColor");
	g_Uniform_LightDir = glGetUniformLocation(g_ShaderProgramObject_ocean, "u_LightDir");

	g_uniform_TextureSampler = glGetUniformLocation(g_ShaderProgramObject_ocean, "u_texture0_sampler");
	/*Setup Uniforms End*/

	/*	Setup Buffers Start	*/
	cufftErrs = cufftPlan2d(&fftPlan,meshSize,meshSize,CUFFT_C2C);
	if (cufftErrs != (int)CUFFT_SUCCESS)
	{
		fprintf_s(g_pFile, "int Initalize, cufftPlan2d failed %d \n", cufftErrs);
	}

	int spectrumSize = spectrumW*spectrumH*sizeof(float2);
	err = cudaMalloc((void**)&d_h0,spectrumSize);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMalloc()\n");
	}

	h_h0 = (float2*)calloc(spectrumW*spectrumH, sizeof(float2));
	if (h_h0 == NULL)
	{
		fprintf_s(g_pFile, "Memory Error : calloc() failed\n");
	}
	
	Generate_h0(h_h0);
	err = cudaMemcpy(d_h0,h_h0,spectrumSize,cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMemcpy()\n");
	}

	int outputSize = meshSize*meshSize*sizeof(float2);
	err = cudaMalloc((void**)&d_ht, outputSize);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMalloc(d_ht)\n");
	}

	err = cudaMalloc((void**)&d_slope,outputSize);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMalloc(d_slope)\n");
	}

	// OpenGL buffers
	glGenVertexArrays(1,&g_VertexArrayObject_ocean);
	glBindVertexArray(g_VertexArrayObject_ocean);

	// Position Buffer
	glGenBuffers(1,&g_VertexBufferObject_ocean_Position);
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_ocean_Position);
	glBufferData(GL_ARRAY_BUFFER, meshSize*meshSize*4*sizeof(float),NULL,GL_DYNAMIC_DRAW);
	float *pos = (float*)glMapNamedBuffer(g_VertexBufferObject_ocean_Position,GL_WRITE_ONLY);
	if (pos != NULL)
	{
		for (int y = 0; y < meshSize; y++) // height
		{
			for (int x = 0; x < meshSize; x++) // width
			{
				float u = x / (float)(meshSize - 1);
				float v = y / (float)(meshSize - 1);

				*pos++ = u*2.0f - 1.0f;
				*pos++ = 0.0f;
				*pos++ = v*2.0f - 1.0f;
				*pos++ = 1.0f;
			}
		}

		glUnmapNamedBuffer(g_VertexBufferObject_ocean_Position);
	}

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Normal Buffer
	glGenBuffers(1, &g_VertexBufferObject_ocean_Normal);
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_ocean_Normal);
	glBufferData(GL_ARRAY_BUFFER, meshSize*meshSize * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	float *norm = (float*)glMapNamedBuffer(g_VertexBufferObject_ocean_Normal, GL_WRITE_ONLY);
	if (norm != NULL)
	{
		for (int y = 0; y < meshSize; y++) // height
		{
			for (int x = 0; x < meshSize; x++) // width
			{
				*norm++ = 0.0;
				*norm++ = 1.0;
				*norm++ = 0.0;
			}
		}

		glUnmapNamedBuffer(g_VertexBufferObject_ocean_Normal);
	}

	glVertexAttribPointer(SAM_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	// Texture Buffer
	glGenBuffers(1, &g_VertexBufferObject_ocean_TexCoord);
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_ocean_TexCoord);
	glBufferData(GL_ARRAY_BUFFER, meshSize*meshSize * 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	float *texCoord = (float*)glMapNamedBuffer(g_VertexBufferObject_ocean_TexCoord, GL_WRITE_ONLY);
	if (pos != NULL)
	{
		for (int y = 0; y < meshSize; y++) // height
		{
			for (int x = 0; x < meshSize; x++) // width
			{
				
				*texCoord++ = (float)x/ ( (float)(meshSize*meshSize)-1.0f);
				*texCoord++ = (float)y / ((float)(meshSize*meshSize) - 1.0f);
			}
		}

		glUnmapNamedBuffer(g_VertexBufferObject_ocean_TexCoord);
	}

	glVertexAttribPointer(SAM_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// heightMap => bind it with cuda
	glGenBuffers(1, &g_VertexBufferObject_ocean_HeightMap);
	glBindBuffer(GL_ARRAY_BUFFER,g_VertexBufferObject_ocean_HeightMap);
	glBufferData(GL_ARRAY_BUFFER, meshSize*meshSize * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(SAM_ATTRIBUTE_HEIGHT, 1, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_HEIGHT);
	glBindBuffer(GL_ARRAY_BUFFER,0);

	// Slope => bind it with cuda
	glGenBuffers(1, &g_VertexBufferObject_ocean_slope);
	glBindBuffer(GL_ARRAY_BUFFER,g_VertexBufferObject_ocean_slope);
	glBufferData(GL_ARRAY_BUFFER, meshSize*meshSize * sizeof(float2), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(SAM_ATTRIBUTE_SLOPE, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_SLOPE);
	glBindBuffer(GL_ARRAY_BUFFER,0);

	// Elements
	int elementSize = ((meshSize * 2) + 2) * (meshSize - 1)* sizeof(GLuint);
	glGenBuffers(1,&g_ElementArray_ocean);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ElementArray_ocean);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, elementSize, NULL, GL_STATIC_DRAW);
	GLuint *indices = (GLuint*)glMapNamedBuffer(g_ElementArray_ocean, GL_WRITE_ONLY);
	if (indices != NULL)
	{
		for (int y = 0; y < meshSize - 1; y++)// height
		{
			for (int x = 0; x < meshSize; x++)// width
			{
				*indices++ = y*meshSize + x;
				*indices++ = (y + 1)*meshSize + x;
			}

			// start new strip with degenerated triangles
			*indices++ = (y + 1)*meshSize + (meshSize - 1);
			*indices++ = (y + 1)*meshSize;
		}

		glUnmapNamedBuffer(g_ElementArray_ocean);
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	// Bind Opengl Buffers with CUDA
	// Height Map
	err = cudaGraphicsGLRegisterBuffer(&heightMap_resource, g_VertexBufferObject_ocean_HeightMap,cudaGraphicsMapFlagsWriteDiscard);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaGraphicsGLRegisterBuffer(heightMap_resource)\n");
	}
	// Slope
	err = cudaGraphicsGLRegisterBuffer(&slope_resource, g_VertexBufferObject_ocean_slope, cudaGraphicsMapFlagsWriteDiscard);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaGraphicsGLRegisterBuffer(slope_resource)\n");
	}
	/*	Setup Buffers Stop 	*/
	/** Ocean stop  **/
	/// Sam : all Shader Code End

	LoadGLTextures(&g_Texture_Smilie, MAKEINTRESOURCE(IDB_SMILIE));

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	//glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	//glEnable(GL_CULL_FACE);

	glClearColor(0.25f, 0.25f, 0.25f, 1.0f);

	g_PersPectiveProjectionMatrix = vmath::mat4::identity();

	Resize(WIN_WIDTH, WIN_HEIGHT);

	return INIT_ALL_OK;
}


int LoadGLTextures(GLuint *texture, TCHAR imageResourceId[])
{
	fprintf_s(g_pFile, "Inside LoadGLTextures....!!!\n");
	HBITMAP hBitmap = NULL;
	BITMAP bmp;
	int iStatus = FALSE;

	glGenTextures(1, texture);
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL), imageResourceId, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);

	if (hBitmap)
	{
		fprintf_s(g_pFile, "EXEC : Inside LoadGLTextures - Image Obtained\n");
		iStatus = TRUE;
		GetObject(hBitmap, sizeof(bmp), &bmp);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glBindTexture(GL_TEXTURE_2D, *texture);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp.bmWidth, bmp.bmHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, bmp.bmBits);

		glGenerateMipmap(GL_TEXTURE_2D);

		DeleteObject(hBitmap);

	}
	fprintf_s(g_pFile, "Leaving LoadGLTextures....!!!\n");
	return iStatus;
}

void Generate_h0(float2 *h_h0)
{
	float phillips(float kx,float ky,float Vdir,float v, float A,float dir_depend);
	float gauss();

	for (unsigned int y = 0; y <= meshSize; y++)
	{
		for (unsigned int x = 0; x <= meshSize; x++)
		{
			float kx = (-(int)meshSize / 2.0f + x)*(2.0f*(float)M_PI / patchSize);
			float ky = (-(int)meshSize / 2.0f + y)*(2.0f*(float)M_PI / patchSize);

			float p = sqrtf(phillips(kx,ky,windDir,windSpeed,A,dirDepend));

			if (kx == 0.0f && ky == 0.0f)
			{
				p = 0.0f;
			}

			float Er = gauss();
			float Ei = gauss();

			float h0_re = Er * p * ONE_BY_ROOT_TWO;
			float h0_im = Ei * p * ONE_BY_ROOT_TWO;

			int i = y * spectrumW + x;
			h_h0[i].x = h0_re;
			h_h0[i].y = h0_im;

		}
	}
}

/*
Generate Gaussian Random number with mean 0 and Standard deviation 1.
*/
float gauss()
{
	float u1 = rand() / (float)RAND_MAX;
	float u2 = rand() / (float)RAND_MAX;
	if (u1 < 1e-6f)
	{
		u1 = 1e-6f;
	}
	return sqrtf(-2 * logf(u1)) * cosf(2* (float)M_PI*u2);
}


/*  Philips Spectrum 
(kx,ky) - normalized wave vector
vDir - Wind Andle in radians
v - Wind Speed
A - constant
*/
float phillips(float kx, float ky, float Vdir, float v, float A, float dir_depend)
{
	float k_squared = kx * kx + ky * ky;

	if (k_squared == 0.0f)
	{
		return 0.0;
	}
	
	// largest possible wave from constant wind of velocity v
	float LargestWave = v * v / g;

	float k_x = kx / sqrtf(k_squared);
	float k_y = ky / sqrtf(k_squared);
	float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

	float phillips_value = A * expf(-1.0f/(k_squared * LargestWave*LargestWave)) / (k_squared * k_squared)*w_dot_k*w_dot_k;

	// filter out waves moving opposite to wind
	if (w_dot_k < 0.0f)
	{
		phillips_value *= dir_depend;
	}

	// damp out waves with very small length w<<l
	//float w = LargestWave / 10000;
	//phillips *= expf(-k_squared * w *w);

	return phillips_value;
}

void Update(void)
{
	void RunCUDAKernels();

	animTime = animTime + 0.05f;

	RunCUDAKernels();
}

void RunCUDAKernels()
{
	cudaError err;
	int cufftErrs;

	size_t num_bytes;

	// generate wave spectrum in Frequency domain
	cuda_GenerateSpectrumKernel(d_h0,d_ht,spectrumW,meshSize,meshSize,animTime,patchSize);

	// calculate inverse FFT to convert to spatial domain
	cufftErrs = cufftExecC2C(fftPlan,d_ht,d_ht,CUFFT_INVERSE);
	if (cufftErrs != (int)CUFFT_SUCCESS)
	{
		fprintf_s(g_pFile, "int RunCUDAKernels, cufftExecC2C failed %d \n", cufftErrs);
	}

	// Update height map values
	err = cudaGraphicsMapResources(1,&heightMap_resource,0);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int RunCUDAKernels, cudaGraphicsMapResources(heightMap_resource) failed %s \n", cudaGetErrorString(err));
	}
	err = cudaGraphicsResourceGetMappedPointer((void**)&g_hptr,&num_bytes,heightMap_resource);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int RunCUDAKernels, cudaGraphicsResourceGetMappedPointer(heightMap_resource) failed %s \n", cudaGetErrorString(err));
	}

	cuda_UpdateHeightMapKernel(g_hptr,d_ht,meshSize,meshSize,false);

	// Calculate slope for shading
	err = cudaGraphicsMapResources(1, &slope_resource, 0);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int RunCUDAKernels, cudaGraphicsMapResources(slope_resource) failed %s \n", cudaGetErrorString(err));
	}

	err = cudaGraphicsResourceGetMappedPointer((void**)&g_sptr, &num_bytes, slope_resource);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int RunCUDAKernels, cudaGraphicsResourceGetMappedPointer(slope_resource) failed %s \n", cudaGetErrorString(err));
	}

	cuda_CalculateSlopKernel(g_hptr,g_sptr,meshSize,meshSize);

	err = cudaGraphicsUnmapResources(1, &heightMap_resource);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int RunCUDAKernels, cudaGraphicsUnmapResources(heightMap_resource) failed %s \n", cudaGetErrorString(err));
	}

	err = cudaGraphicsUnmapResources(1, &slope_resource);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "int RunCUDAKernels, cudaGraphicsUnmapResources(slope_resource) failed %s \n", cudaGetErrorString(err));
	}
}

void Render(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	vmath::mat4 modelMatrix = vmath::mat4::identity();
	vmath::mat4 viewMatrix = vmath::mat4::identity();
	vmath::mat4 m4PersPectiveProjectionMatrix = vmath::perspective(camera.GetZoom(), (float)g_fCurrrentWidth / (float)g_fCurrrentHeight, 0.1f, 100.0f);

	glUseProgram(g_ShaderProgramObject_ocean);
	modelMatrix = vmath::translate(0.0f, 0.0f, -3.0f);

	glUniformMatrix4fv(g_Uniform_Model_Matrix_ocean, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(g_Uniform_View_Matrix_ocean, 1, GL_FALSE, camera.GetViewMatrix());
	glUniformMatrix4fv(g_Uniform_Projection_Matrix_ocean, 1, GL_FALSE, m4PersPectiveProjectionMatrix);

	glUniform1f(g_Uniform_HeightScale_ocean, 0.5f);
	glUniform1f(g_Uniform_Choppiness,1.0f);
	glUniform2f(g_Uniform_Size, (float)meshSize,(float)meshSize);

	glUniform4f(g_Uniform_DeepColor, 0.0f, 0.1f, 0.4f, 1.0f);
	glUniform4f(g_Uniform_ShalowColor, 0.1f, 0.3f, 0.3f, 1.0f);
	glUniform4f(g_Uniform_SkyColor, 1.0f, 1.0f, 1.0f, 1.0f);

	glUniform3f(g_Uniform_LightDir, 0.0f, 1.0f, 0.0f);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g_Texture_Smilie);
	glUniform1i(g_uniform_TextureSampler, 0);

	glBindVertexArray(g_VertexArrayObject_ocean);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ElementArray_ocean);
	//glDrawArrays(GL_POINTS, 0, meshSize * meshSize);
	glDrawElements(GL_TRIANGLE_STRIP, ((meshSize * 2) + 2)*(meshSize - 1), GL_UNSIGNED_INT, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
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

	if (fftPlan)
	{
		cufftDestroy(fftPlan);
		fftPlan = NULL;
	}

	if (d_h0)
	{
		cudaFree(d_h0);
		d_h0 = NULL;
	}

	if (heightMap_resource)
	{
		cudaGraphicsUnmapResources(1, &heightMap_resource);
		cudaGraphicsUnregisterResource(heightMap_resource);
	}

	if (slope_resource)
	{
		cudaGraphicsUnmapResources(1, &slope_resource);
		cudaGraphicsUnregisterResource(slope_resource);
	}

	// Fail safe reset for this process
	cudaDeviceReset();

	/* Ocean Unintialize Start*/
	if (g_VertexBufferObject_ocean_Position)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_ocean_Position);
		g_VertexBufferObject_ocean_Position = NULL;
	}

	if (g_VertexBufferObject_ocean_HeightMap)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_ocean_HeightMap);
		g_VertexBufferObject_ocean_HeightMap = NULL;
	}

	if (g_VertexBufferObject_ocean_slope)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_ocean_slope);
		g_VertexBufferObject_ocean_slope = NULL;
	}

	if (g_VertexBufferObject_ocean_Normal)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_ocean_Normal);
		g_VertexBufferObject_ocean_Normal = NULL;
	}

	if (g_VertexBufferObject_ocean_TexCoord)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_ocean_TexCoord);
		g_VertexBufferObject_ocean_TexCoord = NULL;
	}

	if (g_VertexArrayObject_ocean)
	{
		glDeleteVertexArrays(1, &g_VertexArrayObject_ocean);
		g_VertexArrayObject_ocean = NULL;
	}

	if (g_ShaderProgramObject_ocean)
	{
		GLsizei iShaderCount;
		GLsizei iShaderNumber;

		glUseProgram(g_ShaderProgramObject_ocean);
		glGetProgramiv(g_ShaderProgramObject_ocean, GL_ATTACHED_SHADERS, &iShaderCount);
		GLuint *pShaders = (GLuint*)calloc(iShaderCount, sizeof(GLuint));

		if (pShaders)
		{
			glGetAttachedShaders(g_ShaderProgramObject_ocean, iShaderCount, &iShaderCount, pShaders);
			for (iShaderNumber = 0; iShaderNumber < iShaderCount; iShaderNumber++)
			{
				glDetachShader(g_ShaderProgramObject_ocean, pShaders[iShaderNumber]);
				glDeleteShader(pShaders[iShaderNumber]);
				pShaders[iShaderNumber] = 0;
			}
			free(pShaders);
			pShaders = NULL;
		}
		glUseProgram(0);

		glDeleteProgram(g_ShaderProgramObject_ocean);
		g_ShaderProgramObject_ocean = NULL;
	}
	/*  Ocean Unintialize Stop */

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

