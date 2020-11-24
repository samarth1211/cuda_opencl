#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>

#include <gl\glew.h>
#include <gl\GL.h>


#include "vmath.h"


// CUDA standard includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "Particles_kernel.cuh"
#include "ParticleSystem.cuh"

#define WIN_WIDTH	800
#define WIN_HEIGHT	600

#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"cudart.lib")



enum InitErrorCodes
{
	INIT_CUDA_MALLOC_FAILED = -12,
	INIT_CUDA_REGISTER_FAILED = -11,
	INIT_GEOMETRY_SHADER_COMPILATION_FAILED = -10,
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
	SAM_ATTRIBUTE_NORMAL,
	SAM_ATTRIBUTE_TEXTURE0,
	SAM_ATTRIBUTE_VELOCITY,
	SAM_ATTRIBUTE_LIFETIME,
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


typedef struct smVertexBufferzObj
{
	size_t iSizeOfBufer;
	GLuint ogl_vbo[2];
	float4 *fp_CUDADevicePointer[2];
	float4 *fp_HostPointer;
	struct cudaGraphicsResource *vbo_Reg_to_CUDA[2];
	unsigned int m_currentRead, m_currentWrite;

	// m_currentRead => used to get current device pointer
} smVBOData;



// Projection
vmath::mat4 g_PersPectiveProjectionMatrix;

// Shader Programs
GLuint g_ShaderProgramObject_ParticleProg = 0;


//g_ShaderProgramObject_ParticleProg
GLuint g_Uniform_Model_Matrix_ParticleProg = 0;
GLuint g_Uniform_View_Matrix_ParticleProg = 0;
GLuint g_Uniform_Projection_Matrix_ParticleProg = 0;
GLuint g_Uniform_PointRadius_ParticleProg = 0;
GLuint g_Uniform_TimeStep_ParticleProg = 0;
GLuint g_Uniform_AlphaValue_ParticleProg = 0;


/**
	Position 
	Velocity => 'w' is used as lifetime
	Color
	Elements
**/
GLuint g_VertexArrayObject_Particle;
smVBOData g_VBO_Particle_Position;
smVBOData g_VBO_Particle_Velocity;
GLuint g_VertexBufferObject_Particle_Element =0;
struct cudaGraphicsResource *elements_reg_to_CUDA = NULL;
GLuint *gp_fDevicePointer_Elements = NULL;
GLuint *gp_fHostPointer_Elements = NULL;

//float *gp_iSortIndex = NULL; akin to sortkeys
float *gp_fDevicePointer_SortIndex = NULL;
float *gp_fHostPointer_SortIndex = NULL;

bool g_bDoDepthSort;//=> this is always true for working in cuda is expected

// Smoke rendewrer Clac vectors
float3 g_fSortVector;

/*Interop Params Start*/

/*Interop Params Stop */


// All Simulation Parameters

SimParams g_SimParams;

UINT g_iNumParticles = 1 << 16;
float g_EmitterVel = 0.0f;
UINT g_iEmitterRate = 1000;
float g_EmitterRadius = 0.25;
float g_EmitterSpread = 0.0f;
UINT g_EmitterIndex = 0;

float g_fTimeStep = 0.5f;
float g_fCurrentTime = 0.0;
float g_fSpriteSize = 0.05f;
float g_fAlpha = 0.01f;
float g_fShadowAlpha = 0.01f;
float g_fParticleLifetime = (float)g_iNumParticles/(float)g_iEmitterRate;
GLfloat g_fLightColor[3] = { 1.0f, 1.0f, 0.8f };
GLfloat g_fColorAttenuation[3] = { 0.5f, 0.75f, 1.0f };
float g_fBlurRadius = 2.0f;
float g_fparticleRadius = 0.005f;

vmath::vec3 g_v3CursorPos; // for the movement of cloud

int g_iNumSlices = 64;
int g_iNumDisplayedSlices = g_iNumSlices;

float g_floatTimeStep = 0.0f;

bool g_bEmitterOn = true;


/**** SmokeDemo variables Stop  ****/


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
	case INIT_GEOMETRY_SHADER_COMPILATION_FAILED:
		fprintf_s(g_pFile, "Failed to Compile geometry Shader \n");
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
	void MakeSphere(GLfloat fRadius, GLint iSlices, GLint iStacks);
	GLuint GetIndexCountSphere();
	GLuint GetVertexCountSphere();

	// Particle Init
	void InitCubeRandom(float4* pPos,float4* pVeloc,vmath::vec3 origin,vmath::vec3 size,vmath::vec3 vel,float lifetime);

	int iPixelIndex = 0;
	PIXELFORMATDESCRIPTOR pfd;

	// Shader Objects
	//g_ShaderProgramObject_ParticleProg
	GLuint iVertexShaderObject_ParticleProg = 0;
	GLuint iGeometryShaderObject_ParticleProg = 0;
	GLuint iFragmentShaderObject_ParticleProg = 0;

	GLenum glewErr = NULL; // GLEW Error codes

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

	// Init Cuda
	cudaError cuErr; 
	cuErr = cudaSetDevice(0);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile,"setting default cuda device failed");
	}

	/* Smoke Init Variables Start */

	ZeroMemory((void**)&g_VBO_Particle_Position, sizeof(g_VBO_Particle_Position));
	ZeroMemory((void**)&g_VBO_Particle_Velocity, sizeof(g_VBO_Particle_Velocity));

	g_VBO_Particle_Position.iSizeOfBufer = g_iNumParticles*sizeof(float4);
	g_VBO_Particle_Velocity.iSizeOfBufer = g_iNumParticles*sizeof(float4);

	g_VBO_Particle_Position.fp_HostPointer = NULL;
	g_VBO_Particle_Velocity.fp_HostPointer = NULL;

	g_VBO_Particle_Position.fp_HostPointer = (float4*)calloc(g_iNumParticles, sizeof(float4));
	if (g_VBO_Particle_Position.fp_HostPointer == NULL)
	{
		fprintf_s(g_pFile, "Memory Allocation Failed...!!\n");
	}

	g_VBO_Particle_Velocity.fp_HostPointer = (float4*)calloc(g_iNumParticles, sizeof(float4));
	if (g_VBO_Particle_Velocity.fp_HostPointer == NULL)
	{
		fprintf_s(g_pFile, "Memory Allocation Failed...!!\n");
	}

	gp_fHostPointer_Elements = (GLuint*)calloc(g_iNumParticles, sizeof(GLuint));
	if (gp_fHostPointer_Elements == NULL)
	{
		fprintf_s(g_pFile, "Memory Allocation Failed...!!'gp_fHostPointer_Elements'\n");
	}

	gp_fHostPointer_SortIndex = (float*)calloc(g_iNumParticles,sizeof(float));
	if (gp_fHostPointer_SortIndex == NULL)
	{
		fprintf_s(g_pFile, "Memory Allocation Failed...!!'gp_fHostPointer_SortIndex'\n");
	}
	
	/*cuErr = cudaMalloc((void**)&gp_fDevicePointer_Elements,g_iNumParticles*sizeof(GLuint));
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMalloc(gp_fDevicePointer_Elements) : %s\n", cudaGetErrorString(cuErr));
		return	INIT_CUDA_MALLOC_FAILED;
	}*/

	gp_fHostPointer_SortIndex = (float*)calloc(g_iNumParticles, sizeof(float));
	if (gp_fHostPointer_SortIndex == NULL)
	{
		fprintf_s(g_pFile, "Memory Allocation Failed...!!'gp_fHostPointer_SortIndex'\n");
	}

	cuErr = cudaMalloc((void**)&gp_fDevicePointer_SortIndex, g_iNumParticles*sizeof(float));
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMalloc(gp_fDevicePointer_SortIndex) : %s\n", cudaGetErrorString(cuErr));
		return	INIT_CUDA_MALLOC_FAILED;
	}

	/* Smoke Init Variables Stop  */

	// GL information Start
	fprintf_s(g_pFile, "SHADER_INFO : Vendor is : %s\n", glGetString(GL_VENDOR));
	fprintf_s(g_pFile, "SHADER_INFO : Renderer is : %s\n", glGetString(GL_RENDER));
	fprintf_s(g_pFile, "SHADER_INFO : OpenGL Version is : %s\n", glGetString(GL_VERSION));
	fprintf_s(g_pFile, "SHADER_INFO : GLSL Version is : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	//fprintf_s(g_pFile, "SHADER_INFO : Extention is : %s \n", glGetString(GL_EXTENSIONS));
	// GL information End

	/// Sam : all Shader Code Start
	
	/**		Psrticle Prog Start 	**/
	/*Vertex Shader Start*/ //mblurVS
	// Check if need to remove projection matrix
	/*
	Conversion notes:

	Sample					==>			Our code

	gl_MultiTexCoord0.xyz			vVelocity
	gl_MultiTexCoord0.w				vVelocity.w/vLifeTime
	gl_TexCoord[0]					out_PreviousPos // in eye Space
	gl_TexCoord[1].x				out_Phase
	gl_FrontColor					out_Color
	*/

	iVertexShaderObject_ParticleProg = glCreateShader(GL_VERTEX_SHADER);
	const GLchar *vertexShaderParticleProgSourceCode = "#version 450 core"	\
		"\n" \
		"layout (location = 0)in vec4 vPosition;\n" \
		"layout (location = 1)in vec4 vColor;\n" \
		"layout (location = 4)in vec4 vVelocity;\n" \
		"layout (location = 5)in float vLifeTime;\n" \
		"out vec4 out_Color;\n" \
		"out vec4 out_PreviousPos;\n" \
		"out float out_Phase;\n" \
		"uniform float timeStep;\n"	\
		"uniform mat4 u_model_matrix,u_view_matrix,u_projection_matrix;" \
		"void main(void)" \
		"{\n" \
		"	vec3 pos = vPosition.xyz;\n"	\
		"	vec3 vel = vVelocity.xyz ;\n"	\
		"	vec3 pos2 = (pos - vel*timeStep);\n"	\
		"	gl_Position =  u_view_matrix * u_model_matrix * vec4(pos,1.0);\n" \
		"	out_PreviousPos = u_view_matrix * u_model_matrix * vec4(pos2,1.0);\n"	\
		"	float lifeTime = vVelocity.w;\n"	\
		"	float age = vPosition.w;"	\
		"	float phase = (lifeTime > 0.0) ? (age/lifeTime):1.0;\n"	\
		"	out_Phase  = phase;\n"	\
		"	float fade = 1.0 - phase;\n"	\
		"	out_Color = vec4(vColor.xyz,vColor.w*fade);\n"	\
		"	out_Color = vec4(vColor.xyz,1.0);\n"	\
		"}";

	glShaderSource(iVertexShaderObject_ParticleProg, 1, (const GLchar**)&vertexShaderParticleProgSourceCode, NULL);

	// Compile it
	glCompileShader(iVertexShaderObject_ParticleProg);
	GLint iInfoLogLength = 0;
	GLint iShaderCompileStatus = 0;
	char *szInfoLog = NULL;
	glGetShaderiv(iVertexShaderObject_ParticleProg, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(iVertexShaderObject_ParticleProg, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(iVertexShaderObject_ParticleProg, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf_s(g_pFile, "ERROR : Vertex Shader Display Particle Prog Compilation Log : %s \n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				return INIT_VERTEX_SHADER_COMPILATION_FAILED;
				//DestroyWindow(g_hwnd);
				//exit(EXIT_FAILURE);
			}
		}
	}
	/*Vertex Shader End*/
	/*Geometry Shader Start*/
	// motion blur geometry shader
	// outputs streached quad between prevoius and current positions
	/*
	Updations in geometry shader
	Conversion notes:

	Sample					==>			Our code
	gl_TexCoord[0]						Phase(with Transformation)
	gl_TexCoord[1]						PreviousPos
	
	*/
	iGeometryShaderObject_ParticleProg = glCreateShader(GL_GEOMETRY_SHADER);
	const GLchar *geometryShaderParticleProgSourceCode =
		"#version 450 core"	\
		"\n" \
		"layout(points) in;\n"	\
		"layout(triangle_strip,max_vertices=4)out;\n"	\
		"in vec4 out_Color[];\n" \
		"in vec4 out_PreviousPos[];\n" \
		"in float out_Phase[];\n" \
		"out vec4 gsOut_Color;\n"	\
		"out vec4 gsOut_PreviousPos;\n"	\
		"out vec4 gsOut_Phase;\n"	\
		"uniform mat4 u_model_matrix,u_view_matrix,u_projection_matrix;\n" \
		"uniform float pointRadius;\n"	\
		"void main()"	\
		"{\n"	\
		"	float phase = out_Phase[0];\n"	\
		"	float radius = pointRadius;\n"	\
		"	vec3 pos = gl_in[0].gl_Position.xyz;\n"	\
		"	vec3 pos2 = out_PreviousPos[0].xyz;\n"	\
		"	vec3 motion = pos - pos2;\n"	\
		"	vec3 dir = normalize(motion);\n"	\
		"	float len = length(motion);\n"	\
		"	vec3 x = dir * radius;\n"	\
		"	vec3 view = normalize(-pos);\n"	\
		"	vec3 y = normalize(cross(dir,view))*radius;\n"	\
		"	float facing = dot(view,dir);\n"	\
		"	float threshold = 0.01;\n"	\
		"	if((len < threshold)||(facing > 0.95)||(facing < -0.95))\n"	\
		"	{\n"	\
		"		pos2 = pos;					\n"	\
		"		x = vec3(radius,0.0,0.0);	\n"	\
		"		y=vec3(0.0,-radius,0.0);	\n"	\
		"	}\n"	\
		"	gsOut_Color = out_Color[0];		\n"	\
		"	\n"	\
		"	gsOut_Phase = vec4(0.0,0.0,0.0,phase);						\n"	\
		"	gsOut_PreviousPos = gl_in[0].gl_Position;					\n"	\
		"	gl_Position = u_projection_matrix * vec4(pos + x + y,1.0);	\n"	\
		"	EmitVertex();\n"	\
		"	\n"	\
		"	gsOut_Phase = vec4(0.0,1.0,0.0,phase);\n"	\
		"	gsOut_PreviousPos = gl_in[0].gl_Position;\n"	\
		"	gl_Position = u_projection_matrix * vec4(pos + x - y,1.0);\n"	\
		"	EmitVertex();\n"	\
		"	\n"	\
		"	gsOut_Phase = vec4(1.0,0.0,0.0,phase);\n"	\
		"	gsOut_PreviousPos = gl_in[0].gl_Position;\n"	\
		"	gl_Position = u_projection_matrix * vec4(pos - x + y,1.0);\n"	\
		"	EmitVertex();\n"	\
		"	\n"	\
		"	gsOut_Phase = vec4(1.0,1.0,0.0,phase);\n"	\
		"	gsOut_PreviousPos = gl_in[0].gl_Position;\n"	\
		"	gl_Position = u_projection_matrix * vec4(pos - x - y,1.0);\n"	\
		"	EmitVertex();\n"	\
		"	\n"	\
		"	EndPrimitive();\n"	\
		"	\n"	\
		"}";
	glShaderSource(iGeometryShaderObject_ParticleProg, 1, (const GLchar**)&geometryShaderParticleProgSourceCode, NULL);

	// Compile Source Code
	glCompileShader(iGeometryShaderObject_ParticleProg);
	iInfoLogLength = 0;
	iShaderCompileStatus = 0;
	szInfoLog = NULL;
	glGetShaderiv(iGeometryShaderObject_ParticleProg, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(iGeometryShaderObject_ParticleProg, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(iGeometryShaderObject_ParticleProg, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf_s(g_pFile, "Error : Geometry Shader ParticleProg Compilation Log : %s \n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				return INIT_GEOMETRY_SHADER_COMPILATION_FAILED;
				/*UnInitialize();
				exit(EXIT_FAILURE);*/
			}

		}

	}
	/*Geometry Shader Stop*/
	/*Fragment Shader Start*/ //texture2DPS
	iFragmentShaderObject_ParticleProg = glCreateShader(GL_FRAGMENT_SHADER);
	const GLchar *fragmentShaderParticleProgSourceCode =
		"#version 450 core"	\
		"\n"	\
		"in vec4 gsOut_Color;\n"	\
		"in vec4 gsOut_PreviousPos;\n"	\
		"in vec4 gsOut_Phase;\n"	\
		"out vec4 FragColor;\n"	\
		"uniform float pointRadius;\n"	\
		"uniform float alphaValue;\n"	\
		"void main(void)"	\
		"{\n"	\
		"	vec3 N;\n"	\
		"	N.xy = gsOut_Phase.xy * vec2(2.0,-2.0) + vec2(-1.0,1.0);\n"	\
		"	float r2 = dot(N.xy,N.xy);\n"	\
		"	if(r2 > 1.0)\n"	\
		"	{\n"	\
		"		//discard;\n"	\
		"	}\n"	\
		"	N.z = sqrt(1.0 - r2);\n"	\
		"	float alpha = clamp((1.0-r2),0.0,1.0);\n"	\
		"	alpha *= gsOut_Color.w;\n"	\
		"	FragColor = vec4(gsOut_Color.rgb * alpha,alphaValue);\n"	\
		"}";

	glShaderSource(iFragmentShaderObject_ParticleProg, 1, (const GLchar**)&fragmentShaderParticleProgSourceCode, NULL);
	glCompileShader(iFragmentShaderObject_ParticleProg);
	iInfoLogLength = 0;
	iShaderCompileStatus = 0;
	szInfoLog = NULL;
	glGetShaderiv(iFragmentShaderObject_ParticleProg, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(iFragmentShaderObject_ParticleProg, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(iFragmentShaderObject_ParticleProg, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf(g_pFile, "ERROR: Fragment Shader Display Particle Prog Compilation Log : %s \n", szInfoLog);
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
	g_ShaderProgramObject_ParticleProg = glCreateProgram();
	glAttachShader(g_ShaderProgramObject_ParticleProg, iVertexShaderObject_ParticleProg);
	glAttachShader(g_ShaderProgramObject_ParticleProg, iGeometryShaderObject_ParticleProg);
	glAttachShader(g_ShaderProgramObject_ParticleProg, iFragmentShaderObject_ParticleProg);
	glBindAttribLocation(g_ShaderProgramObject_ParticleProg, SAM_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(g_ShaderProgramObject_ParticleProg, SAM_ATTRIBUTE_COLOR, "vColor");
	glBindAttribLocation(g_ShaderProgramObject_ParticleProg, SAM_ATTRIBUTE_VELOCITY, "vVelocity");
	glBindAttribLocation(g_ShaderProgramObject_ParticleProg, SAM_ATTRIBUTE_LIFETIME, "vLifeTime");

	glLinkProgram(g_ShaderProgramObject_ParticleProg);

	GLint iShaderLinkStatus = 0;
	iInfoLogLength = 0;
	glGetProgramiv(g_ShaderProgramObject_ParticleProg, GL_LINK_STATUS, &iShaderLinkStatus);
	if (iShaderLinkStatus == GL_FALSE)
	{
		glGetProgramiv(g_ShaderProgramObject_ParticleProg, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(g_ShaderProgramObject_ParticleProg, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf_s(g_pFile, "ERROR : Linking Shader Program Particle Prog Object Blur Prog Failed %s \n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				return INIT_LINK_SHADER_PROGRAM_FAILED;
				//DestroyWindow(g_hwnd);
				//exit(EXIT_FAILURE);
			}
		}
	}
	/* Shader Program End */
	/* Set up uniforms Start */
	g_Uniform_Model_Matrix_ParticleProg = glGetUniformLocation(g_ShaderProgramObject_ParticleProg, "u_model_matrix");
	g_Uniform_View_Matrix_ParticleProg = glGetUniformLocation(g_ShaderProgramObject_ParticleProg, "u_view_matrix");
	g_Uniform_Projection_Matrix_ParticleProg = glGetUniformLocation(g_ShaderProgramObject_ParticleProg, "u_projection_matrix");
	g_Uniform_PointRadius_ParticleProg = glGetUniformLocation(g_ShaderProgramObject_ParticleProg, "pointRadius");
	g_Uniform_TimeStep_ParticleProg = glGetUniformLocation(g_ShaderProgramObject_ParticleProg, "timeStep");

	g_Uniform_AlphaValue_ParticleProg = glGetUniformLocation(g_ShaderProgramObject_ParticleProg, "alphaValue");
	/* Set up uniforms Stop  */
	/**		Psrticle Prog Stop  	**/

	
	/****	Implementation of Shaders in SmokeRenderer.cpp Stop  	****/

	
	/* Fill Buffers Start*/

	//InitCubeRandom(g_VBO_Particle_Position.fp_HostPointer, g_VBO_Particle_Velocity.fp_HostPointer,vmath::vec3(0.0, 1.0, 0.0),vmath::vec3(1.0, 1.0, 1.0), vmath::vec3(0.0f, 0.0f, 0.0f), 100.0);

	// 0 will be used for rendering
	glGenVertexArrays(1, &g_VertexArrayObject_Particle);
	glBindVertexArray(g_VertexArrayObject_Particle);//VAO

	glGenBuffers(2, g_VBO_Particle_Position.ogl_vbo);
	g_VBO_Particle_Position.m_currentWrite = 1;
	glBindBuffer(GL_ARRAY_BUFFER, g_VBO_Particle_Position.ogl_vbo[0]);// vbo position
	glBufferData(GL_ARRAY_BUFFER, g_VBO_Particle_Position.iSizeOfBufer, g_VBO_Particle_Position.fp_HostPointer, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glVertexAttrib3f(SAM_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);// vbo color


	glGenBuffers(2, g_VBO_Particle_Velocity.ogl_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, g_VBO_Particle_Velocity.ogl_vbo[0]);// vbo velocity
	glBufferData(GL_ARRAY_BUFFER, g_VBO_Particle_Velocity.iSizeOfBufer, g_VBO_Particle_Velocity.fp_HostPointer, GL_DYNAMIC_DRAW);

	g_VBO_Particle_Velocity.m_currentWrite = 1;
	glVertexAttribPointer(SAM_ATTRIBUTE_VELOCITY, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_VELOCITY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	glBindBuffer(GL_ARRAY_BUFFER, g_VBO_Particle_Position.ogl_vbo[1]);// vbo position
	glBufferData(GL_ARRAY_BUFFER, g_VBO_Particle_Position.iSizeOfBufer, g_VBO_Particle_Position.fp_HostPointer, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glVertexAttrib3f(SAM_ATTRIBUTE_COLOR, 1.0f, 1.0f, 1.0f);// vbo color

	glBindBuffer(GL_ARRAY_BUFFER, g_VBO_Particle_Velocity.ogl_vbo[1]);// vbo velocity
	glBufferData(GL_ARRAY_BUFFER, g_VBO_Particle_Velocity.iSizeOfBufer, g_VBO_Particle_Velocity.fp_HostPointer, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(SAM_ATTRIBUTE_VELOCITY, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_VELOCITY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	glGenBuffers(1, &g_VertexBufferObject_Particle_Element);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_VertexBufferObject_Particle_Element);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, g_iNumParticles * sizeof(GLuint), gp_fHostPointer_Elements, GL_DYNAMIC_DRAW);//gp_fHostPointer_Elements is zeros for inital value, acula values will come from CUDA
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	/*Register all buffers witg CUDA*/
	cuErr = cudaGraphicsGLRegisterBuffer(&g_VBO_Particle_Position.vbo_Reg_to_CUDA[0], g_VBO_Particle_Position.ogl_vbo[0], cudaGraphicsMapFlagsWriteDiscard);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaGraphicsGLRegisterBuffer(g_VBO_Particle_Position.vbo_Reg_to_CUDA[0]) : %s\n", cudaGetErrorString(cuErr));
		return	INIT_CUDA_REGISTER_FAILED;
	}

	cuErr = cudaGraphicsGLRegisterBuffer(&g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[0], g_VBO_Particle_Velocity.ogl_vbo[0], cudaGraphicsMapFlagsWriteDiscard);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaGraphicsGLRegisterBuffer(g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[0]) : %s\n", cudaGetErrorString(cuErr));
		return	INIT_CUDA_REGISTER_FAILED;
	}

	cuErr = cudaGraphicsGLRegisterBuffer(&g_VBO_Particle_Position.vbo_Reg_to_CUDA[1], g_VBO_Particle_Position.ogl_vbo[1], cudaGraphicsMapFlagsWriteDiscard);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaGraphicsGLRegisterBuffer(g_VBO_Particle_Position.vbo_Reg_to_CUDA[1]) : %s\n", cudaGetErrorString(cuErr));
		return	INIT_CUDA_REGISTER_FAILED;
	}

	cuErr = cudaGraphicsGLRegisterBuffer(&g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[1], g_VBO_Particle_Velocity.ogl_vbo[1], cudaGraphicsMapFlagsWriteDiscard);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaGraphicsGLRegisterBuffer(g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[1]) : %s\n", cudaGetErrorString(cuErr));
		return	INIT_CUDA_REGISTER_FAILED;
	}
	// Register elements Array
	cuErr = cudaGraphicsGLRegisterBuffer(&elements_reg_to_CUDA, g_VertexBufferObject_Particle_Element, cudaGraphicsMapFlagsWriteDiscard);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaGraphicsGLRegisterBuffer(elements_reg_to_CUDA) : %s\n", cudaGetErrorString(cuErr));
		return	INIT_CUDA_REGISTER_FAILED;
	}


	/* Fill Buffers End*/
	/// Sam : all Shader Code End


	createNoiseTexture(64, 64, 64);

	glEnable(GL_TEXTURE_2D);
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


void Update(void)
{
	void RunEmitter();
	void Step_ParticleSystem(float deltaTime);
	void DepthSort_ParticleSystem();

	const float speed = 0.02f;


	g_floatTimeStep = g_floatTimeStep + g_fTimeStep;


	g_v3CursorPos[0] = 2.0f*sinf(g_floatTimeStep*speed)*1.5f;//x
	g_v3CursorPos[1] = 1.5f + sinf(g_floatTimeStep*speed*1.3f);//y
	g_v3CursorPos[2] = cosf(g_floatTimeStep*speed)*1.5f;//z

	

	if (g_bEmitterOn)
	{
		RunEmitter();
	}

	// Fill SimParams with appro priate values
	g_SimParams.f3Gravity = make_float3(0.0f,0.0f,0.0f);
	g_SimParams.fGlobalDamping = 0.99f;
	g_SimParams.fNoiseFreq = 0.1f;
	g_SimParams.fNoiseAmp = 0.00100000005f;
	//g_SimParams.f3CursorPos = make_float3(0.0638938025, 1.39843607, 1.02703369);
	g_SimParams.f3CursorPos = make_float3(g_v3CursorPos[0], g_v3CursorPos[1], g_v3CursorPos[2]);
	g_SimParams.fTime = g_floatTimeStep;

	Step_ParticleSystem(g_fTimeStep);

	DepthSort_ParticleSystem();
}


void Render(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	vmath::mat4 modelMatrix = vmath::mat4::identity();
	vmath::mat4 viewMatrix = vmath::mat4::identity();

	/** Particles Added **/
	
	modelMatrix = vmath::mat4::identity();
	viewMatrix = vmath::mat4::identity();
	glUseProgram(g_ShaderProgramObject_ParticleProg);
	glBindVertexArray(g_VertexArrayObject_Particle);
	glBindBuffer(GL_ARRAY_BUFFER, g_VBO_Particle_Position.ogl_vbo[0]);
	glBindBuffer(GL_ARRAY_BUFFER, g_VBO_Particle_Velocity.ogl_vbo[0]);

	modelMatrix = vmath::translate(0.0f, 0.0f, -10.0f);

	glUniformMatrix4fv(g_Uniform_Model_Matrix_ParticleProg, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(g_Uniform_View_Matrix_ParticleProg, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(g_Uniform_Projection_Matrix_ParticleProg, 1, GL_FALSE, g_PersPectiveProjectionMatrix);
	glUniform1f(g_Uniform_PointRadius_ParticleProg, g_fparticleRadius);
	glUniform1f(g_Uniform_TimeStep_ParticleProg, g_floatTimeStep);
	glUniform1f(g_Uniform_AlphaValue_ParticleProg, 0.1f);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
	glDrawArrays(GL_POINTS, 0, g_iNumParticles);
	glDisable(GL_BLEND);


	glBindBuffer(GL_ARRAY_BUFFER, 0);
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
	//fprintf_s(g_pFile, "IN Resize() \n");
	if (iHeight <= 0)
	{
		iHeight = 1;
	}

	glViewport(0, 0, (GLsizei)iWidth, (GLsizei)iHeight);

	g_PersPectiveProjectionMatrix = vmath::perspective(45.0f, (float)iWidth / (float)iHeight, 0.1f, 100.0f);

	//fprintf_s(g_pFile, "OUT Resize() \n\n");
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

	
	if (g_VBO_Particle_Position.vbo_Reg_to_CUDA[0])
	{
		cudaGraphicsUnmapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[0]);
		cudaGraphicsUnregisterResource(g_VBO_Particle_Position.vbo_Reg_to_CUDA[0]);
	}

	if (g_VBO_Particle_Position.vbo_Reg_to_CUDA[1])
	{
		cudaGraphicsUnmapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[1]);
		cudaGraphicsUnregisterResource(g_VBO_Particle_Position.vbo_Reg_to_CUDA[1]);
	}

	if (g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[0])
	{
		cudaGraphicsUnmapResources(1, &g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[0]);
		cudaGraphicsUnregisterResource(g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[0]);
	}

	if (g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[1])
	{
		cudaGraphicsUnmapResources(1, &g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[1]);
		cudaGraphicsUnregisterResource(g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[1]);
	}

	if (g_VBO_Particle_Position.fp_CUDADevicePointer[0])
	{
		cudaFree(g_VBO_Particle_Position.fp_CUDADevicePointer[0]);
		g_VBO_Particle_Position.fp_CUDADevicePointer[0] = NULL;
	}
	
	if (g_VBO_Particle_Position.fp_CUDADevicePointer[1])
	{
		cudaFree(g_VBO_Particle_Position.fp_CUDADevicePointer[1]);
		g_VBO_Particle_Position.fp_CUDADevicePointer[1] = NULL;
	}

	if (g_VBO_Particle_Position.fp_HostPointer)
	{
		free(g_VBO_Particle_Position.fp_HostPointer);
		g_VBO_Particle_Position.fp_HostPointer = NULL;
	}

	if (g_VBO_Particle_Velocity.fp_CUDADevicePointer[0])
	{
		cudaFree(g_VBO_Particle_Velocity.fp_CUDADevicePointer[0]);
		g_VBO_Particle_Velocity.fp_CUDADevicePointer[0] = NULL;
	}

	if (g_VBO_Particle_Velocity.fp_CUDADevicePointer[1])
	{
		cudaFree(g_VBO_Particle_Velocity.fp_CUDADevicePointer[1]);
		g_VBO_Particle_Velocity.fp_CUDADevicePointer[1] = NULL;
	}

	if (g_VBO_Particle_Velocity.fp_HostPointer)
	{
		free(g_VBO_Particle_Velocity.fp_HostPointer);
		g_VBO_Particle_Velocity.fp_HostPointer = NULL;
	}

	if (gp_fDevicePointer_Elements)
	{
		cudaFree(gp_fDevicePointer_Elements);
		gp_fDevicePointer_Elements = NULL;
	}

	if (gp_fHostPointer_Elements)
	{
		free(gp_fHostPointer_Elements);
		gp_fHostPointer_Elements = NULL;
	}

	if (gp_fDevicePointer_SortIndex)
	{
		cudaFree(gp_fDevicePointer_SortIndex);
		gp_fDevicePointer_SortIndex = NULL;
	}

	if (gp_fHostPointer_SortIndex)
	{
		free(gp_fHostPointer_SortIndex);
		gp_fHostPointer_SortIndex = NULL;
	}

	// Fail safe CUDA reset for this process
	cudaDeviceReset();

	// can be done in one call only....!!
	if (g_VBO_Particle_Position.ogl_vbo[0])
	{
		glDeleteBuffers(1, &g_VBO_Particle_Position.ogl_vbo[0]);
		g_VBO_Particle_Position.ogl_vbo[0] = NULL;
	}

	if (g_VBO_Particle_Position.ogl_vbo[1])
	{
		glDeleteBuffers(1, &g_VBO_Particle_Position.ogl_vbo[0]);
		g_VBO_Particle_Position.ogl_vbo[1] = NULL;
	}

	if (g_VBO_Particle_Velocity.ogl_vbo[0])
	{
		glDeleteBuffers(1, &g_VBO_Particle_Position.ogl_vbo[0]);
		g_VBO_Particle_Position.ogl_vbo[0] = NULL;
	}

	if (g_VBO_Particle_Velocity.ogl_vbo[1])
	{
		glDeleteBuffers(1, &g_VBO_Particle_Position.ogl_vbo[0]);
		g_VBO_Particle_Position.ogl_vbo[1] = NULL;
	}

	if (g_VertexBufferObject_Particle_Element)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_Particle_Element);
		g_VertexBufferObject_Particle_Element = NULL;
	}

	if (g_VertexArrayObject_Particle) 
	{
		glDeleteVertexArrays(1, &g_VertexArrayObject_Particle);
		g_VertexArrayObject_Particle = NULL;
	}

	//g_ShaderProgramObject_ParticleProg
	if (g_ShaderProgramObject_ParticleProg)
	{
		GLsizei iShaderCount;
		GLsizei iShaderNumber;

		glUseProgram(g_ShaderProgramObject_ParticleProg);
		glGetProgramiv(g_ShaderProgramObject_ParticleProg, GL_ATTACHED_SHADERS, &iShaderCount);
		GLuint *pShaders = (GLuint*)calloc(iShaderCount, sizeof(GLuint));

		if (pShaders)
		{
			glGetAttachedShaders(g_ShaderProgramObject_ParticleProg, iShaderCount, &iShaderCount, pShaders);
			for (iShaderNumber = 0; iShaderNumber < iShaderCount; iShaderNumber++)
			{
				glDetachShader(g_ShaderProgramObject_ParticleProg, pShaders[iShaderNumber]);
				glDeleteShader(pShaders[iShaderNumber]);
				pShaders[iShaderNumber] = 0;
			}
			free(pShaders);
			pShaders = NULL;
		}
		glUseProgram(0);

		glDeleteProgram(g_ShaderProgramObject_ParticleProg);
		g_ShaderProgramObject_ParticleProg = NULL;
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

/**		Smoke Renderer Init Start	**/
inline float frand()
{
	return rand() / (float)RAND_MAX;
}

inline float sfrand()
{
	return frand()*2.0f - 1.0f;
}

inline vmath::vec3 svrand()
{
	return vmath::vec3(sfrand(), sfrand(), sfrand());
}

void InitCubeRandom(float4* pPos, float4* pVeloc, vmath::vec3 origin, vmath::vec3 size, vmath::vec3 vel, float lifetime)
{
	float4 *posPtr = pPos;
	float4 *velPtr = pVeloc;

	for (UINT i = 0; i < g_iNumParticles; i++)
	{
		vmath::vec3 pos = origin + svrand()*size;
		posPtr[i] = make_float4(pos[0], pos[1], pos[2], 0.0f);
		velPtr[i] = make_float4(vel[0], vel[1], vel[2], lifetime);
	}
}

vmath::vec3 randSphere()
{
	//vmath::vec3 svrand();

	vmath::vec3 retVal;

	do
	{
		retVal = svrand();
	} while (vmath::length(retVal) > 1.0f);

	return retVal;
}

// Smoke Interaction with CUDA Part

// Float To Int conversion
inline int ftoi(float value) 
{
	return (value >= 0 ? static_cast<int>(value + 0.5)
		: static_cast<int>(value - 0.5));
}

void RunEmitter()
{

	void SphereEmitter_ParticleSystem(float4 *pPosition, float4 *pVelocity, unsigned int &index, vmath::vec3 pos, vmath::vec3 vel, vmath::vec3 spread, float r, int n, float lifeTime, float lifeTimeVarience);

	vmath::vec3 vel = vmath::vec3(0.0f, g_EmitterVel, 0.0f);
	vmath::vec3 vx = vmath::vec3(1.0f, 0.0f, 0.0f);
	vmath::vec3 vy = vmath::vec3(0.0f, 0.0f, 1.0f);
	vmath::vec3 spread = vmath::vec3(g_EmitterSpread, g_EmitterSpread, g_EmitterSpread);

	//sphereEmitter or different styles of emmiters
	SphereEmitter_ParticleSystem(g_VBO_Particle_Position.fp_HostPointer, g_VBO_Particle_Velocity.fp_HostPointer, g_EmitterIndex, g_v3CursorPos, vel, spread, g_EmitterRadius, ftoi(g_iEmitterRate*g_fTimeStep), g_fParticleLifetime, g_fParticleLifetime*0.1f);

	if (g_EmitterIndex > g_iNumParticles - 1)
	{
		g_EmitterIndex = 0;
	}
}

void SphereEmitter_ParticleSystem(float4 *pPosition_host, float4 *pVelocity_host, unsigned int &index, vmath::vec3 pos, vmath::vec3 vel, vmath::vec3 spread, float r, int n, float lifeTime, float lifeTimeVarience)
{
	float frand();

	vmath::vec3 randSphere();

	float4 *posPtr = pPosition_host;
	float4 *velPtr = pVelocity_host;

	unsigned int start = index;
	unsigned int count = 0;

	for (int i = 0; i < n; i++)
	{
		vmath::vec3 x = randSphere();

		//float dist = vmath::length(x);
		if (index < g_iNumParticles)
		{
			vmath::vec3 p = pos + x * r;
			float age = 0.0f;

			float lt = lifeTime + frand() * lifeTimeVarience;

			vmath::vec3 dir = randSphere();
			dir[1] = fabsf(dir[1]); // y posn of dir
			vmath::vec3 v = vel + dir*spread;

			posPtr[index] = make_float4(p[0], p[1], p[2], age);
			velPtr[index] = make_float4(v[0], v[1], v[2], lt);

			index++;
			count++;
		}
	}

	// Copy everything in gpu
	cudaError cuErr;
	size_t num_bytes;
	// Map the location on GPU
	cuErr = cudaGraphicsMapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsMapResources(Particle_Position[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsResourceGetMappedPointer((void**)&g_VBO_Particle_Position.fp_CUDADevicePointer[0], &num_bytes, g_VBO_Particle_Position.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsResourceGetMappedPointer(Particle_Position[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsMapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsMapResources(Particle_Position[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsResourceGetMappedPointer((void**)&g_VBO_Particle_Position.fp_CUDADevicePointer[1], &num_bytes, g_VBO_Particle_Position.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsResourceGetMappedPointer(Particle_Position[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsMapResources(1, &g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsMapResources(Particle_Velocity[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsResourceGetMappedPointer((void**)&g_VBO_Particle_Velocity.fp_CUDADevicePointer[0], &num_bytes, g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsResourceGetMappedPointer(g_VBO_Particle_Velocity[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsMapResources(1, &g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsMapResources(g_VBO_Particle_Velocity[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsResourceGetMappedPointer((void**)&g_VBO_Particle_Velocity.fp_CUDADevicePointer[1], &num_bytes, g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsResourceGetMappedPointer(g_VBO_Particle_Velocity[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}
	// Copy the contents to GPU

	cuErr = cudaMemcpy((void*)(g_VBO_Particle_Position.fp_CUDADevicePointer[g_VBO_Particle_Position.m_currentRead] + start), (void*)(g_VBO_Particle_Position.fp_HostPointer + start), count * sizeof(float4), cudaMemcpyHostToDevice);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaMemcpy(g_VBO_Particle_Position[currentRead]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaMemcpy((void*)(g_VBO_Particle_Velocity.fp_CUDADevicePointer[g_VBO_Particle_Velocity.m_currentRead] + start), (void*)(g_VBO_Particle_Velocity.fp_HostPointer + start), count * sizeof(float4), cudaMemcpyHostToDevice);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaMemcpy(g_VBO_Particle_Velocity[currentRead]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	// UnMap GPu content
	cuErr = cudaGraphicsUnmapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsUnmapResources(g_VBO_Particle_Position[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsUnmapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsUnmapResources(g_VBO_Particle_Position[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}


	cuErr = cudaGraphicsUnmapResources(1, &g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsUnmapResources(g_VBO_Particle_Velocity[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsUnmapResources(1, &g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsUnmapResources(g_VBO_Particle_Velocity[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}
}

void Step_ParticleSystem(float deltaTime)
{
	setParameters(&g_SimParams);

	// Map Pointers
	cudaError cuErr;
	size_t num_bytes;
	// Map the location on GPU
	cuErr = cudaGraphicsMapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsMapResources(Particle_Position[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsResourceGetMappedPointer((void**)&g_VBO_Particle_Position.fp_CUDADevicePointer[0], &num_bytes, g_VBO_Particle_Position.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsResourceGetMappedPointer(Particle_Position[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsMapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsMapResources(Particle_Position[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsResourceGetMappedPointer((void**)&g_VBO_Particle_Position.fp_CUDADevicePointer[1], &num_bytes, g_VBO_Particle_Position.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsResourceGetMappedPointer(Particle_Position[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsMapResources(1, &g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsMapResources(Particle_Velocity[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsResourceGetMappedPointer((void**)&g_VBO_Particle_Velocity.fp_CUDADevicePointer[0], &num_bytes, g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsResourceGetMappedPointer(g_VBO_Particle_Velocity[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsMapResources(1, &g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsMapResources(g_VBO_Particle_Velocity[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsResourceGetMappedPointer((void**)&g_VBO_Particle_Velocity.fp_CUDADevicePointer[1], &num_bytes, g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsResourceGetMappedPointer(g_VBO_Particle_Velocity[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	// Integrate Systems
	// getDevicePtr ==>  m_currentRead
	// getDeviceWritePtr ==> m_currentWrite
	// g_fTimeStep
	integrateSystem(g_VBO_Particle_Position.fp_CUDADevicePointer[g_VBO_Particle_Position.m_currentRead],g_VBO_Particle_Position.fp_CUDADevicePointer[g_VBO_Particle_Position.m_currentWrite],
		g_VBO_Particle_Velocity.fp_CUDADevicePointer[g_VBO_Particle_Velocity.m_currentRead], g_VBO_Particle_Velocity.fp_CUDADevicePointer[g_VBO_Particle_Velocity.m_currentWrite], deltaTime,g_iNumParticles
		);
	// UnMap GPu content
	cuErr = cudaGraphicsUnmapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsUnmapResources(g_VBO_Particle_Position[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsUnmapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsUnmapResources(g_VBO_Particle_Position[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}


	cuErr = cudaGraphicsUnmapResources(1, &g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsUnmapResources(g_VBO_Particle_Velocity[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsUnmapResources(1, &g_VBO_Particle_Velocity.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsUnmapResources(g_VBO_Particle_Velocity[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}


	// Swap
	int temp = g_VBO_Particle_Velocity.m_currentRead;
	g_VBO_Particle_Velocity.m_currentRead = g_VBO_Particle_Velocity.m_currentWrite;
	g_VBO_Particle_Velocity.m_currentWrite = temp;

	temp = g_VBO_Particle_Position.m_currentRead;
	g_VBO_Particle_Position.m_currentRead = g_VBO_Particle_Position.m_currentWrite;
	g_VBO_Particle_Position.m_currentWrite = temp;

}

void DepthSort_ParticleSystem()
{
	//g_bDoDepthSort => this is always true for working in cuda is expected

	// map position vbos
	// Map Pointers
	cudaError cuErr;
	size_t num_bytes;
	// Map the location on GPU
	cuErr = cudaGraphicsMapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsMapResources(Particle_Position[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsResourceGetMappedPointer((void**)&g_VBO_Particle_Position.fp_CUDADevicePointer[0], &num_bytes, g_VBO_Particle_Position.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsResourceGetMappedPointer(Particle_Position[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsMapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsMapResources(Particle_Position[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsResourceGetMappedPointer((void**)&g_VBO_Particle_Position.fp_CUDADevicePointer[1], &num_bytes, g_VBO_Particle_Position.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsResourceGetMappedPointer(Particle_Position[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	// element
	cuErr = cudaGraphicsMapResources(1, &elements_reg_to_CUDA);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsMapResources(elements_reg_to_CUDA) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsResourceGetMappedPointer((void**)&gp_fDevicePointer_Elements, &num_bytes, elements_reg_to_CUDA);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsResourceGetMappedPointer(g_VertexBufferObject_Particle_Element) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	calcDepth(g_VBO_Particle_Position.fp_CUDADevicePointer[g_VBO_Particle_Position.m_currentRead], gp_fDevicePointer_SortIndex, gp_fDevicePointer_Elements, g_fSortVector,g_iNumParticles);

	// UnMap GPu content
		cuErr = cudaGraphicsUnmapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[0]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsUnmapResources(g_VBO_Particle_Position[0]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsUnmapResources(1, &g_VBO_Particle_Position.vbo_Reg_to_CUDA[1]);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsUnmapResources(g_VBO_Particle_Position[1]) Failed : %s\n", cudaGetErrorString(cuErr));
	}

	cuErr = cudaGraphicsUnmapResources(1, &elements_reg_to_CUDA);
	if (cuErr != cudaSuccess)
	{
		fprintf_s(g_pFile, "IN SphereEmitter_ParticleSystem : cudaGraphicsUnmapResources(elements_reg_to_CUDA) Failed : %s\n", cudaGetErrorString(cuErr));
	}
}

/**		Smoke Renderer Init Stop 	**/
