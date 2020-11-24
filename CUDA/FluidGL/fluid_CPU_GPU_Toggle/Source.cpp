#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>

#include <gl\glew.h>
#include <gl\GL.h>

#include "vmath.h"

#include <iostream>
#include <iterator> 
#include <map>
#include <ft2build.h>
#include FT_FREETYPE_H

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
#include <cufft.h>
#include <fftw3.h>
#include "my_helper_timer.h"
#include "Defines.h"
#include "FluidsGL_Kernels.cuh"

#define WIN_WIDTH	800
#define WIN_HEIGHT	600

#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib,"cudart.lib")
#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"freetype.lib")
#pragma comment(lib,"fftw3.lib")
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
	SAM_ATTRIBUTE_NORMAL,
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
GLuint g_ShaderProgramObject_Font = 0;
GLuint g_ShaderProgramObject_Square;
GLuint g_ShaderProgramObject_GPU_Fluid = 0;

// All Vertex Buffers
GLuint g_VertexArrayObject_Square;
GLuint g_VertexBufferObject_Position_Square;
GLuint g_VertexBufferObject_Texture_Square;

GLuint g_VertexArrayObject_GPU_Fluid = 0;
GLuint g_VertexBufferObject_Position_GPU_Fluid = 0;
GLuint g_VertexBufferObject_Color = 0;
// Uniforms
GLuint g_Uniform_Model_Matrix_Square;
GLuint g_Uniform_View_Matrix_Square;
GLuint g_Uniform_Projection_Matrix_Square;

GLuint g_Uniform_Model_Matrix_fluid = 0;
GLuint g_Uniform_View_Matrix_fluid = 0;
GLuint g_Uniform_Projection_Matrix_fluid = 0;

// sampler
GLuint g_uniform_TextureSampler = 0;
GLuint g_uniform_TextureSampler_FB = 0;

// Projection
vmath::mat4 g_PersPectiveProjectionMatrix;
vmath::mat4 g_OrthoProjectionMatrix;

/*		Frambuffer Start		*/
// Framebuffer
GLuint g_FrameBuffer;
GLuint g_ColorTexture;
GLuint g_DepthTexture;

const GLenum g_DrawBuffers[] = { GL_COLOR_ATTACHMENT0 };

/// Current Window Sizes
GLfloat g_fCurrentWidth = 0;
GLfloat g_fCurrentHeight = 0;
/*		Frambuffer Stop 		*/

// Freetype start

GLuint g_VertexArrayObject_font = 0;
GLuint g_VertexBufferObject_Position_font = 0;


GLuint g_Uniform_Model_Matrix_font = 0;
GLuint g_Uniform_View_Matrix_font = 0;
GLuint g_Uniform_Projection_Matrix_font = 0;

FT_Library ft;
FT_Face face;

struct Character
{
	GLuint		TexyureID;	// ID handle of the glyph texture
	vmath::vec2 Size;		// Size of glyph
	vmath::vec2 Bearing;	// Offset from baseline to left/top of glyph
	GLuint      Advance;	// Offset to advance to next glyph
};

std::map<GLchar, Character> Characters;
float zDirn = -10.0f;
bool g_bShowHelp = true;

char tempString[256];
char flopString[256];
// Freetype stop

bool gpu_cpu_Switch = false;// default on cpu

GLfloat g_fCurrrentWidth;
GLfloat g_fCurrrentHeight;

/*		Simulation Variables Start		*/
cufftHandle planr2c;
cufftHandle planc2r;
static cData_float *xvField_GPU = NULL;
static cData_float *yvField_GPU = NULL;

cData_float *hvField_GPU = NULL;  // Host Vector
cData_float *dvField = NULL;  // Device Vector

static int wWidth = MAX(512, DIM);
static int wHeight = MAX(512, DIM);

/*static int wWidth = MAX(1024, DIM);
static int wHeight = MAX(1024, DIM);*/

static int clicked = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
IStopWatchTimer *g_Timer = NULL;

/*		Particle Data Start		*/
GLuint g_iVertexBufferObject_GPU = 0;
GLuint g_iVertexArrayObject_fluid_GPU = 0;
struct cudaGraphicsResource *cuda_vbo_resource;
static cData_float *particles = NULL; // particle position in host memory
									  // Texture pitch
size_t tPitch = 0;
/*		Particle Data Stop		*/

static int lastx = 0, lasty = 0;
/*		Simulation Variables Stop		*/

int g_iFlopsPerInteraction = 40;

/** CPU Side variables Start **/
typedef double2 cData;
// this must be treated as 2D array
double2 refTexArray[DIM][DIM]; // used in place of cudaTexture

static cData *xvField_CPU = NULL;
static cData *yvField_CPU = NULL;

cData *hvField_CPU = NULL;  // Host Vector
cData *cpuBuffer = NULL;

static cData *particles_CPU = NULL;
GLuint g_iVertexBufferObject_CPU = 0;
GLuint g_iVertexArrayObject_fluid_CPU = 0;
size_t tPitch_CPU = sizeof(cData)*DIM;

static int currentX = 0, currentY = 0;
/** CPU Side variables Start **/

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
	void motion_CPU(int x, int y);


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

		case 0x48: // 'h' or 'H'
			gpu_cpu_Switch = (gpu_cpu_Switch) ? false : true;
			break;

		default:
			break;
		}
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
		if (gpu_cpu_Switch)
		{
			motion(LOWORD(lParam), HIWORD(lParam));
		}
		else
		{
			currentX = LOWORD(lParam), currentY = HIWORD(lParam);
			motion_CPU(currentX, currentY);
		}

		
		break;

	case WM_SIZE:
		g_fCurrrentWidth = LOWORD(lParam);
		g_fCurrrentHeight = HIWORD(lParam);

		wWidth = LOWORD(lParam);
		wHeight = HIWORD(lParam);
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

void motion_CPU(int x, int y)
{
	void addForces_CPU(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r);

	// convert motion_CPU coordinates to domain
	float fx = (lastx / (float)wWidth);
	float fy = (lasty / (float)wHeight);
	int nx = (int)(fx*DIM);
	int ny = (int)(fy*DIM);

	if (clicked && nx  < DIM - FR && nx>FR - 1 && ny < DIM - FR && ny > FR - 1)
	{
		int ddx = x - lastx;
		int ddy = y - lasty;
		fx = ddx / (float)wWidth;
		fy = ddy / (float)wHeight;
		int spy = ny - FR;
		int spx = nx - FR;
		addForces_CPU(hvField_CPU, DIM, DIM, spx, spy, FORCE*DT*fx, FORCE*DT*fy, FR);

		lastx = x;
		lasty = y;
	}
}

int Initialize(void)
{//particles_CPU
	bool Resize(int, int);
	void initParticles(cData_float *p, int dx, int dy);
	void initParticlesCPU(cData *p, int dx, int dy);


	int iPixelIndex = 0;
	PIXELFORMATDESCRIPTOR pfd;

	// Shader Programs
	GLuint iVertexShaderObject_Font = 0;
	GLuint iFragmentShaderObject_Font = 0;

	GLuint g_VertexShaderObject_Square;
	GLuint g_FragmentShaderObject_Square;
	// for simulation
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
	// GL information End

	/* Set up Free type Start */
	if (FT_Init_FreeType(&ft))
	{
		fprintf_s(g_pFile, "ERROR FreeType : FT_Init_FreeType failed\n");
	}

	if (FT_New_Face(ft, "fonts/ubuntuMono.ttf", 0, &face))
	{
		fprintf_s(g_pFile, "ERROR FreeType : FT_New_Face failed\n");
	}

	FT_Set_Pixel_Sizes(face, 0, 120);

	if (FT_Load_Char(face, 'X', FT_LOAD_RENDER))
	{
		fprintf_s(g_pFile, "ERROR FreeType : FT_Load_Char failed\n");
	}

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	for (GLubyte c = 0; c < 128; c++)
	{
		// Load character glyph 
		if (FT_Load_Char(face, c, FT_LOAD_RENDER))
		{
			std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
			continue;
		}
		// Generate texture
		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, face->glyph->bitmap.width, face->glyph->bitmap.rows, 0, GL_RED,
			GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer);
		// Set texture options
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// Now store character for later use
		Character character = { texture,
			vmath::vec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
			vmath::vec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
			face->glyph->advance.x };
		Characters.insert(std::pair<GLchar, Character>(c, character));
	}

	glBindTexture(GL_TEXTURE_2D, 0);

	FT_Done_Face(face);
	FT_Done_FreeType(ft);

	iVertexShaderObject_Font = glCreateShader(GL_VERTEX_SHADER);
	const GLchar *vertexShaderSourceCode_font = "#version 450 core"	\
		"\n" \
		"layout (location = 0)in vec4 vVertex;\n" \
		"layout (location = 0)out vec2 out_TexCoords;\n" \
		"layout (location = 1)out float shift;"	\
		"uniform mat4 u_model_matrix,u_view_matrix,u_projection_matrix;\n" \
		"void main(void)" \
		"{\n" \
		"	shift = 2.0/3.0;"	\
		"	gl_Position =  u_projection_matrix * u_view_matrix * u_model_matrix * vec4(vVertex.xy,0.0,1.0);\n" \
		"	out_TexCoords =  vVertex.zw;\n" \
		"}";

	glShaderSource(iVertexShaderObject_Font, 1, (const GLchar**)&vertexShaderSourceCode_font, NULL);

	// Compile it
	glCompileShader(iVertexShaderObject_Font);
	int iInfoLogLength = 0;
	int iShaderCompileStatus = 0;
	char* szInfoLog = NULL;
	glGetShaderiv(iVertexShaderObject_Font, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(iVertexShaderObject_Font, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(iVertexShaderObject_Font, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf_s(g_pFile, "ERROR : Vertex Shader Compilation Log : %s \n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				return INIT_VERTEX_SHADER_COMPILATION_FAILED;
				//DestroyWindow(g_hwnd);
				//exit(EXIT_FAILURE);
			}
		}
	}

	iFragmentShaderObject_Font = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar *fragmentShaderSourceCode_font = "#version 450 core"	\
		"\n"
		"layout (location = 0)in vec2 out_TexCoords;\n" \
		"layout (location = 1)in float shift;\n"	\
		"layout (location = 0)out vec4 FragColor;\n"	\
		"uniform sampler2D texture;\n"	\
		"uniform vec2 pixel;\n"	\
		"uniform vec3 textColor;\n"	\
		"void main()"	\
		"{"	\
		"    vec2 uv = out_TexCoords.xy;\n"	\
		"    vec4 current = texture2D(texture, uv);\n"	\
		"    vec4 previous = texture2D(texture, uv+vec2(-1,0)*pixel);\n"	\
		"    float r = current.r;\n"	\
		"    float g = current.g;\n"	\
		"    float b = current.b;\n"	\
		"    float a = current.a;\n"	\
		"    if( shift <= 1.0/3.0 )\n"	\
		"    {"	\
		"        float z = 3.0*shift;\n"	\
		"        r = mix(current.r, previous.b, z);\n"	\
		"        g = mix(current.g, current.r, z);\n"	\
		"        b = mix(current.b, current.g, z);\n"	\
		"    }"	\
		"    else if( shift <= 2.0/3.0 )"	\
		"    {"	\
		"        float z = 3.0*shift-1.0;\n"	\
		"        r = mix(previous.b, previous.g, z);\n"	\
		"        g = mix(current.r, previous.b, z);\n"	\
		"        b = mix(current.g, current.r, z);\n"	\
		"    }"	\
		"    else if( shift < 1.0 )\n"	\
		"    {"	\
		"        float z = 3.0*shift-2.0;\n"	\
		"        r = mix(previous.g, previous.r, z);\n"	\
		"        g = mix(previous.b, previous.g, z);\n"	\
		"        b = mix(current.r, previous.b, z);\n"	\
		"    }"	\
		"	vec4 finalFrag = vec4(r,g,b,a);\n"	\
		"	if((finalFrag.r+finalFrag.g+finalFrag.b) < 0.7)\n"	\
		"	{"	\
		"		discard;\n"	\
		"	}"	\
		"    FragColor = vec4(textColor.r,textColor.g,textColor.b,a);\n"	\
		"}";

	glShaderSource(iFragmentShaderObject_Font, 1, (const GLchar**)&fragmentShaderSourceCode_font, NULL);
	glCompileShader(iFragmentShaderObject_Font);
	iInfoLogLength = 0;
	iShaderCompileStatus = 0;
	szInfoLog = NULL;
	glGetShaderiv(iFragmentShaderObject_Font, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(iFragmentShaderObject_Font, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(iFragmentShaderObject_Font, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf(g_pFile, "ERROR: Fragment Shader Compilation Log : %s \n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				return INIT_FRAGMENT_SHADER_COMPILATION_FAILED;
				//DestroyWindow(g_hwnd);
				//exit(EXIT_FAILURE);
			}
		}
	}

	g_ShaderProgramObject_Font = glCreateProgram();
	glAttachShader(g_ShaderProgramObject_Font, iVertexShaderObject_Font);
	glAttachShader(g_ShaderProgramObject_Font, iFragmentShaderObject_Font);
	glBindAttribLocation(g_ShaderProgramObject_Font, SAM_ATTRIBUTE_POSITION, "vVertex");

	glLinkProgram(g_ShaderProgramObject_Font);

	int iShaderLinkStatus = 0;
	iInfoLogLength = 0;
	glGetProgramiv(g_ShaderProgramObject_Font, GL_LINK_STATUS, &iShaderLinkStatus);
	if (iShaderLinkStatus == GL_FALSE)
	{
		glGetProgramiv(g_ShaderProgramObject_Font, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(g_ShaderProgramObject_Font, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf_s(g_pFile, "ERROR : Linking Shader Program Objects Failed %s \n", szInfoLog);
				free(szInfoLog);
				szInfoLog = NULL;
				return INIT_LINK_SHADER_PROGRAM_FAILED;
				//DestroyWindow(g_hwnd);
				//exit(EXIT_FAILURE);
			}
		}
	}

	g_Uniform_Model_Matrix_font = glGetUniformLocation(g_ShaderProgramObject_Font, "u_model_matrix");
	g_Uniform_Projection_Matrix_font = glGetUniformLocation(g_ShaderProgramObject_Font, "u_projection_matrix");
	g_Uniform_View_Matrix_font = glGetUniformLocation(g_ShaderProgramObject_Font, "u_view_matrix");

	glGenVertexArrays(1, &g_VertexArrayObject_font);
	glBindVertexArray(g_VertexArrayObject_font);

	glGenBuffers(1, &g_VertexBufferObject_Position_font);
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_Position_font);

	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), NULL);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	/* Set up Free type Stop  */


	/*		Square Setup Start		*/
	g_VertexShaderObject_Square = glCreateShader(GL_VERTEX_SHADER);

	// give source code to shader
	const GLchar *vertexShaderSourceCode_Square = "#version 450 core"	\
		"\n" \
		"layout (location = 0)in vec4 vPosition;"	\
		"layout (location = 1)in vec2 vTexture0_Coord;"	\
		"uniform mat4 u_model_matrix;"	\
		"uniform mat4 u_view_matrix;"	\
		"uniform mat4 u_projection_matrix;"	\
		"layout (location = 0)out vec2 out_Texture0_Coord;"	\
		"void main (void)"	\
		"{"	\
		"	gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;"	\
		"	out_Texture0_Coord = vec2(vTexture0_Coord.x,vTexture0_Coord.y);"	\
		"}";
	glShaderSource(g_VertexShaderObject_Square, 1, (const GLchar**)&vertexShaderSourceCode_Square, NULL);

	// Compile Source Code
	glCompileShader(g_VertexShaderObject_Square);
	iInfoLogLength = 0;
	iShaderCompileStatus = 0;
	szInfoLog = NULL;
	glGetShaderiv(g_VertexShaderObject_Square, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(g_VertexShaderObject_Square, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(g_VertexShaderObject_Square, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf_s(g_pFile, "Error : Vertex Shader Video Compilation Log : %s \n", szInfoLog);
				free(szInfoLog);
				return INIT_VERTEX_SHADER_COMPILATION_FAILED;
			}

		}

	}

	//***** Fragment Shader *****
	g_FragmentShaderObject_Square = glCreateShader(GL_FRAGMENT_SHADER);
	//out_Texture0_Coord
	const GLchar *fragmentShaderSourceCode_Square = "#version 450 core"	\
		"\n"	\
		"layout (location = 0)out vec4 FragColor;"	\
		"layout (location = 0)in vec2 out_Texture0_Coord;"	\
		"uniform sampler2D u_texture0_sampler;"	\
		"void main (void)" \
		"{"	\
		"FragColor = texture(u_texture0_sampler,out_Texture0_Coord);"	\
		"}";
	glShaderSource(g_FragmentShaderObject_Square, 1, (const GLchar **)&fragmentShaderSourceCode_Square, NULL);
	//
	// Compile Source Code
	glCompileShader(g_FragmentShaderObject_Square);
	iInfoLogLength = 0;
	iShaderCompileStatus = 0;
	szInfoLog = NULL;
	glGetShaderiv(g_FragmentShaderObject_Square, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(g_FragmentShaderObject_Square, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(g_FragmentShaderObject_Square, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf_s(g_pFile, "Error : Fragment Shader Video Compilation Log : %s \n", szInfoLog);
				free(szInfoLog);
				return INIT_FRAGMENT_SHADER_COMPILATION_FAILED;
			}
		}
	}

	//***** Shader Program *****
	// Create
	g_ShaderProgramObject_Square = glCreateProgram();
	// Attach Vertex Shader
	glAttachShader(g_ShaderProgramObject_Square, g_VertexShaderObject_Square);
	// Attach Fragment Shader
	glAttachShader(g_ShaderProgramObject_Square, g_FragmentShaderObject_Square);
	// pre-link Program object with Vertex Sahder position attribute
	glBindAttribLocation(g_ShaderProgramObject_Square, 0, "vPosition");
	glBindAttribLocation(g_ShaderProgramObject_Square, 1, "vTexture0_Coord");

	// link Shader 
	glLinkProgram(g_ShaderProgramObject_Square);

	GLint iShaderProgramLinkStatus = 0;
	glGetProgramiv(g_ShaderProgramObject_Square, GL_LINK_STATUS, &iShaderProgramLinkStatus);
	szInfoLog = NULL;
	iInfoLogLength = 0;
	if (iShaderProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(g_ShaderProgramObject_Square, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (char*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(g_ShaderProgramObject_Square, GL_INFO_LOG_LENGTH, &written, szInfoLog);
				fprintf_s(g_pFile, "Error : Shader Program Link Log : %s \n", szInfoLog);
				free(szInfoLog);
				return INIT_LINK_SHADER_PROGRAM_FAILED;
			}
		}
	}

	//g_Uniform_ModelViewProjection = glGetUniformLocation(g_ShaderProgramObject_Square, "u_mvp_matrix");

	g_Uniform_Model_Matrix_Square = glGetUniformLocation(g_ShaderProgramObject_Square, "u_model_matrix");
	g_Uniform_View_Matrix_Square = glGetUniformLocation(g_ShaderProgramObject_Square, "u_view_matrix");
	g_Uniform_Projection_Matrix_Square = glGetUniformLocation(g_ShaderProgramObject_Square, "u_projection_matrix");


	//g_uniform_TextureSampler = glGetUniformLocation(g_ShaderProgramObject_Square, "u_texture0_sampler");

	// **** Verttices, Colors, Shader Attribs, Vbo, Vao Initializations ****

	//// Cube Section Start
	const GLfloat squareVertices[] = {
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,

	};

	const GLfloat squareTexCords[] =
	{
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f,0.0f,
		1.0f,1.0f

	};

	glGenVertexArrays(1, &g_VertexArrayObject_Square);
	glBindVertexArray(g_VertexArrayObject_Square);

	//vbo creation and binding for Square
	glGenBuffers(1, &g_VertexBufferObject_Position_Square);
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_Position_Square);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//Texture for Square
	glGenBuffers(1, &g_VertexBufferObject_Texture_Square);
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_Texture_Square);

	glBufferData(GL_ARRAY_BUFFER, sizeof(squareTexCords), squareTexCords, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	/*		Square Setup Stop			*/

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
	iInfoLogLength = 0;
	iShaderCompileStatus = 0;
	szInfoLog = NULL;
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
		"layout (location = 0)out vec4 FragColor;\n"	\
		"void main(void)"	\
		"{\n"	\
		"	FragColor = vec4(0.0, 1.0, 0.0, 0.5);\n"	\
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
	g_ShaderProgramObject_GPU_Fluid = glCreateProgram();
	glAttachShader(g_ShaderProgramObject_GPU_Fluid, iVertexShaderObject);
	glAttachShader(g_ShaderProgramObject_GPU_Fluid, iFragmentShaderObject);
	glBindAttribLocation(g_ShaderProgramObject_GPU_Fluid, SAM_ATTRIBUTE_POSITION, "vPosition");
	//glBindAttribLocation(g_ShaderProgramObject_GPU_Fluid, SAM_ATTRIBUTE_COLOR, "vColor");
	glLinkProgram(g_ShaderProgramObject_GPU_Fluid);

	iShaderLinkStatus = 0;
	iInfoLogLength = 0;
	glGetProgramiv(g_ShaderProgramObject_GPU_Fluid, GL_LINK_STATUS, &iShaderLinkStatus);
	if (iShaderLinkStatus == GL_FALSE)
	{
		glGetProgramiv(g_ShaderProgramObject_GPU_Fluid, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength>0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength * sizeof(GLchar));
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(g_ShaderProgramObject_GPU_Fluid, GL_INFO_LOG_LENGTH, &written, szInfoLog);
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
	g_Uniform_Model_Matrix_fluid = glGetUniformLocation(g_ShaderProgramObject_GPU_Fluid, "u_model_matrix");
	g_Uniform_Projection_Matrix_fluid = glGetUniformLocation(g_ShaderProgramObject_GPU_Fluid, "u_projection_matrix");
	g_Uniform_View_Matrix_fluid = glGetUniformLocation(g_ShaderProgramObject_GPU_Fluid, "u_view_matrix");
	/*Setup Uniforms End*/
	/// Sam : all Shader Code End

	/* Simmulation stuff start  */
	//GLint bsize;

	sdkCreateTimer(&g_Timer);
	sdkResetTimer(&g_Timer);

	/* GPU SetUP  Start*/
	hvField_GPU = (cData_float *)malloc(DS*sizeof(cData_float));
	memset(hvField_GPU, 0, sizeof(cData_float)*DS);
	if (hvField_GPU == NULL)
	{
		fprintf_s(g_pFile, "Memory allocation Failed...!!\n");
	}

	err = cudaMallocPitch((void**)&dvField, &tPitch, sizeof(cData_float)*DIM, DIM);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMallocPitch()\n");
	}

	err = cudaMemcpy(dvField, hvField_GPU, sizeof(cData_float)*DS, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMemcpy()\n");
	}

	// Temporary complex velocity field data
	err = cudaMalloc((void**)&xvField_GPU, sizeof(cData_float)*PDS);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMalloc()\n");
	}

	err = cudaMalloc((void**)&yvField_GPU, sizeof(cData_float)*PDS);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMalloc()\n");
	}

	setupTexture(DIM, DIM);
	bindTexture();

	// Create Particle Array
	particles = (cData_float*)malloc(DS*sizeof(cData_float));
	memset(particles, 0, sizeof(cData_float)*DS);
	if (particles == NULL)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaMalloc()\n");
	}

	initParticles(particles, DIM, DIM);

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

	glGenVertexArrays(1, &g_iVertexArrayObject_fluid_GPU);
	glBindVertexArray(g_iVertexArrayObject_fluid_GPU);

	glGenBuffers(1, &g_iVertexBufferObject_GPU);
	glBindBuffer(GL_ARRAY_BUFFER, g_iVertexBufferObject_GPU);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cData_float)*DS, particles, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	err = cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, g_iVertexBufferObject_GPU, cudaGraphicsMapFlagsNone);
	if (err != cudaSuccess)
	{
		fprintf_s(g_pFile, "CUDA Error : cudaGraphicsGLRegisterBuffer()\n");
	}
	/* GPU SetUP  Stop */

	/* CPU SetUP  Start*/
	hvField_CPU = (cData *)malloc(DS*sizeof(cData));
	if (hvField_CPU == NULL)
	{
		fprintf_s(g_pFile, "Memory allocation Failed...!!\n");
	}
	memset(hvField_CPU, 0, sizeof(cData)*DS);

	cpuBuffer = (cData *)malloc(DS * sizeof(cData));
	if (cpuBuffer == NULL)
	{
		fprintf_s(g_pFile, "Memory allocation Failed...!!\n");
	}
	memset(cpuBuffer, 0, sizeof(cData)*DS);

	// Temporary complex velocity field data
	xvField_CPU = (cData*)malloc(PDS*sizeof(cData));
	if (xvField_CPU == NULL)
	{
		fprintf_s(g_pFile, "Memory allocation Failed...!!\n");
	}
	memset(xvField_CPU, 0xff, sizeof(cData)*PDS);

	yvField_CPU = (cData*)malloc(PDS*sizeof(cData));
	if (yvField_CPU == NULL)
	{
		fprintf_s(g_pFile, "Memory allocation Failed...!!\n");
	}
	memset(yvField_CPU, 0xff, sizeof(cData)*PDS);

	particles_CPU = (cData*)malloc(DS*sizeof(cData));
	memset(particles_CPU, 0, sizeof(cData)*DS);
	if (particles_CPU == NULL)
	{
		fprintf_s(g_pFile, "Memory Error : cudaMalloc()\n");
	}

	initParticlesCPU(particles_CPU, DIM, DIM);

	memcpy_s(hvField_CPU, DS * sizeof(cData), particles_CPU, DS * sizeof(cData));
	memcpy_s(cpuBuffer, DS*sizeof(cData), particles_CPU, DS * sizeof(cData));
	memcpy_s(refTexArray, DIM*DIM * sizeof(cData), particles_CPU, DIM*DIM * sizeof(cData));
	//memcpy(hvField_CPU,particles,DS*sizeof(cData));

	glGenVertexArrays(1, &g_iVertexArrayObject_fluid_CPU);
	glBindVertexArray(g_iVertexArrayObject_fluid_CPU);

	glGenBuffers(1, &g_iVertexBufferObject_CPU);
	glBindBuffer(GL_ARRAY_BUFFER, g_iVertexBufferObject_CPU);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cData)*DS, particles_CPU, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION, 2, GL_DOUBLE, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	/* CPU SetUP  Stop */
	/* Simmulation stuff stop   */

	
	// Frame buffer Start
	glGenFramebuffers(1, &g_FrameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, g_FrameBuffer);

	glGenTextures(1, &g_ColorTexture);
	glBindTexture(GL_TEXTURE_2D, g_ColorTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1920, 1080, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA, 1920, 1080);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenTextures(1, &g_DepthTexture);
	glBindTexture(GL_TEXTURE_2D, g_DepthTexture);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, 1920, 1080);
	glBindTexture(GL_TEXTURE_2D, 0);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, g_ColorTexture, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, g_DepthTexture, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glDrawBuffers(1, g_DrawBuffers);
	// Frame buffer Stop


	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_CULL_FACE);

	glClearColor(1.0, 0.0, 1.0, 1.0f);


	g_PersPectiveProjectionMatrix = vmath::mat4::identity();
	g_OrthoProjectionMatrix = vmath::mat4::identity();

	Resize(WIN_WIDTH, WIN_HEIGHT);

	return INIT_ALL_OK;
}

void initParticles(cData_float *p, int dx, int dy)
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

void initParticlesCPU(cData *p, int dx, int dy)
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
	void SimulateFluids_GPU();
	void SimulateFluids_CPU();
	void computePerfStats(double &interactionsPerSecond, double &gflops,
		float milliseconds, int iterations);

	static double sd_gflops = 0;
	static double sd_ifps = 0;
	static double sd_interactionsPerSecond = 0;

	if (gpu_cpu_Switch) // ture -> gpu
	{
		_int64 currentTime = 0, lastTime = 0, frequency;
		float elapsedTime;

		double gpu_interactionsPerSecond = 0;
		double gpu_gflops = 0;

		QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);
		QueryPerformanceCounter((LARGE_INTEGER*)&lastTime);

		SimulateFluids_GPU();

		QueryPerformanceCounter((LARGE_INTEGER*)&currentTime);
		elapsedTime = ((float)(currentTime - lastTime) / (float)frequency) * 100.0f;

		computePerfStats(gpu_interactionsPerSecond, gpu_gflops, elapsedTime, 1);

		sprintf_s(tempString, "GPU Time: %03.3fms", elapsedTime);
		sprintf_s(flopString, "GPU GFLOPS: %03.3f", gpu_gflops);
	}
	else // false
	{
		_int64 currentTime = 0, lastTime = 0;
		double frequency;
		LARGE_INTEGER temp;
		float elapsedTime;

		double cpu_interactionsPerSecond = 0;
		double cpu_gflops = 0;

		QueryPerformanceFrequency((LARGE_INTEGER*)&temp);
		frequency = (static_cast<double>(temp.QuadPart)); // converting to required type
		QueryPerformanceCounter((LARGE_INTEGER*)&lastTime); // Start Time
															//Update in Cpu Logic


		// simulate CPU
		SimulateFluids_CPU();

		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&currentTime)); // Current Time
		elapsedTime = ((float)(currentTime - lastTime) / (float)frequency) * 100.0f;

		computePerfStats(cpu_interactionsPerSecond, cpu_gflops, elapsedTime, 1);

		sprintf_s(tempString, "CPU Time: %03.3f ms", elapsedTime);
		sprintf_s(flopString, "CPU GFLOPS: %03.3f", cpu_gflops);
	}

	
}

void computePerfStats(double &interactionsPerSecond, double &gflops,
	float milliseconds, int iterations)
{
	interactionsPerSecond = (float)DIM * (float)DIM * 6;
	interactionsPerSecond *= 1e-9 * iterations * 1000 / milliseconds;
	gflops = interactionsPerSecond * (float)g_iFlopsPerInteraction;
}


void SimulateFluids_GPU()
{

	fprintf_s(g_pFile, "\nStart Simmulation\n");
	fprintf_s(g_pFile, "++++++++++++++++++++++++++++++++++++\n");

	addvectVelocity(dvField, (float*)xvField_GPU, (float*)yvField_GPU, DIM, RPADW, DIM, DT);
	diffuseProject(xvField_GPU, yvField_GPU, CPADW, DIM, DT, VIS);
	updateVelocity(dvField, (float*)xvField_GPU, (float*)yvField_GPU, DIM, RPADW, DIM);
	advectParticles(g_iVertexBufferObject_GPU, dvField, DIM, DIM, DT);

	fprintf_s(g_pFile, "++++++++++++++++++++++++++++++++++++\n");
	fprintf_s(g_pFile, "Stop  Simmulation\n");
}


void Render(void)
{
	void RenderText(GLuint programObjFont, std::string text, GLfloat x, GLfloat y, GLfloat scale, vmath::vec3 color);


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	vmath::mat4 modelMatrix = vmath::mat4::identity();
	vmath::mat4 viewMatrix = vmath::mat4::identity();
	vmath::mat4 rotationMatrix = vmath::mat4::identity();
	vmath::mat4 scaleMatrix = vmath::mat4::identity();
	vmath::mat4 m4PersPectiveProjectionMatrix = vmath::perspective(45.0f, (float)g_fCurrrentWidth / (float)g_fCurrrentHeight, 0.1f, 100.0f);

	glViewport(0, 0, (GLsizei)g_fCurrrentWidth, (GLsizei)g_fCurrrentHeight);

	// render on Frame Buffer Start
	glBindFramebuffer(GL_FRAMEBUFFER, g_FrameBuffer);
	glViewport(0, 0, (GLsizei)g_fCurrrentWidth, (GLsizei)g_fCurrrentHeight);
	glClearBufferfv(GL_COLOR, 0, vmath::vec4(0.0f, 0.0f, 0.0f, 1.0f));
	glClearBufferfv(GL_DEPTH, 0, vmath::vec4(1.0f, 1.0f, 1.0f, 1.0f));
	
	// Fluid Rendering Start
	if (gpu_cpu_Switch == true) // gpu
	{
		glUseProgram(g_ShaderProgramObject_GPU_Fluid);

		//modelMatrix = vmath::translate(0.0f, 0.0f, -3.0f);

		glUniformMatrix4fv(g_Uniform_Model_Matrix_fluid, 1, GL_FALSE, modelMatrix);
		glUniformMatrix4fv(g_Uniform_View_Matrix_fluid, 1, GL_FALSE, viewMatrix);
		glUniformMatrix4fv(g_Uniform_Projection_Matrix_fluid, 1, GL_FALSE, g_OrthoProjectionMatrix);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


		glBindVertexArray(g_iVertexArrayObject_fluid_GPU);
		glDrawArrays(GL_POINTS, 0, DS);
		glBindVertexArray(0);

		glDisable(GL_BLEND);
		glUseProgram(0);
	}
	else // cpu
	{
		glUseProgram(g_ShaderProgramObject_GPU_Fluid);

		//modelMatrix = vmath::translate(0.0f, 0.0f, -3.0f);

		glUniformMatrix4fv(g_Uniform_Model_Matrix_fluid, 1, GL_FALSE, modelMatrix);
		glUniformMatrix4fv(g_Uniform_View_Matrix_fluid, 1, GL_FALSE, viewMatrix);
		glUniformMatrix4fv(g_Uniform_Projection_Matrix_fluid, 1, GL_FALSE, g_OrthoProjectionMatrix);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


		glBindVertexArray(g_iVertexArrayObject_fluid_CPU);

		glDrawArrays(GL_POINTS, 0, DS);
		glBindVertexArray(0);

		glDisable(GL_BLEND);
		glUseProgram(0);
	}

	glUseProgram(0);
	// Fluid Rendering Stop

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	// render on Frame Buffer Stop	

	modelMatrix = vmath::mat4::identity();
	viewMatrix = vmath::mat4::identity();
	rotationMatrix = vmath::mat4::identity();
	scaleMatrix = vmath::mat4::identity();
	// Draw Quad  g_ColorTexture
	glUseProgram(g_ShaderProgramObject_Square);

	modelMatrix = vmath::translate(0.0f, 0.0f, -80.00f);
	modelMatrix = modelMatrix * vmath::scale(58.9f, 33.3f, 0.0f);
	glUniformMatrix4fv(g_Uniform_Model_Matrix_Square, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(g_Uniform_View_Matrix_Square, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(g_Uniform_Projection_Matrix_Square, 1, GL_FALSE, g_PersPectiveProjectionMatrix);

	//glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, g_ColorTexture);

	// Full Screen Quad
	glBindVertexArray(g_VertexArrayObject_Square);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);
	glUseProgram(0);
	

	// Font Renderind Start
	glUseProgram(g_ShaderProgramObject_Font);

	modelMatrix = vmath::translate(0.0f, 0.0f, -78.00f);
	//modelMatrix = modelMatrix * vmath::scale(0.30f, 0.30f, 0.30f);
	glBindVertexArray(g_VertexArrayObject_font);
	glUniformMatrix4fv(g_Uniform_Model_Matrix_font, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(g_Uniform_View_Matrix_font, 1, GL_FALSE, viewMatrix);
	glUniformMatrix4fv(g_Uniform_Projection_Matrix_font, 1, GL_FALSE, g_PersPectiveProjectionMatrix);
	if (gpu_cpu_Switch == true)
	{
		RenderText(g_ShaderProgramObject_Font, "GPU", 48.0f, 28.0f, 0.032f, vmath::vec3(1.0f, 1.0f, 0.0f));// vmath::vec3(0.49f, 0.76f, 0.0f)
	}
	else
	{
		RenderText(g_ShaderProgramObject_Font, "CPU", 38.0f, 28.0f, 0.032f, vmath::vec3(1.0f, 1.0f, 0.0f));//vmath::vec3(0.0f, 0.46f, 0.9f)
	}

	RenderText(g_ShaderProgramObject_Font, flopString, -57.0f, -27.0f, 0.028f, vmath::vec3(1.0f, 1.0f, 0.0f));
	RenderText(g_ShaderProgramObject_Font, tempString, -57.0f, -30.0f, 0.028f, vmath::vec3(1.0f, 1.0f, 0.0f));
	RenderText(g_ShaderProgramObject_Font, "Press 'H' for toggle", 18.0f, 24.0f, 0.032f, vmath::vec3(1.0f, 1.0f, 0.0f));

	glBindVertexArray(0);

	glUseProgram(0);
	// Font rendering Stop 


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
	g_OrthoProjectionMatrix = vmath::ortho(0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f);

	if (g_FrameBuffer)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, g_FrameBuffer);
		if (g_ColorTexture)
		{
			glDeleteTextures(1, &g_ColorTexture);
			g_ColorTexture = 0;
			glGenTextures(1, &g_ColorTexture);
			glBindTexture(GL_TEXTURE_2D, g_ColorTexture);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, iWidth, iHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTextureStorage2D(GL_TEXTURE_2D, 1, GL_RGBA, iWidth, iHeight);
			glBindTexture(GL_TEXTURE_2D, 0);
		}

		if (g_DepthTexture)
		{
			glDeleteTextures(1, &g_DepthTexture);
			g_DepthTexture = 0;
			glGenTextures(1, &g_DepthTexture);
			glBindTexture(GL_TEXTURE_2D, g_DepthTexture);
			glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, iWidth, iHeight);
			glBindTexture(GL_TEXTURE_2D, 0);
		}

		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, g_ColorTexture, 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, g_DepthTexture, 0);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	return true;
}

int UnInitialize(void)
{

	std::map<GLchar, Character>::iterator it = Characters.begin();
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
	if (hvField_GPU)
	{
		free(hvField_GPU);
		hvField_GPU = NULL;
	}

	if (hvField_CPU)
	{
		free(hvField_CPU);
		hvField_CPU = NULL;
	}
	
	if (particles)
	{
		free(particles);
		particles = NULL;
	}

	if (particles_CPU)
	{
		free(particles_CPU);
		particles_CPU = NULL;
	}
	
	if (dvField)
	{
		cudaFree(dvField);
		dvField = NULL;
	}

	if (xvField_GPU)
	{
		cudaFree(xvField_GPU);
		xvField_GPU = NULL;
	}
	
	if (yvField_GPU)
	{
		cudaFree(yvField_GPU);
		yvField_GPU = NULL;
	}

	if (xvField_CPU)
	{
		cudaFree(xvField_CPU);
		xvField_CPU = NULL;
	}

	if (yvField_CPU)
	{
		cudaFree(yvField_CPU);
		yvField_CPU = NULL;
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

	while (it != Characters.end())
	{
		glDeleteTextures(1, &it->second.TexyureID);
		it->second.TexyureID = 0;
		it++;
	}

	if (g_ColorTexture)
	{
		glDeleteTextures(1, &g_ColorTexture);
		g_ColorTexture = 0;
	}

	if (g_DepthTexture)
	{
		glDeleteTextures(1, &g_DepthTexture);
		g_DepthTexture = 0;
	}

	if (g_FrameBuffer)
	{
		glDeleteFramebuffers(1, &g_FrameBuffer);
		g_FrameBuffer = 0;
	}

	if (g_VertexBufferObject_Texture_Square)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_Texture_Square);
		g_VertexBufferObject_Texture_Square = NULL;
	}

	if (g_VertexBufferObject_Position_Square)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_Position_Square);
		g_VertexBufferObject_Position_Square = NULL;
	}

	if (g_VertexArrayObject_Square)
	{
		glDeleteVertexArrays(1, &g_VertexArrayObject_Square);
		g_VertexArrayObject_Square = NULL;
	}

	if (g_VertexBufferObject_Position_font)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_Position_font);
		g_VertexBufferObject_Position_font = NULL;
	}

	if (g_VertexArrayObject_font)
	{
		glDeleteVertexArrays(1, &g_VertexArrayObject_font);
		g_VertexArrayObject_font = NULL;
	}

	if (g_iVertexBufferObject_GPU)
	{
		glDeleteVertexArrays(1, &g_iVertexBufferObject_GPU);
		g_iVertexBufferObject_GPU = NULL;
	}

	if (g_VertexBufferObject_Color)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_Color);
		g_VertexBufferObject_Color = NULL;
	}

	if (g_VertexBufferObject_Position_GPU_Fluid)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_Position_GPU_Fluid);
		g_VertexBufferObject_Position_GPU_Fluid = NULL;
	}

	if (g_VertexArrayObject_GPU_Fluid)
	{
		glDeleteVertexArrays(1, &g_VertexArrayObject_GPU_Fluid);
		g_VertexArrayObject_GPU_Fluid = NULL;
	}

	if (g_iVertexBufferObject_CPU)
	{
		glDeleteBuffers(1, &g_iVertexBufferObject_CPU);
		g_iVertexBufferObject_CPU = NULL;
	}

	if (g_iVertexArrayObject_fluid_CPU)
	{
		glDeleteVertexArrays(1, &g_iVertexArrayObject_fluid_CPU);
		g_iVertexArrayObject_fluid_CPU = NULL;
	}

	if (g_Timer)
	{
		sdkDeleteTimer(&g_Timer);
		g_Timer = NULL;
	}
	

	glUseProgram(0);
	//g_ShaderProgramObject_Font
	if (g_ShaderProgramObject_Font)
	{
		GLsizei iShaderCount;
		GLsizei iShaderNumber;

		glUseProgram(g_ShaderProgramObject_Font);
		glGetProgramiv(g_ShaderProgramObject_Font, GL_ATTACHED_SHADERS, &iShaderCount);
		GLuint *pShaders = (GLuint*)calloc(iShaderCount, sizeof(GLuint));

		if (pShaders)
		{
			glGetAttachedShaders(g_ShaderProgramObject_Font, iShaderCount, &iShaderCount, pShaders);
			for (iShaderNumber = 0; iShaderNumber < iShaderCount; iShaderNumber++)
			{
				glDetachShader(g_ShaderProgramObject_Font, pShaders[iShaderNumber]);
				glDeleteShader(pShaders[iShaderNumber]);
				pShaders[iShaderNumber] = 0;
			}
			free(pShaders);
			pShaders = NULL;
		}

		glDeleteProgram(g_ShaderProgramObject_Font);
		g_ShaderProgramObject_Font = NULL;

		glUseProgram(0);

	}

	//g_ShaderProgramObject_Square
	if (g_ShaderProgramObject_Square)
	{
		GLsizei iShaderCount;
		GLsizei iShaderNumber;

		glUseProgram(g_ShaderProgramObject_Square);
		glGetProgramiv(g_ShaderProgramObject_Square, GL_ATTACHED_SHADERS, &iShaderCount);
		GLuint *pShaders = (GLuint*)calloc(iShaderCount, sizeof(GLuint));

		if (pShaders)
		{
			glGetAttachedShaders(g_ShaderProgramObject_Square, iShaderCount, &iShaderCount, pShaders);
			for (iShaderNumber = 0; iShaderNumber < iShaderCount; iShaderNumber++)
			{
				glDetachShader(g_ShaderProgramObject_Square, pShaders[iShaderNumber]);
				glDeleteShader(pShaders[iShaderNumber]);
				pShaders[iShaderNumber] = 0;
			}
			free(pShaders);
			pShaders = NULL;
		}

		glDeleteProgram(g_ShaderProgramObject_Square);
		g_ShaderProgramObject_Square = NULL;

		glUseProgram(0);
	}

	//g_ShaderProgramObject_GPU_Fluid
	if (g_ShaderProgramObject_GPU_Fluid)
	{
		GLsizei iShaderCount;
		GLsizei iShaderNumber;

		glUseProgram(g_ShaderProgramObject_GPU_Fluid);
		glGetProgramiv(g_ShaderProgramObject_GPU_Fluid, GL_ATTACHED_SHADERS, &iShaderCount);
		GLuint *pShaders = (GLuint*)calloc(iShaderCount, sizeof(GLuint));
		if (pShaders)
		{
			glGetAttachedShaders(g_ShaderProgramObject_GPU_Fluid, iShaderCount, &iShaderCount, pShaders);
			for (iShaderNumber = 0; iShaderNumber < iShaderCount; iShaderNumber++)
			{
				glDetachShader(g_ShaderProgramObject_GPU_Fluid, pShaders[iShaderNumber]);
				glDeleteShader(pShaders[iShaderNumber]);
				pShaders[iShaderNumber] = 0;
			}
			free(pShaders);
			pShaders = NULL;
		}
		glUseProgram(0);

		glDeleteProgram(g_ShaderProgramObject_GPU_Fluid);
		g_ShaderProgramObject_GPU_Fluid = NULL;
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

void RenderText(GLuint programObjFont, std::string text, GLfloat x, GLfloat y, GLfloat scale, vmath::vec3 color)
{
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//glUseProgram(programObjFont);
	glUniform3f(glGetUniformLocation(programObjFont, "textColor"), color[0], color[1], color[2]);
	glActiveTexture(GL_TEXTURE0);

	glBindVertexArray(g_VertexArrayObject_font);
	// Iterate through all characters
	std::string::const_iterator c;

	fprintf_s(g_pFile, "FreeType : x = %f, y = %f, scale = %f\n", x, y, scale);

	for (c = text.begin(); c != text.end(); c++)
	{
		Character ch = Characters[*c];

		GLfloat xpos = x + ch.Bearing[0] * scale;
		GLfloat ypos = y - (ch.Size[1] - ch.Bearing[1]) * scale;

		GLfloat w = ch.Size[0] * scale;
		GLfloat h = ch.Size[1] * scale;

		fprintf_s(g_pFile, "FreeType : char = %c \n", *c);
		fprintf_s(g_pFile, "FreeType : xpos = %f, ypos = %f, w = %f , h = %f \n", xpos, ypos, w, h);

		// Update VBO for each character
		GLfloat vertices[6][4] = {
			{ xpos,     ypos + h,   0.0, 0.0 },
			{ xpos,     ypos,       0.0, 1.0 },
			{ xpos + w, ypos,       1.0, 1.0 },

			{ xpos,     ypos + h,   0.0, 0.0 },
			{ xpos + w, ypos,       1.0, 1.0 },
			{ xpos + w, ypos + h,   1.0, 0.0 }
		};

		for (int i = 0; i < 6; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				fprintf_s(g_pFile, "FreeType : xpos, ypos = %f \n", vertices[i][j]);
			}
		}
		// Render glyph texture over quad
		glBindTexture(GL_TEXTURE_2D, ch.TexyureID);
		// Update content of VBO memory
		glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_Position_font);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		// Render quad
		glDrawArrays(GL_TRIANGLES, 0, 6);
		// Now advance cursors for next glyph (note that advance is number of 1/64 pixels)
		x += (ch.Advance >> 6) * scale; // Bitshift by 6 to get value in pixels (2^6 = 64)
	}
	glBindVertexArray(0);
	//glUseProgram(0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_BLEND);
}

/** CPU Simulation Start **/
void addForces_CPU(cData *v, int dx, int dy, int spx, int spy, float fx, float fy, int r)
{

	int totalWidth = 2 * r + 1;
	int totalHeight = 2 * r + 1;

	for (int i = 0; i < totalWidth; i++)
	{
		for (int j = 0; j < totalHeight; j++)
		{
			int tx = i;
			int ty = j;

			cData *fj = (cData*)((char*)v + (ty + spy)*tPitch_CPU) + tx + spx;

			cData vterm = *fj;
			tx -= r;
			ty -= r;
			float s = 1.0f / (1.0f + tx*tx*tx*tx + ty*ty*ty*ty);
			vterm.x += s*fx;
			vterm.y += s*fy;
			*fj = vterm;

		}
	}
}

void SimulateFluids_CPU()
{
	void addvectVelocity_CPU(cData *v, double *vx, double *vy, int dx, int pdx, int dy, float dt);

	void diffuseProject_CPU(cData *vx, cData *vy, int dx, int dy, float dt, float visc);
	void updateVelocity_CPU(cData *v, double *vx, double *vy, int dx, int pdx, int dy);
	void advectParticles_CPU(GLuint vbo, cData * v, int dx, int dy, float dt);


	addvectVelocity_CPU(&hvField_CPU[0], (double*)xvField_CPU, (double*)yvField_CPU, DIM, RPADW, DIM, DT);
	diffuseProject_CPU(xvField_CPU, yvField_CPU, CPADW, DIM, DT, VIS);
	updateVelocity_CPU(hvField_CPU, (double*)xvField_CPU, (double*)yvField_CPU, DIM, RPADW, DIM);
	advectParticles_CPU(g_iVertexBufferObject_CPU, hvField_CPU, DIM, DIM, DT);
}

void addvectVelocity_CPU(cData *v, double *vx, double *vy, int dx, int pdx, int dy, float dt)
{

	int blockDim_x = (dx / TILEX) + (!(dx%TILEX) ? 0 : 1);
	int blockDim_y = (dy / TILEY) + (!(dy%TILEY) ? 0 : 1);
	int lb = TILEY / -TIDSY;

	int p;
	cData vterm, ploc;
	double vxterm, vyterm;

	// update 2d array with hvField_CPU
	// Update texture
	memcpy_s(refTexArray, DIM*DIM*sizeof(cData), v, DIM*DIM*sizeof(cData));
	//memcpy(refTexArray,&hvField_CPU[0],DIM*DIM*sizeof(cData));

	// grid calculation
	for (int blockId_x = 0; blockId_x < blockDim_x; blockId_x++)
	{
		for (int blockId_y = 0; blockId_y < blockDim_y; blockId_y++)
		{
			// thread calculation
			for (int threadId_x = 0; threadId_x < TIDSX; threadId_x++)
			{
				for (int threadId_y = 0; threadId_y < TIDSY; threadId_y++)
				{
					int gtidx = blockId_x * blockDim_x + threadId_x;
					int gtidy = blockId_y * (lb*blockDim_y) + lb*threadId_y;

					if (gtidx < dx)
					{
						for (p = 0; p < lb; p++)
						{
							int fi = gtidy + p;

							if (fi < dy)
							{
								int fj = fi * pdx + gtidx;
								vterm = refTexArray[gtidx][fi];
								ploc.x = (gtidx + 0.5f) - (dt * vterm.x * dx);
								ploc.y = (fi + 0.5f) - (dt * vterm.y * dy);
								//vterm = refTexArray[gtidx][fi];
								vxterm = vterm.x;
								vyterm = vterm.y;
								vx[fj] = vxterm;
								vy[fj] = vyterm;
							}
						}
					}

				}//threadId_y
			}//threadId_x

		}//blockId_y
	}//blockId_x

}


void diffuseProject_CPU(cData * vx, cData * vy, int dx, int dy, float dt, float visc)
{

	int p;
	cData xterm, yterm;
	//int cufftErrs;

	int blockDim_x = (dx / TILEX) + (!(dx%TILEX) ? 0 : 1);
	int blockDim_y = (dy / TILEY) + (!(dy%TILEY) ? 0 : 1);
	int lb = TILEY / -TIDSY;

	fftw_plan forward_x = fftw_plan_dft_2d(dx, dy, (fftw_complex*)vx, (fftw_complex*)vx, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan forward_y = fftw_plan_dft_2d(dx, dy, (fftw_complex*)vy, (fftw_complex*)vy, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_plan inverse_x = fftw_plan_dft_2d(dx, dy, (fftw_complex*)vx, (fftw_complex*)vx, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_plan inverse_y = fftw_plan_dft_2d(dx, dy, (fftw_complex*)vy, (fftw_complex*)vy, FFTW_BACKWARD, FFTW_ESTIMATE);

	// FFT r2c vx
	fftw_execute(forward_x);
	fftw_execute(forward_y);

	//Loops
	// grid calculation
	for (int blockId_x = 0; blockId_x < blockDim_x; blockId_x++)
	{
		for (int blockId_y = 0; blockId_y < blockDim_y; blockId_y++)
		{
			// thread calculation
			for (int threadId_x = 0; threadId_x < TIDSX; threadId_x++)
			{
				for (int threadId_y = 0; threadId_y < TIDSY; threadId_y++)
				{
					int gtidx = blockId_x * blockDim_x + threadId_x;
					int gtidy = blockId_y * (lb*blockDim_y) + (lb*threadId_y);

					if (gtidx < dx)
					{
						for (p = 0; p < lb; p++)
						{
							int fi = gtidy + p;

							if (fi < dy)
							{
								int fj = fi * dx + gtidx;
								xterm = vx[fj];
								yterm = vy[fj];

								int iix = gtidx;
								int iiy = (fi>dy / 2) ? (fi - (dy)) : fi;

								// Velocity diffusion
								float kk = (float)(iix * iix + iiy * iiy); // k^2
								float diff = 1.0f / (1.0f + visc * dt * kk);
								xterm.x *= diff;
								xterm.y *= diff;
								yterm.x *= diff;
								yterm.y *= diff;

								if (kk > 0.0f)
								{
									float rkk = 1.0f / kk;
									// Real portion of velocity projection
									double rkp = (iix * xterm.x + iiy * yterm.x);
									// Imaginary portion of velocity projection
									double ikp = (iix * xterm.y + iiy * yterm.y);
									xterm.x -= rkk * rkp * iix;
									xterm.y -= rkk * ikp * iix;
									yterm.x -= rkk * rkp * iiy;
									yterm.y -= rkk * ikp * iiy;
								}

								vx[fj] = xterm;
								vy[fj] = yterm;
							}
						}
					}

				}//threadId_y
			}//threadId_x

		}//blockId_y
	}//blockId_x

	 // FFT c2r vx
	fftw_execute(inverse_x);
	fftw_execute(inverse_y);

	fftw_destroy_plan(forward_x);
	fftw_destroy_plan(forward_y);
	fftw_destroy_plan(inverse_x);
	fftw_destroy_plan(inverse_y);

}

void updateVelocity_CPU(cData *v, double *vx, double *vy, int dx, int pdx, int dy)
{
	double vxterm, vyterm;
	cData nvterm;

	int blockDim_x = (dx / TILEX) + (!(dx%TILEX) ? 0 : 1);
	int blockDim_y = (dy / TILEY) + (!(dy%TILEY) ? 0 : 1);
	int lb = TILEY / -TIDSY;

	int p;

	//Loops
	// grid calculation
	for (int blockId_x = 0; blockId_x < blockDim_x; blockId_x++)
	{
		for (int blockId_y = 0; blockId_y < blockDim_y; blockId_y++)
		{
			// thread calculation
			for (int threadId_x = 0; threadId_x < TIDSX; threadId_x++)
			{
				for (int threadId_y = 0; threadId_y < TIDSY; threadId_y++)
				{
					int gtidx = blockId_x * blockDim_x + threadId_x;
					int gtidy = blockId_y * lb*blockDim_y + (lb*threadId_y);

					if (gtidx < dx)
					{
						for (p = 0; p < lb; p++)
						{
							int fi = gtidy + p;

							if (fi < dy)
							{
								int fjr = fi * pdx + gtidx;
								vxterm = vx[fjr];
								vyterm = vy[fjr];

								// Normalize the result of the inverse FFT
								double scale = 1.0 / (dx * dy);
								nvterm.x = vxterm * scale;
								nvterm.y = vyterm * scale;

								cData *fj = (cData*)((char*)v + fi * tPitch_CPU) + gtidx;
								*fj = nvterm;
							}
						}
					}

				}//threadId_y
			}//threadId_x

		}//blockId_y
	}//blockId_x

}

void advectParticles_CPU(GLuint vbo, cData * v, int dx, int dy, float dt)
{
	//(p,v,dx,dy,dt,TILEY/TIDSY,tPitch_CPU)
	cData pterm, vterm;
	int blockDim_x = (dx / TILEX) + (!(dx%TILEX) ? 0 : 1);
	int blockDim_y = (dy / TILEY) + (!(dy%TILEY) ? 0 : 1);
	int lb = TILEY / -TIDSY;

	int p;

	//Loops  cpuBuffer
	// grid calculation
	for (int blockId_x = 0; blockId_x < blockDim_x; blockId_x++)
	{
		for (int blockId_y = 0; blockId_y < blockDim_y; blockId_y++)
		{
			// thread calculation
			for (int threadId_x = 0; threadId_x < TIDSX; threadId_x++)
			{
				for (int threadId_y = 0; threadId_y < TIDSY; threadId_y++)
				{
					int gtidx = blockId_x * blockDim_x + threadId_x;
					int gtidy = blockId_y * lb*blockDim_y + lb*threadId_y;

					if (gtidx < dx)
					{
						for (p = 0; p < lb; p++)
						{
							int fi = gtidy + p;

							if (fi < dy)
							{
								int fj = fi * dx + gtidx;
								pterm = v[fj];

								int xvi = ((int)(pterm.x * dx));
								int yvi = ((int)(pterm.y * dy));
								vterm = *((cData *)((char *)v + yvi * tPitch_CPU) + xvi);

								pterm.x += dt * vterm.x;
								pterm.x = pterm.x - (int)pterm.x;
								pterm.x += 1.0f;
								pterm.x = pterm.x - (int)pterm.x;
								pterm.y += dt * vterm.y;
								pterm.y = pterm.y - (int)pterm.y;
								pterm.y += 1.0f;
								pterm.y = pterm.y - (int)pterm.y;

								v[fj] = pterm;
							}
						}
					}

				}//threadId_y
			}//threadId_x

		}//blockId_y
	}//blockId_x

	glBindVertexArray(g_iVertexArrayObject_fluid_CPU);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cData)*DIM*DIM, v, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

}

/** CPU Simulation Stop  **/