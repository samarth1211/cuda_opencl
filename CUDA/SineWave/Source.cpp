

#include "CommonHeader.h"
#include "vmath.h"
#include "Camera.h"
#include "SimpleGL.cuh"

#include<iostream>
#include <iterator> 
#include <map>
#include <ft2build.h>
#include FT_FREETYPE_H


#define WIN_WIDTH		800
#define WIN_HEIGHT		600
#define DELTA			0.0166666666666667f


#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib,"cudart.lib")
#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"freetype.lib")

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
GLuint g_ShaderProgramObject = 0;
GLuint g_ShaderProgramObject_Font = 0;
GLuint g_ShaderProgramObject_Square;

// All Vertex Buffers
GLuint g_VertexArrayObject_wave = 0;
GLuint g_VertexBufferObject_Position_wave = 0;

GLuint g_VertexArrayObject_Quad = 0;
GLuint g_VertexBufferObject_Position_Quad = 0;
GLuint g_VertexBufferObject_TexCoords_Quad = 0;

GLuint g_VertexArrayObject_FB = 0;
GLuint g_VertexBufferObject_Position_FB = 0;
GLuint g_VertexBufferObject_TexCoords_FB = 0;

GLuint g_VertexArrayObject_Square;
GLuint g_VertexBufferObject_Position_Square;
GLuint g_VertexBufferObject_Texture_Square;

// Uniforms
GLuint g_Uniform_Model_Matrix = 0;
GLuint g_Uniform_View_Matrix = 0;
GLuint g_Uniform_Projection_Matrix = 0;
GLuint g_Uniform_Color = 0;
GLuint g_Uniform_Selection_Mode = 0;

GLuint g_Uniform_Model_Matrix_Quad = 0;
GLuint g_Uniform_View_Matrix_Quad = 0;
GLuint g_Uniform_Projection_Matrix_Quad = 0;

GLuint g_Uniform_Model_Matrix_FB = 0;
GLuint g_Uniform_View_Matrix_FB = 0;
GLuint g_Uniform_Projection_Matrix_FB = 0;

GLuint g_Uniform_Model_Matrix_Square;
GLuint g_Uniform_View_Matrix_Square;
GLuint g_Uniform_Projection_Matrix_Square;


// sampler
GLuint g_uniform_TextureSampler = 0;
GLuint g_uniform_TextureSampler_FB=0;


// Projection
vmath::mat4 g_PersPectiveProjectionMatrix;

// Camera Keys
Camera camera(vmath::vec3(0.0f, 0.0f, -1.5f));
GLfloat g_fLastX = WIN_WIDTH / 2;
GLfloat g_fLastY = WIN_HEIGHT / 2;

GLfloat g_DeltaTime = 0.0f;
GLboolean g_bFirstMouse = true;
GLfloat g_fCurrrentWidth;
GLfloat g_fCurrrentHeight;

int mesh_height = MESH_HEIGHT;
int mesh_width = MESH_WIDTH;

float g_fAnime = 0.0f;

struct cudaGraphicsResource *g_cuda_vbo_resource;

int iSelectionMode = 0;
int iCurrentNumBytes = 0;

GLfloat fvPointColors[3] = {0.6f, 0.2f, 0.0f};
unsigned int bufferSize = 128 * 128 * 4 * sizeof(float) +
							256 * 256 * 4 * sizeof(float) +
							512 * 512 * 4 * sizeof(float) +
							1024 * 1024 * 4 * sizeof(float) +
							2048 * 2048 * 4 * sizeof(float) +
							4096 * 4096 * 4 * sizeof(float);

GLfloat *cpuLocation = NULL;
bool gpu_cpu_Switch = false;// default on cpu

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

// FLOP Calculation Start
int g_iFlopsPerInteraction = 20;
// FLOP Calculation Stop

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

		case VK_F1:
			g_bShowHelp = (g_bShowHelp) ? false : true;
			break;

		
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		case 0x50: // 'p' or 'P'
			fprintf_s(g_pFile, " SAMA : z Position => %f\n", zDirn);
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
		case 0x46: // 'f' or 'F'
			//MessageBox(hwnd, TEXT("F is pressed"), TEXT("Status"), MB_OK);
			FullScreen();
			break;
			// cases for rgb//fvPointColors

		case 0x48: // 'h' or 'H'
			gpu_cpu_Switch = (gpu_cpu_Switch) ? false : true;
			break;

		case 0x52://R
			fvPointColors[0] = 1.0f;
			fvPointColors[1] = 0.0f;
			fvPointColors[2] = 0.0f;
			break;
		case 0x47://G
			fvPointColors[0] = 0.0f;
			fvPointColors[1] = 1.0f;
			fvPointColors[2] = 0.0f;
			break;
		case 0x42://B
			fvPointColors[0] = 0.0f;
			fvPointColors[1] = 0.0f;
			fvPointColors[2] = 1.0f;
			break;

		case 0x43://Cyan
			fvPointColors[0] = 0.0f;
			fvPointColors[1] = 1.0f;
			fvPointColors[2] = 1.0f;
			break;

		case 0x59://Yellow
			fvPointColors[0] = 1.0f;
			fvPointColors[1] = 1.0f;
			fvPointColors[2] = 0.0f;
			break;

		case 0x4D://Magenta
			fvPointColors[0] = 1.0f;
			fvPointColors[1] = 0.0f;
			fvPointColors[2] = 1.0f;
			break;

		case 0x4B://White=>k
			fvPointColors[0] = 1.0f;
			fvPointColors[1] = 1.0f;
			fvPointColors[2] = 1.0f;
			break;
			case 0x4F://o
				fvPointColors[0] = 1.0f;
				fvPointColors[1] = 0.65f;
				fvPointColors[2] = 0.0f;
				break;
		case 0x31: //1
			iSelectionMode = 0;
			mesh_height = 128;
			mesh_width = 128;
			break;
		case 0x32: //2
			iSelectionMode = 1;
			mesh_height = 256;
			mesh_width = 256;
			break;
		case 0x33: //3
			iSelectionMode =2;
			mesh_height = 512;
			mesh_width = 512;
			break;
		case 0x34: //4
			iSelectionMode = 3;
			mesh_height = 1024;
			mesh_width = 1024;
			break;
		case 0x35: //5
			iSelectionMode = 4;
			mesh_height = 2048;
			mesh_width = 2048;
			break;
		case 0x36: //6
			iSelectionMode = 5;
			mesh_height = 4096;
			mesh_width = 4096;
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
	int iPixelIndex = 0;
	PIXELFORMATDESCRIPTOR pfd;

	// Shader Programs
	GLuint iVertexShaderObject = 0;
	GLuint iFragmentShaderObject = 0;

	GLuint iVertexShaderObject_Font = 0;
	GLuint iFragmentShaderObject_Font = 0;

	GLuint g_VertexShaderObject_Square;
	GLuint g_FragmentShaderObject_Square;

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

	/// Sam : all Shader Code Start

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
	glBindAttribLocation(g_ShaderProgramObject_Font, SAM_ATTRIBUTE_POSITION0, "vVertex");

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

	/*		Cube Setup Start		*/
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
	/*		Cube Setup Stop			*/

	/*Vertex Shader Start*/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
	
	const GLchar *vertexShaderSourceCode = "#version 450 core"	\
		"\n" \
		"layout (location = 0)in vec4 vPosition0;\n" \
		"layout (location = 1)in vec4 vPosition1;\n" \
		"layout (location = 2)in vec4 vPosition2;\n" \
		"layout (location = 3)in vec4 vPosition3;\n" \
		"layout (location = 4)in vec4 vPosition4;\n" \
		"layout (location = 5)in vec4 vPosition5;\n" \
		"uniform mat4 u_model_matrix,u_view_matrix,u_projection_matrix;\n" \
		"uniform int iSelMode;\n"	\
		"void main(void)" \
		"{\n" \
		"	switch(iSelMode)\n"	\
		"	{"	\
		"		case 0:"	\
		"			gl_Position =  u_projection_matrix * u_view_matrix * u_model_matrix * vPosition0;"	\
		"		break;\n"	\

		"		case 1:"	\
		"			gl_Position =  u_projection_matrix * u_view_matrix * u_model_matrix * vPosition1;"	\
		"		break;\n"	\

		"		case 2:"	\
		"			gl_Position =  u_projection_matrix * u_view_matrix * u_model_matrix * vPosition2;"	\
		"		break;\n"	\

		"		case 3:"	\
		"			gl_Position =  u_projection_matrix * u_view_matrix * u_model_matrix * vPosition3;"	\
		"		break;\n"	\

		"		case 4:"	\
		"			gl_Position =  u_projection_matrix * u_view_matrix * u_model_matrix * vPosition4;"	\
		"		break;\n"	\

		"		case 5:"	\
		"			gl_Position =  u_projection_matrix * u_view_matrix * u_model_matrix * vPosition5;"	\
		"		break;\n"	\

		"	}"	\
		"}";

	glShaderSource(iVertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);

	// Compile it
	glCompileShader(iVertexShaderObject);
	iInfoLogLength = 0;
	iShaderCompileStatus = 0;
	szInfoLog = NULL;
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
		"layout (location = 0)out vec4 FragColor;"	\
		"uniform vec3 u_color;" \
		"void main(void)"	\
		"{"	\
		"	FragColor = vec4(u_color.r,u_color.g,u_color.b,1.0);"	\
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
	glBindAttribLocation(g_ShaderProgramObject, 0, "vPosition0");
	glBindAttribLocation(g_ShaderProgramObject, 1, "vPosition1");
	glBindAttribLocation(g_ShaderProgramObject, 2, "vPosition2");
	glBindAttribLocation(g_ShaderProgramObject, 3, "vPosition3");
	glBindAttribLocation(g_ShaderProgramObject, 4, "vPosition4");
	glBindAttribLocation(g_ShaderProgramObject, 5, "vPosition5");
	//glBindAttribLocation(g_ShaderProgramObject, 1, "vColor");
	glLinkProgram(g_ShaderProgramObject);

	iShaderLinkStatus = 0;
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

	/*Setup Uniforms Start*/ // 
	g_Uniform_Model_Matrix = glGetUniformLocation(g_ShaderProgramObject,"u_model_matrix");
	g_Uniform_Projection_Matrix = glGetUniformLocation(g_ShaderProgramObject, "u_projection_matrix");
	g_Uniform_View_Matrix = glGetUniformLocation(g_ShaderProgramObject, "u_view_matrix");
	g_Uniform_Selection_Mode = glGetUniformLocation(g_ShaderProgramObject, "iSelMode");
	g_Uniform_Color = glGetUniformLocation(g_ShaderProgramObject, "u_color");
	/*Setup Uniforms End*/

	/* Fill Buffers Start*/

	glGenBuffers(1, &g_VertexArrayObject_wave);//VAO
	glBindVertexArray(g_VertexArrayObject_wave);

	glGenBuffers(1, &g_VertexBufferObject_Position_wave);// vbo position
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_Position_wave);
	glBufferData(GL_ARRAY_BUFFER, bufferSize, NULL, GL_STREAM_DRAW);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(0) );
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION0);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(128 * 128 * 4 * sizeof(float)));
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION1);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(128 * 128 * 4 * sizeof(float) +
																										256 * 256 * 4 * sizeof(float) ));
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION2);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION3, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(128 * 128 * 4 * sizeof(float) +
																										256 * 256 * 4 * sizeof(float) +
																										512 * 512 * 4 * sizeof(float) ));
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION3);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION4, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(128 * 128 * 4 * sizeof(float) +
																										256 * 256 * 4 * sizeof(float) +
																										512 * 512 * 4 * sizeof(float) +
																										1024 * 1024 * 4 * sizeof(float)  ));
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION4);

	glVertexAttribPointer(SAM_ATTRIBUTE_POSITION5, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(128 * 128 * 4 * sizeof(float) +
																										256 * 256 * 4 * sizeof(float) +
																										512 * 512 * 4 * sizeof(float) +
																										1024 * 1024 * 4 * sizeof(float) +
																										2048 * 2048 * 4 * sizeof(float) ));
	glEnableVertexAttribArray(SAM_ATTRIBUTE_POSITION5);


	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
	
	/// Sam : all Shader Code End
	// vbo code for CUDA start
	cudaError status;
	status = cudaGraphicsGLRegisterBuffer(&g_cuda_vbo_resource, g_VertexBufferObject_Position_wave, cudaGraphicsMapFlagsWriteDiscard);
	if (status != cudaSuccess)
	{
		fprintf_s(g_pFile, "cudaGraphicsGLRegisterBuffer Failed. \n");
		return -10;
	}
	// vbo code for CUDA End

	// Frame buffer
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

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	//glEnable(GL_CULL_FACE);

	glPointSize(2.0f);
	glClearColor(0.25f, 0.25f, 0.25f, 1.0f);

	g_PersPectiveProjectionMatrix = vmath::mat4::identity();

	Resize(WIN_WIDTH, WIN_HEIGHT);

	return INIT_ALL_OK;
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

void computePerfStats(double &interactionsPerSecond, double &gflops,
	float milliseconds, int iterations)
{
	interactionsPerSecond = (float)mesh_width * (float)mesh_height;
	interactionsPerSecond *= 1e-9 * iterations * 1000 / milliseconds;
	gflops = interactionsPerSecond * (float)g_iFlopsPerInteraction;
}

void Update(void)
{
	void UpdateByCPU(GLfloat *startingPointer,size_t width, size_t height,GLfloat animation);

	g_fAnime = g_fAnime + 0.01f;
	
	static double sd_gflops = 0;
	static double sd_ifps = 0;
	static double sd_interactionsPerSecond = 0;

	if (gpu_cpu_Switch == true) // Use cuda events here
	{
		_int64 currentTime = 0, lastTime = 0, frequency;
		float elapsedTime;

		double gpu_interactionsPerSecond = 0;
		double gpu_gflops = 0;

		QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);
		QueryPerformanceCounter((LARGE_INTEGER*)&lastTime);
		
		RunCUDA(&g_cuda_vbo_resource, mesh_width, mesh_height, g_fAnime);
		
		QueryPerformanceCounter((LARGE_INTEGER*)&currentTime);
		elapsedTime = ((float)(currentTime - lastTime) / (float)frequency) * 100.0f;

		computePerfStats(gpu_interactionsPerSecond, gpu_gflops, elapsedTime, 1);

		sprintf_s(tempString, "GPU Time: %03.3fms", elapsedTime);
		sprintf_s(flopString, "GPU GFLOPS: %03.3f", gpu_gflops);
		//fprintf_s(g_pFile,"GPU Time: %3.5f\n", elapsedTime);
	}
	else
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
		glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_Position_wave);
		cpuLocation = (GLfloat*)glMapNamedBufferRange(g_VertexBufferObject_Position_wave, 0, bufferSize, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
		if (cpuLocation != NULL)
		{
			UpdateByCPU(cpuLocation, mesh_width, mesh_width, g_fAnime);
			glUnmapNamedBuffer(g_VertexBufferObject_Position_wave);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&currentTime)); // Current Time
		elapsedTime = ((float)(currentTime - lastTime) / (float)frequency) * 100.0f ;

		computePerfStats(cpu_interactionsPerSecond, cpu_gflops, elapsedTime,1);

		sprintf_s(tempString, "CPU Time: %03.3f ms", elapsedTime);
		sprintf_s(flopString, "CPU GFLOPS: %03.3f", cpu_gflops);
		//fprintf_s(g_pFile, "CPU Time: %3.5f\n", elapsedTime);
	}
	

}

void UpdateByCPU(GLfloat *startingPointer, size_t width, size_t height, GLfloat animation)
{
	int offset = 0;
	float4 *ptr = (float4*)startingPointer;
	for (size_t i = 0; i < height; i++)
	{
		for (size_t j = 0; j < width;j++)
		{
			float u = (float)i / (float)width;
			float v = (float)j / (float)height;

			u = u*2.0f - 1.0f;
			v = v*2.0f - 1.0f;

			float freq = 5.0f;
			float w = cosf(u*freq + animation) * sinf(v*freq + animation) * 0.5f;

			ptr[j*width + i] = make_float4(u,w,v,1.0f);

		}
	}
}

void Render(void)
{
	void RenderText(GLuint programObjFont, std::string text, GLfloat x, GLfloat y, GLfloat scale, vmath::vec3 color);


	vmath::mat4 modelMatrix = vmath::mat4::identity();
	vmath::mat4 viewMatrix = vmath::mat4::identity();
	vmath::mat4 rotationMatrix = vmath::mat4::identity();
	vmath::mat4 scaleMatrix = vmath::mat4::identity();
	vmath::mat4 m4PersPectiveProjectionMatrix = vmath::perspective(camera.GetZoom(), (float)g_fCurrrentWidth / (float)g_fCurrrentHeight, 0.1f, 100.0f);

	// render on Frame Buffer Start
	glBindFramebuffer(GL_FRAMEBUFFER, g_FrameBuffer);
	glViewport(0, 0, g_fCurrrentWidth, g_fCurrrentHeight);
	glClearBufferfv(GL_COLOR, 0, vmath::vec4(0.0f, 0.0f, 0.0f, 1.0f));
	glClearBufferfv(GL_DEPTH, 0, vmath::vec4(1.0f, 1.0f, 1.0f, 1.0f));

	// Mesh Rendering Strat

	modelMatrix = vmath::mat4::identity();
	viewMatrix = vmath::mat4::identity();
	rotationMatrix = vmath::mat4::identity();
	scaleMatrix = vmath::mat4::identity();
	glUseProgram(g_ShaderProgramObject);

	modelMatrix = vmath::translate(0.0f, 0.0f, -3.0f);

	glUniformMatrix4fv(g_Uniform_Model_Matrix, 1, GL_FALSE, modelMatrix);
	glUniformMatrix4fv(g_Uniform_View_Matrix, 1, GL_FALSE, camera.GetViewMatrix());
	glUniformMatrix4fv(g_Uniform_Projection_Matrix, 1, GL_FALSE, g_PersPectiveProjectionMatrix);
	glUniform3fv(g_Uniform_Color, 1, fvPointColors);

	glBindVertexArray(g_VertexArrayObject_wave);
	glDrawArrays(GL_POINTS, 0, mesh_height*mesh_width);
	glBindVertexArray(0);

	glUseProgram(0);
	// Mesh Rendering Stop

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	// render on Frame Buffer Stop	
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	
	

	modelMatrix = vmath::mat4::identity();
	viewMatrix = vmath::mat4::identity();
	rotationMatrix = vmath::mat4::identity();
	scaleMatrix = vmath::mat4::identity();

	glViewport(0, 0, g_fCurrrentWidth, g_fCurrrentHeight);

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
	if (g_bShowHelp)
	{
		RenderText(g_ShaderProgramObject_Font, "Options:", -57.0f, 29.0f, 0.03f, vmath::vec3(1.0f, 1.0f, 1.0f));
		RenderText(g_ShaderProgramObject_Font, "1. 128 x 128 x 4 ", -57.0f, 25.0f, 0.03f, (iSelectionMode == 0) ? vmath::vec3(1.0f, 0.0f, 0.0f) : vmath::vec3(1.0f, 1.0f, 1.0f));
		RenderText(g_ShaderProgramObject_Font, "2. 256 x 256 x 4 ", -57.0f, 21.0f, 0.03f, (iSelectionMode == 1) ? vmath::vec3(1.0f, 0.0f, 0.0f) : vmath::vec3(1.0f, 1.0f, 1.0f));
		RenderText(g_ShaderProgramObject_Font, "3. 512 x 512 x 4 ", -57.0f, 17.0f, 0.03f, (iSelectionMode == 2) ? vmath::vec3(1.0f, 0.0f, 0.0f) : vmath::vec3(1.0f, 1.0f, 1.0f));
		RenderText(g_ShaderProgramObject_Font, "4. 1024 x 1024 x 4 ", -57.0f, 13.0f, 0.03f, (iSelectionMode == 3) ? vmath::vec3(1.0f, 0.0f, 0.0f) : vmath::vec3(1.0f, 1.0f, 1.0f));
		RenderText(g_ShaderProgramObject_Font, "5. 2048 x 2048 x 4 ", -57.0f, 9.0f, 0.03f, (iSelectionMode == 4) ? vmath::vec3(1.0f, 0.0f, 0.0f) : vmath::vec3(1.0f, 1.0f, 1.0f));
		RenderText(g_ShaderProgramObject_Font, "6. 4096 x 4096 x 4 ", -57.0f, 5.0f, 0.03f, (iSelectionMode == 5) ? vmath::vec3(1.0f, 0.0f, 0.0f) : vmath::vec3(1.0f, 1.0f, 1.0f));
		RenderText(g_ShaderProgramObject_Font, "Press 'H' or 'h' for toggle", -57.0f, 1.0f, 0.03f, vmath::vec3(1.0f, 1.0f, 1.0f));
	}

	if (gpu_cpu_Switch == true)
	{
		RenderText(g_ShaderProgramObject_Font, "GPU", 48.0f, 28.0f, 0.032f, vmath::vec3(0.49f, 0.76f, 0.0f));
	}
	else
	{
		RenderText(g_ShaderProgramObject_Font, "CPU", 38.0f, 28.0f, 0.032f, vmath::vec3(0.0f, 0.46f, 0.9f));
	}
	
	RenderText(g_ShaderProgramObject_Font, flopString, -57.0f, -27.0f, 0.028f, vmath::vec3(1.0f, 1.0f, 0.0f));
	RenderText(g_ShaderProgramObject_Font, tempString, -57.0f, -30.0f, 0.028f, vmath::vec3(1.0f, 1.0f, 0.0f));

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
	if (g_bFullScreen == true)
	{
		SetWindowLong(g_hwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(g_hwnd, &wpPrev);
		SetWindowPos(g_hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOOWNERZORDER | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE);
		ShowCursor(TRUE);
		g_bFullScreen = false;
	}

	for (GLubyte c = 0; c < 128; c++)
	{
		GLuint texture = Characters[c].TexyureID;
		glDeleteTextures(1, &texture);
		texture = 0;

		Characters.erase(c);
	}
	
	if (cpuLocation != NULL)
	{
		glUnmapNamedBuffer(g_VertexBufferObject_Position_wave);
	}

	CleanUPCuda(&g_cuda_vbo_resource);

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

	if (g_VertexBufferObject_Position_wave)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_Position_wave);
		g_VertexBufferObject_Position_wave = NULL;
	}

	if (g_VertexBufferObject_Position_wave)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_Position_wave);
		g_VertexBufferObject_Position_wave = NULL;
	}

	if (g_VertexArrayObject_wave)
	{
		glDeleteVertexArrays(1, &g_VertexArrayObject_wave);
		g_VertexArrayObject_wave = NULL;
	}

	if (g_VertexBufferObject_TexCoords_FB)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_TexCoords_FB);
		g_VertexBufferObject_TexCoords_FB = NULL;
	}

	if (g_VertexBufferObject_Position_FB)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_Position_FB);
		g_VertexBufferObject_Position_FB = NULL;
	}

	if (g_VertexArrayObject_FB)
	{
		glDeleteVertexArrays(1, &g_VertexArrayObject_FB);
		g_VertexArrayObject_FB = NULL;
	}

	////////////////////////////////
	if (g_VertexBufferObject_TexCoords_Quad)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_TexCoords_Quad);
		g_VertexBufferObject_TexCoords_Quad = NULL;
	}

	if (g_VertexBufferObject_Position_Quad)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_Position_Quad);
		g_VertexBufferObject_Position_Quad = NULL;
	}

	if (g_VertexArrayObject_Quad)
	{
		glDeleteVertexArrays(1, &g_VertexArrayObject_Quad);
		g_VertexArrayObject_Quad = NULL;
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

		glDeleteProgram(g_ShaderProgramObject);
		g_ShaderProgramObject = NULL;

		glUseProgram(0);

	}

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
