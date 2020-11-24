#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>

#include<gl\glew.h>
#include<gl\GL.h>
#include<math.h>

#include<cuda.h>
#include<cuda_gl_interop.h>

#include"vmath.h"


#define		WIN_WIDTH		800
#define		WIN_HEIGHT		600
#define		DIM				4096

#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"opengl32.lib")

//using namespace std;

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

enum CUDAInitErrorCodes
{ 
	/* min no -10 */
	INIT_CUDA_SETGLDEVICE_FAILED = -21,
	INIT_CUDA_REGISTER_BUFFER_FAILED,
	CUDA_INIT_GRAPHICS_MAPPED_RES_FAILED,
	CUDA_INIT_GRAPHICS_MAPPED_RES_POINTER_FAILED,
	CUDA_INIT_GRAPHICS_UNMAPP_RES_FAILED,
	INIT_CUDA_CHOOSEDEVICE_FAILED = -10,
	CUDA_INIT_ALL_OK=0,
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
GLuint g_VertexBufferObject_TexCoord = 0;

// Uniforms
GLuint g_Uniform_Model_Matrix = 0;
GLuint g_Uniform_View_Matrix = 0;
GLuint g_Uniform_Projection_Matrix = 0;
// sampler
GLuint g_uniform_TextureSampler;
GLuint g_TextureID;


// Projection
vmath::mat4 g_PersPectiveProjectionMatrix;

// CUDA Res
GLuint bufferOBJ;
cudaGraphicsResource *resource=NULL;

bool gpu_cpu_Switch = false;// default on cpu

float g_fanimate = 0.0f;
bool animation_flag = false;

__global__ void kernel(uchar4 *ptr, float time)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// now calculate the value at the position 
	float fx = x / (float)DIM - 0.5f;
	float fy = y / (float)DIM - 0.5f;
	unsigned char  green = 128 + 127 * sinf(fabsf(fx * 100 * time) - fabsf(fy * 100* time));

	//ptr[(y * DIM) + x].x = 0;
	ptr[offset].x = 0;
	ptr[offset].y = green;
	ptr[offset].z = 0;
	ptr[offset].w = 255;

}


__global__ void kernel1(uchar4 *ptr, float time)
{
	__shared__ unsigned char color;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// now calculate the value at the position 
	float fx = x / (float)DIM - 0.5f;
	float fy = y / (float)DIM - 0.5f;
	color = y * fabsf(fy * 100 * time);
	color += 25 ;
	__syncthreads();

	//ptr[(y * DIM) + x].x = 0;
	ptr[offset].x = color;
	ptr[offset].y = color;
	ptr[offset].z = color;
	ptr[offset].w = 255;

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
	
	default:// For Other issues than OpenGL
		{
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
		}
		
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
		case 0x48: // 'h' or 'H'
			gpu_cpu_Switch = (gpu_cpu_Switch) ? false : true;
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

	// Shader Programs
	GLuint iVertexShaderObject = 0;
	GLuint iFragmentShaderObject = 0;

	// CUDA Init Vriables
	cudaDeviceProp prop;
	int dev;
	cudaError status;

	int iPixelIndex = 0;
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

	// Cuda Init Start
	SecureZeroMemory((void*)&prop, sizeof(prop));

	// Compute Capability 2.0 and beyond
	prop.major = 2;
	prop.minor = 0;

	status = cudaChooseDevice(&dev,&prop);
	if (status != cudaSuccess)
	{
		return INIT_CUDA_CHOOSEDEVICE_FAILED;
	}

	/*// Declared Depricated....!!
	status = cudaGLSetGLDevice(dev);
	if (status != cudaSuccess)
	{
		return INIT_CUDA_SETGLDEVICE_FAILED;
	}*/

	// Cuda Init Stop

	/// Sam : all Shader Code Start

	/*Vertex Shader Start*/
	iVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
	const GLchar *vertexShaderSourceCode = "#version 450 core"	\
		"\n" \
		"layout (location = 0)in vec4 vPosition;" \
		"layout (location = 3)in vec2 vTexture0_Coord;;" \
		"layout (location = 0)out vec2 out_Texture0_Coord;" \
		"uniform mat4 u_model_matrix,u_view_matrix,u_projection_matrix;" \
		"void main(void)" \
		"{" \
		/*"	gl_Position =  u_projection_matrix * u_view_matrix * u_model_matrix * vPosition;" \*/
		"	gl_Position = vPosition;" \
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

	glGenBuffers(1, &g_VertexBufferObject_TexCoord); // vbo texture
	glBindBuffer(GL_ARRAY_BUFFER, g_VertexBufferObject_TexCoord);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareTexCords), squareTexCords, GL_STATIC_DRAW);

	glVertexAttribPointer(SAM_ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SAM_ATTRIBUTE_TEXTURE0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
	/* Fill Buffers End*/
	/// Sam : all Shader Code End
	
	glGenBuffers(1, &bufferOBJ);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferOBJ);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, DIM * DIM * 4, NULL, GL_STREAM_DRAW);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	status = cudaGraphicsGLRegisterBuffer(&resource, bufferOBJ,cudaGraphicsMapFlagsNone);
	if (status != cudaSuccess)
	{
		return INIT_CUDA_REGISTER_BUFFER_FAILED;
	}
	/* Fill Buffers End*/
	/// Sam : all Shader Code End

	// Prepare Texture to take from PBO
	glGenTextures(1,&g_TextureID);
	glBindTexture(GL_TEXTURE_2D, g_TextureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, DIM, DIM, 0, GL_BGRA,
		GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	glEnable(GL_TEXTURE_2D);
	glShadeModel(GL_SMOOTH);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);
	
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glClearColor(0.125f, 0.125f, 0.125f, 1.0f);

	g_PersPectiveProjectionMatrix = vmath::mat4::identity();
	

	Resize(WIN_WIDTH, WIN_HEIGHT);

	return INIT_ALL_OK;
}


int Update(void)
{
	void updatePixels(GLubyte* dst, int size, float animate);
	void updatePixels1(GLubyte* dst, int size, float animate);
	

	if (animation_flag)
	{
		g_fanimate = g_fanimate + 0.005f;
		if ( (g_fanimate >1.0f) )
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

	
	if (gpu_cpu_Switch==true)
	{
		uchar4 *devPtr = NULL;
		size_t size;
		cudaError status;

		status = cudaGraphicsMapResources(1, &resource, NULL);
		if (status != cudaSuccess)
		{
			return CUDA_INIT_GRAPHICS_MAPPED_RES_FAILED;
		}

		status = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);
		if (status != cudaSuccess)
		{
			return CUDA_INIT_GRAPHICS_MAPPED_RES_POINTER_FAILED;
		}

		// Run the kernel
		dim3 grids(DIM / 8, DIM / 8);
		dim3 threads(8, 8);


		kernel<<< grids, threads >>>(devPtr,g_fanimate);
		//kernel1 << < grids, threads >> >(devPtr,g_fanimate);


		// Unmap the resource for use
		status = cudaGraphicsUnmapResources(1, &resource, NULL);
		if (status != cudaSuccess)
		{
			return CUDA_INIT_GRAPHICS_UNMAPP_RES_FAILED;
		}
	}
	else
	{
		// cpu updatation
		/*glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferOBJ);
		GLubyte *ptr = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER,GL_WRITE_ONLY);
		if (ptr)
		{
			updatePixels(ptr, DIM*DIM*4, g_fanimate);
			glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
		}
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);*/

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferOBJ);
		GLubyte *ptr = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
		if (ptr)
		{
			updatePixels1(ptr, DIM*DIM * 4, g_fanimate);
			glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
		}
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}


	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferOBJ);
	glBindTexture(GL_TEXTURE_2D, g_TextureID);

	glTexSubImage2D(GL_TEXTURE_2D,0,0,0,DIM,DIM,GL_BGRA,GL_UNSIGNED_BYTE,NULL);
	
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	return CUDA_INIT_ALL_OK;
}

void updatePixels1(GLubyte* dst, int size,float animate)
{
	if (!dst)
		return;

	int *ptr = (int *)dst;// to operate 4 bytes at once

	for (int i = 0; i < DIM; i++)
	{
		for (int j = 0; j < DIM; j++)
		{
			int offset = j + i * DIM;
			float fx = i / (float)DIM - 0.5f;
			float fy = j / (float)DIM - 0.5f;

			unsigned char  green = ((unsigned char)128) + ((unsigned char)127) * (sinf(fabsf(fx * 100 * animate) - fabsf(fy * 100 * animate)) );

			*(ptr) = 255 - green;

			ptr++;
		}
	}

}

void updatePixels(GLubyte* dst, int size, float animate)
{
	static int color = 0;

	if (!dst)
		return;

	int *ptr = (int *)dst;// to operate 4 bytes at once

	for (int i = 0; i < DIM; i++)
	{
		for (int j = 0; j < DIM; j++)
		{
			*ptr = color;
			++ptr;
		}
		color += 257;
	}
	++color;
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
	glBindTexture(GL_TEXTURE_2D, g_TextureID);
	glUniform1i(g_uniform_TextureSampler, 0);

	glBindVertexArray(g_VertexArrayObject);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindVertexArray(0);

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

	/*		Clear CUDA Resources Start		*/
	cudaGraphicsUnmapResources(1, &resource, NULL);
	cudaGraphicsUnregisterResource(resource);

	cudaDeviceReset();
	/*		Clear CUDA Resources End  		*/

	/*		Clear OpenGL Resources Start		*/
	if (bufferOBJ)
	{
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferOBJ);
		glDeleteBuffers(1, &bufferOBJ);
		bufferOBJ = NULL;
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}

	if (g_VertexBufferObject_TexCoord)
	{
		glDeleteBuffers(1, &g_VertexBufferObject_TexCoord);
		g_VertexBufferObject_TexCoord = NULL;
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
	/*		Clear OpenGL Resources End		*/

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

