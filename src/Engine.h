#pragma once

#include "Core.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "glm/vec2.hpp"
#include <string>
#include <chrono>

#include "DAGTracer.h"
#include "DAGCompression/DAGCompression.h"

class FTimer
{
	using clock_t = std::chrono::high_resolution_clock;
	using timepoint_t = std::chrono::time_point<clock_t>;
	using seconds_ft = std::chrono::duration<float>;
	using milliseconds_t = std::chrono::milliseconds;
	using nanoseconds_t = std::chrono::nanoseconds;
	timepoint_t StartTp;
	timepoint_t EndTp;
	nanoseconds_t DiffNs;

public:
	FTimer()
		: StartTp(clock_t::now())
		, EndTp(clock_t::now())
		, DiffNs(0)
	{
	}

	float DeltaTime() const
	{
		return std::chrono::duration_cast<seconds_ft>(DiffNs).count();
	}
	void Start()
	{
		StartTp = clock_t::now();
	}
	void End()
	{
		EndTp = clock_t::now();
		DiffNs = EndTp - StartTp;
	}
	void Reset()
	{
		StartTp = clock_t::now();
		EndTp = clock_t::now();
		DiffNs = nanoseconds_t{ 0 };
	}
};

struct FAppState
{
	FCameraView Camera;
	bool Loop = true;
	FTimer FrameTimer;
	uint32 DebugLevel = 0xFFFFFFFF;

	void HandleEvents(GLFWwindow* MainWindow)
	{
		glfwPollEvents();
		
		const float DeltaTime = FrameTimer.DeltaTime();

		const auto IsPressed = [&](auto Key)
		{
			return glfwGetKey(MainWindow, Key) == GLFW_PRESS;
		};

		if (glfwWindowShouldClose(MainWindow))
		{
			Loop = false;
		}
		
		if (IsPressed(GLFW_KEY_ESCAPE)) { Loop = false; }

		if (IsPressed(GLFW_KEY_1)) { DebugLevel = 0; }
		if (IsPressed(GLFW_KEY_2)) { DebugLevel = 1; }
		if (IsPressed(GLFW_KEY_3)) { DebugLevel = 2; }
		if (IsPressed(GLFW_KEY_4)) { DebugLevel = 3; }
		if (IsPressed(GLFW_KEY_5)) { DebugLevel = 4; }
		if (IsPressed(GLFW_KEY_6)) { DebugLevel = 5; }
		if (IsPressed(GLFW_KEY_7)) { DebugLevel = 6; }
		if (IsPressed(GLFW_KEY_8)) { DebugLevel = 7; }
		if (IsPressed(GLFW_KEY_9)) { DebugLevel = 8; }
		if (IsPressed(GLFW_KEY_0)) { DebugLevel = 0xFFFFFFFF; }
		if (IsPressed(GLFW_KEY_L)) { DebugLevel = 0xFFFFFFFE; }
		if (IsPressed(GLFW_KEY_C)) { DebugLevel = 0xFFFFFFFD; }
		if (IsPressed(GLFW_KEY_N)) { DebugLevel = 0xFFFFFFFC; }
		
        double speed = 100 * DeltaTime;
		double rotationSpeed = 2 * DeltaTime;

		if (IsPressed(GLFW_KEY_LEFT_SHIFT))
			speed *= 10;

		if (IsPressed(GLFW_KEY_W))
        {
            Camera.Position += speed * Camera.Forward();
        }
		if (IsPressed(GLFW_KEY_S))
        {
			Camera.Position -= speed * Camera.Forward();
        }
		if (IsPressed(GLFW_KEY_D))
        {
			Camera.Position += speed * Camera.Right();
        }
		if (IsPressed(GLFW_KEY_A))
        {
			Camera.Position -= speed * Camera.Right();
        }
		if (IsPressed(GLFW_KEY_SPACE))
        {
            Camera.Position += speed * Camera.Up();
        }
        if (IsPressed(GLFW_KEY_LEFT_CONTROL))
        {
            Camera.Position -= speed * Camera.Up();
        }

        if (IsPressed(GLFW_KEY_RIGHT) || IsPressed(GLFW_KEY_E))
        {
            Camera.Rotation *= Matrix3x3::FromQuaternion(Quaternion::FromAngleAxis(rotationSpeed, Vector3::Up()));
        }
        if (IsPressed(GLFW_KEY_LEFT) || IsPressed(GLFW_KEY_Q))
        {
            Camera.Rotation *= Matrix3x3::FromQuaternion(Quaternion::FromAngleAxis(-rotationSpeed, Vector3::Up()));
        }
        if (IsPressed(GLFW_KEY_DOWN))
        {
            Camera.Rotation *= Matrix3x3::FromQuaternion(Quaternion::FromAngleAxis(rotationSpeed, Camera.Right()));
        }
        if (IsPressed(GLFW_KEY_UP))
        {
            Camera.Rotation *= Matrix3x3::FromQuaternion(Quaternion::FromAngleAxis(-rotationSpeed, Camera.Right()));
        }
	}
};

struct FEngine
{
	GLFWwindow* MainWindow = nullptr;
	glm::ivec2 ScreenDim{ 1024, 1024 };
	GLuint CopyShader = 0;
	GLint RenderBufferUniform = -1;

	static GLuint CompileShader(GLenum shader_type, const std::string& src)
	{
		const GLuint ShaderId = glCreateShader(shader_type);
		const GLchar* Source = static_cast<const GLchar*>(src.c_str());
		glShaderSource(ShaderId, 1, &Source, 0);
		glCompileShader(ShaderId);

		GLint CompileStatus = GL_FALSE;
		glGetShaderiv(ShaderId, GL_COMPILE_STATUS, &CompileStatus);

		checkAlways(CompileStatus == GL_TRUE);

		return ShaderId;
	}

	static GLuint LinkShaders(GLuint VertexShader, GLuint FragmentShader)
	{
		const GLuint ProgramId = glCreateProgram();
		glAttachShader(ProgramId, VertexShader);
		glAttachShader(ProgramId, FragmentShader);
		glLinkProgram(ProgramId);

		GLint LinkStatus = GL_FALSE;
		glGetProgramiv(ProgramId, GL_LINK_STATUS, &LinkStatus);

		checkAlways(LinkStatus == GL_TRUE);

		glDetachShader(ProgramId, VertexShader);
		glDetachShader(ProgramId, FragmentShader);

		glDeleteShader(VertexShader);
		glDeleteShader(FragmentShader);

		return ProgramId;
	}

	inline void Init()
	{
		// Initialize GLFW
		if (!glfwInit())
		{
			fprintf(stderr, "Failed to initialize GLFW\n");
			exit(1);
		}

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
		// Open a window and create its OpenGL context
		MainWindow = glfwCreateWindow(ScreenDim.x, ScreenDim.y, "DAG", NULL, NULL);
		checkAlways(MainWindow);
		glfwMakeContextCurrent(MainWindow);

		glfwSwapInterval(0);

		// Initialize GLEW
		glewExperimental = true; // Needed for core profile
		if (glewInit() != GLEW_OK)
		{
			fprintf(stderr, "Failed to initialize GLEW\n");
			exit(1);
		}

		// Ensure we can capture the escape key being pressed below
		glfwSetInputMode(MainWindow, GLFW_STICKY_KEYS, GL_TRUE);

		// Dark blue background
		glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

		const std::string CopyVertexShader =
			"#version 400 compatibility                                           \n"
			"out vec2 TexCoord;                                                   \n"
			"void main()                                                          \n"
			"{                                                                    \n"
			"   if (gl_VertexID == 0) TexCoord = vec2(0.0, 2.0);                  \n"
			"   if (gl_VertexID == 1) TexCoord = vec2(0.0, 0.0);                  \n"
			"   if (gl_VertexID == 2) TexCoord = vec2(2.0, 0.0);                  \n"
			"   if (gl_VertexID == 0) gl_Position = vec4(-1.0,  3.0, 0.0, 1.0);   \n"
			"   if (gl_VertexID == 1) gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);   \n"
			"   if (gl_VertexID == 2) gl_Position = vec4( 3.0, -1.0, 0.0, 1.0);   \n"
			"}                                                                    \n";

		const std::string CopyFragmentShader =
			"#version 400 compatibility                                  \n"
			"in vec2 TexCoord;                                           \n"
			"uniform sampler2D RenderBuffer;                             \n"
			"void main()                                                 \n"
			"{                                                           \n"
			"    gl_FragColor.xyz = texture(RenderBuffer, TexCoord).xyz; \n"
			"}                                                           \n";

		const GLuint VertexShader = CompileShader(GL_VERTEX_SHADER, CopyVertexShader);
		const GLuint FragmentShader = CompileShader(GL_FRAGMENT_SHADER, CopyFragmentShader);
		CopyShader = LinkShaders(VertexShader, FragmentShader);
		RenderBufferUniform = glGetUniformLocation(CopyShader, "RenderBuffer");
	}

	void Loop(const FAABB& AABB, const FFinalDag& Dag) const
	{
		// Need a dummy VAO to work
		GLuint VAO = 0;
		{
			float Points[] = { 0.0f,  0.0f,  0.0f };

			GLuint VBO = 0;
			glGenBuffers(1, &VBO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(float), Points, GL_STATIC_DRAW);

			glGenVertexArrays(1, &VAO);
			glBindVertexArray(VAO);
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		}
		
		const uint32 Width = ScreenDim.x;
		const uint32 Height = ScreenDim.y;
		FDAGTracer DagTracer(Width, Height);
		FAppState App;
		while (App.Loop)
		{
			App.FrameTimer.Start();
			App.HandleEvents(MainWindow);

			auto Params = FDAGTracer::GetTraceParams(App.DebugLevel, App.Camera, Width, Height, AABB);
			DagTracer.ResolvePaths(Params, Dag.Dag, Dag.Colors, Dag.EnclosedLeaves);

			glClear(GL_COLOR_BUFFER_BIT);

			glUseProgram(CopyShader);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, DagTracer.GetColorsImage());
			glUniform1i(RenderBufferUniform, 0);
			glActiveTexture(GL_TEXTURE0);

			glBindVertexArray(VAO);
			glDrawArrays(GL_TRIANGLES, 0, 3);

			glfwSwapBuffers(MainWindow);
			App.FrameTimer.End();
		}
	}
};