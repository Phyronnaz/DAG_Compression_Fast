#include "Core.h"
#include <SDL.h>
#undef main
#include <iostream>
#include <chrono>
#include "Utils/View.h"
#include "GLTFLoader.h"
#include "DAGTracer.h"
#include "Voxelizer.h"
#include "DAGCompression.h"
#include "Serializer.h"

SDL_Window* MainWindow = nullptr;
SDL_GLContext MainContext;
glm::ivec2 ScreenDim{ 1024, 1024 };
GLuint CopyShader = 0;
GLint RenderBufferUniform = -1;

GLuint CompileShader(GLenum shader_type, const std::string& src)
{
	GLuint shader_id = glCreateShader(shader_type);
	const GLchar* source = static_cast<const GLchar*>(src.c_str());
	glShaderSource(shader_id, 1, &source, 0);
	glCompileShader(shader_id);

	GLint compile_status = GL_FALSE;
	glGetShaderiv(shader_id, GL_COMPILE_STATUS, &compile_status);

	checkAlways(compile_status == GL_TRUE);

	return shader_id;
}

GLuint LinkShaders(GLuint VertexShader, GLuint FragmentShader)
{
	const GLuint ProgramId = glCreateProgram();
	glAttachShader(ProgramId, VertexShader);
	glAttachShader(ProgramId, FragmentShader);
	glLinkProgram(ProgramId);

	GLint LinkStatus = GL_FALSE;
	glGetProgramiv(ProgramId, GL_LINK_STATUS, &LinkStatus);

	checkAlways(LinkStatus == GL_TRUE);

	return ProgramId;
}

void Init()
{
	MainWindow = SDL_CreateWindow(
		"DAG",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		ScreenDim.x,
		ScreenDim.y,
		SDL_WINDOW_OPENGL
	);

	MainContext = SDL_GL_CreateContext(MainWindow);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetSwapInterval(1);

	int32 Value = 0;
	SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &Value);
	std::cout << "SDL_GL_CONTEXT_MAJOR_VERSION : " << Value << std::endl;

	SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &Value);
	std::cout << "SDL_GL_CONTEXT_MINOR_VERSION: " << Value << std::endl;

	glewInit();

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
	FView Camera;
	glm::ivec2 OldMouse{ 0, 0 };
	glm::ivec2 NewMouse{ 0, 0 };
	bool Loop = true;
	FTimer FrameTimer;

	void HandleEvents()
	{
		const float DeltaTime = FrameTimer.DeltaTime();
		SDL_Event Event;
		while (SDL_PollEvent(&Event))
		{
			if (Event.type == SDL_QUIT)
			{
				Loop = false;
				break;
			}
		}

		const float MoveScaleFactor{ 1000.0f * DeltaTime };
		const uint8* KeyState = SDL_GetKeyboardState(NULL);
		if (KeyState[SDL_SCANCODE_ESCAPE]) { Loop = false; }
		if (KeyState[SDL_SCANCODE_W]) { Camera.Position += MoveScaleFactor * Camera.Rotation[2]; }
		if (KeyState[SDL_SCANCODE_S]) { Camera.Position -= MoveScaleFactor * Camera.Rotation[2]; }
		if (KeyState[SDL_SCANCODE_D]) { Camera.Position += MoveScaleFactor * Camera.Rotation[0]; }
		if (KeyState[SDL_SCANCODE_A]) { Camera.Position -= MoveScaleFactor * Camera.Rotation[0]; }
		if (KeyState[SDL_SCANCODE_E]) { Camera.Position += MoveScaleFactor * Camera.Rotation[1]; }
		if (KeyState[SDL_SCANCODE_Q]) { Camera.Position -= MoveScaleFactor * Camera.Rotation[1]; }

		const uint32 MouseState = SDL_GetMouseState(&NewMouse.x, &NewMouse.y);
		constexpr uint32 LeftMouseButton  { 1 << 0 };
		constexpr uint32 MiddleMouseButton{ 1 << 1 };
		constexpr uint32 RightMouseButton { 1 << 2 };
		const bool LeftDown   = (LeftMouseButton   & MouseState) != 0;
		const bool MiddleDown = (MiddleMouseButton & MouseState) != 0;
		const bool RightDown  = (RightMouseButton  & MouseState) != 0;

		const auto DeltaMouse = NewMouse - OldMouse;

		const float MouseScaleFactor{ 0.25f * DeltaTime };
		if (LeftDown && RightDown) 
		{
			Camera.Position += DeltaMouse.y * MouseScaleFactor * Camera.Rotation[2];
		}
		else if (LeftDown) 
		{
			Camera.Pitch(-DeltaMouse.y * MouseScaleFactor);
			Camera.Yaw(-DeltaMouse.x * MouseScaleFactor);
		}
		else if (RightDown) 
		{
			Camera.Roll(-DeltaMouse.x * MouseScaleFactor);
		}
		else if (MiddleDown) 
		{
			Camera.Position -= DeltaMouse.y * MouseScaleFactor * Camera.Rotation[1];
			Camera.Position += DeltaMouse.x * MouseScaleFactor * Camera.Rotation[0];
		}
		OldMouse = NewMouse;
	}
};

uint64_t splitBy3_64(uint32_t a)
{
	uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
	x = (x | x << 32) & 0x1f00000000fffful; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = (x | x << 16) & 0x1f0000ff0000fful; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = (x | x << 8) & 0x100f00f00f00f00ful; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = (x | x << 4) & 0x10c30c30c30c30c3ul; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = (x | x << 2) & 0x1249249249249249ul;
	return x;
};
uint64_t mortonEncode64(uint32_t x, uint32_t y, uint32_t z)
{
	return splitBy3_64(x) << 2 | splitBy3_64(y) << 1 | splitBy3_64(z);
};

int main()
{
	Init();

	const uint32 Width = ScreenDim.x;
	const uint32 Height = ScreenDim.y;

	FAABB AABB;
	TStaticArray<uint32, EMemoryType::GPU> Dag;

#if 1
	{
		ZoneScopedN("Main");
		
		FGLTFLoader Loader("GLTF/Sponza/glTF/", "Sponza.gltf");
		const auto Scene = Loader.GetScene();
		AABB = MakeSquareAABB(Scene.AABB);

		FVoxelizer Voxelizer(1 << LEVELS, Scene);
		
		const double GenerateFragmentsStartTime = Utils::Seconds();
		const auto Fragments = Voxelizer.GenerateFragments(AABB, 1 << LEVELS);
		LOG("GenerateFragments took %fs", Utils::Seconds() - GenerateFragmentsStartTime);

		LOG("%llu fragments", Fragments.Num());
		LOG("Depth = %d", LEVELS);

		const double CreateDAGStartTime = Utils::Seconds();
		Dag = DAGCompression::CreateDAG(Fragments, LEVELS);
		LOG("CreateDAG took %fs", Utils::Seconds() - CreateDAGStartTime);
		LOG("Peak GPU memory usage: %f MB", Utils::ToMB(FMemory::GetGpuMaxAllocatedMemory()));
		
		FreeScene(Scene);
	}
#else
	FFileReader Reader("C:/ROOT/DAG_Compression/cache/result.basic_dag.dag.bin");
	const auto ReadDouble = [&]() { double X; Reader.Read(X); return float(X); };
	AABB.Min.x = ReadDouble();
	AABB.Min.y = ReadDouble();
	AABB.Min.z = ReadDouble();
	AABB.Max.x = ReadDouble();
	AABB.Max.y = ReadDouble();
	AABB.Max.z = ReadDouble();

	uint32 Levels;
	Reader.Read(Levels);
	check(Levels == LEVELS);

	TStaticArray<uint32, EMemoryType::CPU> DagCpu;
	Reader.Read(DagCpu, "DagCpu");

	Dag = DagCpu.CreateGPU();
#endif

	{
		FDAGTracer DagTracer(Width, Height);
		FAppState App;
		App.Camera.LookAt(glm::vec3{ 0.0f, 1.0f, 0.0f }, glm::vec3{ 0.0f, 0.0f, 0.0f });
		while (App.Loop)
		{
			App.FrameTimer.Start();
			App.HandleEvents();

			auto Params = FDAGTracer::GetTraceParams(App.Camera, Width, Height, AABB);
			DagTracer.ResolvePaths(Params, Dag);

			glViewport(0, 0, ScreenDim.x, ScreenDim.y);
			glUseProgram(CopyShader);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, DagTracer.GetColorsImage());
			glUniform1i(RenderBufferUniform, 0);
			glActiveTexture(GL_TEXTURE0);
			glDrawArrays(GL_TRIANGLES, 0, 3);

			SDL_GL_SwapWindow(MainWindow);
			App.FrameTimer.End();
		}
	}

	Dag.Free();

	SDL_GL_DeleteContext(MainContext);
	SDL_DestroyWindow(MainWindow);
	SDL_Quit();

	return 0;
}
