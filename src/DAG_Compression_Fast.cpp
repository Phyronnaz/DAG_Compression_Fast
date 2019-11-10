#include "Core.h"
#include <SDL.h>
#undef main
#include <iostream>
#include <chrono>
#include "Utils/View.h"
#include "GLTFLoader.h"
#include "DAGTracer.h"
#include "Voxelizer.h"

SDL_Window* mainWindow;
SDL_GLContext mainContext;
glm::ivec2 screen_dim{ 1024, 1024 };
GLuint copy_shader{ 0 };
GLint renderbuffer_uniform{ -1 };

GLuint compile_shader(GLenum shader_type, const std::string& src) {
	GLuint shader_id = glCreateShader(shader_type);
	const GLchar* source = static_cast<const GLchar*>(src.c_str());
	glShaderSource(shader_id, 1, &source, 0);
	glCompileShader(shader_id);

	GLint compile_status = GL_FALSE;
	glGetShaderiv(shader_id, GL_COMPILE_STATUS, &compile_status);

	if (compile_status != GL_TRUE) { throw std::runtime_error("Failed to compile shaders"); }

	return shader_id;
}

GLuint link_shaders(GLuint vs, GLuint fs) {
	GLuint program_id = glCreateProgram();
	glAttachShader(program_id, vs);
	glAttachShader(program_id, fs);
	glLinkProgram(program_id);

	GLint linkStatus = GL_FALSE;
	glGetProgramiv(program_id, GL_LINK_STATUS, &linkStatus);

	if (linkStatus != GL_TRUE) { throw std::runtime_error("Failed to link shaders"); }

	return program_id;
}

void init() {
	mainWindow = SDL_CreateWindow(
		"DAG",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		screen_dim.x,
		screen_dim.y,
		SDL_WINDOW_OPENGL
	);

	mainContext = SDL_GL_CreateContext(mainWindow);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetSwapInterval(1);

	int value = 0;
	SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &value);
	std::cout << "SDL_GL_CONTEXT_MAJOR_VERSION : " << value << std::endl;

	SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &value);
	std::cout << "SDL_GL_CONTEXT_MINOR_VERSION: " << value << std::endl;

	glewInit();

	std::string copy_vertex_shader =
		"#version 400 compatibility                                           \n"
		"out vec2 texcoord;                                                   \n"
		"void main() {                                                        \n"
		"   if(gl_VertexID == 0){ texcoord = vec2(0.0, 2.0); }                \n"
		"   if(gl_VertexID == 1){ texcoord = vec2(0.0, 0.0); }                \n"
		"   if(gl_VertexID == 2){ texcoord = vec2(2.0, 0.0); }                \n"
		"   if(gl_VertexID == 0){ gl_Position = vec4(-1.0,  3.0, 0.0, 1.0); } \n"
		"   if(gl_VertexID == 1){ gl_Position = vec4(-1.0, -1.0, 0.0, 1.0); } \n"
		"   if(gl_VertexID == 2){ gl_Position = vec4( 3.0, -1.0, 0.0, 1.0); } \n"
		"}                                                                    \n";

	std::string copy_fragment_shader =
		"#version 400 compatibility                                  \n"
		"in vec2 texcoord;                                           \n"
		"uniform sampler2D renderbuffer;                             \n"
		"void main() {                                               \n"
		"    gl_FragColor.xyz = texture(renderbuffer, texcoord).xyz; \n"
		"}                                                           \n";

	GLuint vs = compile_shader(GL_VERTEX_SHADER, copy_vertex_shader);
	GLuint fs = compile_shader(GL_FRAGMENT_SHADER, copy_fragment_shader);
	copy_shader = link_shaders(vs, fs);
	renderbuffer_uniform = glGetUniformLocation(copy_shader, "renderbuffer");
}

class Timer {
	using clock_t        = std::chrono::high_resolution_clock;
	using timepoint_t    = std::chrono::time_point<clock_t>;
	using seconds_ft     = std::chrono::duration<float>;
	using milliseconds_t = std::chrono::milliseconds;
	using nanoseconds_t  = std::chrono::nanoseconds;
	timepoint_t start_tp;
	timepoint_t end_tp;
	nanoseconds_t diff_ns;

 public:
	Timer() : start_tp(clock_t::now()), end_tp(clock_t::now()), diff_ns(0) {}
	float dt_seconds() const { return std::chrono::duration_cast<seconds_ft>(diff_ns).count(); }
	void start() { start_tp = clock_t::now(); }
	void end() {
		end_tp  = clock_t::now();
		diff_ns = end_tp - start_tp;
	}
	void reset() {
		start_tp = clock_t::now();
		end_tp   = clock_t::now();
		diff_ns  = nanoseconds_t{0};
	}
};

struct AppState {
	FView camera;
	glm::ivec2 old_mouse{0, 0};
	glm::ivec2 new_mouse{0, 0};
	bool loop = true;
	Timer frame_timer;

	void handle_events() {
		const float dt = frame_timer.dt_seconds();
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT) {
				loop = false;
				break;
			}
		}

		const float move_scale_factor{1000.0f * dt};
		const Uint8 *key_state = SDL_GetKeyboardState(NULL);
		if (key_state[SDL_SCANCODE_ESCAPE]) { loop = false; }
		if (key_state[SDL_SCANCODE_W]) { camera.Position -= move_scale_factor * camera.Rotation[2]; }
		if (key_state[SDL_SCANCODE_S]) { camera.Position += move_scale_factor * camera.Rotation[2]; }
		if (key_state[SDL_SCANCODE_D]) { camera.Position += move_scale_factor * camera.Rotation[0]; }
		if (key_state[SDL_SCANCODE_A]) { camera.Position -= move_scale_factor * camera.Rotation[0]; }
		if (key_state[SDL_SCANCODE_E]) { camera.Position += move_scale_factor * camera.Rotation[1]; }
		if (key_state[SDL_SCANCODE_Q]) { camera.Position -= move_scale_factor * camera.Rotation[1]; }

		Uint32 mouse_state = SDL_GetMouseState(&(new_mouse.x), &(new_mouse.y));
		constexpr Uint32 lmb{1 << 0};
		constexpr Uint32 mmb{1 << 1};
		constexpr Uint32 rmb{1 << 2};
		const bool ld = (lmb & mouse_state) != 0;
		const bool md = (mmb & mouse_state) != 0;
		const bool rd = (rmb & mouse_state) != 0;

		glm::vec2 delta = (glm::vec2(new_mouse) - glm::vec2(old_mouse));

		const float mouse_scale_factor{0.25f * dt};
		if (ld && rd) {
			camera.Position += delta.y * mouse_scale_factor * camera.Rotation[2];
		} else if (ld) {
			camera.Pitch(-delta.y * mouse_scale_factor);
			camera.Yaw(-delta.x * mouse_scale_factor);
		} else if (rd) {
			camera.Roll(-delta.x * mouse_scale_factor);
		} else if (md) {
			camera.Position -= delta.y * mouse_scale_factor * camera.Rotation[1];
			camera.Position += delta.x * mouse_scale_factor * camera.Rotation[0];
		}
		old_mouse = new_mouse;
	}
};

int main()
{
	init();

	const uint32 Width = 1280;
	const uint32 Height = 720;

	FAABB AABB;
	TStaticArray<uint32, EMemoryType::GPU> Dag;
	{
		FGLTFLoader Loader("GLTF/Sponza/glTF/", "Sponza.gltf");
		const auto Scene = Loader.GetScene();
		AABB = Scene.AABB;

		FVoxelizer Voxelizer(1 << LEVELS, Scene);
		
		auto Data = Voxelizer.GenerateFragments(AABB, 1 << LEVELS);
	}
	
	
	FDAGTracer DagTracer(Width, Height);
	AppState app;
	app.camera.LookAt(glm::vec3{ 0.0f, 1.0f, 0.0f }, glm::vec3{ 0.0f, 0.0f, 0.0f });
	while (app.loop)
	{
		app.frame_timer.start();
		app.handle_events();

		auto Params = FDAGTracer::GetTraceParams(app.camera, Width, Height, AABB);
		DagTracer.ResolvePaths(Params, Dag);

		glViewport(0, 0, screen_dim.x, screen_dim.y);
		glUseProgram(copy_shader);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, DagTracer.GetColorsImage());
		glUniform1i(renderbuffer_uniform, 0);
		glActiveTexture(GL_TEXTURE0);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		SDL_GL_SwapWindow(mainWindow);
		app.frame_timer.end();
	}

	SDL_GL_DeleteContext(mainContext);
	SDL_DestroyWindow(mainWindow);
	SDL_Quit();

	return 0;
}
