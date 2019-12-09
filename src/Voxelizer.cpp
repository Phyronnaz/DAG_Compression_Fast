#include "Voxelizer.h"
#include "Memory.h"
#include "GLTFLoader.h"
#include "Utils/View.h"

#include "glm/gtc/type_ptr.hpp"

#include <cuda_gl_interop.h>

#include <fstream>
#include <vector>
#include <array>
#include <stack>

#define FRAMEBUFFER_PROGRAMMABLE_SAMPLE_LOCATIONS_NV 0x9342

static std::string GetFileContents(const char* Filename)
{
	std::ifstream In(Filename, std::ios::in | std::ios::binary);
	checkAlways(In);

	std::string Contents;
	In.seekg(0, std::ios::end);
	Contents.resize(In.tellg());
	In.seekg(0, std::ios::beg);
	In.read(&Contents[0], Contents.size());
	In.close();
	return (Contents);
}

static std::string AddDefine(const std::string& File, const std::string& Define, const std::string& Value)
{
	return "#define " + Define + " " + Value + "\n" + File;
}

static GLuint CompileShader(const GLenum ShaderType, const std::string& Source)
{
	const auto ShaderCode = "#version 450 compatibility\n" + Source;
	const GLchar* ShaderCodePtr = ShaderCode.c_str();
	
	const GLuint ShaderId = glCreateShader(ShaderType);
	glShaderSource(ShaderId, 1, &ShaderCodePtr, nullptr);
	glCompileShader(ShaderId);

	GLint BufferLength = 0;
	glGetShaderiv(ShaderId, GL_INFO_LOG_LENGTH, &BufferLength);
	if (BufferLength > 1)
	{
		std::vector<GLchar> LogString(std::size_t(BufferLength) + 1);
		glGetShaderInfoLog(ShaderId, BufferLength, nullptr, LogString.data());
		LOG("Log found for '%s.':\n%s", ShaderCodePtr, LogString.data());
		checkAlways(false);
	}

	GLint CompileStatus = GL_FALSE;
	glGetShaderiv(ShaderId, GL_COMPILE_STATUS, &CompileStatus);
	checkf(CompileStatus == 1, "Failed to compile shaders");

	return ShaderId;
}

static GLuint LinkShaders(GLuint VertexShader, GLuint FragmentShader, GLuint GeometryShader)
{
	const GLuint ProgramId = glCreateProgram();
	glAttachShader(ProgramId, VertexShader);
	glAttachShader(ProgramId, FragmentShader);
	glAttachShader(ProgramId, GeometryShader);

	glProgramParameteriEXT(ProgramId, GL_GEOMETRY_INPUT_TYPE, GL_TRIANGLES);
	glProgramParameteriEXT(ProgramId, GL_GEOMETRY_OUTPUT_TYPE, GL_TRIANGLE_STRIP);
	glProgramParameteriEXT(ProgramId, GL_GEOMETRY_VERTICES_OUT, 3);

	glLinkProgram(ProgramId);

	GLint LinkStatus = GL_FALSE;
	glGetProgramiv(ProgramId, GL_LINK_STATUS, &LinkStatus);

	checkfAlways(LinkStatus == GL_TRUE, "Failed to link shaders");

	return ProgramId;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

enum class EAxis
{
	X,
	Y,
	Z
};

inline glm::vec3 ToVec3(Vector3 V)
{
	return { V.X, V.Y, V.Z };
}

static FOrthographicView GetCamera(const FAABB& AABB, const EAxis Axis)
{
	const glm::vec3 Position =
		Axis == EAxis::X
		? ToVec3(AABB.GetCentre()) - ToVec3(AABB.GetHalfSize()).x * glm::vec3{ 1.f, 0.f, 0.f }
		: Axis == EAxis::Y
		? ToVec3(AABB.GetCentre()) - ToVec3(AABB.GetHalfSize()).y * glm::vec3{ 0.f, 1.f, 0.f }
		: ToVec3(AABB.GetCentre()) - ToVec3(AABB.GetHalfSize()).z * glm::vec3{ 0.f, 0.f, 1.f };
	const glm::vec3 Up =
		Axis == EAxis::X
		? glm::vec3{ 0.f, 1.f, 0.f }
		: Axis == EAxis::Y
		? glm::vec3{ 0.f, 0.f, 1.f }
		: glm::vec3{ 1.f, 0.f, 0.f };

	// Figure out clipping planes.
	const std::array<const glm::vec3, 8> Points
	{
		glm::vec3{AABB.Min.X, AABB.Max.Y, AABB.Min.Z}, glm::vec3{AABB.Min.X, AABB.Max.Y, AABB.Max.Z},
		glm::vec3{AABB.Min.X, AABB.Min.Y, AABB.Min.Z}, glm::vec3{AABB.Min.X, AABB.Min.Y, AABB.Max.Z},
		glm::vec3{AABB.Max.X, AABB.Max.Y, AABB.Min.Z}, glm::vec3{AABB.Max.X, AABB.Max.Y, AABB.Max.Z},
		glm::vec3{AABB.Max.X, AABB.Min.Y, AABB.Min.Z}, glm::vec3{AABB.Max.X, AABB.Min.Y, AABB.Max.Z}
	};

	float MinX = std::numeric_limits<float>::max();
	float MinY = std::numeric_limits<float>::max();
	float MinZ = std::numeric_limits<float>::max();
	float MaxX = std::numeric_limits<float>::lowest();
	float MaxY = std::numeric_limits<float>::lowest();
	float MaxZ = std::numeric_limits<float>::lowest();

	FOrthographicView Result;
	Result.LookAt(Position, ToVec3(AABB.GetCentre()), Up);
	{
		const glm::mat4 ViewMatrix = Result.GetViewMatrix();
		for (const auto& Point : Points)
		{
			const glm::vec4 Vector = ViewMatrix * glm::vec4{ Point, 1.0f };

			MinX = std::min(MinX, Vector.x);
			MinY = std::min(MinY, Vector.y);
			MinZ = std::min(MinZ, Vector.z);
			MaxX = std::max(MaxX, Vector.x);
			MaxY = std::max(MaxY, Vector.y);
			MaxZ = std::max(MaxZ, Vector.z);
		}
	}

	Result.Right = MaxX;
	Result.Left = MinX;
	Result.Top = MaxY;
	Result.Bottom = MinY;
	Result.Far = MinZ;
	Result.Near = MaxZ;

	return Result;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::vector<std::size_t> GetNodesToRender(const FScene& Scene)
{
	struct FStackElement
	{
		glm::mat4 ModelMatrix{1.0f};
		std::size_t NodeId = 0;
	};
	
	std::vector<std::size_t> NodesToRender;
	std::stack<FStackElement> Stack;
	for (const auto& Node : Scene.SceneNodesRoots)
	{
		Stack.push({ glm::mat4{1.0f}, Node });
	}

	while (!Stack.empty())
	{
		FStackElement Node = Stack.top();
		Stack.pop();
		const FSceneNode& SceneNode = Scene.SceneNodes[Node.NodeId];

		Node.ModelMatrix = SceneNode.GetMatrix();

		if (SceneNode.HasProperty(FSceneNode::CHILDREN))
		{
			for (std::size_t ChildIndex : SceneNode.Children)
			{
				Stack.push({ Node.ModelMatrix, ChildIndex });
			}
		}

		if (SceneNode.HasProperty(FSceneNode::MESH))
		{
			NodesToRender.push_back(Node.NodeId);
		}
	}
	return NodesToRender;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

FVoxelizer::FVoxelizer(uint32 TexDim, uint32 SubGridSize, const FScene& Scene)
	: TexDim(TexDim)
	, SubGridSize(SubGridSize)
	, Scene(Scene)
	, NodesToRender(GetNodesToRender(Scene))
{
	PROFILE_FUNCTION();

	check(SubGridSize > 0);
	const GLuint VertexShader = CompileShader(GL_VERTEX_SHADER, GetFileContents("Shaders/VertexShader.glsl"));
	const GLuint FragmentShader = CompileShader(GL_FRAGMENT_SHADER,
		AddDefine(
			AddDefine(
				GetFileContents("Shaders/FragmentShader.glsl"),
				"MORTON_BITS", "64"),
			"ENABLE_COLORS", STR(ENABLE_COLORS)));
	const GLuint GeometryShader = CompileShader(GL_GEOMETRY_SHADER, GetFileContents("Shaders/GeometryShader.glsl"));
	VoxelizeShader = LinkShaders(VertexShader, FragmentShader, GeometryShader);

	// Atomic counter.
	glGenBuffers(1, &FragCtrBuffer);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, FragCtrBuffer);
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), nullptr, GL_DYNAMIC_DRAW);
	
	GLuint Zero(0);
	glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &Zero);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

	// Data buffer (pos).
	glGenBuffers(1, &PositionSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, PositionSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER, TexDim * sizeof(uint64), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	// Dummy framebuffer.
	glGenFramebuffers(1, &DummyFBO);
	glNamedFramebufferParameteri(DummyFBO, GL_FRAMEBUFFER_DEFAULT_WIDTH, SubGridSize);
	glNamedFramebufferParameteri(DummyFBO, GL_FRAMEBUFFER_DEFAULT_HEIGHT, SubGridSize);
	glBindFramebuffer(GL_FRAMEBUFFER, DummyFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	std::size_t NumBytes = 0;
	CUDA_CHECKED_CALL cudaGraphicsGLRegisterBuffer(&CudaPositionResource, PositionSSBO, cudaGraphicsMapFlagsNone);
	CUDA_CHECKED_CALL cudaGraphicsMapResources(1, &CudaPositionResource);
	CUDA_CHECKED_CALL cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&Positions), &NumBytes, CudaPositionResource);

	FMemory::RegisterCustomAlloc(Positions, "OpenGL Fragments", TexDim * sizeof(uint64), EMemoryType::GPU);
}

FVoxelizer::~FVoxelizer()
{
	PROFILE_FUNCTION();
	
	glDeleteBuffers(1, &FragCtrBuffer);
	glDeleteBuffers(1, &PositionSSBO);
	glDeleteFramebuffers(1, &DummyFBO);

	CUDA_CHECKED_CALL cudaGraphicsUnmapResources(1, &CudaPositionResource);

	FMemory::UnregisterCustomAlloc(Positions);
}

TGpuArray<uint64> FVoxelizer::GenerateFragments(const FAABB& AABB) const
{
	PROFILE_FUNCTION();
	
	// Get ortho camera for given axis.
	const FOrthographicView ViewX = GetCamera(AABB, EAxis::X);
	const FOrthographicView ViewY = GetCamera(AABB, EAxis::Y);
	const FOrthographicView ViewZ = GetCamera(AABB, EAxis::Z);

	// Note: we'll run out of memory before overflowing the 32 bit counter
	GLuint FragCount = 0;

	glUseProgram(VoxelizeShader);
	{
		GLuint Location;
		Location = glGetUniformLocation(VoxelizeShader, "proj_x");
		glUniformMatrix4fv(Location, 1, false, value_ptr(GetProjectionViewMatrix(ViewX)));
		Location = glGetUniformLocation(VoxelizeShader, "proj_y");
		glUniformMatrix4fv(Location, 1, false, value_ptr(GetProjectionViewMatrix(ViewY)));
		Location = glGetUniformLocation(VoxelizeShader, "proj_z");
		glUniformMatrix4fv(Location, 1, false, value_ptr(GetProjectionViewMatrix(ViewZ)));

		Location = glGetUniformLocation(VoxelizeShader, "grid_dim");
		glUniform1i(Location, SubGridSize);
		Location = glGetUniformLocation(VoxelizeShader, "aabb_size");
		glUniform3fv(Location, 1, value_ptr(ToVec3(AABB.GetHalfSize())));

		glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, FragCtrBuffer);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, PositionSSBO);

		glBindFramebuffer(GL_FRAMEBUFFER, DummyFBO);
		{
			glViewport(0, 0, SubGridSize, SubGridSize);
			glDisable(GL_CULL_FACE);
			glDisable(GL_DEPTH_TEST);
			glDisable(GL_MULTISAMPLE);

			if (glewIsExtensionSupported("GL_NV_conservative_raster")) glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				Draw();
			}
			if (glewIsExtensionSupported("GL_NV_conservative_raster")) glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// Get number of written fragments.
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, FragCtrBuffer);
		glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &FragCount);
		uint32 zero_counter(0);
		glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &zero_counter);
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
	}
	glUseProgram(0);
	checkfAlways(FragCount <= TexDim, "TexDim too low: is %u, needs to be >= %u", TexDim, FragCount);
	return { Positions, FragCount };
}

void FVoxelizer::Draw() const
{
	PROFILE_FUNCTION_TRACY();
	
	for (const auto& NodeToRender : NodesToRender) 
	{
		const auto& Node = Scene.SceneNodes[NodeToRender];
		const auto& Mesh = Scene.Meshes[Node.MeshId];
		const auto& Primitives = Mesh.Primitives;

		glBindVertexArray(MASTER_VAO);
		for (const auto& Primitive : Primitives) 
		{
			for (const auto& Attribute : Primitive.Attributes) 
			{
				if (Attribute.Index == FAttribute::ERROR_TYPE) continue;
				const auto& Accessor = Scene.Accessors[Attribute.Accessor];
				const auto& BufferView = Scene.BufferViews[Accessor.BufferView];
				const auto& Buffer = Scene.BufferObjects[BufferView.Buffer];
				glEnableVertexAttribArray(Attribute.Index);
				glBindBuffer(GL_ARRAY_BUFFER, Buffer);
				void* Offset = reinterpret_cast<void*>(Accessor.ByteOffset + BufferView.ByteOffset);
				glVertexAttribPointer(Attribute.Index, Accessor.TypeSize, Accessor.ComponentType, GL_FALSE, 0, Offset);
			}

			const auto BindTexture = [&](GLuint TextureIndex, GLint Unit)
			{
				if (TextureIndex != NO_ID_SENTINEL)
				{
					const auto Texture = Scene.Textures[TextureIndex];
					glActiveTexture(GL_TEXTURE0 + Unit);
					glBindTexture(GL_TEXTURE_2D, Texture);
				}
			};

			const GLuint Location = glGetUniformLocation(VoxelizeShader, "modelMatrix");
			glUniformMatrix4fv(Location, 1, false, value_ptr(Node.GetMatrix()));

			const auto& Material = Scene.Materials[Primitive.Material];
			if (Material.Type == FMaterial::EType::METALLIC_ROUGHNESS)
			{
				const auto& PBR = Material.MetallicRoughness;
				BindTexture(PBR.BaseColorTexture, 0);
			}
			else 
			{
				checkAlways(Material.Type == FMaterial::EType::SPECULAR_GLOSSINESS);
				const auto& PBR = Material.SpecularGlossiness;
				BindTexture(PBR.DiffuseTexture, 0);
			}

			const auto& Indices = Scene.Accessors[Primitive.Indices];
			const auto& BufferView = Scene.BufferViews[Indices.BufferView];
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Scene.BufferObjects[BufferView.Buffer]);
			void* offset = reinterpret_cast<void*>(Indices.ByteOffset + BufferView.ByteOffset);
			glDrawElements(GL_TRIANGLES, Indices.Count, Indices.ComponentType, offset);
		}
	}
}