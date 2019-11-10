#pragma once

#include "Core.h"
#include <optional>
#include <GL/glew.h>
#include "nlohmann/json.hpp"
#include "glm/detail/type_quat.hpp"
#include "glm/gtc/quaternion.hpp"
#include "Utils/Aabb.h"

struct FBufferView
{
	GLuint Buffer = 0;
	GLsizei ByteLength = 0;
	GLsizeiptr ByteOffset = 0;
	std::optional<GLbitfield> Target;
};

struct FAccessor
{
	enum class EType
	{
		SCALAR,
		VEC2,
		VEC3,
		VEC4,
		MAT2,
		MAT3,
		MAT4,
		ERROR_TYPE,
	};
	std::size_t BufferView;
	GLbitfield ComponentType;
	GLsizei Count;
	GLsizeiptr ByteOffset;
	EType Type;
	GLint TypeSize;
};

struct FAttribute
{
	enum EType : GLint
	{
		POSITION = 0,
		NORMAL = 1,
		TANGENT = 2,
		TEXCOORD_0 = 3,
		TEXCOORD_1 = 4,
		ERROR_TYPE = -1,
	};
	EType Index;
	std::size_t Accessor;
};

struct FPrimitive
{
	std::vector<FAttribute> Attributes;
	std::size_t Material;
	std::size_t Indices;
};

struct FMesh
{
	std::vector<FPrimitive> Primitives;
	std::string Name;
};

struct FMaterial
{
	enum class EAlphaMode
	{
		ALPHA_OPAQUE,
		ALPHA_MASK,
		ALPHA_BLEND
	};
	enum class EType
	{
		METALLIC_ROUGHNESS,
		SPECULAR_GLOSSINESS
	};
	struct FPBRMetallicRoughness
	{
		GLuint BaseColorTexture;
		GLuint MetallicRoughnessTexture;
		float BaseColorFactor[3];
	};
	struct FPBRSpecularGlossiness
	{
		GLuint DiffuseTexture;
		GLuint SpecularGlossinessTexture;
		float DiffuseFactor[3];
	};
	union
	{
		FPBRMetallicRoughness MetallicRoughness;
		FPBRSpecularGlossiness SpecularGlossiness;
	};
	GLuint NormalTexture;
	GLuint OcclusionTexture;
	GLuint BakedTexture;
	EAlphaMode AlphaMode;
	EType Type;
	std::string Name;
};

struct FSceneNode
{
	using FlagType = uint16;
	enum EFlag : FlagType
	{
		MESH = 1 << 0,
		ROTATION = 1 << 1,
		TRANSLATION = 1 << 2,
		SCALE = 1 << 3,
		MATRIX = 1 << 4,
		CHILDREN = 1 << 5,
	};

	bool HasProperty(EFlag Flag) const
	{
		return Properties & Flag;
	}

	glm::quat Rotation{ 1.0f, 0.0f, 0.0f, 0.0f };
	glm::vec3 Translation{ 0.0f };
	glm::vec3 Scale{ 1.0f };
	glm::mat4 Matrix{ 1.0f };
	std::size_t MeshId;
	FlagType Properties = 0;
	std::vector<std::size_t> Children;
	std::string Name;

	inline glm::mat4 GetMatrix() const
	{
#if 0
		return
			glm::translate(glm::mat4{ 1.0f }, Translation) *
			glm::mat4_cast(Rotation) *
			glm::scale(glm::mat4{ 1.0f }, Scale);
#else
		// Just use identity to skip transforming the AABBs
		return glm::mat4{ 1.f };
#endif
	}
};

struct FScene
{
	std::vector<FMesh> Meshes;
	std::vector<FBufferView> BufferViews;
	std::vector<FAccessor> Accessors;
	std::vector<FMaterial> Materials;
	std::vector<GLuint> Textures;
	std::vector<FSceneNode> SceneNodes;
	std::vector<std::size_t> SceneNodesRoots;
	std::vector<GLuint> BufferObjects;
	FAABB AABB;
};

extern GLuint MASTER_VAO;
constexpr GLuint NO_ID_SENTINEL = 0xFFFFFFFF;

inline void FreeScene(FScene& scene)
{
	glDeleteBuffers(uint32(scene.BufferObjects.size()), scene.BufferObjects.data());
	glDeleteTextures(uint32(scene.Textures.size()), scene.Textures.data());
	glDeleteVertexArrays(1, &MASTER_VAO);
	MASTER_VAO = 0;
}

class FGLTFLoader
{
public:
	using json = nlohmann::json;
	
	const std::string Path;
	const json Root;

	FGLTFLoader(const std::string& Path, const std::string& Filename);
	
	FScene GetScene();
	GLuint GetTextureId(const json& Material, const std::string& Name) const;
};