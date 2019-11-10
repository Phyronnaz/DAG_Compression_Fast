#include "GLTFLoader.h"

#include <fstream>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <glm/gtc/type_ptr.hpp>
#include <stb/stb_image.h>

GLuint MASTER_VAO;

using json = nlohmann::json;

template<typename T>
std::optional<T> GetOptionalField(const json& Json, const std::string& Field)
{
	const auto It = Json.find(Field);
	if (It != Json.end()) return { *It };
	return {};
}

FAttribute::EType GetAttributeType(const std::string& TypeString)
{
	if (TypeString == "POSITION") return FAttribute::EType::POSITION;
	else if (TypeString == "NORMAL") return FAttribute::EType::NORMAL;
	else if (TypeString == "TANGENT") return FAttribute::EType::TANGENT;
	else if (TypeString == "TEXCOORD_0") return FAttribute::EType::TEXCOORD_0;
	else if (TypeString == "TEXCOORD_1") return FAttribute::EType::TEXCOORD_1;
	checkAlways(false); return FAttribute::EType::ERROR_TYPE;
}
FAccessor::EType GetAccessorType(const std::string& TypeString)
{
	if (TypeString == "SCALAR") return FAccessor::EType::SCALAR;
	else if (TypeString == "VEC2") return FAccessor::EType::VEC2;
	else if (TypeString == "VEC3") return FAccessor::EType::VEC3;
	else if (TypeString == "VEC4") return FAccessor::EType::VEC4;
	else if (TypeString == "MAT2") return FAccessor::EType::MAT2;
	else if (TypeString == "MAT3") return FAccessor::EType::MAT3;
	else if (TypeString == "MAT4") return FAccessor::EType::MAT4;
	checkAlways(false); return FAccessor::EType::ERROR_TYPE;
}

FMaterial::EAlphaMode GetAlphaMode(const std::optional<std::string>& Mode)
{
	if (Mode)
	{
		if (Mode == "OPAQUE") return FMaterial::EAlphaMode::ALPHA_OPAQUE;
		if (Mode == "MASK") return FMaterial::EAlphaMode::ALPHA_OPAQUE;
		if (Mode == "BLEND") return FMaterial::EAlphaMode::ALPHA_BLEND;
		checkAlways(false);
	}
	return FMaterial::EAlphaMode::ALPHA_OPAQUE;
}

GLint getTypeSize(const FAccessor::EType Type)
{
	switch (Type)
	{
	case FAccessor::EType::SCALAR: return 1;
	case FAccessor::EType::VEC2: return 2;
	case FAccessor::EType::VEC3: return 3;
	case FAccessor::EType::VEC4: return 4;
	case FAccessor::EType::MAT2: return 4;
	case FAccessor::EType::MAT3: return 9;
	case FAccessor::EType::MAT4: return 16;
	case FAccessor::EType::ERROR_TYPE:
	default:
		checkAlways(false);
		return -1;
	}
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

inline FGLTFLoader*& GlobalLoader()
{
	static FGLTFLoader* Loader = nullptr;
	return Loader;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void from_json(const json& Json, FBufferView& BufferView)
{
	Json.at("buffer").get_to(BufferView.Buffer);
	Json.at("byteLength").get_to(BufferView.ByteLength);
	BufferView.ByteOffset = GetOptionalField<GLuint>(Json, "byteOffset").value_or(0);
	BufferView.Target = GetOptionalField<GLuint>(Json, "target");
}

void from_json(const json& Json, FAccessor& Accessor)
{
	Json.at("bufferView").get_to(Accessor.BufferView);
	Json.at("componentType").get_to(Accessor.ComponentType);
	Json.at("count").get_to(Accessor.Count);
	Accessor.Type = GetAccessorType(Json.at("type"));
	Accessor.TypeSize = getTypeSize(Accessor.Type);
	Accessor.ByteOffset = GetOptionalField<GLuint>(Json, "byteOffset").value_or(0);
}

void from_json(const json& Json, FPrimitive& Primitive)
{
	for (auto It = Json["attributes"].begin(); It != Json["attributes"].end(); ++It)
	{
		Primitive.Attributes.emplace_back(FAttribute{GetAttributeType(It.key()), It.value()});
	}
	Json.at("indices").get_to(Primitive.Indices);
	Json.at("material").get_to(Primitive.Material);
}

void from_json(const json& Json, FMesh& Mesh)
{
	Mesh.Name = GetOptionalField<std::string>(Json, "name").value_or("Unnamed");
	Json.at("primitives").get_to(Mesh.Primitives);
}

void from_json(const json& Json, FSceneNode& SceneNode)
{
	SceneNode.Name = GetOptionalField<std::string>(Json, "name").value_or("Unnamed");

	if (const auto It = Json.find("mesh"); It != Json.end())
	{
		It->get_to(SceneNode.MeshId);
		SceneNode.Properties |= FSceneNode::EFlag::MESH;
	}

	if (const auto It = Json.find("children"); It != Json.end())
	{
		It->get_to(SceneNode.Children);
		SceneNode.Properties |= FSceneNode::EFlag::CHILDREN;
	}

	if (const auto It = Json.find("translation"); It != Json.end())
	{
		SceneNode.Translation = glm::vec3{(*It)[0], (*It)[1], (*It)[2]};
		SceneNode.Properties |= FSceneNode::EFlag::TRANSLATION;
	}

	if (const auto It = Json.find("rotation"); It != Json.end())
	{
		// NOTE: Constructor for glm::quat is quat(w,x,y,z), whereas the glTF quaternion is stored as xyzw.
		SceneNode.Rotation = glm::quat{(*It)[3], (*It)[0], (*It)[1], (*It)[2]};
		SceneNode.Properties |= FSceneNode::EFlag::ROTATION;
	}

	if (const auto It = Json.find("scale"); It != Json.end())
	{
		SceneNode.Scale = glm::vec3{(*It)[0], (*It)[1], (*It)[2]};
		SceneNode.Properties |= FSceneNode::EFlag::SCALE;
	}

	if (const auto It = Json.find("matrix"); It != Json.end())
	{
		std::vector<float> V = *It;
		SceneNode.Matrix = glm::make_mat4(V.data());
		SceneNode.Properties |= FSceneNode::EFlag::MATRIX;
	}
}

void from_json(const json& Json, FMaterial& Material)
{
	//j.at("name").get_to(m.name);
	// FIXME: Move optional part into getAlphaMode
	Material.AlphaMode = GetAlphaMode(GetOptionalField<std::string>(Json, "alphaMode"));

	// TODO: A glTF material can have both MetallicRoughness and SpecularGlossiness defined.
	//       Add support for both. Now we default to SpecularGlossiness.
	if (const auto Ext = Json.find("extensions"); Ext != Json.end())
	{
		if (const auto SpecPbr = Ext->find("KHR_materials_pbrSpecularGlossiness"); SpecPbr != Ext->end())
		{
			Material.Type = FMaterial::EType::SPECULAR_GLOSSINESS;
			Material.SpecularGlossiness.DiffuseTexture = GlobalLoader()->GetTextureId(*SpecPbr, "diffuseTexture");
			Material.SpecularGlossiness.SpecularGlossinessTexture = GlobalLoader()->GetTextureId(*SpecPbr, "specularGlossinessTexture");
		}
		else if (const auto MetallicPbr = Json.find("pbrMetallicRoughness"); MetallicPbr != Json.end())
		{
			Material.Type = FMaterial::EType::METALLIC_ROUGHNESS;
			Material.MetallicRoughness.BaseColorTexture = GlobalLoader()->GetTextureId(*MetallicPbr, "baseColorTexture");
			Material.MetallicRoughness.MetallicRoughnessTexture = GlobalLoader()->GetTextureId(*MetallicPbr, "metallicRoughnessTexture");
			const auto It = MetallicPbr->find("baseColorFactor");
			checkAlways(It != MetallicPbr->end());
			if (It != MetallicPbr->end())
			{
				Material.MetallicRoughness.BaseColorFactor[0] = (*It)[0];
				Material.MetallicRoughness.BaseColorFactor[1] = (*It)[1];
				Material.MetallicRoughness.BaseColorFactor[2] = (*It)[2];
			}
		}
	}
	else if (const auto Pbr = Json.find("pbrMetallicRoughness"); Pbr != Json.end())
	{
		Material.Type = FMaterial::EType::METALLIC_ROUGHNESS;
		Material.MetallicRoughness.BaseColorTexture = GlobalLoader()->GetTextureId(*Pbr, "baseColorTexture");
		Material.MetallicRoughness.MetallicRoughnessTexture = GlobalLoader()->GetTextureId(*Pbr, "metallicRoughnessTexture");
		const auto It = Pbr->find("baseColorFactor");
		if (It != Pbr->end())
		{
			Material.MetallicRoughness.BaseColorFactor[0] = (*It)[0];
			Material.MetallicRoughness.BaseColorFactor[1] = (*It)[1];
			Material.MetallicRoughness.BaseColorFactor[2] = (*It)[2];
		}
	}
	Material.NormalTexture = GlobalLoader()->GetTextureId(Json, "normalTexture");
	Material.OcclusionTexture = GlobalLoader()->GetTextureId(Json, "occlusionTexture");
	Material.BakedTexture = GlobalLoader()->GetTextureId(Json, "bakedTexture");
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

FAABB GetAABB(const FScene& Scene, const FAccessor& Accessor)
{
	checkAlways(Accessor.Type == FAccessor::EType::VEC3);

	const auto& BufferView = Scene.BufferViews[Accessor.BufferView];
	const std::size_t Offset = Accessor.ByteOffset + BufferView.ByteOffset;

	const GLuint VBOId = Scene.BufferObjects[BufferView.Buffer];

	glBindBuffer(GL_ARRAY_BUFFER, VBOId);
	std::byte* Bytes = reinterpret_cast<std::byte*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY)) + Offset;
	const float* Buffer = reinterpret_cast<float*>(Bytes);
	FAABB AABB = MakeInverseExtremeAABB();
	for (int32 Index = 0; Index < Accessor.Count; ++Index)
	{
		AABB = Combine(AABB, glm::vec3(Buffer[0 + 3 * Index], Buffer[1 + 3 * Index], Buffer[2 + 3 * Index]));
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
	return AABB;
}

FAABB GetAABB(const FScene& Scene, const FPrimitive& Primitive)
{
	FAABB AABB = MakeInverseExtremeAABB();
	for (const auto& attr : Primitive.Attributes)
	{
		if (attr.Index == FAttribute::POSITION)
		{
			AABB = Combine(AABB, GetAABB(Scene, Scene.Accessors[attr.Accessor]));
		}
	}
	return AABB;
}

FAABB GetAABB(const FScene& Scene, const FSceneNode& Node)
{
	FAABB AABB = MakeInverseExtremeAABB();
	// Construct AABB containing all primitive AABB.
	if (Node.HasProperty(FSceneNode::MESH))
	{
		const FMesh& Mesh = Scene.Meshes[Node.MeshId];
		for (auto& Primitive : Mesh.Primitives)
		{
			AABB = Combine(AABB, GetAABB(Scene, Primitive));
		}
	}
	// Combine with children AABB.
	for (auto& Child : Node.Children)
	{
		// Note: highly inefficient, but w/e
		AABB = Combine(AABB, GetAABB(Scene, Scene.SceneNodes[Child]));
	}
	return AABB;
}

FAABB GetAABB(const FScene& Scene)
{
	FAABB AABB = MakeInverseExtremeAABB();
	for (auto& SceneNode : Scene.SceneNodesRoots)
	{
		AABB = Combine(AABB, GetAABB(Scene, Scene.SceneNodes[SceneNode]));
	}
	return AABB;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

GLuint ReadTexture_RGBA8888(const std::string& Filename)
{
	int32 Width, Height, Components;
	stbi_set_flip_vertically_on_load(false);

	struct SafeFree
	{
		void operator()(stbi_uc* data) const
		{
			if (data) free(data);
		}
	};
	const std::unique_ptr<stbi_uc, SafeFree> Data(stbi_load(Filename.c_str(), &Width, &Height, &Components, 4));

	if (!Data)
	{
		std::cout << "Failed to load texture: " << Filename << std::endl;
		checkfAlways(false, "Error: %s", stbi_failure_reason());
	}

	GLuint Texture;
	glGenTextures(1, &Texture);
	glBindTexture(GL_TEXTURE_2D, Texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, Width, Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, Data.get());
	glGenerateMipmap(GL_TEXTURE_2D);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	return Texture;
}

GLuint UploadBin(const std::string& File, GLsizeiptr Count)
{
	// If count is zero, figure out file size.
	std::ifstream BinaryFile(File, ((Count == 0) ? (std::ios::binary | std::ios::ate) : std::ios::binary));
	if (Count == 0)
	{
		Count = BinaryFile.tellg();
		BinaryFile.seekg(0);
	}

	if (MASTER_VAO == 0)
	{
		glGenVertexArrays(1, &MASTER_VAO);
	}
	glBindVertexArray(MASTER_VAO);

	GLuint VBOId;
	glGenBuffers(1, &VBOId);
	glBindBuffer(GL_ARRAY_BUFFER, VBOId);
	glBufferData(GL_ARRAY_BUFFER, Count, nullptr, GL_STATIC_DRAW);
	void* Tmp = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	BinaryFile.read(reinterpret_cast<char*>(Tmp), Count);
	glUnmapBuffer(GL_ARRAY_BUFFER);
	return VBOId;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

FGLTFLoader::FGLTFLoader(const std::string& Path, const std::string& Filename)
	: Path(Path)
{
	std::ifstream File(Path + Filename);
	checkAlways(File.is_open());

	File >> const_cast<json&>(Root);

	GlobalLoader() = this;
}

FScene FGLTFLoader::GetScene()
{
	FScene Result;
	Root["bufferViews"].get_to(Result.BufferViews);
	Root["accessors"].get_to(Result.Accessors);
	Root["meshes"].get_to(Result.Meshes);
	Root["nodes"].get_to(Result.SceneNodes);
	Root["materials"].get_to(Result.Materials);
	const unsigned SceneId = Root["scene"];
	const json& Nodes = Root["scenes"][SceneId]["nodes"];
	Nodes.get_to(Result.SceneNodesRoots);

	for (const auto& Image : Root["images"])
	{
		const std::string Filename = Image["uri"];
		Result.Textures.emplace_back(ReadTexture_RGBA8888(Path + Filename));
	}

	for (auto& Buffer : Root["buffers"])
	{
		std::string Filename = Path + std::string(Buffer["uri"]);
		const GLsizeiptr Count = GetOptionalField<GLsizeiptr>(Buffer, "byteLength").value_or(0);
		Result.BufferObjects.push_back(UploadBin(Filename, Count));
	}

	Result.AABB = GetAABB(Result);
	return Result;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

GLuint FGLTFLoader::GetTextureId(const json& Material, const std::string& Name) const
{
	if (const auto It = Material.find(Name); It != Material.end())
	{
		const unsigned SourceId = (*It)["index"];
		const json& Texture = Root["textures"][SourceId];
		/* Sampler etc... */
		return Texture["source"];
	}
	return NO_ID_SENTINEL;
}