#pragma once

#include "Core.h"
#include "GL/glew.h"
#include "Utils/Aabb.h"
#include "GLTFLoader.h"

class FVoxelizer
{
public:
	const int32 GridSize;
	const FScene& Scene;
	static constexpr int32 TexDim = 128 * 1024 * 1024;

private:
	const std::vector<std::size_t> NodesToRender;

	GLuint VoxelizeShader = 0;
	GLuint FragCtrBuffer = 0;
	GLuint PositionSSBO = 0;
	GLuint DummyFBO = 0;
	
	cudaGraphicsResource* CudaPositionResource = nullptr;
	uint64* Positions = nullptr;

public:
	explicit FVoxelizer(int32 GridSize, const FScene& Scene);
	~FVoxelizer();

	struct FResult
	{
		// GPU array!
		uint64* Positions = nullptr;
		uint64 Count = 0;
	};
	// FResult is only valid while the voxelizer object hasn't been destroyed!
	FResult GenerateFragments(const FAABB& AABB, int32 GridResolution) const;

private:
	void Draw() const;
};