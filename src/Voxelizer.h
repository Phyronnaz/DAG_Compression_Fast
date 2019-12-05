#pragma once

#include "Core.h"
#include "GL/glew.h"
#include "Utils/Aabb.h"
#include "GLTFLoader.h"
#include "Array.h"

class FVoxelizer
{
public:
	const int32 SubGridSize;
	const FScene& Scene;
	static constexpr int32 TexDim = VOXELIZER_MAX_NUM_FRAGMENTS;

private:
	const std::vector<std::size_t> NodesToRender;

	GLuint VoxelizeShader = 0;
	GLuint FragCtrBuffer = 0;
	GLuint PositionSSBO = 0;
	GLuint DummyFBO = 0;
	
	cudaGraphicsResource* CudaPositionResource = nullptr;
	uint64* Positions = nullptr;

public:
	explicit FVoxelizer(int32 SubGridSize, const FScene& Scene);
	~FVoxelizer();

	// Array is only valid while the voxelizer object hasn't been destroyed!
	TGpuArray<uint64> GenerateFragments(const FAABB& AABB) const;

private:
	void Draw() const;
};
