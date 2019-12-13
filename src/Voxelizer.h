#pragma once

#include "Core.h"
#include "GL/glew.h"
#include "Array.h"

class FAABB;
struct FScene;

class FVoxelizer
{
public:
	const uint32 TexDim;
	const uint32 SubGridSize;
	const FScene& Scene;

private:
	const std::vector<std::size_t> NodesToRender;

	GLuint VoxelizeShader = 0;
	GLuint FragCtrBuffer = 0;
	GLuint DummyFBO = 0;

	static constexpr uint32 NumBuffers = 2;
	GLuint PositionSSBOs[NumBuffers];
	uint64* Positions[NumBuffers];
	cudaGraphicsResource* PositionsCudaResource[NumBuffers];
	int32 ActiveBuffer = 0;

public:
	explicit FVoxelizer(uint32 TexDim, uint32 SubGridSize, const FScene& Scene);
	~FVoxelizer();

	// Array is only valid while the voxelizer object hasn't been destroyed!
	TGpuArray<uint64> GenerateFragments(const FAABB& AABB) const;

	inline void SwapBuffers()
	{
		ActiveBuffer = (ActiveBuffer + 1) % NumBuffers;
	}

private:
	void Draw() const;
};
