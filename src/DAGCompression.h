#pragma once

#include "Core.h"
#include "Array.h"

struct FChildrenIndices
{
	uint32 Indices[8];
};
static_assert(sizeof(FChildrenIndices) == 8 * sizeof(uint32), "");

struct FCpuLevel
{
	TStaticArray<uint8, EMemoryType::CPU> ChildMasks;
	TStaticArray<FChildrenIndices, EMemoryType::CPU> ChildrenIndices;
	TStaticArray<uint64, EMemoryType::CPU> Hashes;

	inline void Free()
	{
		ChildMasks.Free();
		ChildrenIndices.Free();
		Hashes.Free();
	}
};

struct FCpuDag
{
	std::vector<FCpuLevel> Levels;
	TStaticArray<uint64, EMemoryType::CPU> Leaves;

	inline void Free()
	{
		for(auto& Level : Levels)
		{
			Level.Free();
		}
		Levels.clear();
		Leaves.Free();
	}
};

namespace DAGCompression
{
	FCpuDag CreateDAG(const TStaticArray<uint64, EMemoryType::GPU>& Fragments, int32 Depth);
	FCpuDag MergeDAGs(std::vector<FCpuDag>&& CpuDags);
	TStaticArray<uint32, EMemoryType::GPU> CreateFinalDAG(FCpuDag&& CpuDag);
	void FreeAll();
}