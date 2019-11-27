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
	TStaticArray<uint32, EMemoryType::CPU> Colors;

	inline void Free()
	{
		for(auto& Level : Levels)
		{
			Level.Free();
		}
		Levels.clear();
		Leaves.Free();
		Colors.Free();
	}
};

struct FFinalDag
{
	TStaticArray<uint32, EMemoryType::GPU> Dag;
	TStaticArray<uint32, EMemoryType::GPU> Colors;
	TStaticArray<uint64, EMemoryType::GPU> EnclosedLeaves;
};

namespace DAGCompression
{
	FCpuDag CreateSubDAG(const TStaticArray<FMortonCode, EMemoryType::GPU>& Fragments);
	FCpuDag MergeDAGs(std::vector<FCpuDag>&& CpuDags);
	FFinalDag CreateFinalDAG(FCpuDag&& CpuDag);
	void FreeAll();

	void Test();
}