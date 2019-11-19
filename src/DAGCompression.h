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
};

struct FCpuDag
{
	std::vector<FCpuLevel> Levels;
	TStaticArray<uint64, EMemoryType::CPU> Leaves;
};

namespace DAGCompression
{
	FCpuDag CreateDAG(const TStaticArray<uint64, EMemoryType::GPU>& Fragments, int32 Depth);
	FCpuDag MergeDAGs(const std::vector<FCpuDag>& CpuDags);
	TStaticArray<uint32, EMemoryType::GPU> CreateFinalDAG(FCpuDag&& CpuDag);
	void FreeAll();
}