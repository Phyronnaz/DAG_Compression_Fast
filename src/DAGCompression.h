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

struct FGpuLevel
{
	TStaticArray<uint8, EMemoryType::GPU> ChildMasks;
	TStaticArray<FChildrenIndices, EMemoryType::GPU> ChildrenIndices;
	TStaticArray<uint64, EMemoryType::GPU> Hashes;

	FCpuLevel ToCPU() const
	{
		FCpuLevel CpuLevel;

		CpuLevel.ChildMasks = TStaticArray<uint8, EMemoryType::CPU>("ChildMasks", ChildMasks.Num());
		CpuLevel.ChildrenIndices = TStaticArray<FChildrenIndices, EMemoryType::CPU>("ChildrenIndices", ChildrenIndices.Num());
		CpuLevel.Hashes = TStaticArray<uint64, EMemoryType::CPU>("Hashes", Hashes.Num());

		ChildMasks.CopyToCPU(CpuLevel.ChildMasks);
		ChildrenIndices.CopyToCPU(CpuLevel.ChildrenIndices);
		Hashes.CopyToCPU(CpuLevel.Hashes);

		return CpuLevel;
	}
	void FromCPU(const FCpuLevel& CpuLevel, const bool LoadHashes = true)
	{
		ChildMasks = TStaticArray<uint8, EMemoryType::GPU>("ChildMasks", CpuLevel.ChildMasks.Num());
		ChildrenIndices = TStaticArray<FChildrenIndices, EMemoryType::GPU>("ChildrenIndices", CpuLevel.ChildrenIndices.Num());
		if (LoadHashes) Hashes = TStaticArray<uint64, EMemoryType::GPU>("Hashes", CpuLevel.Hashes.Num());

		CpuLevel.ChildMasks.CopyToGPU(ChildMasks);
		CpuLevel.ChildrenIndices.CopyToGPU(ChildrenIndices);
		if (LoadHashes) CpuLevel.Hashes.CopyToGPU(Hashes);
	}

	void Free(bool FreeHashes = true)
	{
		ChildMasks.Free();
		ChildrenIndices.Free();
		if (FreeHashes) Hashes.Free();
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
	FCpuDag CreateSubDAG(TStaticArray<uint64, EMemoryType::GPU>& Fragments);
	FCpuDag MergeDAGs(std::vector<FCpuDag>&& CpuDags);
	FFinalDag CreateFinalDAG(FCpuDag&& CpuDag);

	// Will trash Fragments
	TStaticArray<uint64, EMemoryType::GPU> SortFragmentsAndRemoveDuplicates(TStaticArray<uint64, EMemoryType::GPU>& Fragments);
	TStaticArray<uint32, EMemoryType::GPU> ExtractColorsAndFixFragments(TStaticArray<uint64, EMemoryType::GPU>& Fragments);
	TStaticArray<uint64, EMemoryType::GPU> ExtractLeavesAndShiftReduceFragments(TStaticArray<uint64, EMemoryType::GPU>& Fragments);
	// If leaves: Level only has Hashes valid
	void SortLevel(FGpuLevel& Level, TStaticArray<uint32, EMemoryType::GPU>& OutHashesToSortedUniqueHashes);
	FGpuLevel ExtractLevelAndShiftReduceFragments(TStaticArray<uint64, EMemoryType::GPU>& Fragments, const TStaticArray<uint32, EMemoryType::GPU>& FragmentIndicesToChildrenIndices);
	void ComputeHashes(FGpuLevel& Level, const TStaticArray<uint64, EMemoryType::GPU>& LowerLevelHashes);
}