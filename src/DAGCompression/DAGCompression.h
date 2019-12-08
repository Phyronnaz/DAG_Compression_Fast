#pragma once

#include "Core.h"
#include "Array.h"

class FAABB;
struct FScene;

struct FChildrenIndices
{
	uint32 Indices[8];
};
static_assert(sizeof(FChildrenIndices) == 8 * sizeof(uint32), "");

HOST_DEVICE uint64 HashChildrenHashes(uint64 ChildrenHashes[8])
{
	const uint64 Hash = Utils::MurmurHash128xN(reinterpret_cast<const uint128*>(ChildrenHashes), 4).Data[0];
	check(Hash != 0); // Can remove this check, just want to see if it actually happens
	return Hash == 0 ? 1 : Hash;
}

struct FCpuLevel
{
	TCpuArray<uint8> ChildMasks;
	TCpuArray<FChildrenIndices> ChildrenIndices;
	TCpuArray<uint64> Hashes;

	template<typename T>
	inline void Free(T& Allocator)
	{
		ChildMasks.Free(Allocator);
		ChildrenIndices.Free(Allocator);
		Hashes.Free(Allocator);
	}
};

struct FGpuLevel
{
	TGpuArray<uint8> ChildMasks;
	TGpuArray<FChildrenIndices> ChildrenIndices;
	TGpuArray<uint64> Hashes;

	FCpuLevel ToCPU() const
	{
		FCpuLevel CpuLevel;

		CpuLevel.ChildMasks = TCpuArray<uint8>("ChildMasks", ChildMasks.Num());
		CpuLevel.ChildrenIndices = TCpuArray<FChildrenIndices>("ChildrenIndices", ChildrenIndices.Num());
		CpuLevel.Hashes = TCpuArray<uint64>("Hashes", Hashes.Num());

		ChildMasks.CopyToCPU(CpuLevel.ChildMasks);
		ChildrenIndices.CopyToCPU(CpuLevel.ChildrenIndices);
		Hashes.CopyToCPU(CpuLevel.Hashes);

		return CpuLevel;
	}
	void FromCPU(const FCpuLevel& CpuLevel, const bool LoadHashes = true)
	{
		ChildMasks = TGpuArray<uint8>("ChildMasks", CpuLevel.ChildMasks.Num());
		ChildrenIndices = TGpuArray<FChildrenIndices>("ChildrenIndices", CpuLevel.ChildrenIndices.Num());
		if (LoadHashes) Hashes = TGpuArray<uint64>("Hashes", CpuLevel.Hashes.Num());

		CpuLevel.ChildMasks.CopyToGPU(ChildMasks);
		CpuLevel.ChildrenIndices.CopyToGPU(ChildrenIndices);
		if (LoadHashes) CpuLevel.Hashes.CopyToGPU(Hashes);
	}

	template<typename T>
	void Free(T& Allocator, bool FreeHashes = true)
	{
		ChildMasks.Free(Allocator);
		ChildrenIndices.Free(Allocator);
		if (FreeHashes) Hashes.Free(Allocator);
	}
};

struct FCpuDag
{
	std::vector<FCpuLevel> Levels;
	TCpuArray<uint64> Leaves;
	TCpuArray<uint32> Colors;

	template<typename T>
	inline void Free(T& Allocators)
	{
		for(auto& Level : Levels)
		{
			Level.Free(Allocators);
		}
		Levels.clear();
		Leaves.Free(Allocators);
		Colors.Free(Allocators);
	}
};

struct FFinalDag
{
	TGpuArray<uint32> Dag;
	TGpuArray<uint32> Colors;
	TGpuArray<uint64> EnclosedLeaves;
};

namespace DAGCompression
{
	FFinalDag Compress_SingleThreaded(const FScene& Scene, const FAABB& AABB);
	FFinalDag Compress_MultiThreaded(const FScene& Scene, const FAABB& AABB);
	
	FCpuDag CreateSubDAG(TGpuArray<uint64>& Fragments);
	FCpuDag MergeDAGs(std::vector<FCpuDag> CpuDags);
	FFinalDag CreateFinalDAG(FCpuDag CpuDag);

	// Will free Fragments
	TGpuArray<uint64> SortFragmentsAndRemoveDuplicates(TGpuArray<uint64> Fragments);
	TGpuArray<uint32> ExtractColorsAndFixFragments(TGpuArray<uint64>& Fragments);
	TGpuArray<uint64> ExtractLeavesAndShiftReduceFragments(TGpuArray<uint64>& Fragments);
	void SortLeaves(TGpuArray<uint64>& Leaves, TGpuArray<uint32>& OutHashesToSortedUniqueHashes);
	void SortLevel(FGpuLevel& Level, TGpuArray<uint32>& OutHashesToSortedUniqueHashes);
	FGpuLevel ExtractLevelAndShiftReduceFragments(TGpuArray<uint64>& Fragments, const TGpuArray<uint32>& FragmentIndicesToChildrenIndices);
	void ComputeHashes(FGpuLevel& Level, const TGpuArray<uint64>& LowerLevelHashes);

	FCpuDag MergeDAGs(FCpuDag A, FCpuDag B);
	std::thread MergeColors(std::vector<TCpuArray<uint32>> Colors, TCpuArray<uint32>& MergedColors);
	
	template<typename T>
	void CheckLevelIndices(const T& Level);
	void CheckDag(const FFinalDag& FinalDag);
	// Returns total num leaves
	uint64 CheckDag(const TCpuArray<uint32>& Dag, const TCpuArray<uint64>& EnclosedLeaves);

	void CheckGPU();
}