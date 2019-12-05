#include "DAGCompression/DAGCompression.h"
#include "GpuPrimitives.h"

// Returns NumWords
uint32 ComputeChildPositions(const FGpuLevel& Level, TGpuArray<uint32>& ChildPositions)
{
	const int32 Num = Cast<int32>(Level.ChildMasks.Num());
	check(Num > 0);

	ChildPositions = TGpuArray<uint32>("ChildPositions", Num);

	const auto NodeSizes = thrust::make_transform_iterator(Level.ChildMasks.GetData(),  [] GPU_LAMBDA (uint8 ChildMask) { return Utils::TotalSize(ChildMask); });

	{
		FTempStorage Storage;
		CUDA_CHECKED_CALL cub::DeviceScan::ExclusiveSum(Storage.Ptr, Storage.Bytes, NodeSizes, ChildPositions.GetData(), Num, DEFAULT_STREAM);
		Storage.Allocate();
		CUDA_CHECKED_CALL cub::DeviceScan::ExclusiveSum(Storage.Ptr, Storage.Bytes, NodeSizes, ChildPositions.GetData(), Num, DEFAULT_STREAM);
		Storage.SyncAndFree();
	}
	
	return GetElement(ChildPositions, Num - 1) + Utils::TotalSize(GetElement(Level.ChildMasks, Num - 1));
}

template<typename T>
void WriteLevelTo(const FGpuLevel& Level, const TGpuArray<uint32>& ChildPositions, TGpuArray<uint32>& Data, T GetChildIndex)
{
	DAGCompression::CheckLevelIndices(Level);
	
	Scatter(Level.ChildMasks, ChildPositions, Data);

	const auto ChildrenIndices = Level.ChildrenIndices.CastTo<uint32>();
	const auto ChildMasks = Level.ChildMasks;

	ScatterIfWithTransform(
		ChildrenIndices, [=] GPU_LAMBDA(uint32 Index) { return Index == 0xFFFFFFFF ? Index : GetChildIndex(Index); },
		[=] GPU_LAMBDA (uint64 Index)
		{
			return ChildPositions[Index / 8] + Utils::ChildOffset(ChildMasks[Index / 8], Index % 8);
		},
		Data,
		[=] GPU_LAMBDA (uint64 Index)
		{
			checkf((ChildrenIndices[Index] == 0xFFFFFFFF) == !(ChildMasks[Index / 8] & (1 << (Index % 8))), "Index: %" PRIu64, Index);
			return ChildrenIndices[Index] != 0xFFFFFFFF;
		});
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<uint32 Level>
HOST_DEVICE_RECURSIVE uint64 SetLeavesCounts(
	TGpuArray<uint64> Counts,
	TGpuArray<uint32> Dag,
	uint32 Index)
{
	// No need to be thread safe here, worst case we do the same computation twice
	if (Counts[Index] != 0)
	{
		return Counts[Index];
	}

	const uint32 ChildMask = Utils::ChildMask(Dag[Index]);
	uint64 Count = 0;
	for (uint32 ChildIndex = 0; ChildIndex < Utils::Popc(ChildMask); ChildIndex++)
	{
		Count += SetLeavesCounts<Level + 1>(
			Counts,
			Dag,
			Dag[Index + 1 + ChildIndex]);
	}
	checkInf(Count, (1u << 24));
	Dag[Index] |= Count << 8;
	Counts[Index] = Count;
	return Count;
}

template<>
HOST_DEVICE_RECURSIVE uint64 SetLeavesCounts<LEVELS - 2>(
	TGpuArray<uint64> Counts,
	TGpuArray<uint32> Dag,
	uint32 Index)
{
	(void)Counts;
	return Utils::Popc(Dag[Index + 0]) + Utils::Popc(Dag[Index + 1]);
}

__global__ void SetLeavesCountsKernel(
	TGpuArray<uint32> Indices,
	TGpuArray<uint64> Counts,
	TGpuArray<uint32> Dag)
{
	const uint32 Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (!Indices.IsValidIndex(Index)) return;

	const uint64 Count = SetLeavesCounts<TOP_LEVEL>(Counts, Dag, Indices[Index]);
	checkEqual(Count, Counts[Indices[Index]]);
}

template<bool IsFirstPass>
uint64 SetLeavesCountsCpuImpl(
	TCpuArray<uint32> Dag,
	TCpuArray<uint64> Counts,
	std::vector<uint64>& OutEnclosedLeaves,
	std::vector<uint32>& OutIndices,
	uint32 Index,
	uint32 Level)
{
	if (!IsFirstPass && Counts[Index] != 0)
	{
		return Counts[Index];
	}

	if (Level == TOP_LEVEL)
	{
		// Must have Counts[Index] != 0 in second pass
		check(IsFirstPass);
		OutIndices.push_back(Index);
		return 0;
	}
	else 
	{
		const uint32 ChildMask = Utils::ChildMask(Dag[Index]);
		uint64 Count = 0;
		for (uint32 ChildIndex = 0; ChildIndex < Utils::Popc(ChildMask); ChildIndex++)
		{
			Count += SetLeavesCountsCpuImpl<IsFirstPass>(
				Dag,
				Counts,
				OutEnclosedLeaves,
				OutIndices,
				Dag[Index + 1 + ChildIndex],
				Level + 1);
		}
		if (!IsFirstPass)
		{
			const uint64 EnclosedLeavesIndex = OutEnclosedLeaves.size();
			checkInf(EnclosedLeavesIndex, 1u << 24);
			checkEqual(Dag[Index] >> 8, 0);
			Dag[Index] |= EnclosedLeavesIndex << 8;
			OutEnclosedLeaves.push_back(Count);
			Counts[Index] = Count;
		}
		return Count;
	}
}

void SetLeavesCountsCpu(FFinalDag& Dag)
{
	PROFILE_FUNCTION();

	std::vector<uint64> EnclosedLeaves;
	std::vector<uint32> Indices;

	auto GpuDag = Dag.Dag;
	auto CpuDag = GpuDag.CreateCPU();
	TCpuArray<uint64> CpuCounts("Counts", CpuDag.Num());
	CpuCounts.MemSet(0);
	SetLeavesCountsCpuImpl<true>(CpuDag, CpuCounts, EnclosedLeaves, Indices, 0, 0);

	TGpuArray<uint32> GpuIndices("Indices", Indices.size());
	{
		PROFILE_SCOPE("Memcpy");
		FMemory::CudaMemcpy(GpuIndices.GetData(), &Indices[0], GpuIndices.SizeInBytes(), cudaMemcpyHostToDevice);
	}

	auto GpuCounts = CpuCounts.CreateGPU();
	SetLeavesCountsKernel <<< uint32(Indices.size()), 1 >>> (GpuIndices, GpuCounts, GpuDag);
	GpuIndices.Free();

	GpuCounts.CopyToCPU(CpuCounts);
	GpuCounts.Free();

	GpuDag.CopyToCPU(CpuDag);

	SetLeavesCountsCpuImpl<false>(CpuDag, CpuCounts, EnclosedLeaves, Indices, 0, 0);
	CpuCounts.Free();
	
	CpuDag.CopyToGPU(Dag.Dag);
	CpuDag.Free();

	if (!EnclosedLeaves.empty())
	{
		Dag.EnclosedLeaves = TGpuArray<uint64>("EnclosedLeaves", EnclosedLeaves.size());
		FMemory::CudaMemcpy(Dag.EnclosedLeaves.GetData(), &EnclosedLeaves[0], Dag.EnclosedLeaves.SizeInBytes(), cudaMemcpyHostToDevice);
	}
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

FFinalDag DAGCompression::CreateFinalDAG(FCpuDag CpuDag)
{
	PROFILE_FUNCTION();
	CUDA_CHECK_LAST_ERROR();

	checkfAlways(CpuDag.Levels[0].Hashes.Num() > 0, "Empty Dag!");
	
	std::vector<FGpuLevel> Levels;
	for (auto& Level : CpuDag.Levels)
	{
		Levels.push_back({});
		Levels.back().FromCPU(Level, false);
		Level.Free();
	}
	auto Leaves = CpuDag.Leaves.CreateGPU();
	CpuDag.Leaves.Free();

	struct FLevelInfo
	{
		uint32 NumWords;
		TGpuArray<uint32> ChildPositions;
	};

	std::vector<FLevelInfo> LevelsInfo;
	LevelsInfo.reserve(Levels.size());
	
	uint64 Num = 0;
	for (auto& Level : Levels)
	{
		FLevelInfo Info;
		Info.NumWords = ComputeChildPositions(Level, Info.ChildPositions);
		Num += Info.NumWords;
		LevelsInfo.push_back(Info);
	}
	Num += Leaves.Num() * 2;

	TGpuArray<uint32> Dag("Dag", Num);
	Dag.MemSet(0xFF); // To spot errors
	uint64 Offset = 0;
	for (uint64 LevelIndex = 0; LevelIndex < Levels.size(); LevelIndex++)
	{
		const auto& Level = Levels[LevelIndex];
		const auto& LevelInfo = LevelsInfo[LevelIndex];
		const uint32 NextLevelStart = Cast<uint32>(Offset + LevelInfo.NumWords);
		checkInfEqual(Offset, Num);
		auto DagWithOffset = TGpuArray<uint32>(Dag.GetData() + Offset, Num - Offset);
		if (LevelIndex == Levels.size() - 1)
		{
			const auto GetChildIndex = [=] GPU_LAMBDA (uint32 Index) { return NextLevelStart + 2 * Index; };
			WriteLevelTo(Level, LevelInfo.ChildPositions, DagWithOffset, GetChildIndex);
		}
		else
		{
			const auto NextLevelChildPositions = LevelsInfo[LevelIndex + 1].ChildPositions;
			const auto GetChildIndex = [=] GPU_LAMBDA (uint32 Index) { return NextLevelStart + NextLevelChildPositions[Index]; };
			WriteLevelTo(Level, LevelInfo.ChildPositions, DagWithOffset, GetChildIndex);
		}
		Offset += LevelInfo.NumWords;
	}
	checkEqual(Offset + 2 * Leaves.Num(), Num);
	FMemory::CudaMemcpy(Dag.GetData() + Offset, reinterpret_cast<uint32*>(Leaves.GetData()), Leaves.Num() * sizeof(uint64), cudaMemcpyDeviceToDevice);

	for (auto& Level : Levels)
	{
		Level.Free(false);
	}	
	for (auto& LevelInfo : LevelsInfo)
	{
		LevelInfo.ChildPositions.Free();
	}
	Leaves.Free();

	FFinalDag FinalDag;
	FinalDag.Dag = Dag;

#if ENABLE_COLORS
	SetLeavesCountsCpu(FinalDag);
	{
		PROFILE_SCOPE("Copy Colors");
		FinalDag.Colors = TGpuArray<uint32>("Colors", CpuDag.Colors.Num());
		CpuDag.Colors.CopyToGPU(FinalDag.Colors);
		CpuDag.Colors.Free();
	}
#endif

	CUDA_SYNCHRONIZE_STREAM();

	return FinalDag;
}