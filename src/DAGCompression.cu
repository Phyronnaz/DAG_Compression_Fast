#include "DAGCompression.h"
#include "GpuPrimitives.h"
#include "Threadpool.h"

inline void LogRemoved(const char* Name, uint64 Before, uint64 After)
{
	checkInfEqual(After, Before);
	LOG_DEBUG("%s: %llu elements (%f%%) removed", Name, Before - After, double(Before - After) * 100. / Before);
}

HOST_DEVICE uint64 HashChildrenHashes(uint64 ChildrenHashes[8])
{
	const uint64 Hash = Utils::MurmurHash128xN(reinterpret_cast<const uint128*>(ChildrenHashes), 4).Data[0];
	check(Hash != 0); // Can remove this check, just want to see if it actually happens
	return Hash == 0 ? 1 : Hash;
}

HOST_DEVICE uint64 HashChildren(const FChildrenIndices& ChildrenIndices, const TStaticArray<uint64, EMemoryType::GPU>& ChildrenArray)
{
	uint64 ChildrenHashes[8];
	for (int32 Index = 0; Index < 8; Index++)
	{
		const uint32 ChildIndex = ChildrenIndices.Indices[Index];
		ChildrenHashes[Index] = ChildIndex == 0xFFFFFFFF ? 0 : ChildrenArray[ChildIndex];
	}
	return HashChildrenHashes(ChildrenHashes);
}

template<typename T>
inline void CheckLevelIndices(const T& Level)
{
	(void)Level;
#if ENABLE_CHECKS
	const auto ChildrenIndices = Level.ChildrenIndices.template CastTo<uint32>();
	const auto ChildMasks = Level.ChildMasks;
	const auto Check = [=] GPU_LAMBDA (uint64 Index)
	{
		const uint64 NodeIndex = Index / 8;
		const uint64 ChildIndex = Index % 8;
		const uint8 ChildMask = ChildMasks[NodeIndex];
		if ((ChildrenIndices[Index] == 0xFFFFFFFF) != !(ChildMask & (1 << ChildIndex)))
		{
			printf("Invalid Children Indices:\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n",
				ChildrenIndices[8 * NodeIndex + 0], bool(ChildMask & 0x01),
				ChildrenIndices[8 * NodeIndex + 1], bool(ChildMask & 0x02),
				ChildrenIndices[8 * NodeIndex + 2], bool(ChildMask & 0x04),
				ChildrenIndices[8 * NodeIndex + 3], bool(ChildMask & 0x08),
				ChildrenIndices[8 * NodeIndex + 4], bool(ChildMask & 0x10),
				ChildrenIndices[8 * NodeIndex + 5], bool(ChildMask & 0x20),
				ChildrenIndices[8 * NodeIndex + 6], bool(ChildMask & 0x40),
				ChildrenIndices[8 * NodeIndex + 7], bool(ChildMask & 0x80));
			DEBUG_BREAK();
		}
	};
	if (ChildrenIndices.GetMemoryType() == EMemoryType::GPU && !DEBUG_GPU_ARRAYS)
	{
		const auto It = thrust::make_counting_iterator<uint64>(0);
		thrust::for_each(ExecutionPolicy, It, It + ChildrenIndices.Num(), Check);
		CUDA_CHECK_ERROR();
	}
	else
	{
		for (uint64 Index = 0; Index < ChildrenIndices.Num(); Index++)
		{
			Check(Index);
		}
	}
#endif	
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// Returns NumWords
uint32 ComputeChildPositions(const FGpuLevel& Level, TStaticArray<uint32, EMemoryType::GPU>& ChildPositions)
{
	const int32 Num = Cast<int32>(Level.ChildMasks.Num());
	check(Num > 0);

	ChildPositions = TStaticArray<uint32, EMemoryType::GPU>("ChildPositions", Num);

	const auto NodeSizes = thrust::make_transform_iterator(Level.ChildMasks.GetData(),  [] GPU_LAMBDA (uint8 ChildMask) { return Utils::TotalSize(ChildMask); });

	{
		void* TempStorage = nullptr;
		size_t TempStorageBytes;
		CUDA_CHECKED_CALL cub::DeviceScan::ExclusiveSum(TempStorage, TempStorageBytes, NodeSizes, ChildPositions.GetData(), Num);
		CUDA_CHECKED_CALL cnmemMalloc(&TempStorage, TempStorageBytes, DEFAULT_STREAM);
		CUDA_CHECKED_CALL cub::DeviceScan::ExclusiveSum(TempStorage, TempStorageBytes, NodeSizes, ChildPositions.GetData(), Num);
		CUDA_CHECK_ERROR();
		CUDA_CHECKED_CALL cnmemFree(TempStorage, DEFAULT_STREAM);
	}
	
	return GetElement(ChildPositions, Num - 1) + Utils::TotalSize(GetElement(Level.ChildMasks, Num - 1));
}

template<typename T>
void WriteLevelTo(const FGpuLevel& Level, const TStaticArray<uint32, EMemoryType::GPU>& ChildPositions, TStaticArray<uint32, EMemoryType::GPU>& Data, T GetChildIndex)
{
	CheckLevelIndices(Level);
	
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

uint64 CheckDag(const TStaticArray<uint32, EMemoryType::CPU>& Dag, const TStaticArray<uint64, EMemoryType::CPU>& EnclosedLeaves, uint32 Index, uint32 Level)
{
	if (Level == LEVELS - 2)
	{
		return Utils::Popc(Dag[Index + 0]) + Utils::Popc(Dag[Index + 1]);
	}
	else
	{
		const uint32 ChildMask = Utils::ChildMask(Dag[Index]);
		checkInf(ChildMask, 256);
		uint64 Count = 0;
		for (uint32 ChildIndex = 0; ChildIndex < Utils::Popc(ChildMask); ChildIndex++)
		{
			Count += CheckDag(Dag, EnclosedLeaves, Dag[Index + 1 + ChildIndex], Level + 1);
		}
#if ENABLE_COLORS
		if (Level < TOP_LEVEL)
		{
			checkEqual(Count, EnclosedLeaves[Dag[Index] >> 8]);
		}
		else
		{
			checkEqual(Count, Dag[Index] >> 8);
		}
#endif
		return Count;
	}
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<uint32 Level>
HOST_DEVICE_RECURSIVE uint64 SetLeavesCounts(
	TStaticArray<uint64, EMemoryType::GPU> Counts,
	TStaticArray<uint32, EMemoryType::GPU> Dag,
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
	TStaticArray<uint64, EMemoryType::GPU> Counts,
	TStaticArray<uint32, EMemoryType::GPU> Dag,
	uint32 Index)
{
	(void)Counts;
	return Utils::Popc(Dag[Index + 0]) + Utils::Popc(Dag[Index + 1]);
}

__global__ void SetLeavesCountsKernel(
	TStaticArray<uint32, EMemoryType::GPU> Indices,
	TStaticArray<uint64, EMemoryType::GPU> Counts,
	TStaticArray<uint32, EMemoryType::GPU> Dag)
{
	const uint32 Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (!Indices.IsValidIndex(Index)) return;

	const uint64 Count = SetLeavesCounts<TOP_LEVEL>(Counts, Dag, Indices[Index]);
	checkEqual(Count, Counts[Indices[Index]]);
}

template<bool IsFirstPass>
uint64 SetLeavesCountsCpuImpl(
	TStaticArray<uint32, EMemoryType::CPU> Dag,
	TStaticArray<uint64, EMemoryType::CPU> Counts,
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
	TStaticArray<uint64, EMemoryType::CPU> CpuCounts("Counts", CpuDag.Num());
	CpuCounts.MemSet(0);
	SetLeavesCountsCpuImpl<true>(CpuDag, CpuCounts, EnclosedLeaves, Indices, 0, 0);

	TStaticArray<uint32, EMemoryType::GPU> GpuIndices("Indices", Indices.size());
	{
		PROFILE_SCOPE("Memcpy");
		CUDA_CHECKED_CALL cudaMemcpyAsync(GpuIndices.GetData(), &Indices[0], GpuIndices.SizeInBytes(), cudaMemcpyHostToDevice);
		CUDA_SYNCHRONIZE_STREAM();
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
		Dag.EnclosedLeaves = TStaticArray<uint64, EMemoryType::GPU>("EnclosedLeaves", EnclosedLeaves.size());
		{
			PROFILE_SCOPE("Memcpy");
			CUDA_CHECKED_CALL cudaMemcpyAsync(Dag.EnclosedLeaves.GetData(), &EnclosedLeaves[0], Dag.EnclosedLeaves.SizeInBytes(), cudaMemcpyHostToDevice);
			CUDA_SYNCHRONIZE_STREAM();
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

FCpuDag DAGCompression::CreateSubDAG(TStaticArray<uint64, EMemoryType::GPU>& InFragments)
{
	PROFILE_FUNCTION();
	
	FCpuDag CpuDag;
	CpuDag.Levels.resize(SUBDAG_LEVELS - 2);

	if (InFragments.Num() == 0)
	{
		return CpuDag;
	}
	
	auto Fragments = SortFragmentsAndRemoveDuplicates(InFragments);
	
#if ENABLE_COLORS
	auto Colors = ExtractColorsAndFixFragments(Fragments);
	CpuDag.Colors = Colors.CreateCPU();
	Colors.Free();
#endif

	auto Leaves = ExtractLeavesAndShiftReduceFragments(Fragments);

	TStaticArray<uint32, EMemoryType::GPU> FragmentIndicesToChildrenIndices;
	{
		FGpuLevel Level;
		Level.Hashes = std::move(Leaves);
		SortLevel(Level, FragmentIndicesToChildrenIndices);
		Leaves = Level.Hashes;
	}
	CpuDag.Leaves = Leaves.CreateCPU();
	
	auto PreviousLevelHashes = Leaves;
	for (int32 LevelDepth = SUBDAG_LEVELS - 3; LevelDepth >= 0; LevelDepth--)
	{
		PROFILE_SCOPE("Level %d", LevelDepth);
		LOG_DEBUG("Level %d", LevelDepth);
		
		FGpuLevel Level = ExtractLevelAndShiftReduceFragments(Fragments, FragmentIndicesToChildrenIndices);
		FragmentIndicesToChildrenIndices.Free();

		ComputeHashes(Level, PreviousLevelHashes);
		PreviousLevelHashes.Free();

		SortLevel(Level, FragmentIndicesToChildrenIndices);

		CpuDag.Levels[LevelDepth] = Level.ToCPU();
		Level.Free(false);
		PreviousLevelHashes = Level.Hashes;
	}
	check(PreviousLevelHashes.Num() == 1);
	PreviousLevelHashes.Free();

	FragmentIndicesToChildrenIndices.Free();
	Fragments.Free();
	
	return CpuDag;
}

uint32 Merge(
	const TStaticArray<uint64, EMemoryType::GPU>& KeysA,
	const TStaticArray<uint64, EMemoryType::GPU>& KeysB,
	TStaticArray<uint32, EMemoryType::GPU>& OutIndicesAToMergedIndices,
	TStaticArray<uint32, EMemoryType::GPU>& OutIndicesBToMergedIndices)
{
	PROFILE_FUNCTION();

	check(!OutIndicesAToMergedIndices.IsValid());
	check(!OutIndicesBToMergedIndices.IsValid());

	CheckIsSorted(KeysA);
	CheckIsSorted(KeysB);

	const int32 NumToMerge = Cast<int32>(KeysA.Num() + KeysB.Num());
	auto KeysAB = TStaticArray<uint64, EMemoryType::GPU>("KeysAB", NumToMerge);
	auto Permutation = TStaticArray<uint32, EMemoryType::GPU>("Permutation", NumToMerge);
	MergePairs(
		KeysA,
		thrust::make_counting_iterator<uint32>(0u),
		KeysB,
		thrust::make_counting_iterator<uint32>(1u << 31),
		KeysAB,
		Permutation,
		thrust::less<uint64>());
	CheckIsSorted(KeysAB);

	auto UniqueSum = TStaticArray<uint32, EMemoryType::GPU>("UniqueSum", NumToMerge);
	{
		auto UniqueFlags = TStaticArray<uint32, EMemoryType::GPU>("UniqueFlags", NumToMerge);
		AdjacentDifference(KeysAB, UniqueFlags, thrust::not_equal_to<uint64>(), 0u);
		{
			void* TempStorage = nullptr;
			size_t TempStorageBytes;
			CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(TempStorage, TempStorageBytes, UniqueFlags.GetData(), UniqueSum.GetData(), NumToMerge);
			CUDA_CHECKED_CALL cnmemMalloc(&TempStorage, TempStorageBytes, DEFAULT_STREAM);
			CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(TempStorage, TempStorageBytes, UniqueFlags.GetData(), UniqueSum.GetData(), NumToMerge);
			CUDA_CHECK_ERROR();
			CUDA_CHECKED_CALL cnmemFree(TempStorage, DEFAULT_STREAM);
		}
		UniqueFlags.Free();
	}
	KeysAB.Free();

	const uint32 NumUniques = GetElement(UniqueSum, UniqueSum.Num() - 1) + 1;

	OutIndicesAToMergedIndices = TStaticArray<uint32, EMemoryType::GPU>("IndicesAToMergedIndices", KeysA.Num());
	OutIndicesBToMergedIndices = TStaticArray<uint32, EMemoryType::GPU>("IndicesBToMergedIndices", KeysB.Num());

	ScatterIf(
		UniqueSum, 
		[=] GPU_LAMBDA (uint64 Index) { return Utils::ClearFlag(Permutation[Index]); },
		OutIndicesAToMergedIndices,
		[=] GPU_LAMBDA (uint64 Index) { return !Utils::HasFlag(Permutation[Index]); });
	ScatterIf(
		UniqueSum, 
		[=] GPU_LAMBDA (uint64 Index) { return Utils::ClearFlag(Permutation[Index]); },
		OutIndicesBToMergedIndices,
		[=] GPU_LAMBDA (uint64 Index) { return Utils::HasFlag(Permutation[Index]); });

	CheckArrayBounds<uint32>(OutIndicesAToMergedIndices, 0, NumUniques - 1);
	CheckArrayBounds<uint32>(OutIndicesBToMergedIndices, 0, NumUniques - 1);

	UniqueSum.Free();
	Permutation.Free();

	return NumUniques;
}

inline FCpuDag MergeDAGs(
	FCpuDag A,
	FCpuDag B)
{
	PROFILE_FUNCTION();
	
	checkEqual(A.Levels.size(), B.Levels.size());

	if (A.Leaves.Num() == 0)
	{
		for (auto& Level : A.Levels) check(Level.Hashes.Num() == 0);
		return B;
	}
	if (B.Leaves.Num() == 0)
	{
		for (auto& Level : B.Levels) check(Level.Hashes.Num() == 0);
		return A;
	}

	for (auto& Level : A.Levels) check(Level.Hashes.Num() != 0);
	for (auto& Level : B.Levels) check(Level.Hashes.Num() != 0);

	const int32 NumLevels = int32(A.Levels.size());
	
	FCpuDag MergedCpuDag;
	MergedCpuDag.Levels.resize(NumLevels);

	TStaticArray<uint32, EMemoryType::GPU> IndicesAToMergedIndices;
	TStaticArray<uint32, EMemoryType::GPU> IndicesBToMergedIndices;
	{
		auto LeavesA = A.Leaves.CreateGPU();
		auto LeavesB = B.Leaves.CreateGPU();
		A.Leaves.Free();
		B.Leaves.Free();

		const uint32 NumMergedLeaves = Merge(
			LeavesA,
			LeavesB,
			IndicesAToMergedIndices,
			IndicesBToMergedIndices);

		auto MergedLeaves = TStaticArray<uint64, EMemoryType::GPU>("MergedLeaves", NumMergedLeaves);
		Scatter(LeavesA, IndicesAToMergedIndices, MergedLeaves);
		Scatter(LeavesB, IndicesBToMergedIndices, MergedLeaves);

		LeavesA.Free();
		LeavesB.Free();

		MergedCpuDag.Leaves = MergedLeaves.CreateCPU();
		MergedLeaves.Free();
	}

	auto PreviousIndicesAToMergedIndices = IndicesAToMergedIndices;
	auto PreviousIndicesBToMergedIndices = IndicesBToMergedIndices;
	IndicesAToMergedIndices.Reset();
	IndicesBToMergedIndices.Reset();

	for (int32 LevelIndex = NumLevels - 1; LevelIndex >= 0; LevelIndex--)
	{
		auto& LevelA = A.Levels[LevelIndex];
		auto& LevelB = B.Levels[LevelIndex];

		auto HashesA = LevelA.Hashes.CreateGPU();
		auto HashesB = LevelB.Hashes.CreateGPU();
		LevelA.Hashes.Free();
		LevelB.Hashes.Free();
		
		const uint32 NumMerged = Merge(
			HashesA,
			HashesB,
			IndicesAToMergedIndices,
			IndicesBToMergedIndices);

		FCpuLevel MergedLevel;

		{
			auto MergedHashes = TStaticArray<uint64, EMemoryType::GPU>("MergedHashes", NumMerged);

			Scatter(HashesA, IndicesAToMergedIndices, MergedHashes);
			Scatter(HashesB, IndicesBToMergedIndices, MergedHashes);

			HashesA.Free();
			HashesB.Free();

			MergedLevel.Hashes = MergedHashes.CreateCPU();
			MergedHashes.Free();
		}

		{
			auto MergedChildMasks = TStaticArray<uint8, EMemoryType::GPU>("MergedChildMasks", NumMerged);
			
			auto ChildMasksA = LevelA.ChildMasks.CreateGPU();
			auto ChildMasksB = LevelB.ChildMasks.CreateGPU();
			LevelA.ChildMasks.Free();
			LevelB.ChildMasks.Free();

			Scatter(ChildMasksA, IndicesAToMergedIndices, MergedChildMasks);
			Scatter(ChildMasksB, IndicesBToMergedIndices, MergedChildMasks);

			ChildMasksA.Free();
			ChildMasksB.Free();

			MergedLevel.ChildMasks = MergedChildMasks.CreateCPU();
			MergedChildMasks.Free();
		}

		{
			const auto Transform = [] GPU_LAMBDA(const TStaticArray<uint32, EMemoryType::GPU>& IndicesMap, FChildrenIndices Indices)
			{
				for (uint32& Index : Indices.Indices)
				{
					if (Index != 0xFFFFFFFF)
					{
						Index = IndicesMap[Index];
					}
				}
				return Indices;
			};
			const auto TransformA = [=] GPU_LAMBDA(FChildrenIndices Indices) { return Transform(PreviousIndicesAToMergedIndices, Indices); };
			const auto TransformB = [=] GPU_LAMBDA(FChildrenIndices Indices) { return Transform(PreviousIndicesBToMergedIndices, Indices); };

			auto MergedChildrenIndices = TStaticArray<FChildrenIndices, EMemoryType::GPU>("MergedChildrenIndices", NumMerged);

			auto ChildrenIndicesA = LevelA.ChildrenIndices.CreateGPU();
			auto ChildrenIndicesB = LevelB.ChildrenIndices.CreateGPU();
			LevelA.ChildrenIndices.Free();
			LevelB.ChildrenIndices.Free();

			ScatterWithTransform(ChildrenIndicesA, TransformA, IndicesAToMergedIndices, MergedChildrenIndices);
			ScatterWithTransform(ChildrenIndicesB, TransformB, IndicesBToMergedIndices, MergedChildrenIndices);
			
			ChildrenIndicesA.Free();
			ChildrenIndicesB.Free();

			MergedLevel.ChildrenIndices = MergedChildrenIndices.CreateCPU();
			MergedChildrenIndices.Free();
		}
		
		CheckLevelIndices(MergedLevel);
		CheckIsSorted(MergedLevel.Hashes);
		MergedCpuDag.Levels[LevelIndex] = MergedLevel;

		PreviousIndicesAToMergedIndices.Free();
		PreviousIndicesBToMergedIndices.Free();
		PreviousIndicesAToMergedIndices = IndicesAToMergedIndices;
		PreviousIndicesBToMergedIndices = IndicesBToMergedIndices;
		IndicesAToMergedIndices.Reset();
		IndicesBToMergedIndices.Reset();
	}

	PreviousIndicesAToMergedIndices.Free();
	PreviousIndicesBToMergedIndices.Free();

	return MergedCpuDag;
}

// Note: dags are split in morton order
FCpuDag DAGCompression::MergeDAGs(std::vector<FCpuDag>&& CpuDags)
{
	PROFILE_FUNCTION();
	
	check(Utils::IsPowerOf2(CpuDags.size()));
	const int32 NumLevels = int32(CpuDags[0].Levels.size());
	for (auto& CpuDag : CpuDags)
	{
		checkEqual(CpuDag.Levels.size(), NumLevels);
	}

#if ENABLE_COLORS
	TStaticArray<uint32, EMemoryType::CPU> MergedColors;
	std::vector<TStaticArray<uint32, EMemoryType::CPU>> ColorsToMerge;
	ColorsToMerge.reserve(CpuDags.size());
	for (auto& Dag : CpuDags)
	{
		ColorsToMerge.push_back(Dag.Colors);
		Dag.Colors.Reset();
	}
	std::thread MergeColorThread([&ColorsToMerge, &MergedColors]()
	{
		NAME_THREAD("MergeColorThread");
		PROFILE_SCOPE("Merge Colors");
		uint64 Num = 0;
		std::vector<uint64> Offsets;
		for(auto& Colors : ColorsToMerge)
		{
			Offsets.push_back(Num);
			Num += Colors.Num();
		}
		MergedColors = TStaticArray<uint32, EMemoryType::CPU>("Colors", Num);

		std::vector<std::thread> Threads;
		for (uint32 Index = 0; Index < ColorsToMerge.size(); Index++)
		{
			if (ColorsToMerge[Index].Num() > 0) // Else we might memcpy an invalid ptr
			{
				Threads.emplace_back([Index, &ColorsToMerge, &Offsets, &MergedColors]()
					{
						NAME_THREAD("MergeColorThreadPool");
						std::memcpy(&MergedColors[Offsets[Index]], ColorsToMerge[Index].GetData(), ColorsToMerge[Index].SizeInBytes());
						ColorsToMerge[Index].Free();
					});
				Threads.back().join(); // TODO HACK
			}
		}
		for(auto& Thread : Threads)
		{
			//Thread.join(); TODO
		}
	});
	MergeColorThread.join(); // TODO HACK
#endif

	FThreadPool ThreadPool(NUM_MERGE_THREADS);

	int32 PassIndex = 0;
	while (CpuDags.size() > 1)
	{
		LOG("Pass %d", PassIndex);
		PROFILE_SCOPE("Pass %d", PassIndex);
		PassIndex++;
		
		check(CpuDags.size() % 8 == 0);
		
		std::vector<FCpuDag> NewCpuDags;
		NewCpuDags.resize(CpuDags.size() / 8);
		
		for (int32 SubDagIndex = 0; SubDagIndex < CpuDags.size() / 8; SubDagIndex++)
		{
			ThreadPool.Enqueue([&, SubDagIndex]() 
			{
				PROFILE_SCOPE("Sub Dag %d", SubDagIndex);

				const auto GetDag = [&](int32 Index) -> FCpuDag& { return CpuDags[8 * SubDagIndex + Index]; };

				uint64 Hashes[8];
				const auto CopyHash = [&](int32 Index)
				{
					auto& TopLevelHashes = GetDag(Index).Levels[0].Hashes;
					checkInfEqual(TopLevelHashes.Num(), 1);
					if (TopLevelHashes.Num() > 0)
					{
						check(TopLevelHashes[0] != 0);
						Hashes[Index] = TopLevelHashes[0];
					}
					else
					{
						Hashes[Index] = 0;
					}
				};

				CopyHash(0);
				FCpuDag MergedDag = GetDag(0);
				for (int32 Index = 1; Index < 8; Index++)
				{
					CopyHash(Index);
					MergedDag = MergeDAGs(std::move(MergedDag), std::move(GetDag(Index)));
				}

				// MergedHashes contains all the root subdags hashes
				// Use the stored hashes to find back the roots indices
				// The new children will be in the correct order as dags are in morton order (see AABB split)

				uint8 ChildMask = 0;
				FChildrenIndices ChildrenIndices;
				const auto& MergedHashes = MergedDag.Levels[0].Hashes;
				for (int32 Index = 0; Index < 8; Index++)
				{
					const uint64 Hash = Hashes[Index];
					if (Hash == 0)
					{
						ChildrenIndices.Indices[Index] = 0xFFFFFFFF;
					}
					else
					{
						ChildMask |= 1 << Index;
						ChildrenIndices.Indices[Index] = uint32(MergedHashes.FindChecked(Hash));
					}
				}
				const uint64 Hash = HashChildrenHashes(Hashes);

				FCpuLevel TopLevel;
				if (ChildMask != 0)
				{
					TopLevel.ChildMasks = TStaticArray<uint8, EMemoryType::CPU>("ChildMasks", { ChildMask });
					TopLevel.ChildrenIndices = TStaticArray<FChildrenIndices, EMemoryType::CPU>("ChildrenIndices", { ChildrenIndices });
					TopLevel.Hashes = TStaticArray<uint64, EMemoryType::CPU>("Hashes", { Hash });
				}
				CheckLevelIndices(TopLevel);
				CheckIsSorted(TopLevel.Hashes);
				
				MergedDag.Levels.insert(MergedDag.Levels.begin(), TopLevel);

				NewCpuDags[SubDagIndex] = MergedDag;
			});
		}

		ThreadPool.WaitForCompletion();
		
		CpuDags = std::move(NewCpuDags);
	}
	checkEqual(CpuDags.size(), 1);

	ThreadPool.Destroy();
	
#if ENABLE_COLORS
	{
		PROFILE_SCOPE("Wait For Merge Colors Thread");
		//MergeColorThread.join(); TODO
		CpuDags[0].Colors = MergedColors;
	}
#endif
	
	return CpuDags[0];
}

FFinalDag DAGCompression::CreateFinalDAG(FCpuDag&& CpuDag)
{
	PROFILE_FUNCTION();
	
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
		TStaticArray<uint32, EMemoryType::GPU> ChildPositions;
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

	TStaticArray<uint32, EMemoryType::GPU> Dag("Dag", Num);
	Dag.MemSet(0xFF); // To spot errors
	uint64 Offset = 0;
	for (uint64 LevelIndex = 0; LevelIndex < Levels.size(); LevelIndex++)
	{
		const auto& Level = Levels[LevelIndex];
		const auto& LevelInfo = LevelsInfo[LevelIndex];
		const uint32 NextLevelStart = Cast<uint32>(Offset + LevelInfo.NumWords);
		checkInfEqual(Offset, Num);
		auto DagWithOffset = TStaticArray<uint32, EMemoryType::GPU>(Dag.GetData() + Offset, Num - Offset);
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
	CUDA_CHECKED_CALL cudaMemcpyAsync(Dag.GetData() + Offset, Leaves.GetData(), Leaves.Num() * sizeof(uint64), cudaMemcpyDeviceToDevice);
	CUDA_SYNCHRONIZE_STREAM();

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
		FinalDag.Colors = TStaticArray<uint32, EMemoryType::GPU>("Colors", CpuDag.Colors.Num());
		CpuDag.Colors.CopyToGPU(FinalDag.Colors);
		CpuDag.Colors.Free();
	}
#endif

#if ENABLE_CHECKS
	{
		PROFILE_SCOPE("CheckDag");
		
		auto CpuDagCopy = Dag.CreateCPU();
		auto CpuEnclosedLeavesCopy = FinalDag.EnclosedLeaves.CreateCPU();
		CheckDag(CpuDagCopy, CpuEnclosedLeavesCopy, 0, 0);
		CpuDagCopy.Free();
		CpuEnclosedLeavesCopy.Free();
	}
#endif

	return FinalDag;
}

TStaticArray<uint64, EMemoryType::GPU> DAGCompression::SortFragmentsAndRemoveDuplicates(TStaticArray<uint64, EMemoryType::GPU>& Fragments)
{
	PROFILE_FUNCTION();

	auto NewFragments = TStaticArray<uint64, EMemoryType::GPU>("Fragments", Fragments.Num());

	{
		const int32 Num = Cast<int32>(Fragments.Num());
		const int32 NumBits = 3 * SUBDAG_LEVELS;
		cub::DoubleBuffer<uint64> Keys(Fragments.GetData(), NewFragments.GetData());
		
		void* TempStorage = nullptr;
		size_t TempStorageBytes;
		CUDA_CHECKED_CALL cub::DeviceRadixSort::SortKeys(TempStorage, TempStorageBytes, Keys, Num, 0, NumBits);
		CUDA_CHECKED_CALL cnmemMalloc(&TempStorage, TempStorageBytes, DEFAULT_STREAM);
		CUDA_CHECKED_CALL cub::DeviceRadixSort::SortKeys(TempStorage, TempStorageBytes, Keys, Num, 0, NumBits);
		CUDA_CHECK_ERROR();
		CUDA_CHECKED_CALL cnmemFree(TempStorage, DEFAULT_STREAM);

		// Ends up in different buffers on different GPUs
		if (Keys.Current() != Fragments.GetData())
		{
			CUDA_CHECKED_CALL cudaMemcpyAsync(Fragments.GetData(), NewFragments.GetData(), Num * sizeof(uint64), cudaMemcpyDeviceToDevice);
			CUDA_SYNCHRONIZE_STREAM();
		}
	}

	int32 NumUnique;
	{
		// Don't check the color bits
		const auto EqualityLambda = [] GPU_LAMBDA (uint64 A, uint64 B) { return (A << (64 - 3 * SUBDAG_LEVELS)) == (B << (64 - 3 * SUBDAG_LEVELS)); };
		const int32 Num = Cast<int32>(Fragments.Num());
		TSingleGPUElement<int32> NumUniqueGPU;
		
		void* TempStorage = nullptr;
		size_t TempStorageBytes;
		CUDA_CHECKED_CALL cub::DeviceSelect2::Unique(TempStorage, TempStorageBytes, Fragments.GetData(), NewFragments.GetData(), NumUniqueGPU.GetPtr(), Num, EqualityLambda);
		CUDA_CHECKED_CALL cnmemMalloc(&TempStorage, TempStorageBytes, DEFAULT_STREAM);
		CUDA_CHECKED_CALL cub::DeviceSelect2::Unique(TempStorage, TempStorageBytes, Fragments.GetData(), NewFragments.GetData(), NumUniqueGPU.GetPtr(), Num, EqualityLambda);
		CUDA_CHECK_ERROR();
		CUDA_CHECKED_CALL cnmemFree(TempStorage, DEFAULT_STREAM);

		NumUnique = NumUniqueGPU.GetValue();
	}
	checkInf(NumUnique, Fragments.Num());

	// Shrink
#if 0
	CUDA_CHECKED_CALL cudaMemcpyAsync(Fragments.GetData(), NewFragments.GetData(), NumUnique * sizeof(uint64), cudaMemcpyDeviceToDevice);
	CUDA_SYNCHRONIZE_STREAM();
	NewFragments.Free();
	NewFragments = TStaticArray<uint64, EMemoryType::GPU>("Fragments", NumUnique);
	CUDA_CHECKED_CALL cudaMemcpyAsync(NewFragments.GetData(), Fragments.GetData(), NumUnique * sizeof(uint64), cudaMemcpyDeviceToDevice);
	CUDA_SYNCHRONIZE_STREAM();
#else
	const auto New = TStaticArray<uint64, EMemoryType::GPU>(NewFragments.GetData(), Cast<uint32>(NumUnique));
	NewFragments.Reset();
	NewFragments = New;
#endif

	return NewFragments;
}

TStaticArray<uint32, EMemoryType::GPU> DAGCompression::ExtractColorsAndFixFragments(TStaticArray<uint64, EMemoryType::GPU>& Fragments)
{
	PROFILE_FUNCTION();

	auto Colors = TStaticArray<uint32, EMemoryType::GPU>("Colors", Fragments.Num());
	Transform(Fragments, Colors, [] GPU_LAMBDA (uint64 X) { return X >> 40; });
	Transform(Fragments, Fragments, [] GPU_LAMBDA (uint64 X) { return X & ((uint64(1) << 40) - 1); });
	return Colors;
}

TStaticArray<uint64, EMemoryType::GPU> DAGCompression::ExtractLeavesAndShiftReduceFragments(TStaticArray<uint64, EMemoryType::GPU>& Fragments)
{
	PROFILE_FUNCTION();

	// Use the last 6 bits to create the leaves
	const auto ParentsTransform = [] GPU_LAMBDA (uint64 Code) { return Code >> 6; };
	const auto LeavesTransform = [] GPU_LAMBDA (uint64 Code) { return uint64(1) << (Code & 0b111111); };

	const int32 MinAverageVoxelsPer64Leaf = 1; // TODO higher?
	// TODO compute exact num?
	
	auto NewFragments = TStaticArray<uint64, EMemoryType::GPU>("Fragments", Fragments.Num() / MinAverageVoxelsPer64Leaf);
	auto Leaves = TStaticArray<uint64, EMemoryType::GPU>("Leaves", Fragments.Num() / MinAverageVoxelsPer64Leaf);
	
	const auto ParentsBits = TransformIterator(Fragments.GetData(), ParentsTransform);
	const auto LeavesBits = TransformIterator(Fragments.GetData(), LeavesTransform);

	const int32 Num = Cast<int32>(Fragments.Num());

	TSingleGPUElement<int32> NumRunsGPU;
	{
		void* TempStorage = nullptr;
		size_t TempStorageBytes = 0;
		CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(TempStorage, TempStorageBytes, ParentsBits, NewFragments.GetData(), LeavesBits, Leaves.GetData(), NumRunsGPU.GetPtr(), thrust::bit_or<uint64>(), Num);
		CUDA_CHECKED_CALL cnmemMalloc(&TempStorage, TempStorageBytes, DEFAULT_STREAM);
		CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(TempStorage, TempStorageBytes, ParentsBits, NewFragments.GetData(), LeavesBits, Leaves.GetData(), NumRunsGPU.GetPtr(), thrust::bit_or<uint64>(), Num);
		CUDA_CHECK_ERROR();
		CUDA_CHECKED_CALL cnmemFree(TempStorage, DEFAULT_STREAM);
	}
	
	const int32 NumRuns = NumRunsGPU.GetValue();
	checkfAlways(NumRuns <= NewFragments.Num(), "Need to decrease MinAverageVoxelsPer64Leaf: %d runs, but expected max %d", NumRuns, int32(NewFragments.Num()));

	// Note: these copies & allocs could be avoided if we computed the exact number of runs first

	Fragments.Free();
	Fragments = TStaticArray<uint64, EMemoryType::GPU>("Fragments", NumRuns);
	CUDA_CHECKED_CALL cudaMemcpyAsync(Fragments.GetData(), NewFragments.GetData(), NumRuns * sizeof(uint64), cudaMemcpyDeviceToDevice);
	CUDA_SYNCHRONIZE_STREAM();
	NewFragments.Free();

	auto ShrunkLeaves = TStaticArray<uint64, EMemoryType::GPU>("Leaves", NumRuns);
	CUDA_CHECKED_CALL cudaMemcpyAsync(ShrunkLeaves.GetData(), Leaves.GetData(), NumRuns * sizeof(uint64), cudaMemcpyDeviceToDevice);
	CUDA_SYNCHRONIZE_STREAM();
	Leaves.Free();

	return ShrunkLeaves;
}

void DAGCompression::SortLevel(FGpuLevel& Level, TStaticArray<uint32, EMemoryType::GPU>& OutHashesToSortedUniqueHashes)
{
	PROFILE_FUNCTION();

	CheckLevelIndices(Level);

	const bool IsLeaf = Level.ChildrenIndices.Num() == 0;

	const int32 NumHashes = Cast<int32>(Level.Hashes.Num());
	auto SortedHashes = TStaticArray<uint64, EMemoryType::GPU>("SortedHashes", NumHashes);
	auto SortedHashesToHashes = TStaticArray<uint32, EMemoryType::GPU>("SortedHashesToHashes", NumHashes);

	{
		// Need a real array to use CUB sort
		auto Sequence = TStaticArray<uint32, EMemoryType::GPU>("Sequence", NumHashes);
		MakeSequence(Sequence);

		cub::DoubleBuffer<uint64> Keys(Level.Hashes.GetData(), SortedHashes.GetData());
		cub::DoubleBuffer<uint32> Values(Sequence.GetData(), SortedHashesToHashes.GetData());

		void* TempStorage = nullptr;
		size_t TempStorageBytes;
		CUDA_CHECKED_CALL cub::DeviceRadixSort::SortPairs(TempStorage, TempStorageBytes, Keys, Values, NumHashes, 0, 64);
		CUDA_CHECKED_CALL cnmemMalloc(&TempStorage, TempStorageBytes, DEFAULT_STREAM);
		CUDA_CHECKED_CALL cub::DeviceRadixSort::SortPairs(TempStorage, TempStorageBytes, Keys, Values, NumHashes, 0, 64);
		CUDA_CHECK_ERROR();
		CUDA_CHECKED_CALL cnmemFree(TempStorage, DEFAULT_STREAM);

		if (Keys.Current() != SortedHashes.GetData())
		{
			check(Keys.Current() == Level.Hashes.GetData());
			std::swap(SortedHashes, Level.Hashes);
		}
		if (Values.Current() != SortedHashesToHashes.GetData())
		{
			check(Values.Current() == Sequence.GetData());
			std::swap(SortedHashesToHashes, Sequence);
		}

		// Level hashes are trashed by the sort
		Level.Hashes.Free();
		Sequence.Free();
	}
	CheckIsSorted(SortedHashes);

	auto SortedHashesFlags = TStaticArray<uint32, EMemoryType::GPU>("SortedHashesFlags", NumHashes);
	AdjacentDifference(SortedHashes, SortedHashesFlags, thrust::not_equal_to<uint64>(), 0u);

	auto SortedHashesToUniqueHashes = TStaticArray<uint32, EMemoryType::GPU>("SortedHashesToUniqueHashes", NumHashes);
	{
		void* TempStorage = nullptr;
		size_t TempStorageBytes;
		CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(TempStorage, TempStorageBytes, SortedHashesFlags.GetData(), SortedHashesToUniqueHashes.GetData(), NumHashes);
		CUDA_CHECKED_CALL cnmemMalloc(&TempStorage, TempStorageBytes, DEFAULT_STREAM);
		CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(TempStorage, TempStorageBytes, SortedHashesFlags.GetData(), SortedHashesToUniqueHashes.GetData(), NumHashes);
		CUDA_CHECK_ERROR();
		CUDA_CHECKED_CALL cnmemFree(TempStorage, DEFAULT_STREAM);
	}
	SortedHashesFlags.Free();
	
	const uint32 NumUniques = GetElement(SortedHashesToUniqueHashes, SortedHashesToUniqueHashes.Num() - 1) + 1;

	Level.Hashes = TStaticArray<uint64, EMemoryType::GPU>(IsLeaf ? "Leaves" : "Hashes", NumUniques);
	Scatter(SortedHashes, SortedHashesToUniqueHashes, Level.Hashes);
	SortedHashes.Free();
	CheckIsSorted(Level.Hashes);

	OutHashesToSortedUniqueHashes = TStaticArray<uint32, EMemoryType::GPU>("HashesToSortedUniqueHashes", NumHashes);
	Scatter(SortedHashesToUniqueHashes, SortedHashesToHashes, OutHashesToSortedUniqueHashes);
	CheckArrayBounds(OutHashesToSortedUniqueHashes, 0u, NumUniques - 1);
	SortedHashesToHashes.Free();
	SortedHashesToUniqueHashes.Free();

	if (!IsLeaf)
	{
		{
			auto NewChildrenIndices = TStaticArray<FChildrenIndices, EMemoryType::GPU>("ChildrenIndices", NumUniques);
			Scatter(Level.ChildrenIndices, OutHashesToSortedUniqueHashes, NewChildrenIndices);
			Level.ChildrenIndices.Free();
			Level.ChildrenIndices = NewChildrenIndices;
		}
		{
			auto NewChildMasks = TStaticArray<uint8, EMemoryType::GPU>("ChildMasks", NumUniques);
			Scatter(Level.ChildMasks, OutHashesToSortedUniqueHashes, NewChildMasks);
			Level.ChildMasks.Free();
			Level.ChildMasks = NewChildMasks;
		}
	}

	CheckLevelIndices(Level);
}

FGpuLevel DAGCompression::ExtractLevelAndShiftReduceFragments(TStaticArray<uint64, EMemoryType::GPU>& Fragments, const TStaticArray<uint32, EMemoryType::GPU>& FragmentIndicesToChildrenIndices)
{
	PROFILE_FUNCTION();

	FGpuLevel OutLevel;

	const int32 Num = Cast<int32>(Fragments.Num());

	const auto ParentsTransform = [] GPU_LAMBDA(uint64 Code) { return Code >> 3; };
	const auto ChildrenTransform = [] GPU_LAMBDA(uint64 Code) { return uint64(1) << (Code & 0b111); };

	// First create the mapping & get the number of unique parents
	int32 NumUniqueParents;
	{
		auto UniqueParentsSum = TStaticArray<uint32, EMemoryType::GPU>("UniqueParentsSum", Num);
		{
			auto UniqueParentsFlags = TStaticArray<uint32, EMemoryType::GPU>("UniqueParentsFlags", Num);
			AdjacentDifferenceWithTransform(Fragments, ParentsTransform, UniqueParentsFlags, thrust::not_equal_to<uint64>(), 0u);
			{
				void* TempStorage = nullptr;
				size_t TempStorageBytes;
				CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(TempStorage, TempStorageBytes, UniqueParentsFlags.GetData(), UniqueParentsSum.GetData(), Num);
				CUDA_CHECKED_CALL cnmemMalloc(&TempStorage, TempStorageBytes, DEFAULT_STREAM);
				CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(TempStorage, TempStorageBytes, UniqueParentsFlags.GetData(), UniqueParentsSum.GetData(), Num);
				CUDA_CHECK_ERROR();
				CUDA_CHECKED_CALL cnmemFree(TempStorage, DEFAULT_STREAM);
			}
			UniqueParentsFlags.Free();
		}

		NumUniqueParents = GetElement(UniqueParentsSum, UniqueParentsSum.Num() - 1) + 1;
		checkInfEqual(NumUniqueParents, Num);

		OutLevel.ChildrenIndices = TStaticArray<FChildrenIndices, EMemoryType::GPU>("ChildrenIndices", NumUniqueParents);
		OutLevel.ChildrenIndices.MemSet(0xFF);
		{
			const auto CompactChildrenToExpandedChildren = [=] GPU_LAMBDA(uint64 Index) { return (Fragments[Index] & 0b111) + 8 * UniqueParentsSum[Index]; };
			auto Out = OutLevel.ChildrenIndices.CastTo<uint32>();
			ScatterPred(FragmentIndicesToChildrenIndices, CompactChildrenToExpandedChildren, Out);
		}

		UniqueParentsSum.Free();
	}

	auto NewFragments = TStaticArray<uint64, EMemoryType::GPU>("Fragments", NumUniqueParents);
	OutLevel.ChildMasks = TStaticArray<uint8, EMemoryType::GPU>("ChildMasks", NumUniqueParents);
	{
		const auto ParentsBits = TransformIterator(Fragments.GetData(), ParentsTransform);
		const auto ChildrenBits = TransformIterator(Fragments.GetData(), ChildrenTransform);

		TSingleGPUElement<int32> NumRunsGPU;

		void* TempStorage = nullptr;
		size_t TempStorageBytes = 0;
		CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(TempStorage, TempStorageBytes, ParentsBits, NewFragments.GetData(), ChildrenBits, OutLevel.ChildMasks.GetData(), NumRunsGPU.GetPtr(), thrust::bit_or<uint8>(), Num);
		CUDA_CHECKED_CALL cnmemMalloc(&TempStorage, TempStorageBytes, DEFAULT_STREAM);
		CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(TempStorage, TempStorageBytes, ParentsBits, NewFragments.GetData(), ChildrenBits, OutLevel.ChildMasks.GetData(), NumRunsGPU.GetPtr(), thrust::bit_or<uint8>(), Num);
		CUDA_CHECK_ERROR();
		CUDA_CHECKED_CALL cnmemFree(TempStorage, DEFAULT_STREAM);

		checkEqual(NumUniqueParents, NumRunsGPU.GetValue());
	}

	Fragments.Free();
	Fragments = NewFragments;

	CheckLevelIndices(OutLevel);
	return OutLevel;
}

void DAGCompression::ComputeHashes(FGpuLevel& Level, const TStaticArray<uint64, EMemoryType::GPU>& LowerLevelHashes)
{
	Level.Hashes = TStaticArray<uint64, EMemoryType::GPU>("Hashes", Level.ChildrenIndices.Num());
	Transform(Level.ChildrenIndices, Level.Hashes, [=] GPU_LAMBDA (const FChildrenIndices & Children) { return HashChildren(Children, LowerLevelHashes); });

#if ENABLE_CHECKS && DEBUG_GPU_ARRAYS
	std::unordered_map<uint64, FChildrenIndices> HashesToIndices;
	for (uint64 Index = 0; Index < Level.Hashes.Num(); Index++)
	{
		const uint64 Hash = Level.Hashes[Index];
		const FChildrenIndices ChildrenIndices = Level.ChildrenIndices[Index];

		const auto ExistingIndices = HashesToIndices.find(Hash);
		if (ExistingIndices == HashesToIndices.end())
		{
			HashesToIndices[Hash] = ChildrenIndices;
		}
		else
		{
			for (int32 ChildIndex = 0; ChildIndex < 8; ChildIndex++)
			{
				checkEqual(ChildrenIndices.Indices[ChildIndex], ExistingIndices->second.Indices[ChildIndex]);
			}
		}
	}
#endif
}