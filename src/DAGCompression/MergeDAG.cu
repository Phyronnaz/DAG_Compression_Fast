#include "DAGCompression/DAGCompression.h"
#include "GpuPrimitives.h"
#include "Threadpool.h"

inline uint32 Merge(
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

FCpuDag DAGCompression::MergeDAGs(FCpuDag A, FCpuDag B)
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