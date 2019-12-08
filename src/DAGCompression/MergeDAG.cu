#include "DAGCompression/DAGCompression.h"
#include "GpuPrimitives.h"
#include "Threadpool.h"

inline uint32 Merge(
	const TGpuArray<uint64>& KeysA,
	const TGpuArray<uint64>& KeysB,
	TGpuArray<uint32>& OutIndicesAToMergedIndices,
	TGpuArray<uint32>& OutIndicesBToMergedIndices)
{
	PROFILE_FUNCTION_TRACY();

	FScopeAllocator Allocator;

	check(!OutIndicesAToMergedIndices.IsValid());
	check(!OutIndicesBToMergedIndices.IsValid());

	CheckIsSorted(KeysA);
	CheckIsSorted(KeysB);

	const int32 NumToMerge = Cast<int32>(KeysA.Num() + KeysB.Num());
	auto KeysAB = TGpuArray<uint64>("KeysAB", NumToMerge);
	auto Permutation = TGpuArray<uint32>("Permutation", NumToMerge);
	MergePairs(
		KeysA,
		thrust::make_counting_iterator<uint32>(0u),
		KeysB,
		thrust::make_counting_iterator<uint32>(1u << 31),
		KeysAB,
		Permutation,
		thrust::less<uint64>());
	CheckIsSorted(KeysAB);

	auto UniqueSum = TGpuArray<uint32>("UniqueSum", NumToMerge);
	{
		auto UniqueFlags = TGpuArray<uint32>("UniqueFlags", NumToMerge);
		AdjacentDifference(KeysAB, UniqueFlags, thrust::not_equal_to<uint64>(), 0u);
		{
			FTempStorage Storage;
			CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(Storage.Ptr, Storage.Bytes, UniqueFlags.GetData(), UniqueSum.GetData(), NumToMerge, DEFAULT_STREAM);
			Storage.Allocate();
			CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(Storage.Ptr, Storage.Bytes, UniqueFlags.GetData(), UniqueSum.GetData(), NumToMerge, DEFAULT_STREAM);
			Storage.SyncAndFree();
		}
		UniqueFlags.Free(Allocator);
	}
	KeysAB.Free(Allocator);

	const uint32 NumUniques = GetElement(UniqueSum, UniqueSum.Num() - 1) + 1;

	OutIndicesAToMergedIndices = TGpuArray<uint32>("IndicesAToMergedIndices", KeysA.Num());
	OutIndicesBToMergedIndices = TGpuArray<uint32>("IndicesBToMergedIndices", KeysB.Num());

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

	CheckArrayBounds(OutIndicesAToMergedIndices, 0u, NumUniques - 1);
	CheckArrayBounds(OutIndicesBToMergedIndices, 0u, NumUniques - 1);

	UniqueSum.Free(Allocator);
	Permutation.Free(Allocator);

	return NumUniques;
}

inline uint32 Merge(
	const TCpuArray<uint64>& KeysA,
	const TCpuArray<uint64>& KeysB,
	TCpuArray<uint32>& OutIndicesAToMergedIndices,
	TCpuArray<uint32>& OutIndicesBToMergedIndices)
{
	PROFILE_FUNCTION_TRACY();

	check(KeysA.Num() > 0);
	check(KeysB.Num() > 0);

	check(!OutIndicesAToMergedIndices.IsValid());
	check(!OutIndicesBToMergedIndices.IsValid());

	CheckIsSorted(KeysA);
	CheckIsSorted(KeysB);
	
	OutIndicesAToMergedIndices = { "IndicesAToMergedIndices", KeysA.Num() };
	OutIndicesBToMergedIndices = { "IndicesBToMergedIndices", KeysB.Num() };

	int32 Counter = 0; // Pointer to the last item! Num = Counter + 1
	int32 IndexA = 0;
	int32 IndexB = 0;
	uint64 Last = std::min(KeysA[0], KeysB[0]);
	
	while (IndexA < KeysA.Num() && IndexB < KeysB.Num())
	{
		const uint64 KeyA = KeysA[IndexA];
		const uint64 KeyB = KeysB[IndexB];
		if (KeyA == Last)
		{
			OutIndicesAToMergedIndices[IndexA++] = Counter;
		}
		else if (KeyB == Last)
		{
			OutIndicesBToMergedIndices[IndexB++] = Counter;
		}
		else if (KeyA < KeyB)
		{
			Last = KeyA;
			OutIndicesAToMergedIndices[IndexA++] = ++Counter;
		}
		else
		{
			// KeyB <= KeyA
			Last = KeyB;
			OutIndicesBToMergedIndices[IndexB++] = ++Counter;
		}
	}
	while (IndexA < KeysA.Num())
	{
		const uint64 KeyA = KeysA[IndexA];
		if (KeyA == Last)
		{
			OutIndicesAToMergedIndices[IndexA++] = Counter;
		}
		else
		{
			Last = KeyA;
			OutIndicesAToMergedIndices[IndexA++] = ++Counter;
		}
	}
	while (IndexB < KeysB.Num())
	{
		const uint64 KeyB = KeysB[IndexB];
		if (KeyB == Last)
		{
			OutIndicesBToMergedIndices[IndexB++] = Counter;
		}
		else
		{
			Last = KeyB;
			OutIndicesBToMergedIndices[IndexB++] = ++Counter;
		}
	}
	
	checkEqual(IndexA, KeysA.Num());
	checkEqual(IndexB, KeysB.Num());
	
	return Counter + 1;
}

FCpuDag DAGCompression::MergeDAGs(FCpuDag A, FCpuDag B)
{
	PROFILE_FUNCTION();
	
	checkEqual(A.Levels.size(), B.Levels.size());

	if (A.Leaves.Num() == 0)
	{
		for (auto& Level : A.Levels) { (void)Level; check(Level.Hashes.Num() == 0); }
		return B;
	}
	if (B.Leaves.Num() == 0)
	{
		for (auto& Level : B.Levels) { (void)Level; check(Level.Hashes.Num() == 0); }
		return A;
	}

	for (auto& Level : A.Levels) { (void)Level; check(Level.Hashes.Num() != 0); }
	for (auto& Level : B.Levels) { (void)Level; check(Level.Hashes.Num() != 0); }

	const int32 NumLevels = int32(A.Levels.size());
	
	FCpuDag MergedCpuDag;
	MergedCpuDag.Levels.resize(NumLevels);
	
	TGpuArray<uint32> PreviousIndicesAToMergedIndices;
	TGpuArray<uint32> PreviousIndicesBToMergedIndices;
	{
		TGpuArray<uint32> IndicesAToMergedIndices;
		TGpuArray<uint32> IndicesBToMergedIndices;
		{
			FScopeAllocator Allocator;
			
			auto LeavesA = A.Leaves.CreateGPU();
			auto LeavesB = B.Leaves.CreateGPU();
			A.Leaves.Free(Allocator);
			B.Leaves.Free(Allocator);
			const uint32 NumMergedLeaves = Merge(
				LeavesA,
				LeavesB,
				IndicesAToMergedIndices,
				IndicesBToMergedIndices);
			auto MergedLeaves = TGpuArray<uint64>("MergedLeaves", NumMergedLeaves);
			Scatter(LeavesA, IndicesAToMergedIndices, MergedLeaves);
			Scatter(LeavesB, IndicesBToMergedIndices, MergedLeaves);
			LeavesA.Free(Allocator);
			LeavesB.Free(Allocator);
			MergedCpuDag.Leaves = MergedLeaves.CreateCPU();
			MergedLeaves.Free(Allocator);
		}
		PreviousIndicesAToMergedIndices = IndicesAToMergedIndices;
		PreviousIndicesBToMergedIndices = IndicesBToMergedIndices;
	}
	
	TCpuArray<uint32> PreviousIndicesAToMergedIndices_CPU;
	TCpuArray<uint32> PreviousIndicesBToMergedIndices_CPU;

	for (int32 LevelIndex = NumLevels - 1; LevelIndex >= 0; LevelIndex--)
	{
		FScopeAllocator Allocator;
		
		FCpuLevel& LevelA = A.Levels[LevelIndex];
		FCpuLevel& LevelB = B.Levels[LevelIndex];

		if (LevelA.Hashes.Num() + LevelB.Hashes.Num() > MERGE_MIN_NODES_FOR_GPU)
		{
			check(PreviousIndicesAToMergedIndices_CPU.Num() == 0 && PreviousIndicesBToMergedIndices_CPU.Num() == 0);
			
			auto HashesA = LevelA.Hashes.CreateGPU();
			auto HashesB = LevelB.Hashes.CreateGPU();
			LevelA.Hashes.Free(Allocator);
			LevelB.Hashes.Free(Allocator);

			TGpuArray<uint32> IndicesAToMergedIndices;
			TGpuArray<uint32> IndicesBToMergedIndices;
			const uint32 NumMerged = Merge(
				HashesA,
				HashesB,
				IndicesAToMergedIndices,
				IndicesBToMergedIndices);

			FCpuLevel MergedLevel;

			{
				auto MergedHashes = TGpuArray<uint64>("MergedHashes", NumMerged);

				Scatter(HashesA, IndicesAToMergedIndices, MergedHashes);
				Scatter(HashesB, IndicesBToMergedIndices, MergedHashes);

				HashesA.Free(Allocator);
				HashesB.Free(Allocator);

				MergedLevel.Hashes = MergedHashes.CreateCPU();
				MergedHashes.Free(Allocator);
			}

			{
				auto MergedChildMasks = TGpuArray<uint8>("MergedChildMasks", NumMerged);
				
				// TODO async copy
				
				auto ChildMasksA = LevelA.ChildMasks.CreateGPU();
				auto ChildMasksB = LevelB.ChildMasks.CreateGPU();
				LevelA.ChildMasks.Free(Allocator);
				LevelB.ChildMasks.Free(Allocator);

				Scatter(ChildMasksA, IndicesAToMergedIndices, MergedChildMasks);
				Scatter(ChildMasksB, IndicesBToMergedIndices, MergedChildMasks);

				ChildMasksA.Free(Allocator);
				ChildMasksB.Free(Allocator);

				MergedLevel.ChildMasks = MergedChildMasks.CreateCPU();
				MergedChildMasks.Free(Allocator);
			}

			{
				const auto Transform = [] GPU_LAMBDA(const TGpuArray<uint32> & IndicesMap, FChildrenIndices Indices)
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

				auto MergedChildrenIndices = TGpuArray<FChildrenIndices>("MergedChildrenIndices", NumMerged);

				auto ChildrenIndicesA = LevelA.ChildrenIndices.CreateGPU();
				auto ChildrenIndicesB = LevelB.ChildrenIndices.CreateGPU();
				LevelA.ChildrenIndices.Free(Allocator);
				LevelB.ChildrenIndices.Free(Allocator);

				ScatterWithTransform(ChildrenIndicesA, TransformA, IndicesAToMergedIndices, MergedChildrenIndices);
				ScatterWithTransform(ChildrenIndicesB, TransformB, IndicesBToMergedIndices, MergedChildrenIndices);

				ChildrenIndicesA.Free(Allocator);
				ChildrenIndicesB.Free(Allocator);

				MergedLevel.ChildrenIndices = MergedChildrenIndices.CreateCPU();
				MergedChildrenIndices.Free(Allocator);
			}

			CheckLevelIndices(MergedLevel);
			CheckIsSorted(MergedLevel.Hashes);
			MergedCpuDag.Levels[LevelIndex] = MergedLevel;

			PreviousIndicesAToMergedIndices.Free(Allocator);
			PreviousIndicesBToMergedIndices.Free(Allocator);
			PreviousIndicesAToMergedIndices = IndicesAToMergedIndices;
			PreviousIndicesBToMergedIndices = IndicesBToMergedIndices;
		}
		else
		{
			if (PreviousIndicesAToMergedIndices_CPU.Num() == 0 && PreviousIndicesAToMergedIndices.Num() > 0)
			{
				PreviousIndicesAToMergedIndices_CPU = PreviousIndicesAToMergedIndices.CreateCPU();
				PreviousIndicesAToMergedIndices.Free(Allocator);
			}
			if (PreviousIndicesBToMergedIndices_CPU.Num() == 0 && PreviousIndicesBToMergedIndices.Num() > 0)
			{
				PreviousIndicesBToMergedIndices_CPU = PreviousIndicesBToMergedIndices.CreateCPU();
				PreviousIndicesBToMergedIndices.Free(Allocator);
			}
			
			const auto HashesA = LevelA.Hashes;
			const auto HashesB = LevelB.Hashes;
			
			TCpuArray<uint32> IndicesAToMergedIndices;
			TCpuArray<uint32> IndicesBToMergedIndices;
			const uint32 NumMerged = Merge(
				HashesA,
				HashesB,
				IndicesAToMergedIndices,
				IndicesBToMergedIndices);

			FCpuLevel MergedLevel;

			{
				MergedLevel.Hashes = TCpuArray<uint64>("Hashes", NumMerged);
				Scatter(HashesA, IndicesAToMergedIndices, MergedLevel.Hashes);
				Scatter(HashesB, IndicesBToMergedIndices, MergedLevel.Hashes);
				CheckIsSorted(MergedLevel.Hashes);
			}

			{
				MergedLevel.ChildMasks = TCpuArray<uint8>("ChildMasks", NumMerged);
				Scatter(LevelA.ChildMasks, IndicesAToMergedIndices, MergedLevel.ChildMasks);
				Scatter(LevelB.ChildMasks, IndicesBToMergedIndices, MergedLevel.ChildMasks);
			}

			{
				const auto Transform = [] GPU_LAMBDA (const TCpuArray<uint32>& IndicesMap, FChildrenIndices Indices)
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
				const auto TransformA = [=] GPU_LAMBDA (FChildrenIndices Indices) { return Transform(PreviousIndicesAToMergedIndices_CPU, Indices); };
				const auto TransformB = [=] GPU_LAMBDA (FChildrenIndices Indices) { return Transform(PreviousIndicesBToMergedIndices_CPU, Indices); };

				MergedLevel.ChildrenIndices = TCpuArray<FChildrenIndices>("ChildrenIndices", NumMerged);
				ScatterWithTransform(LevelA.ChildrenIndices, TransformA, IndicesAToMergedIndices, MergedLevel.ChildrenIndices);
				ScatterWithTransform(LevelB.ChildrenIndices, TransformB, IndicesBToMergedIndices, MergedLevel.ChildrenIndices);
			}

			CheckLevelIndices(MergedLevel);
			CheckIsSorted(MergedLevel.Hashes);
			MergedCpuDag.Levels[LevelIndex] = MergedLevel;

			PreviousIndicesAToMergedIndices_CPU.Free(Allocator);
			PreviousIndicesBToMergedIndices_CPU.Free(Allocator);
			PreviousIndicesAToMergedIndices_CPU = IndicesAToMergedIndices;
			PreviousIndicesBToMergedIndices_CPU = IndicesBToMergedIndices;

			LevelA.Free(Allocator);
			LevelB.Free(Allocator);
		}
	}

	PreviousIndicesAToMergedIndices.FreeNow();
	PreviousIndicesBToMergedIndices.FreeNow();
	PreviousIndicesAToMergedIndices_CPU.FreeNow();
	PreviousIndicesBToMergedIndices_CPU.FreeNow();

	return MergedCpuDag;
}

std::thread DAGCompression::MergeColors(std::vector<TCpuArray<uint32>> Colors, TCpuArray<uint32>& MergedColors)
{
	return std::thread([Colors, &MergedColors]()
	{
		NAME_THREAD("Merge Color Thread");
		PROFILE_SCOPE("Merge Colors");
		uint64 Num = 0;
		std::vector<uint64> Offsets;
		for (auto& SubDagColors : Colors)
		{
			Offsets.push_back(Num);
			Num += SubDagColors.Num();
		}
		if (Num == 0) return;

		MergedColors = TCpuArray<uint32>("Colors", Num);

		FThreadPool ThreadPool(NUM_MERGE_COLORS_THREADS, "Merge Colors Thread Pool");
		for (uint32 Index = 0; Index < Colors.size(); Index++)
		{
			if (Colors[Index].Num() > 0) // Else we might memcpy an invalid ptr
			{
				const auto Task = [Index, &MergedColors, &Offsets, &Colors]()
				{
					std::memcpy(&MergedColors[Offsets[Index]], Colors[Index].GetData(), Colors[Index].SizeInBytes());
					const_cast<TCpuArray<uint32>&>(Colors[Index]).FreeNow();
				};
				ThreadPool.Enqueue(Task);
			}
		}
		ThreadPool.WaitForCompletion();
		ThreadPool.Destroy();
	});
}

// Note: dags are split in morton order
FCpuDag DAGCompression::MergeDAGs(std::vector<FCpuDag> CpuDags)
{
	PROFILE_FUNCTION();

#if ENABLE_CHECKS
	check(Utils::IsPowerOf2(CpuDags.size()));
	const int32 NumLevels = int32(CpuDags[0].Levels.size());
	for (auto& CpuDag : CpuDags)
	{
		checkEqual(CpuDag.Levels.size(), NumLevels);
	}
#endif

#if ENABLE_COLORS
	TCpuArray<uint32> MergedColors;
	std::vector<TCpuArray<uint32>> ColorsToMerge;
	ColorsToMerge.reserve(CpuDags.size());
	for (auto& Dag : CpuDags)
	{
		ColorsToMerge.push_back(Dag.Colors);
		Dag.Colors.Reset();
	}
	std::thread MergeColorThread = MergeColors(std::move(ColorsToMerge), MergedColors);
#endif

	FThreadPool ThreadPool(NUM_MERGE_THREADS, "Merge Thread Pool");

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
					TopLevel.ChildMasks = TCpuArray<uint8>("ChildMasks", { ChildMask });
					TopLevel.ChildrenIndices = TCpuArray<FChildrenIndices>("ChildrenIndices", { ChildrenIndices });
					TopLevel.Hashes = TCpuArray<uint64>("Hashes", { Hash });
				}
				CheckLevelIndices(TopLevel);
				
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
		MergeColorThread.join();
		CpuDags[0].Colors = MergedColors;
	}
#endif
	
	return CpuDags[0];
}