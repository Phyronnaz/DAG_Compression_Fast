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
	SCOPED_BANDWIDTH_TRACY(KeysA.Num() + KeysB.Num());

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
			PROFILE_SCOPE_TRACY("InclusiveSum");
			SCOPED_BANDWIDTH_TRACY(NumToMerge);
			
			FTempStorage Storage;
			CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(Storage.Ptr, Storage.Bytes, UniqueFlags.GetData(), UniqueSum.GetData(), NumToMerge, DEFAULT_STREAM);
			Storage.Allocate();
			CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(Storage.Ptr, Storage.Bytes, UniqueFlags.GetData(), UniqueSum.GetData(), NumToMerge, DEFAULT_STREAM);
			Storage.Free();
		}
		UniqueFlags.Free();
	}
	KeysAB.Free();

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

	UniqueSum.Free();
	Permutation.Free();

	return NumUniques;
}

inline uint32 Merge(
	const TCpuArray<uint64>& KeysA,
	const TCpuArray<uint64>& KeysB,
	TCpuArray<uint32>& OutIndicesAToMergedIndices,
	TCpuArray<uint32>& OutIndicesBToMergedIndices)
{
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(KeysA.Num() + KeysB.Num());

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

std::shared_ptr<FCpuDag> DAGCompression::MergeDAGs(FCpuDag A, FCpuDag B, FThreadPool& Pool)
{
	PROFILE_FUNCTION();
	
	checkEqual(A.Levels.size(), B.Levels.size());

	if (A.Leaves.Num() == 0)
	{
		for (auto& Level : A.Levels) { (void)Level; check(Level.Hashes.Num() == 0); }
		return std::make_shared<FCpuDag>(B);
	}
	if (B.Leaves.Num() == 0)
	{
		for (auto& Level : B.Levels) { (void)Level; check(Level.Hashes.Num() == 0); }
		return std::make_shared<FCpuDag>(A);
	}

	for (auto& Level : A.Levels) { (void)Level; check(Level.Hashes.Num() != 0); }
	for (auto& Level : B.Levels) { (void)Level; check(Level.Hashes.Num() != 0); }

	const int32 NumLevels = int32(A.Levels.size());

	std::shared_ptr<FCpuDag> MergedCpuDag = std::make_shared<FCpuDag>();
	MergedCpuDag->Levels.resize(NumLevels);
	
	TGpuArray<uint32> PreviousIndicesAToMergedIndices;
	TGpuArray<uint32> PreviousIndicesBToMergedIndices;
	{
		TGpuArray<uint32> IndicesAToMergedIndices;
		TGpuArray<uint32> IndicesBToMergedIndices;
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
			auto MergedLeaves = TGpuArray<uint64>("MergedLeaves", NumMergedLeaves);
			Scatter(LeavesA, IndicesAToMergedIndices, MergedLeaves);
			Scatter(LeavesB, IndicesBToMergedIndices, MergedLeaves);
			LeavesA.Free();
			LeavesB.Free();

			Pool.Enqueue([MergedCpuDag, InMergedLeaves = MergedLeaves]()
			{
				auto MergedLeaves = InMergedLeaves;
				MergedCpuDag->Leaves = MergedLeaves.CreateCPU();
				MergedLeaves.Free();
			});
		}
		PreviousIndicesAToMergedIndices = IndicesAToMergedIndices;
		PreviousIndicesBToMergedIndices = IndicesBToMergedIndices;
	}

	int32 LevelIndex = NumLevels - 1;
	for (; LevelIndex >= 0; LevelIndex--)
	{
		FCpuLevel& LevelA = A.Levels[LevelIndex];
		FCpuLevel& LevelB = B.Levels[LevelIndex];

		if (LevelA.Hashes.Num() + LevelB.Hashes.Num() > MERGE_MIN_NODES_FOR_GPU)
		{
			break;
		}
	
		PROFILE_SCOPE_TRACY("Level %d GPU", LevelIndex);
		
		auto HashesA = LevelA.Hashes.CreateGPU();
		auto HashesB = LevelB.Hashes.CreateGPU();

		TGpuArray<uint32> IndicesAToMergedIndices;
		TGpuArray<uint32> IndicesBToMergedIndices;
		const uint32 NumMerged = Merge(
			HashesA,
			HashesB,
			IndicesAToMergedIndices,
			IndicesBToMergedIndices);

		FCpuLevel& MergedLevel = MergedCpuDag->Levels[LevelIndex];

		{
			auto MergedHashes = TGpuArray<uint64>("MergedHashes", NumMerged);

			Scatter(HashesA, IndicesAToMergedIndices, MergedHashes);
			Scatter(HashesB, IndicesBToMergedIndices, MergedHashes);

			HashesA.Free();
			HashesB.Free();

			Pool.Enqueue([&MergedLevel, InMergedHashes = MergedHashes]()
			{
				auto MergedHashes = InMergedHashes;
				MergedLevel.Hashes = MergedHashes.CreateCPU();
				MergedHashes.Free();
			});
		}

		{
			auto MergedChildMasks = TGpuArray<uint8>("MergedChildMasks", NumMerged);
			
			auto ChildMasksA = LevelA.ChildMasks.CreateGPU();
			auto ChildMasksB = LevelB.ChildMasks.CreateGPU();

			Scatter(ChildMasksA, IndicesAToMergedIndices, MergedChildMasks);
			Scatter(ChildMasksB, IndicesBToMergedIndices, MergedChildMasks);

			ChildMasksA.Free();
			ChildMasksB.Free();

			Pool.Enqueue([&MergedLevel, InMergedChildMasks = MergedChildMasks]()
			{
				auto MergedChildMasks = InMergedChildMasks;
				MergedLevel.ChildMasks = MergedChildMasks.CreateCPU();
				MergedChildMasks.Free();
			});
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

			ScatterWithTransform(ChildrenIndicesA, TransformA, IndicesAToMergedIndices, MergedChildrenIndices);
			ScatterWithTransform(ChildrenIndicesB, TransformB, IndicesBToMergedIndices, MergedChildrenIndices);

			ChildrenIndicesA.Free();
			ChildrenIndicesB.Free();

			Pool.Enqueue([&MergedLevel, InMergedChildrenIndices = MergedChildrenIndices]()
			{
				auto MergedChildrenIndices = InMergedChildrenIndices;
				MergedLevel.ChildrenIndices = MergedChildrenIndices.CreateCPU();
				MergedChildrenIndices.Free();
			});
		}

		CheckLevelIndices(MergedLevel);
		CheckIsSorted(MergedLevel.Hashes);

		PreviousIndicesAToMergedIndices.Free();
		PreviousIndicesBToMergedIndices.Free();
		PreviousIndicesAToMergedIndices = IndicesAToMergedIndices;
		PreviousIndicesBToMergedIndices = IndicesBToMergedIndices;

		Pool.Enqueue([InLevelA = LevelA, InLevelB = LevelB]()
		{
			auto LevelA = InLevelA;
			auto LevelB = InLevelB;
			LevelA.Free();
			LevelB.Free();
		});
	}

	if (LevelIndex >= 0)
	{
		Pool.Enqueue([
				InLevelIndex = LevelIndex, InA = A, InB = B, MergedCpuDag,
				InPreviousIndicesAToMergedIndices = PreviousIndicesAToMergedIndices, 
				InPreviousIndicesBToMergedIndices = PreviousIndicesBToMergedIndices]()
		{
			auto A = InA;
			auto B = InB;
			auto PreviousIndicesAToMergedIndices = InPreviousIndicesAToMergedIndices;
			auto PreviousIndicesBToMergedIndices = InPreviousIndicesBToMergedIndices;
			TCpuArray<uint32> PreviousIndicesAToMergedIndices_CPU = PreviousIndicesAToMergedIndices.CreateCPU();
			TCpuArray<uint32> PreviousIndicesBToMergedIndices_CPU = PreviousIndicesBToMergedIndices.CreateCPU();
			PreviousIndicesAToMergedIndices.Free();
			PreviousIndicesBToMergedIndices.Free();

			for (int32 LevelIndex = InLevelIndex; LevelIndex >= 0; LevelIndex--)
			{
				PROFILE_SCOPE_TRACY("Level %d CPU", LevelIndex);

				FCpuLevel& LevelA = A.Levels[LevelIndex];
				FCpuLevel& LevelB = B.Levels[LevelIndex];

				const auto HashesA = LevelA.Hashes;
				const auto HashesB = LevelB.Hashes;

				TCpuArray<uint32> IndicesAToMergedIndices;
				TCpuArray<uint32> IndicesBToMergedIndices;
				const uint32 NumMerged = Merge(
					HashesA,
					HashesB,
					IndicesAToMergedIndices,
					IndicesBToMergedIndices);

				FCpuLevel& MergedLevel = MergedCpuDag->Levels[LevelIndex];

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
					const auto Transform = [] GPU_LAMBDA(const TCpuArray<uint32> & IndicesMap, FChildrenIndices Indices)
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
					const auto TransformA = [=] GPU_LAMBDA(FChildrenIndices Indices) { return Transform(PreviousIndicesAToMergedIndices_CPU, Indices); };
					const auto TransformB = [=] GPU_LAMBDA(FChildrenIndices Indices) { return Transform(PreviousIndicesBToMergedIndices_CPU, Indices); };

					MergedLevel.ChildrenIndices = TCpuArray<FChildrenIndices>("ChildrenIndices", NumMerged);
					ScatterWithTransform(LevelA.ChildrenIndices, TransformA, IndicesAToMergedIndices, MergedLevel.ChildrenIndices);
					ScatterWithTransform(LevelB.ChildrenIndices, TransformB, IndicesBToMergedIndices, MergedLevel.ChildrenIndices);
				}

				CheckLevelIndices(MergedLevel);
				CheckIsSorted(MergedLevel.Hashes);
				
				PreviousIndicesAToMergedIndices_CPU.Free();
				PreviousIndicesBToMergedIndices_CPU.Free();
				PreviousIndicesAToMergedIndices_CPU = IndicesAToMergedIndices;
				PreviousIndicesBToMergedIndices_CPU = IndicesBToMergedIndices;

				LevelA.Free();
				LevelB.Free();
			}
			PreviousIndicesAToMergedIndices_CPU.Free();
			PreviousIndicesBToMergedIndices_CPU.Free();
		});
	}
	else
	{
		PreviousIndicesAToMergedIndices.Free();
		PreviousIndicesBToMergedIndices.Free();
	}

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
					const_cast<TCpuArray<uint32>&>(Colors[Index]).Free();
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

	FThreadPool ThreadPool(6, "Merge Thread Pool");

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
			PROFILE_SCOPE("Sub Dag %d", SubDagIndex);

			const auto GetDag = [&](int32 Index) -> FCpuDag& { return CpuDags[8 * SubDagIndex + Index]; };

			uint64 Hashes[8];
			for (int32 Index = 0; Index < 8; Index++)
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
			}

			FCpuDag MergedDag;
			{
				auto Dag01 = MergeDAGs(GetDag(0), GetDag(1), ThreadPool);
				auto Dag23 = MergeDAGs(GetDag(2), GetDag(3), ThreadPool);
				auto Dag45 = MergeDAGs(GetDag(4), GetDag(5), ThreadPool);
				auto Dag67 = MergeDAGs(GetDag(6), GetDag(7), ThreadPool);
				ThreadPool.WaitForCompletion();
				auto Dag0123 = MergeDAGs(*Dag01, *Dag23, ThreadPool);
				auto Dag4567 = MergeDAGs(*Dag45, *Dag67, ThreadPool);
				ThreadPool.WaitForCompletion();
				auto Dag01234567 = MergeDAGs(*Dag0123, *Dag4567, ThreadPool);
				ThreadPool.WaitForCompletion();
				MergedDag = *Dag01234567;
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