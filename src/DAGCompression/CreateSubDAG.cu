#include "DAGCompression/DAGCompression.h"
#include "GpuPrimitives.h"

HOST_DEVICE uint64 HashChildren(const FChildrenIndices& ChildrenIndices, const TGpuArray<uint64>& ChildrenArray)
{
	uint64 ChildrenHashes[8];
	for (int32 Index = 0; Index < 8; Index++)
	{
		const uint32 ChildIndex = ChildrenIndices.Indices[Index];
		ChildrenHashes[Index] = ChildIndex == 0xFFFFFFFF ? 0 : ChildrenArray[ChildIndex];
	}
	return HashChildrenHashes(ChildrenHashes);
}

void DAGCompression::ComputeHashes(FGpuLevel& Level, const TGpuArray<uint64>& LowerLevelHashes)
{
	Level.Hashes = TGpuArray<uint64>("Hashes", Level.ChildrenIndices.Num());
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

TGpuArray<uint64> DAGCompression::SortFragmentsAndRemoveDuplicates(TGpuArray<uint64> Fragments)
{
	PROFILE_FUNCTION();
	SCOPED_BANDWIDTH_TRACY(Fragments.Num());

	auto NewFragments = TGpuArray<uint64>("Fragments", Fragments.Num());

	{
		const int32 Num = Cast<int32>(Fragments.Num());
		const int32 NumBits = 3 * SUBDAG_LEVELS;
		cub::DoubleBuffer<uint64> Keys(Fragments.GetData(), NewFragments.GetData());

		FTempStorage Storage;
		CUDA_CHECKED_CALL cub::DeviceRadixSort::SortKeys(Storage.Ptr, Storage.Bytes, Keys, Num, 0, NumBits, DEFAULT_STREAM);
		Storage.Allocate();
		CUDA_CHECKED_CALL cub::DeviceRadixSort::SortKeys(Storage.Ptr, Storage.Bytes, Keys, Num, 0, NumBits, DEFAULT_STREAM);
		Storage.Free();

		if (Keys.Current() != NewFragments.GetData())
		{
			check(Keys.Current() == Fragments.GetData());
			std::swap(Fragments, NewFragments);
		}
	}

	// Sorted data is in NewFragments

	int32 NumUnique;
	{
		// Don't check the color bits
		const auto EqualityLambda = [] GPU_LAMBDA (uint64 A, uint64 B) { return (A << (64 - 3 * SUBDAG_LEVELS)) == (B << (64 - 3 * SUBDAG_LEVELS)); };
		const int32 Num = Cast<int32>(Fragments.Num());
		TSingleGPUElement<int32> NumUniqueGPU;
		
		FTempStorage Storage;
		CUDA_CHECKED_CALL cub::DeviceSelect2::Unique(Storage.Ptr, Storage.Bytes, NewFragments.GetData(), Fragments.GetData(), NumUniqueGPU.GetPtr(), Num, EqualityLambda, DEFAULT_STREAM);
		Storage.Allocate();
		CUDA_CHECKED_CALL cub::DeviceSelect2::Unique(Storage.Ptr, Storage.Bytes, NewFragments.GetData(), Fragments.GetData(), NumUniqueGPU.GetPtr(), Num, EqualityLambda, DEFAULT_STREAM);
		Storage.Free();

		NumUnique = NumUniqueGPU.GetValue();
	}
	checkInf(0, NumUnique);
	checkInfEqual(NumUnique, Fragments.Num());

	// Unique data is in Fragments

	// Shrink if less than 95% of original size
	if (NumUnique < Fragments.Num() * 0.95)
	{
		NewFragments.Free();
		NewFragments = TGpuArray<uint64>("Fragments", NumUnique);
		FMemory::CudaMemcpy(NewFragments.GetData(), Fragments.GetData(), NumUnique * sizeof(uint64), cudaMemcpyDeviceToDevice);
	}
	else
	{
		std::swap(NewFragments, Fragments);
		auto Temp = NewFragments;
		NewFragments.Reset();
		NewFragments = { Temp.GetData(), uint64(NumUnique) };
	}
	
	Fragments.Free();
	
	return NewFragments;
}

TGpuArray<uint32> DAGCompression::ExtractColorsAndFixFragments(TGpuArray<uint64>& Fragments)
{
	PROFILE_FUNCTION();

	auto Colors = TGpuArray<uint32>("Colors", Fragments.Num());
	Transform(Fragments, Colors, [] GPU_LAMBDA (uint64 X) { return X >> 40; });
	Transform(Fragments, Fragments, [] GPU_LAMBDA (uint64 X) { return X & ((uint64(1) << 40) - 1); });
	return Colors;
}

TGpuArray<uint64> DAGCompression::ExtractLeavesAndShiftReduceFragments(TGpuArray<uint64>& Fragments)
{
	PROFILE_FUNCTION();

	// Use the last 6 bits to create the leaves
	const auto ParentsTransform = [] GPU_LAMBDA (uint64 Code) { return Code >> 6; };
	const auto LeavesTransform = [] GPU_LAMBDA (uint64 Code) { return uint64(1) << (Code & 0b111111); };

	const int32 MinAverageVoxelsPer64Leaf = 1; // TODO higher?
	// TODO compute exact num?
	
	auto NewFragments = TGpuArray<uint64>("Fragments", Fragments.Num() / MinAverageVoxelsPer64Leaf);
	auto Leaves = TGpuArray<uint64>("Leaves", Fragments.Num() / MinAverageVoxelsPer64Leaf);
	
	const auto ParentsBits = TransformIterator(Fragments.GetData(), ParentsTransform);
	const auto LeavesBits = TransformIterator(Fragments.GetData(), LeavesTransform);

	const int32 Num = Cast<int32>(Fragments.Num());

	TSingleGPUElement<int32> NumRunsGPU;
	{
		FTempStorage Storage;
		CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(Storage.Ptr, Storage.Bytes, ParentsBits, NewFragments.GetData(), LeavesBits, Leaves.GetData(), NumRunsGPU.GetPtr(), thrust::bit_or<uint64>(), Num, DEFAULT_STREAM);
		Storage.Allocate();
		CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(Storage.Ptr, Storage.Bytes, ParentsBits, NewFragments.GetData(), LeavesBits, Leaves.GetData(), NumRunsGPU.GetPtr(), thrust::bit_or<uint64>(), Num, DEFAULT_STREAM);
		Storage.Free();
	}
	
	const int32 NumRuns = NumRunsGPU.GetValue();
	checkfAlways(NumRuns <= NewFragments.Num(), "Need to decrease MinAverageVoxelsPer64Leaf: %d runs, but expected max %d", NumRuns, int32(NewFragments.Num()));

	// Note: these copies & allocs could be avoided if we computed the exact number of runs first

	Fragments.Free();
	Fragments = TGpuArray<uint64>("Fragments", NumRuns);
	FMemory::CudaMemcpy(Fragments.GetData(), NewFragments.GetData(), NumRuns * sizeof(uint64), cudaMemcpyDeviceToDevice);
	NewFragments.Free();

	auto ShrunkLeaves = TGpuArray<uint64>("Leaves", NumRuns);
	FMemory::CudaMemcpy(ShrunkLeaves.GetData(), Leaves.GetData(), NumRuns * sizeof(uint64), cudaMemcpyDeviceToDevice);
	Leaves.Free();

	return ShrunkLeaves;
}

void SortLevelImpl(bool IsLeaf, FGpuLevel& Level, TGpuArray<uint32>& OutHashesToSortedUniqueHashes)
{
	PROFILE_FUNCTION_TRACY();

	using namespace DAGCompression;
	
	CheckLevelIndices(Level);

	const int32 NumHashes = Cast<int32>(Level.Hashes.Num());
	auto SortedHashes = TGpuArray<uint64>("SortedHashes", NumHashes);
	auto SortedHashesToHashes = TGpuArray<uint32>("SortedHashesToHashes", NumHashes);

	{
		// Need a real array to use CUB sort
		auto Sequence = TGpuArray<uint32>("Sequence", NumHashes);
		MakeSequence(Sequence);

		cub::DoubleBuffer<uint64> Keys(Level.Hashes.GetData(), SortedHashes.GetData());
		cub::DoubleBuffer<uint32> Values(Sequence.GetData(), SortedHashesToHashes.GetData());
		
		FTempStorage Storage;
		CUDA_CHECKED_CALL cub::DeviceRadixSort::SortPairs(Storage.Ptr, Storage.Bytes, Keys, Values, NumHashes, 0, 64, DEFAULT_STREAM);
		Storage.Allocate();
		CUDA_CHECKED_CALL cub::DeviceRadixSort::SortPairs(Storage.Ptr, Storage.Bytes, Keys, Values, NumHashes, 0, 64, DEFAULT_STREAM);
		Storage.Free();

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

	auto SortedHashesFlags = TGpuArray<uint32>("SortedHashesFlags", NumHashes);
	AdjacentDifference(SortedHashes, SortedHashesFlags, thrust::not_equal_to<uint64>(), 0u);

	auto SortedHashesToUniqueHashes = TGpuArray<uint32>("SortedHashesToUniqueHashes", NumHashes);
	{
		FTempStorage Storage;
		CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(Storage.Ptr, Storage.Bytes, SortedHashesFlags.GetData(), SortedHashesToUniqueHashes.GetData(), NumHashes, DEFAULT_STREAM);
		Storage.Allocate();
		CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(Storage.Ptr, Storage.Bytes, SortedHashesFlags.GetData(), SortedHashesToUniqueHashes.GetData(), NumHashes, DEFAULT_STREAM);
		Storage.Free();
	}
	SortedHashesFlags.Free();
	
	const uint32 NumUniques = GetElement(SortedHashesToUniqueHashes, SortedHashesToUniqueHashes.Num() - 1) + 1;

	Level.Hashes = TGpuArray<uint64>(IsLeaf ? "Leaves" : "Hashes", NumUniques);
	Scatter(SortedHashes, SortedHashesToUniqueHashes, Level.Hashes);
	SortedHashes.Free();
	CheckIsSorted(Level.Hashes);

	OutHashesToSortedUniqueHashes = TGpuArray<uint32>("HashesToSortedUniqueHashes", NumHashes);
	Scatter(SortedHashesToUniqueHashes, SortedHashesToHashes, OutHashesToSortedUniqueHashes);
	CheckArrayBounds(OutHashesToSortedUniqueHashes, 0u, NumUniques - 1);
	SortedHashesToHashes.Free();
	SortedHashesToUniqueHashes.Free();

	if (!IsLeaf)
	{
		{
			auto NewChildrenIndices = TGpuArray<FChildrenIndices>("ChildrenIndices", NumUniques);
			Scatter(Level.ChildrenIndices, OutHashesToSortedUniqueHashes, NewChildrenIndices);
			Level.ChildrenIndices.Free();
			Level.ChildrenIndices = NewChildrenIndices;
		}
		{
			auto NewChildMasks = TGpuArray<uint8>("ChildMasks", NumUniques);
			Scatter(Level.ChildMasks, OutHashesToSortedUniqueHashes, NewChildMasks);
			Level.ChildMasks.Free();
			Level.ChildMasks = NewChildMasks;
		}
	}

	CheckLevelIndices(Level);
}

void DAGCompression::SortLeaves(TGpuArray<uint64>& Leaves, TGpuArray<uint32>& OutHashesToSortedUniqueHashes)
{
	FGpuLevel Level;
	Level.Hashes = std::move(Leaves);
	SortLevelImpl(true, Level, OutHashesToSortedUniqueHashes);
	Leaves = Level.Hashes;
}

void DAGCompression::SortLevel(FGpuLevel& Level, TGpuArray<uint32>& OutHashesToSortedUniqueHashes)
{
	SortLevelImpl(false, Level, OutHashesToSortedUniqueHashes);
}

FGpuLevel DAGCompression::ExtractLevelAndShiftReduceFragments(TGpuArray<uint64>& Fragments, const TGpuArray<uint32>& FragmentIndicesToChildrenIndices)
{
	PROFILE_FUNCTION_TRACY();

	FGpuLevel OutLevel;

	const int32 Num = Cast<int32>(Fragments.Num());

	const auto ParentsTransform = [] GPU_LAMBDA(uint64 Code) { return Code >> 3; };
	const auto ChildrenTransform = [] GPU_LAMBDA(uint64 Code) { return uint64(1) << (Code & 0b111); };

	// First create the mapping & get the number of unique parents
	int32 NumUniqueParents;
	{
		auto UniqueParentsSum = TGpuArray<uint32>("UniqueParentsSum", Num);
		{
			auto UniqueParentsFlags = TGpuArray<uint32>("UniqueParentsFlags", Num);
			AdjacentDifferenceWithTransform(Fragments, ParentsTransform, UniqueParentsFlags, thrust::not_equal_to<uint64>(), 0u);
			{
				FTempStorage Storage;
				CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(Storage.Ptr, Storage.Bytes, UniqueParentsFlags.GetData(), UniqueParentsSum.GetData(), Num, DEFAULT_STREAM);
				Storage.Allocate();
				CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(Storage.Ptr, Storage.Bytes, UniqueParentsFlags.GetData(), UniqueParentsSum.GetData(), Num, DEFAULT_STREAM);
				Storage.Free();
			}
			UniqueParentsFlags.Free();
		}

		NumUniqueParents = GetElement(UniqueParentsSum, UniqueParentsSum.Num() - 1) + 1;
		checkInfEqual(NumUniqueParents, Num);

		OutLevel.ChildrenIndices = TGpuArray<FChildrenIndices>("ChildrenIndices", NumUniqueParents);
		OutLevel.ChildrenIndices.MemSet(0xFF);
		{
			const auto CompactChildrenToExpandedChildren = [=] GPU_LAMBDA(uint64 Index) { return (Fragments[Index] & 0b111) + 8 * UniqueParentsSum[Index]; };
			auto Out = OutLevel.ChildrenIndices.CastTo<uint32>();
			ScatterPred(FragmentIndicesToChildrenIndices, CompactChildrenToExpandedChildren, Out);
		}

		UniqueParentsSum.Free();
	}

	auto NewFragments = TGpuArray<uint64>("Fragments", NumUniqueParents);
	OutLevel.ChildMasks = TGpuArray<uint8>("ChildMasks", NumUniqueParents);
	{
		const auto ParentsBits = TransformIterator(Fragments.GetData(), ParentsTransform);
		const auto ChildrenBits = TransformIterator(Fragments.GetData(), ChildrenTransform);

		TSingleGPUElement<int32> NumRunsGPU;

		FTempStorage Storage;
		CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(Storage.Ptr, Storage.Bytes, ParentsBits, NewFragments.GetData(), ChildrenBits, OutLevel.ChildMasks.GetData(), NumRunsGPU.GetPtr(), thrust::bit_or<uint8>(), Num, DEFAULT_STREAM);
		Storage.Allocate();
		CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(Storage.Ptr, Storage.Bytes, ParentsBits, NewFragments.GetData(), ChildrenBits, OutLevel.ChildMasks.GetData(), NumRunsGPU.GetPtr(), thrust::bit_or<uint8>(), Num, DEFAULT_STREAM);
		Storage.Free();

		checkEqual(NumUniqueParents, NumRunsGPU.GetValue());
	}

	Fragments.Free();
	Fragments = NewFragments;

	CheckLevelIndices(OutLevel);
	return OutLevel;
}

FCpuDag DAGCompression::CreateSubDAG(TGpuArray<uint64> InFragments, FTimeCounter& SortFragmentsTime)
{
	PROFILE_FUNCTION();
	
	FCpuDag CpuDag;
	CpuDag.Levels.resize(SUBDAG_LEVELS - 2);

	if (InFragments.Num() == 0)
	{
		return CpuDag;
	}

	TGpuArray<uint64> Fragments;

	{
		SCOPE_TIME(SortFragmentsTime);
		Fragments = SortFragmentsAndRemoveDuplicates(std::move(InFragments));
	}
		
#if ENABLE_COLORS
	auto Colors = ExtractColorsAndFixFragments(Fragments);
	CpuDag.Colors = Colors.CreateCPU();
	Colors.Free();
#endif

	auto Leaves = ExtractLeavesAndShiftReduceFragments(Fragments);

	TGpuArray<uint32> FragmentIndicesToChildrenIndices;
	SortLeaves(Leaves, FragmentIndicesToChildrenIndices);
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