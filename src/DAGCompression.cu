#include "DAGCompression.h"
#include "CudaPerfRecorder.h"

#pragma push_macro("check")
#undef check
#include "cub/iterator/transform_input_iterator.cuh"
#include "cub/iterator/counting_input_iterator.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_reduce.cuh"
#include "cub/device/device_scan.cuh"
#include "cub/device/device_select.cuh"
#include <thrust/adjacent_difference.h>
#include <thrust/sequence.h>
#include <thrust/scatter.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#pragma pop_macro("check")

template<typename T, typename F>
inline auto TransformIterator(const T* Ptr, F Lambda)
{
	using TReturnType = decltype(Lambda(T()));
	return cub::TransformInputIterator<const T, F, const TReturnType*>(Ptr, Lambda);
}

template<typename T>
inline void SortKeys(
	TDynamicArray<uint8, EMemoryType::GPU>& TempStorage, 
	const TStaticArray<T, EMemoryType::GPU>& Src, 
	TStaticArray<T, EMemoryType::GPU>& Dst, 
	int32 NumBits)
{
	checkEqual(Src.Num(), Dst.Num());

	size_t TempStorageBytes;
	CUDA_CHECKED_CALL cub::DeviceRadixSort::SortKeys(nullptr, TempStorageBytes, Src.GetData(), Dst.GetData(), Cast<int32>(Src.Num()), 0, NumBits);
	TempStorage.ResizeUninitialized(TempStorageBytes);

	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Src.Num());
	CUDA_CHECKED_CALL cub::DeviceRadixSort::SortKeys(TempStorage.GetData(), TempStorageBytes, Src.GetData(), Dst.GetData(), Cast<int32>(Src.Num()), 0, NumBits);
}

template<typename KeyType, typename ValueType>
inline void SortPairs(
	TDynamicArray<uint8, EMemoryType::GPU>& TempStorage,
	const TStaticArray<KeyType, EMemoryType::GPU>& KeysIn,
	TStaticArray<KeyType, EMemoryType::GPU>& KeysOut,
	const TStaticArray<ValueType, EMemoryType::GPU>& ValuesIn,
	TStaticArray<ValueType, EMemoryType::GPU>& ValuesOut,
	int32 NumBits)
{
	checkEqual(KeysIn.Num(), KeysOut.Num());
	checkEqual(ValuesIn.Num(), ValuesOut.Num());
	checkEqual(KeysIn.Num(), ValuesIn.Num());

	size_t TempStorageBytes;
	CUDA_CHECKED_CALL cub::DeviceRadixSort::SortPairs(
		nullptr, TempStorageBytes,
		KeysIn.GetData(), KeysOut.GetData(),
		ValuesIn.GetData(), ValuesOut.GetData(),
		Cast<int32>(KeysIn.Num()),
		0, NumBits);
	TempStorage.ResizeUninitialized(TempStorageBytes);

	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, KeysIn.Num());
	CUDA_CHECKED_CALL cub::DeviceRadixSort::SortPairs(
		TempStorage.GetData(), TempStorageBytes,
		KeysIn.GetData(), KeysOut.GetData(),
		ValuesIn.GetData(), ValuesOut.GetData(),
		Cast<int32>(KeysIn.Num()),
		0, NumBits);
}

template<typename T>
inline void SelectUnique(
	TDynamicArray<uint8, EMemoryType::GPU>& TempStorage, 
	const TStaticArray<T, EMemoryType::GPU>& Src, 
	TDynamicArray<T, EMemoryType::GPU>& Dst)
{
	checkEqual(Src.Num(), Dst.Num());

	SINGLE_GPU_ELEMENT(int32, NumSelected);

	size_t TempStorageBytes;
	CUDA_CHECKED_CALL cub::DeviceSelect::Unique(nullptr, TempStorageBytes, Src.GetData(), Dst.GetData(), NumSelected.ToGPU(), Cast<int32>(Src.Num()));
	TempStorage.ResizeUninitialized(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Src.Num());
		CUDA_CHECKED_CALL cub::DeviceSelect::Unique(TempStorage.GetData(), TempStorageBytes, Src.GetData(), Dst.GetData(), NumSelected.ToGPU(), Cast<int32>(Src.Num()));
	}

	const int32 NumSelected_CPU = NumSelected.ToCPU();
	checkInfEqual(NumSelected_CPU, Dst.Num());
	Dst.Resize(NumSelected_CPU);
}

template<typename KeyIn, typename KeyOut, typename ValueIn, typename ValueOut, typename F1, typename F2, typename F3>
inline void ReduceByKey(
	TDynamicArray<uint8, EMemoryType::GPU>& TempStorage, 
	const TStaticArray<KeyIn, EMemoryType::GPU>& Keys, F1 KeysTransform, 
	const TStaticArray<ValueIn, EMemoryType::GPU>& Values, F2 ValuesTransform,
	TDynamicArray<KeyOut, EMemoryType::GPU>& UniqueKeysOut, 
	TDynamicArray<ValueOut, EMemoryType::GPU>& ReducedValuesOut,
	F3 Reduce)
{
	checkEqual(Keys.Num(), Values.Num());
	checkEqual(Keys.Num(), UniqueKeysOut.Num());
	checkEqual(Keys.Num(), ReducedValuesOut.Num());

	using KeyType = decltype(Keys[0]);
	
	SINGLE_GPU_ELEMENT(int32, NumRuns);

	const auto KeysIt = TransformIterator(Keys.GetData(), KeysTransform);
	const auto ValuesIt = TransformIterator(Values.GetData(), ValuesTransform);

	size_t TempStorageBytes;
	CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(nullptr, TempStorageBytes, KeysIt, UniqueKeysOut.GetData(), ValuesIt, ReducedValuesOut.GetData(), NumRuns.ToGPU(), Reduce, Cast<int32>(Keys.Num()));
	TempStorage.ResizeUninitialized(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Keys.Num());
		CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(TempStorage.GetData(), TempStorageBytes, KeysIt, UniqueKeysOut.GetData(), ValuesIt, ReducedValuesOut.GetData(), NumRuns.ToGPU(), Reduce, Cast<int32>(Keys.Num()));
	}

	const int32 NumRuns_CPU = NumRuns.ToCPU();
	checkInfEqual(NumRuns_CPU, Keys.Num());
	UniqueKeysOut.Resize(NumRuns_CPU);
	ReducedValuesOut.Resize(NumRuns_CPU);
}

template<typename InType, typename OutType>
inline void InclusiveSum(
	TDynamicArray<uint8, EMemoryType::GPU>& TempStorage, 
	const TStaticArray<InType, EMemoryType::GPU>& In, 
	TStaticArray<OutType, EMemoryType::GPU>& Out)
{
	checkEqual(In.Num(), Out.Num());

	size_t TempStorageBytes;
	CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(nullptr, TempStorageBytes, In.GetData(), Out.GetData(), Cast<int32>(In.Num()));
	TempStorage.ResizeUninitialized(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
		CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(TempStorage.GetData(), TempStorageBytes, In.GetData(), Out.GetData(), Cast<int32>(In.Num()));
	}
}

template<typename InType, typename OutType, typename F>
inline void ExclusiveSumWithTransform(
	TDynamicArray<uint8, EMemoryType::GPU>& TempStorage, 
	const TStaticArray<InType, EMemoryType::GPU>& In, F Transform, 
	TStaticArray<OutType, EMemoryType::GPU>& Out)
{
	checkEqual(In.Num(), Out.Num());
	
	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);

	size_t TempStorageBytes;
	CUDA_CHECKED_CALL cub::DeviceScan::ExclusiveSum(nullptr, TempStorageBytes, It, Out.GetData(), Cast<int32>(In.Num()));
	TempStorage.ResizeUninitialized(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
		CUDA_CHECKED_CALL cub::DeviceScan::ExclusiveSum(TempStorage.GetData(), TempStorageBytes, It, Out.GetData(), Cast<int32>(In.Num()));
	}
}

template<typename T>
inline void SetElement(TStaticArray<T, EMemoryType::GPU>& Array, uint64 Index, T Value)
{
	CUDA_CHECKED_CALL cudaMemcpy(&Array[Index], &Value, sizeof(T), cudaMemcpyHostToDevice);
}
template<typename T>
inline T GetElement(const TStaticArray<T, EMemoryType::GPU>& Array, uint64 Index)
{
	T Value;
	CUDA_CHECKED_CALL cudaMemcpy(&Value, &Array[Index], sizeof(T), cudaMemcpyDeviceToHost);
	return Value;
}

template<typename InType, typename OutType, typename F1>
inline void AdjacentDifference(
	const TStaticArray<InType, EMemoryType::GPU>& In,
	TStaticArray<OutType, EMemoryType::GPU>& Out,
	F1 BinaryOp,
	OutType FirstElement)
{
	checkEqual(In.Num(), Out.Num())

	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	thrust::transform(
		In.GetData() + 1,
		In.GetData() + In.Num(),
		In.GetData(),
		Out.GetData() + 1,
		BinaryOp);
	SetElement(Out, 0, FirstElement);
}

template<typename InType, typename OutType, typename F1, typename F2>
inline void AdjacentDifferenceWithTransform(
	const TStaticArray<InType, EMemoryType::GPU>& In, F1 Transform,
	TStaticArray<OutType, EMemoryType::GPU>& Out,
	F2 BinaryOp,
	OutType FirstElement)
{
	checkEqual(In.Num(), Out.Num())

	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);
	thrust::transform(
		It + 1,
		It + In.Num(),
		It,
		Out.GetData() + 1,
		BinaryOp);
	SetElement(Out, 0, FirstElement);
}

template<typename Type, typename MapFunction, typename OutIterator>
inline void ScatterPred(
	const TStaticArray<Type, EMemoryType::GPU>& In,
	MapFunction Map,
	OutIterator Out)
{
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	thrust::scatter(
		In.GetData(),
		In.GetData() + In.Num(),
		MapTransform,
		Out);
}

template<typename Type, typename IndexType, typename OutIterator>
inline void Scatter(
	const TStaticArray<Type, EMemoryType::GPU>& In,
	const TStaticArray<IndexType, EMemoryType::GPU>& Map,
	OutIterator Out)
{
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	thrust::scatter(
		In.GetData(),
		In.GetData() + In.Num(),
		Map.GetData(),
		Out);
}

template<typename Type, typename MapFunction, typename OutIterator, typename F>
inline void ScatterIf(
	const TStaticArray<Type, EMemoryType::GPU>& In,
	MapFunction Map,
	OutIterator Out,
	F Condition)
{
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	thrust::scatter_if(
		In.GetData(),
		In.GetData() + In.Num(),
		MapTransform,
		thrust::make_counting_iterator<uint64>(0),
		Out,
		Condition);
}

template<typename Type, typename InF, typename MapFunction, typename OutIterator, typename F>
inline void ScatterIfWithTransform(
	const TStaticArray<Type, EMemoryType::GPU>& In, InF Transform,
	MapFunction Map,
	OutIterator Out,
	F Condition)
{
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);
	thrust::scatter_if(
		It,
		It + In.Num(),
		MapTransform,
		thrust::make_counting_iterator<uint64>(0),
		Out,
		Condition);
}

template<typename InType, typename OutType, typename F>
inline void Transform(
	const TStaticArray<InType, EMemoryType::GPU>& In,
	TStaticArray<OutType, EMemoryType::GPU>& Out,
	F Lambda)
{
	checkEqual(In.Num(), Out.Num())
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	thrust::transform(
		In.GetData(),
		In.GetData() + In.Num(),
		Out.GetData(),
		Lambda);
}

template<typename InType, typename OutType, typename F1, typename F2>
inline void TransformIf(
	InType In,
	OutType Out,
	const uint64 Num,
	F1 Lambda,
	F2 Condition)
{
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Num);
	thrust::transform_if(
		In,
		In + Num,
		Out,
		Lambda,
		Condition);
}

template<typename T>
inline void MakeSequence(TStaticArray<T, EMemoryType::GPU>& Array)
{
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Array.Num());
	thrust::sequence(Array.GetData(), Array.GetData() + Array.Num());
}

inline void LogRemoved(const char* Name, uint64 Before, uint64 After)
{
	checkInfEqual(After, Before);
	LOG("%s: %llu elements (%f%%) removed", Name, Before - After, double(Before - After) * 100. / Before);
}

struct FChildrenIndices
{
	uint32 Indices[8];
};
static_assert(sizeof(FChildrenIndices) == 8 * sizeof(uint32), "");

HOST_DEVICE uint64 HashChildren(const FChildrenIndices& ChildrenIndices, const TStaticArray<uint64, EMemoryType::GPU>& ChildrenArray)
{
	uint64 Data[8];
	for (int32 Index = 0; Index < 8; Index++)
	{
		const uint32 ChildIndex = ChildrenIndices.Indices[Index];
		Data[Index] = ChildIndex == 0xFFFFFFFF ? 0 : ChildrenArray[ChildIndex];
	}
	return Utils::MurmurHash128xN(reinterpret_cast<const uint128*>(Data), 4).Data[0];
}

struct FLevel
{
	TStaticArray<uint8, EMemoryType::GPU> ChildMasks;
	TStaticArray<FChildrenIndices, EMemoryType::GPU> ChildrenIndices;
	TStaticArray<uint64, EMemoryType::GPU> Hashes;
	TStaticArray<uint32, EMemoryType::GPU> ChildPositions;
	uint32 NumWords = 0;

	void Free()
	{
		ChildMasks.Free();
		ChildrenIndices.Free();
		Hashes.Free();
		ChildPositions.Free();
	}

	void ComputeHashes(const TStaticArray<uint64, EMemoryType::GPU>& LowerLevelHashes)
	{
		Hashes = TStaticArray<uint64, EMemoryType::GPU>("Hashes", ChildrenIndices.Num());
		Transform(ChildrenIndices, Hashes, [=] GPU_LAMBDA(const FChildrenIndices& Children) { return HashChildren(Children, LowerLevelHashes); });
	}
	void ComputeChildPositions(TDynamicArray<uint8, EMemoryType::GPU>& TempStorage)
	{
		ChildPositions = TStaticArray<uint32, EMemoryType::GPU>("ChildPositions", ChildMasks.Num());
		ExclusiveSumWithTransform(
			TempStorage,
			ChildMasks, [] GPU_LAMBDA(uint8 ChildMask) { return Utils::TotalSize(ChildMask); },
			ChildPositions);
		NumWords = GetElement(ChildPositions, ChildPositions.Num() - 1) + Utils::TotalSize(GetElement(ChildMasks, ChildMasks.Num() - 1));
	}
	template<typename T>
	void WriteTo(uint32* Data, T GetChildIndex) const
	{
		Scatter(ChildMasks, ChildPositions, Data);
		const auto ChildrenIndicesArray = ChildrenIndices.CastTo<uint32>();
		ScatterIfWithTransform(
			ChildrenIndicesArray, [=] GPU_LAMBDA (uint32 Index) { return Index == 0xFFFFFFFF ? Index : GetChildIndex(Index); },
			[=] GPU_LAMBDA (uint64 Index) { return ChildPositions[Index / 8] + Utils::ChildOffset(ChildMasks[Index / 8], Index % 8); },
			Data,
			[=] GPU_LAMBDA(uint64 Index) { check((ChildrenIndicesArray[Index] == 0xFFFFFFFF) == !(ChildMasks[Index / 8] & (1 << (Index % 8)))); return ChildrenIndicesArray[Index] != 0xFFFFFFFF; });
	}
};

template<typename F>
void SortLevel(
	TDynamicArray<uint8, EMemoryType::GPU>& TempStorage,
	TStaticArray<uint64, EMemoryType::GPU>& InOutHashes,
	TStaticArray<uint32, EMemoryType::GPU>& OutHashesToSortedUniqueHashes,
	F ApplySortToOtherArray)
{
	const auto Hashes = InOutHashes;
	
	TStaticArray<uint64, EMemoryType::GPU> SortedHashes("SortedHashes", Hashes.Num());
	TStaticArray<uint32, EMemoryType::GPU> SortedHashesToHashes("SortedHashesToHashes", Hashes.Num());

	{
		// Need a real array to use CUB sort
		TStaticArray<uint32, EMemoryType::GPU> Sequence("Sequence", Hashes.Num());
		MakeSequence(Sequence);
		SortPairs(TempStorage, Hashes, SortedHashes, Sequence, SortedHashesToHashes, 64);
		Sequence.Free();
	}

	TStaticArray<uint32, EMemoryType::GPU> SortedHashesFlags("UniqueHashesFlags", Hashes.Num());
	AdjacentDifference(SortedHashes, SortedHashesFlags, thrust::not_equal_to<uint64>(), 0u);
	SortedHashes.Free();
	
	TStaticArray<uint32, EMemoryType::GPU> SortedHashesToUniqueHashes("SortedHashesToUniqueHashes", Hashes.Num());
	InclusiveSum(TempStorage, SortedHashesFlags, SortedHashesToUniqueHashes);
	SortedHashesFlags.Free();
	
	const uint32 NumUniques = GetElement(SortedHashesToUniqueHashes, SortedHashesToUniqueHashes.Num() - 1) + 1;

	OutHashesToSortedUniqueHashes = TStaticArray<uint32, EMemoryType::GPU>("HashesToSortedUniqueHashes", Hashes.Num());
	Scatter(SortedHashesToUniqueHashes, SortedHashesToHashes, OutHashesToSortedUniqueHashes.GetData());
	SortedHashesToHashes.Free();

	const auto Sort = [&](auto& Array)
	{
		checkEqual(Array.Num(), OutHashesToSortedUniqueHashes.Num());
		typename std::remove_reference<decltype(Array)>::type FinalArray(FMemory::GetAllocInfo(Array.GetData()).Name, NumUniques);
		Scatter(Array, OutHashesToSortedUniqueHashes, FinalArray.GetData());
		Array.Free();
		Array = FinalArray;
	};

	ApplySortToOtherArray(Sort);

	Sort(InOutHashes);

	SortedHashesToUniqueHashes.Free();
}

FLevel ExtractLevel(
	TDynamicArray<uint8, EMemoryType::GPU>& TempStorage,
	TStaticArray<uint64, EMemoryType::GPU>& InOutFragments,
	const TStaticArray<uint32, EMemoryType::GPU>& FragmentIndicesToChildrenIndices)
{
	FLevel OutLevel;
	
	const auto Fragments = InOutFragments;

	const auto ParentsTransform = [] GPU_LAMBDA (uint64 Code) { return Code >> 3; };
	const auto ChildIndicesTransform = [] GPU_LAMBDA (uint64 Code) { return uint64(1) << (Code & 0b111); };

	TDynamicArray<uint64, EMemoryType::GPU> Parents("Parents", Fragments.Num());
	{
		auto ChildMasks = TDynamicArray<uint8, EMemoryType::GPU>("ChildMasks", Fragments.Num());
		ReduceByKey(
			TempStorage,
			Fragments, ParentsTransform,
			Fragments, ChildIndicesTransform,
			Parents,
			ChildMasks,
			thrust::bit_or<uint8>());

		ChildMasks.Shrink();
		OutLevel.ChildMasks = TStaticArray<uint8, EMemoryType::GPU>(ChildMasks);
	}

	{
		TStaticArray<uint32, EMemoryType::GPU> UniqueParentsSum("UniqueParentsSum", Fragments.Num());
		{
			TStaticArray<uint32, EMemoryType::GPU> UniqueParentsFlags("UniqueParentsFlags", Fragments.Num());
			AdjacentDifferenceWithTransform(Fragments, ParentsTransform, UniqueParentsFlags, thrust::not_equal_to<uint64>(), 0u);
			InclusiveSum(TempStorage, UniqueParentsFlags, UniqueParentsSum);
			UniqueParentsFlags.Free();
		}

		OutLevel.ChildrenIndices = TStaticArray<FChildrenIndices, EMemoryType::GPU>("ChildrenIndices", Parents.Num());
		OutLevel.ChildrenIndices.MemSet(0xFF);

		const auto CompactChildrenToExpandedChildren = [=] GPU_LAMBDA (uint64 Index) { return (Fragments[Index] & 0b111) + 8 * UniqueParentsSum[Index]; };
		ScatterPred(FragmentIndicesToChildrenIndices, CompactChildrenToExpandedChildren, OutLevel.ChildrenIndices.CastTo<uint32>().GetData());

#if ENABLE_CHECKS && DEBUG_GPU_ARRAYS
		const auto ChildrenIndicesArray = OutLevel.ChildrenIndices.CastTo<uint32>();
		for (uint64 Index = 0; Index < ChildrenIndicesArray.Num(); Index++)
		{
			check((ChildrenIndicesArray[Index] == 0xFFFFFFFF) == !(OutLevel.ChildMasks[Index / 8] & (1 << (Index % 8))));
		}
#endif

		UniqueParentsSum.Free();
	}
	
	Parents.Shrink();
	InOutFragments.Free();
	InOutFragments = TStaticArray<uint64, EMemoryType::GPU>(Parents);

	return OutLevel;
}

void CheckDag(const TStaticArray<uint32, EMemoryType::GPU>& Dag, uint32 Index, uint32 Level)
{
	if (Level == LEVELS - 2)
	{
		Dag[Index + 0];
		Dag[Index + 1];
	}
	else
	{
		const uint32 ChildMask = Dag[Index];
		checkInf(ChildMask, 256);
		for (uint32 ChildIndex = 0; ChildIndex < Utils::Popc(ChildMask); ChildIndex++)
		{
			CheckDag(Dag, Dag[Index + 1 + ChildIndex], Level + 1);
		}
	}
}

TStaticArray<uint32, EMemoryType::GPU> DAGCompression::CreateDAG(const TStaticArray<uint64, EMemoryType::GPU>& InFragments, const int32 Depth)
{
	TDynamicArray<uint8, EMemoryType::GPU> TempStorage("TempStorage", 32);

	TStaticArray<uint64, EMemoryType::GPU> Fragments;
	{
		TStaticArray<uint64, EMemoryType::GPU> SortedFragments("SortedFragments", InFragments.Num());
		SortKeys(TempStorage, InFragments, SortedFragments, 3 * Depth);

		TDynamicArray<uint64, EMemoryType::GPU> UniqueFragments("UniqueFragments", SortedFragments.Num());
		SelectUnique(TempStorage, SortedFragments, UniqueFragments);

		LogRemoved("Fragments", SortedFragments.Num(), UniqueFragments.Num());

		SortedFragments.Free();
		UniqueFragments.Shrink();
		Fragments = TStaticArray<uint64, EMemoryType::GPU>(UniqueFragments);
	}

	TStaticArray<uint64, EMemoryType::GPU> Leaves;
	{
		TDynamicArray<uint64, EMemoryType::GPU> LocalLeaves("Leaves", Fragments.Num());
		// Use the last 6 bits to create the leaves
		const auto ParentsTransform = [] GPU_LAMBDA(uint64 Code) { return Code >> 6; };
		const auto LeavesTransform = [] GPU_LAMBDA(uint64 Code) { return uint64(1) << (Code & 0b111111); };

		TDynamicArray<uint64, EMemoryType::GPU> Parents("Parents", Fragments.Num());
		ReduceByKey(
			TempStorage,
			Fragments, ParentsTransform,
			Fragments, LeavesTransform,
			Parents,
			LocalLeaves,
			thrust::bit_or<uint64>());
		LogRemoved("Parents", Fragments.Num(), Parents.Num());
		LOG("%llu leaves", LocalLeaves.Num());
		LocalLeaves.Shrink();
		Leaves = TStaticArray<uint64, EMemoryType::GPU>(LocalLeaves);
		
		Parents.Shrink();
		Fragments.Free();
		Fragments = TStaticArray<uint64, EMemoryType::GPU>(Parents);
	}

	std::vector<FLevel> Levels(Depth - 2);

	TStaticArray<uint32, EMemoryType::GPU> FragmentIndicesToChildrenIndices;
	SortLevel(TempStorage, Leaves, FragmentIndicesToChildrenIndices, [&](auto) {});

	{
		FLevel Level = ExtractLevel(TempStorage, Fragments, FragmentIndicesToChildrenIndices);
		FragmentIndicesToChildrenIndices.Free();
		
		Level.ComputeHashes(Leaves);

		SortLevel(TempStorage, Level.Hashes, FragmentIndicesToChildrenIndices, [&](auto Sort) { Sort(Level.ChildrenIndices); Sort(Level.ChildMasks); });

		Levels[Depth - 3] = Level;
	}

	for (int32 LevelDepth = Depth - 4; LevelDepth >= 0; LevelDepth--)
	{
		LOG("Level %d", LevelDepth);
		
		FLevel Level = ExtractLevel(TempStorage, Fragments, FragmentIndicesToChildrenIndices);
		FragmentIndicesToChildrenIndices.Free();
		
		Level.ComputeHashes(Levels.back().Hashes);

		SortLevel(TempStorage, Level.Hashes, FragmentIndicesToChildrenIndices, [&](auto Sort) { Sort(Level.ChildrenIndices); Sort(Level.ChildMasks); });

		Levels[LevelDepth] = Level;
	}
	check(Levels[0].Hashes.Num() == 1);

	FragmentIndicesToChildrenIndices.Free();
	Fragments.Free();

	uint64 Num = 0;
	for (auto& Level : Levels)
	{
		Level.ComputeChildPositions(TempStorage);
		Num += Level.NumWords;
	}
	Num += Leaves.Num() * 2;

	TStaticArray<uint32, EMemoryType::GPU> Dag("Dag", Num);
	Dag.MemSet(0xFF); // To spot errors
	uint64 Offset = 0;
	for (uint64 LevelIndex = 0; LevelIndex < Levels.size(); LevelIndex++)
	{
		const auto& Level = Levels[LevelIndex];
		const uint32 NextLevelStart = Cast<uint32>(Offset + Level.NumWords);
		if (LevelIndex == Levels.size() - 1)
		{
			const auto GetChildIndex = [=] GPU_LAMBDA (uint32 Index) { return NextLevelStart + 2 * Index; };
			Level.WriteTo(Dag.GetData() + Offset, GetChildIndex);

		}
		else
		{
			const auto& NextLevel = Levels[LevelIndex + 1];
			const auto GetChildIndex = [=] GPU_LAMBDA (uint32 Index) { return NextLevelStart + NextLevel.ChildPositions[Index]; };
			Level.WriteTo(Dag.GetData() + Offset, GetChildIndex);
		}
		Offset += Levels[LevelIndex].NumWords;
	}
	CUDA_CHECKED_CALL cudaMemcpy(Dag.GetData() + Offset, Leaves.GetData(), Leaves.Num() * sizeof(uint64), cudaMemcpyDeviceToDevice);

#if ENABLE_CHECKS && DEBUG_GPU_ARRAYS
	CheckDag(Dag, 0, 0);
#endif
	
	for (auto& Level : Levels)
	{
		Level.Free();
	}
	Leaves.Free();

	TempStorage.Free();
	
	std::cout << FMemory::GetStatsString();
	
	return Dag;
}
