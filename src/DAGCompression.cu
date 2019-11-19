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
#include <thrust/system/cuda/execution_policy.h>
#pragma pop_macro("check")

#if DEBUG_GPU_ARRAYS
struct my_policy : thrust::system::cpp::execution_policy<my_policy> {};
static auto ExecutionPolicy = my_policy();
#else
static auto ExecutionPolicy = thrust::cuda::par.on(0);
#endif

struct FPool
{
	struct FAlloc
	{
		void* Ptr = nullptr;
		uint64 Size = 0;
	};
	
	std::vector<FAlloc> Allocations;
	std::unordered_map<void*, uint64> Sizes;
	std::unordered_map<void*, bool> IsOpenGLBuffer;

	template<typename T>
	T* Allocate(uint64 Size)
	{
		const uint64 SizeInBytes = Size * sizeof(T);
		FAlloc BestMatch;
		uint64 BestMatchIndex = 0;
		for (uint64 Index = 0; Index < Allocations.size(); Index++)
		{
			auto& Alloc = Allocations[Index];
			if (Alloc.Size >= SizeInBytes && (!BestMatch.Ptr || Alloc.Size < BestMatch.Size))
			{
				BestMatch = Alloc;
				BestMatchIndex = Index;
			}
		}

		if (BestMatch.Ptr)
		{
			Allocations.erase(Allocations.begin() + BestMatchIndex);
			return reinterpret_cast<T*>(BestMatch.Ptr);
		}
		else
		{
			auto* Ptr = FMemory::Malloc<T>("Pool", SizeInBytes, EMemoryType::GPU);
			Sizes[Ptr] = SizeInBytes;
			return Ptr;
		}
	}
	template<typename T>
	void Free(T* Ptr)
	{
		void* VoidPtr = reinterpret_cast<void*>(Ptr);
		check(VoidPtr);
		check(Sizes.find(VoidPtr) != Sizes.end());
		Allocations.push_back({ VoidPtr, Sizes[VoidPtr] });
	}

	template<typename T>
	TStaticArray<T, EMemoryType::GPU> CreateArray(uint64 Size)
	{
		return { Allocate<T>(Size), Size };
	}
	template<typename T>
	TDynamicArray<T, EMemoryType::GPU> CreateDynamicArray(uint64 Size)
	{
		return TDynamicArray<T, EMemoryType::GPU>(CreateArray<T>(Size));
	}
	template<typename T>
	void FreeArray(TStaticArray<T, EMemoryType::GPU>& Array)
	{
		Free(Array.GetData());
		Array.Reset();
	}
	template<typename T>
	TStaticArray<T, EMemoryType::GPU> ShrinkArray(TDynamicArray<T, EMemoryType::GPU>& Array)
	{
		if (Array.GetAllocatedSize() == Array.Num())
		{
			return TStaticArray<T, EMemoryType::GPU>(Array);
		}
		else
		{
			auto NewArray = CreateArray<T>(Array.Num());
			CUDA_CHECKED_CALL cudaMemcpy(NewArray.GetData(), Array.GetData(), Array.Num() * sizeof(T), cudaMemcpyDeviceToDevice);
			FreeArray(Array);
			return NewArray;
		}
	}

	void FreeAll()
	{
		check(Allocations.size() == Sizes.size());
		for (auto& It : Sizes)
		{
			if (!IsOpenGLBuffer[It.first])
			{
				FMemory::Free(It.first);
			}
		}
		Allocations.clear();
		Sizes.clear();
	}
};

template<typename T, typename F>
inline auto TransformIterator(const T* Ptr, F Lambda)
{
	using TReturnType = decltype(Lambda(T()));
	return cub::TransformInputIterator<const T, F, const TReturnType*>(Ptr, Lambda);
}

template<typename T>
inline void SortKeys(
	FPool& Pool,
	const TStaticArray<T, EMemoryType::GPU>& Src, 
	TStaticArray<T, EMemoryType::GPU>& Dst, 
	int32 NumBits)
{
	ZoneScoped;
	
	checkEqual(Src.Num(), Dst.Num());

	size_t TempStorageBytes;
	CUDA_CHECKED_CALL cub::DeviceRadixSort::SortKeys(nullptr, TempStorageBytes, Src.GetData(), Dst.GetData(), Cast<int32>(Src.Num()), 0, NumBits);
	auto TempStorage = Pool.CreateArray<uint8>(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Src.Num());
		CUDA_CHECKED_CALL cub::DeviceRadixSort::SortKeys(TempStorage.GetData(), TempStorageBytes, Src.GetData(), Dst.GetData(), Cast<int32>(Src.Num()), 0, NumBits);
	}

	Pool.FreeArray(TempStorage);
}

template<typename KeyType, typename ValueType>
inline void SortPairs(
	FPool& Pool,
	const TStaticArray<KeyType, EMemoryType::GPU>& KeysIn,
	TStaticArray<KeyType, EMemoryType::GPU>& KeysOut,
	const TStaticArray<ValueType, EMemoryType::GPU>& ValuesIn,
	TStaticArray<ValueType, EMemoryType::GPU>& ValuesOut,
	int32 NumBits)
{
	ZoneScoped;
	
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
	auto TempStorage = Pool.CreateArray<uint8>(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, KeysIn.Num());
		CUDA_CHECKED_CALL cub::DeviceRadixSort::SortPairs(
			TempStorage.GetData(), TempStorageBytes,
			KeysIn.GetData(), KeysOut.GetData(),
			ValuesIn.GetData(), ValuesOut.GetData(),
			Cast<int32>(KeysIn.Num()),
			0, NumBits);
	}

	Pool.FreeArray(TempStorage);
}

template<typename T>
inline void SelectUnique(
	FPool& Pool,
	const TStaticArray<T, EMemoryType::GPU>& Src, 
	TDynamicArray<T, EMemoryType::GPU>& Dst)
{
	ZoneScoped;
	
	checkEqual(Src.Num(), Dst.Num());

	SINGLE_GPU_ELEMENT(int32, NumSelected);

	size_t TempStorageBytes;
	CUDA_CHECKED_CALL cub::DeviceSelect::Unique(nullptr, TempStorageBytes, Src.GetData(), Dst.GetData(), NumSelected.ToGPU(), Cast<int32>(Src.Num()));
	auto TempStorage = Pool.CreateArray<uint8>(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Src.Num());
		CUDA_CHECKED_CALL cub::DeviceSelect::Unique(TempStorage.GetData(), TempStorageBytes, Src.GetData(), Dst.GetData(), NumSelected.ToGPU(), Cast<int32>(Src.Num()));
	}

	Pool.FreeArray(TempStorage);

	const int32 NumSelected_CPU = NumSelected.ToCPU();
	checkInfEqual(NumSelected_CPU, Dst.Num());
	Dst.Resize(NumSelected_CPU);
}

template<typename KeyIn, typename KeyOut, typename ValueIn, typename ValueOut, typename F1, typename F2, typename F3>
inline void ReduceByKey(
	FPool& Pool,
	const TStaticArray<KeyIn, EMemoryType::GPU>& Keys, F1 KeysTransform, 
	const TStaticArray<ValueIn, EMemoryType::GPU>& Values, F2 ValuesTransform,
	TDynamicArray<KeyOut, EMemoryType::GPU>& UniqueKeysOut, 
	TDynamicArray<ValueOut, EMemoryType::GPU>& ReducedValuesOut,
	F3 Reduce)
{
	ZoneScoped;
	
	checkEqual(Keys.Num(), Values.Num());
	// checkEqual(Keys.Num(), UniqueKeysOut.Num());
	// checkEqual(Keys.Num(), ReducedValuesOut.Num());

	using KeyType = decltype(Keys[0]);
	
	SINGLE_GPU_ELEMENT(int32, NumRuns);

	const auto KeysIt = TransformIterator(Keys.GetData(), KeysTransform);
	const auto ValuesIt = TransformIterator(Values.GetData(), ValuesTransform);

	size_t TempStorageBytes;
	CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(nullptr, TempStorageBytes, KeysIt, UniqueKeysOut.GetData(), ValuesIt, ReducedValuesOut.GetData(), NumRuns.ToGPU(), Reduce, Cast<int32>(Keys.Num()));
	auto TempStorage = Pool.CreateArray<uint8>(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Keys.Num());
		CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(TempStorage.GetData(), TempStorageBytes, KeysIt, UniqueKeysOut.GetData(), ValuesIt, ReducedValuesOut.GetData(), NumRuns.ToGPU(), Reduce, Cast<int32>(Keys.Num()));
	}

	Pool.FreeArray(TempStorage);

	const int32 NumRuns_CPU = NumRuns.ToCPU();
	checkInfEqual(NumRuns_CPU, Keys.Num());
	checkInfEqual(NumRuns_CPU, UniqueKeysOut.Num());
	checkInfEqual(NumRuns_CPU, ReducedValuesOut.Num());
	UniqueKeysOut.Resize(NumRuns_CPU);
	ReducedValuesOut.Resize(NumRuns_CPU);
}

template<typename InType, typename OutType>
inline void InclusiveSum(
	FPool& Pool,
	const TStaticArray<InType, EMemoryType::GPU>& In, 
	TStaticArray<OutType, EMemoryType::GPU>& Out)
{
	ZoneScoped;
	
	checkEqual(In.Num(), Out.Num());

	size_t TempStorageBytes;
	CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(nullptr, TempStorageBytes, In.GetData(), Out.GetData(), Cast<int32>(In.Num()));
	auto TempStorage = Pool.CreateArray<uint8>(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
		CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(TempStorage.GetData(), TempStorageBytes, In.GetData(), Out.GetData(), Cast<int32>(In.Num()));
	}

	Pool.FreeArray(TempStorage);
}

template<typename InType, typename OutType, typename F>
inline void ExclusiveSumWithTransform(
	FPool& Pool,
	const TStaticArray<InType, EMemoryType::GPU>& In, F Transform, 
	TStaticArray<OutType, EMemoryType::GPU>& Out)
{
	ZoneScoped;
	
	checkEqual(In.Num(), Out.Num());
	
	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);

	size_t TempStorageBytes;
	CUDA_CHECKED_CALL cub::DeviceScan::ExclusiveSum(nullptr, TempStorageBytes, It, Out.GetData(), Cast<int32>(In.Num()));
	auto TempStorage = Pool.CreateArray<uint8>(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
		CUDA_CHECKED_CALL cub::DeviceScan::ExclusiveSum(TempStorage.GetData(), TempStorageBytes, It, Out.GetData(), Cast<int32>(In.Num()));
	}

	Pool.FreeArray(TempStorage);
}

template<typename T>
inline void SetElement(TStaticArray<T, EMemoryType::GPU>& Array, uint64 Index, T Value)
{
	ZoneScoped;
	
	CUDA_CHECKED_CALL cudaMemcpy(&Array[Index], &Value, sizeof(T), cudaMemcpyHostToDevice);
}
template<typename T>
inline T GetElement(const TStaticArray<T, EMemoryType::GPU>& Array, uint64 Index)
{
	ZoneScoped;
	
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
	ZoneScoped;
	
	checkEqual(In.Num(), Out.Num())

	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	thrust::transform(
		ExecutionPolicy,
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
	ZoneScoped;
	
	checkEqual(In.Num(), Out.Num())

	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);
	thrust::transform(
		ExecutionPolicy,
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
	ZoneScoped;
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	thrust::scatter(
		ExecutionPolicy,
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
	ZoneScoped;
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	thrust::scatter(
		ExecutionPolicy,
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
	ZoneScoped;
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	thrust::scatter_if(
		ExecutionPolicy,
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
	ZoneScoped;
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);
	thrust::scatter_if(
		ExecutionPolicy,
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
	ZoneScoped;
	
	checkEqual(In.Num(), Out.Num())
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	thrust::transform(
		ExecutionPolicy,
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
	ZoneScoped;
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Num);
	thrust::transform_if(
		ExecutionPolicy,
		In,
		In + Num,
		Out,
		Lambda,
		Condition);
}

template<typename T>
inline void MakeSequence(TStaticArray<T, EMemoryType::GPU>& Array)
{
	ZoneScoped;
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Array.Num());
	thrust::sequence(ExecutionPolicy, Array.GetData(), Array.GetData() + Array.Num());
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

	void Free(FPool& Pool)
	{
		ZoneScoped;

		Pool.FreeArray(ChildMasks);
		Pool.FreeArray(ChildrenIndices);
		Pool.FreeArray(Hashes);
		Pool.FreeArray(ChildPositions);
	}

	void ComputeHashes(FPool& Pool, const TStaticArray<uint64, EMemoryType::GPU>& LowerLevelHashes)
	{
		ZoneScoped;

		Hashes = Pool.CreateArray<uint64>(ChildrenIndices.Num());
		Transform(ChildrenIndices, Hashes, [=] GPU_LAMBDA (const FChildrenIndices& Children) { return HashChildren(Children, LowerLevelHashes); });
	}
	void ComputeChildPositions(FPool& Pool)
	{
		ZoneScoped;

		ChildPositions = Pool.CreateArray<uint32>(ChildMasks.Num());
		ExclusiveSumWithTransform(
			Pool,
			ChildMasks, [] GPU_LAMBDA(uint8 ChildMask) { return Utils::TotalSize(ChildMask); },
			ChildPositions);
		NumWords = GetElement(ChildPositions, ChildPositions.Num() - 1) + Utils::TotalSize(GetElement(ChildMasks, ChildMasks.Num() - 1));
	}
	template<typename T>
	void WriteTo(uint32* Data, T GetChildIndex) const
	{
		ZoneScoped;

		Scatter(ChildMasks, ChildPositions, Data);
		
		const auto ChildrenIndicesArray = ChildrenIndices.CastTo<uint32>();
		const auto ChildPositionsC = ChildPositions; // Need to create copies else the GPU lambda is keeping a ref to the object instead of the arrays
		const auto ChildMasksC = ChildMasks;

		ScatterIfWithTransform(
			ChildrenIndicesArray, [=] GPU_LAMBDA (uint32 Index) { return Index == 0xFFFFFFFF ? Index : GetChildIndex(Index); },
			[=] GPU_LAMBDA (uint64 Index) { return ChildPositionsC[Index / 8] + Utils::ChildOffset(ChildMasksC[Index / 8], Index % 8); },
			Data,
			[=] GPU_LAMBDA(uint64 Index) { check((ChildrenIndicesArray[Index] == 0xFFFFFFFF) == !(ChildMasksC[Index / 8] & (1 << (Index % 8)))); return ChildrenIndicesArray[Index] != 0xFFFFFFFF; });
	}
};

template<typename F>
void SortLevel(
	FPool& Pool,
	TStaticArray<uint64, EMemoryType::GPU>& InOutHashes,
	TStaticArray<uint32, EMemoryType::GPU>& OutHashesToSortedUniqueHashes,
	F ApplySortToOtherArray)
{
	ZoneScoped;

	const auto Hashes = InOutHashes;

	auto SortedHashes = Pool.CreateArray<uint64>(Hashes.Num());
	auto SortedHashesToHashes = Pool.CreateArray<uint32>(Hashes.Num());

	{
		// Need a real array to use CUB sort
		auto Sequence = Pool.CreateArray<uint32>(Hashes.Num());
		MakeSequence(Sequence);
		SortPairs(Pool, Hashes, SortedHashes, Sequence, SortedHashesToHashes, 64);
		Pool.FreeArray(Sequence);
	}

	auto SortedHashesFlags = Pool.CreateArray<uint32>(Hashes.Num());
	AdjacentDifference(SortedHashes, SortedHashesFlags, thrust::not_equal_to<uint64>(), 0u);
	Pool.FreeArray(SortedHashes);
	
	auto SortedHashesToUniqueHashes = Pool.CreateArray<uint32>(Hashes.Num());
	InclusiveSum(Pool, SortedHashesFlags, SortedHashesToUniqueHashes);
	Pool.FreeArray(SortedHashesFlags);
	
	const uint32 NumUniques = GetElement(SortedHashesToUniqueHashes, SortedHashesToUniqueHashes.Num() - 1) + 1;

	OutHashesToSortedUniqueHashes = Pool.CreateArray<uint32>(Hashes.Num());
	Scatter(SortedHashesToUniqueHashes, SortedHashesToHashes, OutHashesToSortedUniqueHashes.GetData());
	Pool.FreeArray(SortedHashesToHashes);
	Pool.FreeArray(SortedHashesToUniqueHashes);
	
	const auto Sort = [&](auto& Array)
	{
		checkEqual(Array.Num(), OutHashesToSortedUniqueHashes.Num());
		using T = typename std::remove_reference<decltype(Array[0])>::type;
		auto FinalArray = Pool.CreateArray<T>(NumUniques);
		Scatter(Array, OutHashesToSortedUniqueHashes, FinalArray.GetData());
		Pool.FreeArray(Array);
		Array = FinalArray;
	};

	ApplySortToOtherArray(Sort);

	Sort(InOutHashes);
}

FLevel ExtractLevel(
	FPool& Pool,
	TStaticArray<uint64, EMemoryType::GPU>& InOutFragments,
	const TStaticArray<uint32, EMemoryType::GPU>& FragmentIndicesToChildrenIndices)
{
	ZoneScoped;

	FLevel OutLevel;
	
	const auto Fragments = InOutFragments;

	const auto ParentsTransform = [] GPU_LAMBDA (uint64 Code) { return Code >> 3; };
	const auto ChildIndicesTransform = [] GPU_LAMBDA (uint64 Code) { return uint64(1) << (Code & 0b111); };

	auto Parents = Pool.CreateDynamicArray<uint64>(Fragments.Num());
	{
		auto ChildMasks = Pool.CreateDynamicArray<uint8>(Fragments.Num());
		ReduceByKey(
			Pool,
			Fragments, ParentsTransform,
			Fragments, ChildIndicesTransform,
			Parents,
			ChildMasks,
			thrust::bit_or<uint8>());
		OutLevel.ChildMasks = Pool.ShrinkArray(ChildMasks);
	}

	{
		auto UniqueParentsSum = Pool.CreateArray<uint32>(Fragments.Num());
		{
			auto UniqueParentsFlags = Pool.CreateArray<uint32>(Fragments.Num());
			AdjacentDifferenceWithTransform(Fragments, ParentsTransform, UniqueParentsFlags, thrust::not_equal_to<uint64>(), 0u);
			InclusiveSum(Pool, UniqueParentsFlags, UniqueParentsSum);
			Pool.FreeArray(UniqueParentsFlags);
		}

		OutLevel.ChildrenIndices = Pool.CreateArray<FChildrenIndices>(Parents.Num());
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

		Pool.FreeArray(UniqueParentsSum);
	}
	
	Pool.FreeArray(InOutFragments);
	InOutFragments = Pool.ShrinkArray(Parents);

	return OutLevel;
}

void CheckDag(const TStaticArray<uint32, EMemoryType::GPU>& Dag, uint32 Index, uint32 Level)
{
	ZoneScoped;

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
	ZoneScoped;
	
	FPool Pool;

	TStaticArray<uint64, EMemoryType::GPU> Fragments;
	{
		auto SortedFragments = Pool.CreateArray<uint64>(InFragments.Num());
		SortKeys(Pool, InFragments, SortedFragments, 3 * Depth);

		{
			auto* Ptr = reinterpret_cast<void*>(const_cast<uint64*>(InFragments.GetData()));
			Pool.Sizes[Ptr] = FMemory::GetAllocInfo(Ptr).Size;
			Pool.Free(Ptr);
			Pool.IsOpenGLBuffer[Ptr] = true;
		}
			
		auto UniqueFragments = Pool.CreateDynamicArray<uint64>(InFragments.Num());
		SelectUnique(Pool, SortedFragments, UniqueFragments);

		LogRemoved("Fragments", SortedFragments.Num(), UniqueFragments.Num());

		Pool.FreeArray(SortedFragments);
		Fragments = Pool.ShrinkArray(UniqueFragments);
	}

	TStaticArray<uint64, EMemoryType::GPU> Leaves;
	{
		// Use the last 6 bits to create the leaves
		const auto ParentsTransform = [] GPU_LAMBDA (uint64 Code) { return Code >> 6; };
		const auto LeavesTransform = [] GPU_LAMBDA (uint64 Code) { return uint64(1) << (Code & 0b111111); };

		auto LocalLeaves = Pool.CreateDynamicArray<uint64>(Fragments.Num() / 2); // Should not have the / 2, but that way we can save quite a lot of memory
		auto Parents = Pool.CreateDynamicArray<uint64>(Fragments.Num() / 2);

		ReduceByKey(
			Pool,
			Fragments, ParentsTransform,
			Fragments, LeavesTransform,
			Parents,
			LocalLeaves,
			thrust::bit_or<uint64>());
		LogRemoved("Parents", Fragments.Num(), Parents.Num());
		LOG("%llu leaves", LocalLeaves.Num());
		
		Leaves = Pool.ShrinkArray(LocalLeaves);
		
		Pool.FreeArray(Fragments);
		Fragments = Pool.ShrinkArray(Parents);
	}

	std::vector<FLevel> Levels(Depth - 2);

	TStaticArray<uint32, EMemoryType::GPU> FragmentIndicesToChildrenIndices;
	SortLevel(Pool, Leaves, FragmentIndicesToChildrenIndices, [&](auto) {});

	{
		FLevel Level = ExtractLevel(Pool, Fragments, FragmentIndicesToChildrenIndices);
		Pool.FreeArray(FragmentIndicesToChildrenIndices);
		
		Level.ComputeHashes(Pool, Leaves);

		SortLevel(Pool, Level.Hashes, FragmentIndicesToChildrenIndices, [&](auto Sort) { Sort(Level.ChildrenIndices); Sort(Level.ChildMasks); });

		Levels[Depth - 3] = Level;
	}

	for (int32 LevelDepth = Depth - 4; LevelDepth >= 0; LevelDepth--)
	{
		LOG("Level %d", LevelDepth);
		
		FLevel Level = ExtractLevel(Pool, Fragments, FragmentIndicesToChildrenIndices);
		Pool.FreeArray(FragmentIndicesToChildrenIndices);
		
		Level.ComputeHashes(Pool, Levels.back().Hashes);

		SortLevel(Pool, Level.Hashes, FragmentIndicesToChildrenIndices, [&](auto Sort) { Sort(Level.ChildrenIndices); Sort(Level.ChildMasks); });

		Levels[LevelDepth] = Level;
	}
	check(Levels[0].Hashes.Num() == 1);

	Pool.FreeArray(FragmentIndicesToChildrenIndices);
	Pool.FreeArray(Fragments);

	uint64 Num = 0;
	for (auto& Level : Levels)
	{
		Level.ComputeChildPositions(Pool);
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
			const auto NextLevelChildPositions = Levels[LevelIndex + 1].ChildPositions;
			const auto GetChildIndex = [=] GPU_LAMBDA (uint32 Index) { return NextLevelStart + NextLevelChildPositions[Index]; };
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
		Level.Free(Pool);
	}
	Pool.FreeArray(Leaves);

	Pool.FreeAll();
	
	std::cout << FMemory::GetStatsString();
	
	return Dag;
}