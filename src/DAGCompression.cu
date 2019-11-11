#include "cub/iterator/transform_input_iterator.cuh"
#include "cub/iterator/counting_input_iterator.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_reduce.cuh"
#include "cub/device/device_scan.cuh"
#include "cub/device/device_select.cuh"
#include <thrust/adjacent_difference.h>
#include <thrust/sequence.h>
#undef check

#include "DAGCompression.h"
#include "CudaPerfRecorder.h"

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

template<typename F, typename... TArgs>
struct TLambdaWrapper
{
	F Lambda;

	__host__ __device__ __forceinline__	auto operator()(TArgs... Args) const
	{
		return Lambda(Args...);
	}
};

template<typename KeyOut, typename ValueOut, typename KeysIterator, typename ValuesIterator, typename F>
inline void ReduceByKey(
	TDynamicArray<uint8, EMemoryType::GPU>& TempStorage, 
	const KeysIterator& Keys, 
	const ValuesIterator& Values,
	const int32 NumInputs,
	TDynamicArray<KeyOut, EMemoryType::GPU>& UniqueKeysOut, 
	TDynamicArray<ValueOut, EMemoryType::GPU>& ReducedValuesOut,
	F Reduce)
{
	checkEqual(NumInputs, UniqueKeysOut.Num());
	checkEqual(NumInputs, ReducedValuesOut.Num());

	using KeyType = decltype(Keys[0]);
	
	SINGLE_GPU_ELEMENT(int32, NumRuns);

	TLambdaWrapper<F, const KeyType, const KeyType> Op{ Reduce };
	
	size_t TempStorageBytes;
	CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(nullptr, TempStorageBytes, Keys, UniqueKeysOut.GetData(), Values, ReducedValuesOut.GetData(), NumRuns.ToGPU(), Op, NumInputs);
	TempStorage.ResizeUninitialized(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, NumInputs);
		CUDA_CHECKED_CALL cub::DeviceReduce::ReduceByKey(TempStorage.GetData(), TempStorageBytes, Keys, UniqueKeysOut.GetData(), Values, ReducedValuesOut.GetData(), NumRuns.ToGPU(), Op, NumInputs);
	}

	const int32 NumRuns_CPU = NumRuns.ToCPU();
	checkInfEqual(NumRuns_CPU, NumInputs);
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
	CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(nullptr, TempStorageBytes, In.GetData(), Out.GetData(), In.Num());
	TempStorage.ResizeUninitialized(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
		CUDA_CHECKED_CALL cub::DeviceScan::InclusiveSum(TempStorage.GetData(), TempStorageBytes, In.GetData(), Out.GetData(), In.Num());
	}
}

template<typename InType, typename OutType, typename F>
inline void AdjacentDifference(
	const TStaticArray<InType, EMemoryType::GPU>& In,
	TStaticArray<OutType, EMemoryType::GPU>& Out,
	F BinaryOp)
{
	checkEqual(In.Num(), Out.Num())

	TLambdaWrapper<F, const InType, const InType> Op{ BinaryOp };

	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	thrust::adjacent_difference(
		In.GetData(),
		In.GetData() + In.Num(),
		Out.GetData(),
		Op);
}

template<typename T>
inline void Sequence(TStaticArray<T, EMemoryType::GPU>& Array)
{
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Array.Num());
	thrust::sequence(Array.GetData(), Array.GetData() + Array.Num());
}

template<typename T, typename F>
inline auto TransformIterator(const T* Ptr, F Lambda)
{
	using TReturnType = decltype(Lambda(T()));
	using FOp = TLambdaWrapper<F, const T>;
	return cub::TransformInputIterator<const T, FOp, const TReturnType*>(Ptr, FOp{ Lambda });
}

inline void LogRemoved(const char* Name, uint64 Before, uint64 After)
{
	checkInfEqual(After, Before);
	LOG("%s: %llu elements (%f%%) removed", Name, Before - After, double(Before - After) * 100. / Before);
}

TStaticArray<uint32, EMemoryType::GPU> DAGCompression::CreateDAG(TStaticArray<uint64, EMemoryType::GPU> Fragments, const int32 Depth)
{
	TDynamicArray<uint8, EMemoryType::GPU> TempStorage("TempStorage", 32);

	{
		TStaticArray<uint64, EMemoryType::GPU> SortedFragments("SortedFragments", Fragments.Num());
		SortKeys(TempStorage, Fragments, SortedFragments, 3 * Depth);

		TDynamicArray<uint64, EMemoryType::GPU> UniqueFragments("UniqueFragments", SortedFragments.Num());
		SelectUnique(TempStorage, SortedFragments, UniqueFragments);

		LogRemoved("Fragments", SortedFragments.Num(), UniqueFragments.Num());

		SortedFragments.Free();
		UniqueFragments.Shrink();
		Fragments = TStaticArray<uint64, EMemoryType::GPU>(UniqueFragments);
	}

	TDynamicArray<uint64, EMemoryType::GPU> Leaves("Leaves", Fragments.Num());
	{
		// Use the last 6 bits to create the leaves
		TDynamicArray<uint64, EMemoryType::GPU> Parents("Parents", Fragments.Num());
		
		const auto ParentsIterator = TransformIterator(Fragments.GetData(), [] GPU_LAMBDA(uint64 Code) { return Code >> 6; });
		const auto LeavesIterator = TransformIterator(Fragments.GetData(), [] GPU_LAMBDA(uint64 Code) { return Code & 0b111111; });

		ReduceByKey(TempStorage, ParentsIterator, LeavesIterator, Fragments.Num(), Parents, Leaves, [] GPU_LAMBDA(uint64 A, uint64 B) { return A | B; });
		LogRemoved("Parents", Fragments.Num(), Parents.Num());
		LOG("%llu leaves", Leaves.Num());
		Parents.Shrink();
		Leaves.Shrink();
		Fragments.Free();
		Fragments = TStaticArray<uint64, EMemoryType::GPU>(Parents);
	}

	std::cout << FMemory::GetStatsString();

	{
		TDynamicArray<uint64, EMemoryType::GPU> SortedLeaves("SortedLeaves", Leaves.Num());
		TDynamicArray<uint32, EMemoryType::GPU> LeavesToSortedLeaves("LeavesToSortedLeaves", Leaves.Num());
		TDynamicArray<uint32, EMemoryType::GPU> LeavesSequence("LeavesSequence", Leaves.Num());

		Sequence(LeavesSequence);
		SortPairs(TempStorage, Leaves, SortedLeaves, LeavesSequence, LeavesToSortedLeaves, 64);
		
		TDynamicArray<uint32, EMemoryType::GPU> UniqueLeavesFlags("UniqueLeavesFlags", Leaves.Num());
		AdjacentDifference(SortedLeaves, UniqueLeavesFlags, [] GPU_LAMBDA(uint64 A, uint64 B) { return A != B; });
		TDynamicArray<uint32, EMemoryType::GPU> UniqueLeavesSums("UniqueLeavesSums", Leaves.Num());
		InclusiveSum(TempStorage, UniqueLeavesFlags, UniqueLeavesSums);

		TDynamicArray<uint64, EMemoryType::GPU> UniqueLeaves("UniqueLeaves", Leaves.Num());
		SelectUnique(TempStorage, SortedLeaves, UniqueLeaves);
	}
		
	return {};
}
