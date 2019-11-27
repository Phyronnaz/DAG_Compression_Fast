#include "DAGCompression.h"
#include "CudaPerfRecorder.h"

#include <iomanip>
#include <atomic>
#include <queue>

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
#include "moderngpu/kernel_merge.hxx"
#pragma pop_macro("check")

#if DEBUG_THRUST
struct my_policy : thrust::system::cpp::execution_policy<my_policy> {};
static auto ExecutionPolicy = my_policy();
#else
static auto ExecutionPolicy = thrust::cuda::par.on(0);
#endif

class FPool : public std::enable_shared_from_this<FPool>
{
private:
	// Sizes of all the ptr managed by this pool
	std::unordered_map<void*, uint64> Sizes;

	// Allocated ptrs with the used size (< real size)
	std::unordered_map<void*, uint64> Allocations;
	// Free ptrs with real size
	std::unordered_map<void*, uint64> FreeBuffers;
	
public:
	uint64 NumFragments = 0;

	struct mgpu_context_t : mgpu::standard_context_t
	{
		FPool& Pool;

		mgpu_context_t(FPool& Pool)
			: mgpu::standard_context_t(false, 0)
			, Pool(Pool)
		{
		}

		void* alloc(size_t size, mgpu::memory_space_t space) override
		{
			check(space == mgpu::memory_space_device);

			return Pool.Malloc<uint8>(size);
		}
		void free(void* p, mgpu::memory_space_t space) override
		{
			check(space == mgpu::memory_space_device);

			Pool.Free(p);
		}
		void synchronize() override
		{
			CUDA_CHECK_ERROR();
		}
	};
	mgpu_context_t* const mgpu_context;

	static std::shared_ptr<FPool> Create()
	{
		PROFILE_FUNCTION();
		
		auto Pool = std::shared_ptr<FPool>(new FPool());
		FMemory::RegisterOutOfMemoryEvent([WeakPtr = Pool->weak_from_this()]()
		{
			const auto Pinned = WeakPtr.lock();
			if (Pinned) Pinned->PrintStats();
		});
		FMemory::RegisterOutOfMemoryFallback([WeakPtr = Pool->weak_from_this()](auto FreeFunction)
		{
			const auto Pinned = WeakPtr.lock();
			if (Pinned) Pinned->FreeUnused(FreeFunction);
		});
		return Pool;
	}
	~FPool()
	{
		PROFILE_FUNCTION();
		check(Sizes.empty());
		check(Allocations.empty());
		check(FreeBuffers.empty());
		delete mgpu_context;
	}

private:
	FPool()
		: mgpu_context([&]() { PROFILE_SCOPE("Create MGPU context"); return new mgpu_context_t(*this); }())
	{
	}
	
public:
	template<typename T>
	T* Malloc(uint64 Size)
	{
		PROFILE_FUNCTION();
		
		const uint64 SizeInBytes = Size * sizeof(T);
		uint64 BestMatchSize = 0;
		void* BestMatchPtr = nullptr;
		for (auto& It : FreeBuffers)
		{
			if (It.second >= SizeInBytes && (!BestMatchPtr || It.second < BestMatchSize))
			{
				BestMatchSize = It.second;
				BestMatchPtr = It.first;
			}
		}

		T* Result;
		if (BestMatchPtr)
		{
			FreeBuffers.erase(BestMatchPtr);
			Result = reinterpret_cast<T*>(BestMatchPtr);
		}
		else
		{
			Result = FMemory::Malloc<T>("Pool", SizeInBytes, EMemoryType::GPU);
			Sizes[Result] = SizeInBytes;
		}
		check(Allocations.find(Result) == Allocations.end());
		Allocations[Result] = Size;
		return Result;
	}
	template<typename T>
	void Free(T* Ptr)
	{
		PROFILE_FUNCTION();
		
		void* VoidPtr = reinterpret_cast<void*>(Ptr);
		check(VoidPtr);
		check(Sizes.find(VoidPtr) != Sizes.end());
		check(Allocations.find(VoidPtr) != Allocations.end());
		check(FreeBuffers.find(VoidPtr) == FreeBuffers.end());
		Allocations.erase(VoidPtr);
		FreeBuffers[VoidPtr] = Sizes[VoidPtr];
	}

	void RegisterPtrInternal(void* Ptr, uint64 Size)
	{
		Sizes[Ptr] = Size;
		FreeBuffers[Ptr] = Size;
	}
	void RemovePtrInternal(void* Ptr)
	{
		checkAlways(FreeBuffers.erase(Ptr) == 1);
		checkAlways(Sizes.erase(Ptr) == 1);
	}

	void FreeAll()
	{
		PROFILE_FUNCTION();
		
		check(FreeBuffers.size() == Sizes.size());
		check(Allocations.size() == 0);
		for (auto& It : Sizes)
		{
			FMemory::Free(It.first);
		}
		FreeBuffers.clear();
		Sizes.clear();
	}
	template<typename T>
	void FreeUnused(T FreeFunction)
	{
		PROFILE_FUNCTION();
		
		for (auto& It : FreeBuffers)
		{
			FreeFunction(It.first);
			Sizes.erase(It.first);
		}
		FreeBuffers.clear();
	}

	void PrintStats() const
	{
		std::cout << "############################################################" << std::endl;
		std::cout << "Pool Stats" << std::endl;
		std::cout << "############################################################" << std::endl;
		std::cout << std::endl;
		
		std::cout << "Allocated:" << std::endl;

		std::vector<std::pair<void*, uint64>> AllocationsList(Allocations.begin(), Allocations.end());
		std::sort(AllocationsList.begin(), AllocationsList.end(), [](auto& A, auto& B) { return A.second > B.second; });
		for (auto& It : AllocationsList)
		{
			const uint64 UsedSize = It.second;
			const uint64 RealSize = Sizes.at(It.first);
			std::cout << "\t" << "0x" << std::hex << It.first << "; " <<
				std::dec << std::setw(10) << UsedSize << " B; " <<
				std::dec << std::setw(10) << RealSize << " B; " <<
				std::setw(6) << std::fixed << std::setprecision(2) << 100 * float(UsedSize) / RealSize << "%" <<
				std::endl;
		}
		
		std::cout << "Free:" << std::endl;

		std::vector<std::pair<void*, uint64>> FreeList(FreeBuffers.begin(), FreeBuffers.end());
		std::sort(FreeList.begin(), FreeList.end(), [](auto& A, auto& B) { return A.second > B.second; });
		for (auto& It : FreeList)
		{
			std::cout << "\t" << "0x" << std::hex << It.first << "; " <<
				std::dec << std::setw(10) << It.second << " B" <<
				std::endl;
		}
		std::cout << std::endl;
		std::cout << "############################################################" << std::endl;
	}

public:
	template<typename T>
	TStaticArray<T, EMemoryType::GPU> CreateArray(uint64 Size)
	{
		return { Malloc<T>(Size), Size };
	}
	template<typename T>
	TDynamicArray<T, EMemoryType::GPU> CreateDynamicArray(uint64 Size)
	{
		return TDynamicArray<T, EMemoryType::GPU>(CreateArray<T>(Size));
	}
	template<typename T>
	TStaticArray<T, EMemoryType::GPU> FromCPU(const TStaticArray<T, EMemoryType::CPU>& Array)
	{
		auto NewArray = CreateArray<T>(Array.Num());
		Array.CopyToGPU(NewArray);
		return NewArray;
	}
	template<typename T>
	void FreeArray(TStaticArray<T, EMemoryType::GPU>& Array)
	{
		check(Array.IsValid());
		Free(Array.GetData());
		Array.Reset();
	}
	template<typename T>
	TStaticArray<T, EMemoryType::GPU> ShrinkArray(TDynamicArray<T, EMemoryType::GPU>& Array)
	{
		PROFILE_FUNCTION();
		
		if (Array.GetAllocatedSize() == Array.Num())
		{
			return TStaticArray<T, EMemoryType::GPU>(Array);
		}
		else
		{
			auto NewArray = CreateArray<T>(Array.Num());
			CUDA_CHECKED_CALL cudaMemcpyAsync(NewArray.GetData(), Array.GetData(), Array.Num() * sizeof(T), cudaMemcpyDeviceToDevice);
			CUDA_CHECKED_CALL cudaStreamSynchronize(0);
			FreeArray(Array);
			return NewArray;
		}
	}
};

template<typename T, typename F>
inline auto TransformIterator(const T* Ptr, F Lambda)
{
	using TReturnType = decltype(Lambda(T()));
	return cub::TransformInputIterator<const TReturnType, F, const T*>(Ptr, Lambda);
}

template<typename T>
inline void CheckArrayBounds(const TStaticArray<T, EMemoryType::GPU>& Array, T Min, T Max)
{
#if ENABLE_CHECKS
	const auto Check = [=] GPU_LAMBDA (T Value)
	{
		checkf(Min <= Value, "%" PRIi64 " <= %" PRIi64, int64(Min), int64(Value));
		checkf(Value <= Max, "%" PRIi64 " <= %" PRIi64, int64(Value), int64(Max));
	};
	thrust::for_each(ExecutionPolicy, Array.GetData(), Array.GetData() + Array.Num(), Check);
	CUDA_CHECK_ERROR();
#endif
}

template<typename TIn, typename TOut, typename F>
inline void CheckFunctionBounds(
	F Function, 
	TIn MinInput, 
	TIn MaxInput,
	TOut MinOutput, 
	TOut MaxOutput)
{
#if ENABLE_CHECKS
	const auto Check = [=] GPU_LAMBDA (TIn Value)
	{
		const TOut Output = Function(Value);
		checkf(MinOutput <= Output, "%" PRIi64 " <= %" PRIi64, int64(MinOutput), int64(Output));
		checkf(Output <= MaxOutput, "%" PRIi64 " <= %" PRIi64, int64(Output), int64(MaxOutput));
	};
	const auto It = thrust::make_counting_iterator<TIn>(MinInput);
	thrust::for_each(ExecutionPolicy, It, It + MaxInput - MinInput, Check);
	CUDA_CHECK_ERROR();
#endif
}

template<typename TIn, typename TOut, typename F, typename FCondition>
inline void CheckFunctionBoundsIf(
	F Function,
	FCondition Condition,
	TIn MinInput, 
	TIn MaxInput,
	TOut MinOutput, 
	TOut MaxOutput)
{
#if ENABLE_CHECKS
	const auto Check = [=] GPU_LAMBDA (TIn Value)
	{
		if (!Condition(Value)) return;
		const TOut Output = Function(Value);
		checkf(MinOutput <= Output, "%" PRIi64 " <= %" PRIi64, int64(MinOutput), int64(Output));
		checkf(Output <= MaxOutput, "%" PRIi64 " <= %" PRIi64, int64(Output), int64(MaxOutput));
	};
	const auto It = thrust::make_counting_iterator<TIn>(MinInput);
	thrust::for_each(ExecutionPolicy, It, It + MaxInput - MinInput, Check);
	CUDA_CHECK_ERROR();
#endif
}

template<typename T>
inline void SortKeys(
	FPool& Pool,
	const TStaticArray<T, EMemoryType::GPU>& Src, 
	TStaticArray<T, EMemoryType::GPU>& Dst, 
	int32 NumBits)
{
	PROFILE_FUNCTION();
	
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
	PROFILE_FUNCTION();
	
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

namespace CubImplPrivate
{
	// Cub one doesn't allow custom equality
	template<
		typename InputIteratorT,
		typename OutputIteratorT,
		typename NumSelectedIteratorT,
		typename EqualityOpT>
		CUB_RUNTIME_FUNCTION __forceinline__
		static cudaError_t Unique(
			void* d_temp_storage,
			size_t& temp_storage_bytes,
			InputIteratorT d_in,
			OutputIteratorT d_out,
			NumSelectedIteratorT d_num_selected_out,
			int num_items,
			EqualityOpT Equality,
			cudaStream_t stream = 0,
			bool debug_synchronous = false)
	{
		using namespace cub;

		typedef int                     OffsetT;
		typedef NullType* FlagIterator;
		typedef NullType                SelectOp;

		return DispatchSelectIf<InputIteratorT, FlagIterator, OutputIteratorT, NumSelectedIteratorT, SelectOp, EqualityOpT, OffsetT, false>::Dispatch(
			d_temp_storage,
			temp_storage_bytes,
			d_in,
			NULL,
			d_out,
			d_num_selected_out,
			SelectOp(),
			Equality,
			num_items,
			stream,
			debug_synchronous);
	}
}

template<typename T, typename TEquality>
inline void SelectUnique(
	FPool& Pool,
	const TStaticArray<T, EMemoryType::GPU>& Src, 
	TDynamicArray<T, EMemoryType::GPU>& Dst,
	TEquality EqualityOp)
{
	PROFILE_FUNCTION();
	
	checkEqual(Src.Num(), Dst.Num());

	SINGLE_GPU_ELEMENT(int32, NumSelected, Pool);

	size_t TempStorageBytes;
	CUDA_CHECKED_CALL CubImplPrivate::Unique(nullptr, TempStorageBytes, Src.GetData(), Dst.GetData(), NumSelected.ToGPU(), Cast<int32>(Src.Num()), EqualityOp);
	auto TempStorage = Pool.CreateArray<uint8>(TempStorageBytes);

	{
		FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Src.Num());
		CUDA_CHECKED_CALL CubImplPrivate::Unique(TempStorage.GetData(), TempStorageBytes, Src.GetData(), Dst.GetData(), NumSelected.ToGPU(), Cast<int32>(Src.Num()), EqualityOp);
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
	PROFILE_FUNCTION();
	
	checkEqual(Keys.Num(), Values.Num());
	// checkEqual(Keys.Num(), UniqueKeysOut.Num());
	// checkEqual(Keys.Num(), ReducedValuesOut.Num());

	using KeyType = decltype(Keys[0]);
	
	SINGLE_GPU_ELEMENT(int32, NumRuns, Pool);

	const auto KeysIt = TransformIterator(Keys.GetData(), KeysTransform);
	const auto ValuesIt = TransformIterator(Values.GetData(), ValuesTransform);

	size_t TempStorageBytes = 0;
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
	PROFILE_FUNCTION();
	
	checkEqual(In.Num(), Out.Num());

	size_t TempStorageBytes = 0;
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
	PROFILE_FUNCTION();
	
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
	PROFILE_FUNCTION();
	
	CUDA_CHECKED_CALL cudaMemcpyAsync(&Array[Index], &Value, sizeof(T), cudaMemcpyHostToDevice);
	CUDA_CHECKED_CALL cudaStreamSynchronize(0);
}
template<typename T>
inline T GetElement(const TStaticArray<T, EMemoryType::GPU>& Array, uint64 Index)
{
	PROFILE_FUNCTION();
	
	T Value;
	CUDA_CHECKED_CALL cudaMemcpyAsync(&Value, &Array[Index], sizeof(T), cudaMemcpyDeviceToHost);
	CUDA_CHECKED_CALL cudaStreamSynchronize(0);
	return Value;
}

template<typename InType, typename OutType, typename F1>
inline void AdjacentDifference(
	const TStaticArray<InType, EMemoryType::GPU>& In,
	TStaticArray<OutType, EMemoryType::GPU>& Out,
	F1 BinaryOp,
	OutType FirstElement)
{
	PROFILE_FUNCTION();
	
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
	PROFILE_FUNCTION();
	
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

template<typename TypeIn, typename TypeOut, typename MapFunction>
inline void ScatterPred(
	const TStaticArray<TypeIn, EMemoryType::GPU>& In,
	MapFunction Map,
	TStaticArray<TypeOut, EMemoryType::GPU>& Out)
{
	PROFILE_FUNCTION();

	CheckFunctionBounds<uint64, uint64>(Map, 0, In.Num() - 1, 0, Out.Num() - 1);
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	thrust::scatter(
		ExecutionPolicy,
		In.GetData(),
		In.GetData() + In.Num(),
		MapTransform,
		Out.GetData());
}

template<typename TypeIn, typename TypeOut, typename IndexType>
inline void Scatter(
	const TStaticArray<TypeIn, EMemoryType::GPU>& In,
	const TStaticArray<IndexType, EMemoryType::GPU>& Map,
	TStaticArray<TypeOut, EMemoryType::GPU>& Out)
{
	PROFILE_FUNCTION();

	CheckArrayBounds<IndexType>(Map, 0, IndexType(Out.Num()) - 1);
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	thrust::scatter(
		ExecutionPolicy,
		In.GetData(),
		In.GetData() + In.Num(),
		Map.GetData(),
		Out.GetData());
}

template<typename TypeIn, typename TypeOut, typename InF, typename IndexType>
inline void ScatterWithTransform(
	const TStaticArray<TypeIn, EMemoryType::GPU>& In, InF Transform,
	const TStaticArray<IndexType, EMemoryType::GPU>& Map,
	TStaticArray<TypeOut, EMemoryType::GPU>& Out)
{
	PROFILE_FUNCTION();

	CheckArrayBounds<IndexType>(Map, 0, IndexType(Out.Num()) - 1);
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);
	thrust::scatter(
		ExecutionPolicy,
		It,
		It + In.Num(),
		Map.GetData(),
		Out.GetData());
}

template<typename TypeIn, typename TypeOut, typename MapFunction, typename F>
inline void ScatterIf(
	const TStaticArray<TypeIn, EMemoryType::GPU>& In,
	MapFunction Map,
	TStaticArray<TypeOut, EMemoryType::GPU>& Out,
	F Condition)
{
	PROFILE_FUNCTION();

	CheckFunctionBoundsIf<uint64, uint64>(Map, Condition, 0, In.Num() - 1, 0, Out.Num() - 1);
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	thrust::scatter_if(
		ExecutionPolicy,
		In.GetData(),
		In.GetData() + In.Num(),
		MapTransform,
		thrust::make_counting_iterator<uint64>(0),
		Out.GetData(),
		Condition);
}

template<typename TypeIn, typename TypeOut, typename InF, typename MapFunction, typename F>
inline void ScatterIfWithTransform(
	const TStaticArray<TypeIn, EMemoryType::GPU>& In, InF Transform,
	MapFunction Map,
	TStaticArray<TypeOut, EMemoryType::GPU>& Out,
	F Condition)
{
	PROFILE_FUNCTION();

	CheckFunctionBoundsIf<uint64, uint64>(Map, Condition, 0, In.Num() - 1, 0, Out.Num() - 1);
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, In.Num());
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);
	thrust::scatter_if(
		ExecutionPolicy,
		It,
		It + In.Num(),
		MapTransform,
		thrust::make_counting_iterator<uint64>(0),
		Out.GetData(),
		Condition);
}

template<typename InType, typename OutType, typename F>
inline void Transform(
	const TStaticArray<InType, EMemoryType::GPU>& In,
	TStaticArray<OutType, EMemoryType::GPU>& Out,
	F Lambda)
{
	PROFILE_FUNCTION();
	
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
	PROFILE_FUNCTION();
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Num);
	thrust::transform_if(
		ExecutionPolicy,
		In,
		In + Num,
		Out,
		Lambda,
		Condition);
}

template<typename TValueIt, typename TKey, typename TValue, typename F>
inline void MergePairs(
	FPool& Pool,
	const TStaticArray<TKey, EMemoryType::GPU>& InKeysA,
	TValueIt InValuesA,
	const TStaticArray<TKey, EMemoryType::GPU>& InKeysB,
	TValueIt InValuesB,
	TStaticArray<TKey, EMemoryType::GPU>& OutKeys,
	TStaticArray<TValue, EMemoryType::GPU>& OutValues,
	F Comp)
{
	PROFILE_FUNCTION();

	checkEqual(OutKeys.Num(), OutValues.Num());
	checkEqual(InKeysA.Num() + InKeysB.Num(), OutValues.Num());

	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, OutKeys.Num());
	mgpu::merge(
		InKeysA.GetData(), InValuesA, Cast<int32>(InKeysA.Num()),
		InKeysB.GetData(), InValuesB, Cast<int32>(InKeysB.Num()),
		OutKeys.GetData(), OutValues.GetData(),
		Comp,
		*Pool.mgpu_context);
}

template<typename T>
inline void MakeSequence(TStaticArray<T, EMemoryType::GPU>& Array, T Init = 0)
{
	PROFILE_FUNCTION();
	
	FCudaScopePerfRecorder PerfRecorder(__FUNCTION__, Array.Num());
	thrust::sequence(ExecutionPolicy, Array.GetData(), Array.GetData() + Array.Num(), Init);
}

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

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct FGpuLevel
{
	TStaticArray<uint8, EMemoryType::GPU> ChildMasks;
	TStaticArray<FChildrenIndices, EMemoryType::GPU> ChildrenIndices;
	TStaticArray<uint64, EMemoryType::GPU> Hashes;
	TStaticArray<uint32, EMemoryType::GPU> ChildPositions;
	uint32 NumWords = 0;

	FCpuLevel ToCPU() const
	{
		PROFILE_FUNCTION();

		FCpuLevel CpuLevel;

		CpuLevel.ChildMasks = TStaticArray<uint8, EMemoryType::CPU>("ChildMasks", ChildMasks.Num());
		CpuLevel.ChildrenIndices = TStaticArray<FChildrenIndices, EMemoryType::CPU>("ChildrenIndices", ChildrenIndices.Num());
		CpuLevel.Hashes = TStaticArray<uint64, EMemoryType::CPU>("Hashes", Hashes.Num());

		ChildMasks.CopyToCPU(CpuLevel.ChildMasks);
		ChildrenIndices.CopyToCPU(CpuLevel.ChildrenIndices);
		Hashes.CopyToCPU(CpuLevel.Hashes);

		return CpuLevel;
	}
	void FromCPU(FPool& Pool, const FCpuLevel& CpuLevel, bool bLoadHashes = true)
	{
		PROFILE_FUNCTION();

		ChildMasks = Pool.CreateArray<uint8>(CpuLevel.ChildMasks.Num());
		ChildrenIndices = Pool.CreateArray<FChildrenIndices>(CpuLevel.ChildrenIndices.Num());
		if (bLoadHashes) Hashes = Pool.CreateArray<uint64>(CpuLevel.Hashes.Num());

		CpuLevel.ChildMasks.CopyToGPU(ChildMasks);
		CpuLevel.ChildrenIndices.CopyToGPU(ChildrenIndices);
		if (bLoadHashes) CpuLevel.Hashes.CopyToGPU(Hashes);
	}

	void Free(FPool& Pool, bool bFreeHashes = true)
	{
		PROFILE_FUNCTION();

		Pool.FreeArray(ChildMasks);
		Pool.FreeArray(ChildrenIndices);
		if (bFreeHashes) Pool.FreeArray(Hashes);
		if (ChildPositions.IsValid()) Pool.FreeArray(ChildPositions);
	}

	void ComputeHashes(FPool& Pool, const TStaticArray<uint64, EMemoryType::GPU>& LowerLevelHashes)
	{
		PROFILE_FUNCTION();

		Hashes = Pool.CreateArray<uint64>(ChildrenIndices.Num());
		Transform(ChildrenIndices, Hashes, [=] GPU_LAMBDA (const FChildrenIndices& Children) { return HashChildren(Children, LowerLevelHashes); });
	}
	void ComputeChildPositions(FPool& Pool)
	{
		PROFILE_FUNCTION();

		ChildPositions = Pool.CreateArray<uint32>(ChildMasks.Num());
		ExclusiveSumWithTransform(
			Pool,
			ChildMasks, [] GPU_LAMBDA(uint8 ChildMask) { return Utils::TotalSize(ChildMask); },
			ChildPositions);
		NumWords = GetElement(ChildPositions, ChildPositions.Num() - 1) + Utils::TotalSize(GetElement(ChildMasks, ChildMasks.Num() - 1));
	}
	template<typename T>
	void WriteTo(TStaticArray<uint32, EMemoryType::GPU>& Data, T GetChildIndex) const
	{
		PROFILE_FUNCTION();

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
	PROFILE_FUNCTION();

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
	Scatter(SortedHashesToUniqueHashes, SortedHashesToHashes, OutHashesToSortedUniqueHashes);
	Pool.FreeArray(SortedHashesToHashes);
	Pool.FreeArray(SortedHashesToUniqueHashes);
	
	const auto Sort = [&](auto& Array)
	{
		checkEqual(Array.Num(), OutHashesToSortedUniqueHashes.Num());
		using T = typename std::remove_reference<decltype(Array[0])>::type;
		auto FinalArray = Pool.CreateArray<T>(NumUniques);
		Scatter(Array, OutHashesToSortedUniqueHashes, FinalArray);
		Pool.FreeArray(Array);
		Array = FinalArray;
	};

	ApplySortToOtherArray(Sort);

	Sort(InOutHashes);
}

FGpuLevel ExtractLevel(
	FPool& Pool,
	TStaticArray<FMortonCode, EMemoryType::GPU>& InOutFragments,
	const TStaticArray<uint32, EMemoryType::GPU>& FragmentIndicesToChildrenIndices)
{
	PROFILE_FUNCTION();

	FGpuLevel OutLevel;
	
	const auto Fragments = InOutFragments;

	const auto ParentsTransform = [] GPU_LAMBDA (FMortonCode Code) { return Code >> 3; };
	const auto ChildIndicesTransform = [] GPU_LAMBDA (FMortonCode Code) { return uint64(1) << (Code & 0b111); };

	auto Parents = Pool.CreateDynamicArray<FMortonCode>(Fragments.Num());
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
		auto Out = OutLevel.ChildrenIndices.CastTo<uint32>();
		ScatterPred(FragmentIndicesToChildrenIndices, CompactChildrenToExpandedChildren, Out);

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
		CUDA_CHECKED_CALL cudaStreamSynchronize(0);
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
			CUDA_CHECKED_CALL cudaStreamSynchronize(0);
		}
	}
}

inline FPool& GetGlobalPool()
{
	static std::shared_ptr<FPool> Pool;
	if (!Pool) Pool = FPool::Create();
	return *Pool;
}

FCpuDag DAGCompression::CreateSubDAG(const TStaticArray<FMortonCode, EMemoryType::GPU>& InFragments)
{
	PROFILE_FUNCTION();

#if 0
	{
		const uint32 Size = 1 << 30;
		uint32* PtrACPU;
		uint32* PtrAGPU;
		uint32* PtrBCPU;
		uint32* PtrBGPU;
		CUDA_CHECKED_CALL cudaMallocHost(&PtrACPU, Size);
		CUDA_CHECKED_CALL cudaMallocHost(&PtrBCPU, Size);
		CUDA_CHECKED_CALL cudaMalloc(&PtrAGPU, Size);
		CUDA_CHECKED_CALL cudaMalloc(&PtrBGPU, Size);

		auto ThreadA = std::thread([=]()
		{
			CUDA_CHECKED_CALL cudaMemcpyAsync(PtrAGPU, PtrACPU, Size, cudaMemcpyHostToDevice);
			CUDA_CHECKED_CALL cudaStreamSynchronize(0);
		});
		auto ThreadB = std::thread([=]()
		{
			CUDA_CHECKED_CALL cudaMemcpyAsync(PtrBGPU, PtrBCPU, Size, cudaMemcpyHostToDevice);
			CUDA_CHECKED_CALL cudaStreamSynchronize(0);
		});

		ThreadA.join();
		ThreadB.join();
		
		CUDA_CHECKED_CALL cudaFreeHost(PtrACPU);
		CUDA_CHECKED_CALL cudaFreeHost(PtrBCPU);
		CUDA_CHECKED_CALL cudaFree(PtrAGPU);
		CUDA_CHECKED_CALL cudaFree(PtrBGPU);
	}
#endif
	
	const uint32 Depth = SUBDAG_LEVELS;
	
	FCpuDag CpuDag;
	CpuDag.Levels.resize(Depth - 2);

	if (InFragments.Num() == 0)
	{
		return CpuDag;
	}
	
	FPool& Pool = GetGlobalPool();

	if (Pool.NumFragments < InFragments.Num())
	{
		Pool.FreeAll();
		Pool.NumFragments = InFragments.Num();
	}

	TStaticArray<FMortonCode, EMemoryType::GPU> Fragments;
	{
		PROFILE_SCOPE("Sort & Clean fragments");
		
		auto SortedFragments = Pool.CreateArray<FMortonCode>(InFragments.Num());
		SortKeys(Pool, InFragments, SortedFragments, 3 * Depth);

#if !DEBUG_GPU_ARRAYS
		{
			void* Ptr = reinterpret_cast<void*>(const_cast<FMortonCode*>(InFragments.GetData()));
			Pool.RegisterPtrInternal(Ptr, FMemory::GetAllocInfo(Ptr).Size);
		}
#endif
		
		auto UniqueFragments = Pool.CreateDynamicArray<FMortonCode>(SortedFragments.Num());
		SelectUnique(Pool, SortedFragments, UniqueFragments, [] GPU_LAMBDA (uint64 A, uint64 B) { return (A << (64 - 3 * SUBDAG_LEVELS)) == (B << (64 - 3 * SUBDAG_LEVELS)); });
		
		LogRemoved("Fragments", SortedFragments.Num(), UniqueFragments.Num());
		
		Pool.FreeArray(SortedFragments);
		Fragments = Pool.ShrinkArray(UniqueFragments);
	}

#if ENABLE_COLORS
	std::thread CopyColorsThread;
	{
		PROFILE_SCOPE("Extract colors");

		auto Colors = TStaticArray<uint32, EMemoryType::GPU>("Colors", Fragments.Num());
		Transform(Fragments, Colors, [] GPU_LAMBDA(uint64 X) { return X >> 40; });
		Transform(Fragments, Fragments, [] GPU_LAMBDA(uint64 X) { return X & ((uint64(1) << 40) - 1); });

		CopyColorsThread = std::thread([&CpuDag, Colors]()
		{
			NAME_THREAD("CopyColorsThread");
			PROFILE_SCOPE("Copy colors");

			CpuDag.Colors = { "Colors", Colors.Num() };
			const uint32 BlockSize = 1 << 20;
			for (uint64 Index = 0; Index < Colors.Num() / BlockSize; ++Index)
			{
				const uint64 Offset = Index * BlockSize;
				CUDA_CHECKED_CALL cudaMemcpyAsync(CpuDag.Colors.GetData() + Offset, Colors.GetData() + Offset, BlockSize * sizeof(uint32), cudaMemcpyDeviceToHost);
				CUDA_CHECKED_CALL cudaStreamSynchronize(0);
			}
			const uint64 Offset = BlockSize * (Colors.Num() / BlockSize);
			const uint64 Size = Colors.Num() - Offset;
			if (Size > 0)
			{
				CUDA_CHECKED_CALL cudaMemcpyAsync(CpuDag.Colors.GetData() + Offset, Colors.GetData() + Offset, Size * sizeof(uint32), cudaMemcpyDeviceToHost);
				CUDA_CHECKED_CALL cudaStreamSynchronize(0);
			}

			const_cast<TStaticArray<uint32, EMemoryType::GPU>&>(Colors).Free();
		});
	}
#endif

	TStaticArray<uint64, EMemoryType::GPU> Leaves;
	{
		PROFILE_SCOPE("Extract Leaves")
		
		// Use the last 6 bits to create the leaves
		const auto ParentsTransform = [] GPU_LAMBDA (FMortonCode Code) { return Code >> 6; };
		const auto LeavesTransform = [] GPU_LAMBDA (FMortonCode Code) { return uint64(1) << (Code & 0b111111); };

		auto LocalLeaves = Pool.CreateDynamicArray<uint64>(Fragments.Num() / 2); // Should not have the / 2, but that way we can save quite a lot of memory
		auto Parents = Pool.CreateDynamicArray<FMortonCode>(Fragments.Num() / 2);

		ReduceByKey(
			Pool,
			Fragments, ParentsTransform,
			Fragments, LeavesTransform,
			Parents,
			LocalLeaves,
			thrust::bit_or<uint64>());

		LogRemoved("Parents", Fragments.Num(), Parents.Num());
		LOG_DEBUG("%llu leaves", LocalLeaves.Num());
		
		Leaves = Pool.ShrinkArray(LocalLeaves);
		
		Pool.FreeArray(Fragments);
		Fragments = Pool.ShrinkArray(Parents);
	}

	TStaticArray<uint32, EMemoryType::GPU> FragmentIndicesToChildrenIndices;
	SortLevel(Pool, Leaves, FragmentIndicesToChildrenIndices, [&](auto) {});

	CpuDag.Leaves = TStaticArray<uint64, EMemoryType::CPU>("Leaves", Leaves.Num());
	Leaves.CopyToCPU(CpuDag.Leaves);
	
	auto PreviousLevelHashes = Leaves;
	for (int32 LevelDepth = Depth - 3; LevelDepth >= 0; LevelDepth--)
	{
		PROFILE_SCOPE("Level %d", LevelDepth);
		LOG_DEBUG("Level %d", LevelDepth);
		
		FGpuLevel Level = ExtractLevel(Pool, Fragments, FragmentIndicesToChildrenIndices);
		Pool.FreeArray(FragmentIndicesToChildrenIndices);

		Level.ComputeHashes(Pool, PreviousLevelHashes);
		Pool.FreeArray(PreviousLevelHashes);

		SortLevel(Pool, Level.Hashes, FragmentIndicesToChildrenIndices, [&](auto Sort) { Sort(Level.ChildrenIndices); Sort(Level.ChildMasks); });

		CpuDag.Levels[LevelDepth] = Level.ToCPU();
		Level.Free(Pool, false);
		PreviousLevelHashes = Level.Hashes;
	}
	check(PreviousLevelHashes.Num() == 1);
	Pool.FreeArray(PreviousLevelHashes);

	Pool.FreeArray(FragmentIndicesToChildrenIndices);
	Pool.FreeArray(Fragments);

#if !DEBUG_GPU_ARRAYS
	{
		void* Ptr = reinterpret_cast<void*>(const_cast<FMortonCode*>(InFragments.GetData()));
		Pool.RemovePtrInternal(Ptr);
	}
#endif

#if ENABLE_COLORS
	{
		PROFILE_SCOPE("Wait for copy colors");
		CopyColorsThread.join();
	}
#endif
	
	return CpuDag;
}

uint32 Merge(
	FPool& Pool,
	const TStaticArray<uint64, EMemoryType::GPU>& KeysA,
	const TStaticArray<uint64, EMemoryType::GPU>& KeysB,
	TStaticArray<uint32, EMemoryType::GPU>& OutIndicesAToMergedIndices,
	TStaticArray<uint32, EMemoryType::GPU>& OutIndicesBToMergedIndices)
{
	PROFILE_FUNCTION();

	check(!OutIndicesAToMergedIndices.IsValid());
	check(!OutIndicesBToMergedIndices.IsValid());
	
	auto KeysAB = Pool.CreateArray<uint64>(KeysA.Num() + KeysB.Num());
	auto Permutation = Pool.CreateArray<uint32>(KeysA.Num() + KeysB.Num());
	{
		MergePairs(
			Pool,
			KeysA,
			thrust::make_counting_iterator<uint32>(0u),
			KeysB,
			thrust::make_counting_iterator<uint32>(1u << 31),
			KeysAB,
			Permutation,
			thrust::less<uint64>());
	}

	auto UniqueSum = Pool.CreateArray<uint32>(KeysA.Num() + KeysB.Num());
	{
		auto UniqueFlags = Pool.CreateArray<uint32>(KeysA.Num() + KeysB.Num());
		AdjacentDifference(KeysAB, UniqueFlags, thrust::not_equal_to<uint64>(), 0u);
		InclusiveSum(Pool, UniqueFlags, UniqueSum);
		Pool.FreeArray(UniqueFlags);
	}
	Pool.FreeArray(KeysAB);

	const uint32 NumUniques = GetElement(UniqueSum, UniqueSum.Num() - 1) + 1;

	OutIndicesAToMergedIndices = Pool.CreateArray<uint32>(KeysA.Num());
	OutIndicesBToMergedIndices = Pool.CreateArray<uint32>(KeysB.Num());
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

	Pool.FreeArray(UniqueSum);
	Pool.FreeArray(Permutation);

	return NumUniques;
}

inline FCpuDag MergeDAGs(
	FPool& Pool,
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
		auto LeavesA = Pool.FromCPU(A.Leaves);
		auto LeavesB = Pool.FromCPU(B.Leaves);
		A.Leaves.Free();
		B.Leaves.Free();

		const uint32 NumMergedLeaves = Merge(
			Pool,
			LeavesA,
			LeavesB,
			IndicesAToMergedIndices,
			IndicesBToMergedIndices);

		auto MergedLeaves = Pool.CreateArray<uint64>(NumMergedLeaves);
		Scatter(LeavesA, IndicesAToMergedIndices, MergedLeaves);
		Scatter(LeavesB, IndicesBToMergedIndices, MergedLeaves);

		Pool.FreeArray(LeavesA);
		Pool.FreeArray(LeavesB);

		MergedCpuDag.Leaves = MergedLeaves.CreateCPU();
		Pool.FreeArray(MergedLeaves);
	}

	auto PreviousIndicesAToMergedIndices = IndicesAToMergedIndices;
	auto PreviousIndicesBToMergedIndices = IndicesBToMergedIndices;
	IndicesAToMergedIndices.Reset();
	IndicesBToMergedIndices.Reset();

	for (int32 LevelIndex = NumLevels - 1; LevelIndex >= 0; LevelIndex--)
	{
		auto& LevelA = A.Levels[LevelIndex];
		auto& LevelB = B.Levels[LevelIndex];

		auto HashesA = Pool.FromCPU(LevelA.Hashes);
		auto HashesB = Pool.FromCPU(LevelB.Hashes);
		LevelA.Hashes.Free();
		LevelB.Hashes.Free();
		
		const uint32 NumMerged = Merge(
			Pool,
			HashesA,
			HashesB,
			IndicesAToMergedIndices,
			IndicesBToMergedIndices);

		FCpuLevel MergedLevel;

		{
			auto MergedHashes = Pool.CreateArray<uint64>(NumMerged);

			Scatter(HashesA, IndicesAToMergedIndices, MergedHashes);
			Scatter(HashesB, IndicesBToMergedIndices, MergedHashes);

			Pool.FreeArray(HashesA);
			Pool.FreeArray(HashesB);

			MergedLevel.Hashes = MergedHashes.CreateCPU();
			Pool.FreeArray(MergedHashes);
		}

		{
			auto MergedChildMasks = Pool.CreateArray<uint8>(NumMerged);
			
			auto ChildMasksA = Pool.FromCPU(LevelA.ChildMasks);
			auto ChildMasksB = Pool.FromCPU(LevelB.ChildMasks);
			LevelA.ChildMasks.Free();
			LevelB.ChildMasks.Free();

			Scatter(ChildMasksA, IndicesAToMergedIndices, MergedChildMasks);
			Scatter(ChildMasksB, IndicesBToMergedIndices, MergedChildMasks);

			Pool.FreeArray(ChildMasksA);
			Pool.FreeArray(ChildMasksB);

			MergedLevel.ChildMasks = MergedChildMasks.CreateCPU();
			Pool.FreeArray(MergedChildMasks);
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

			auto MergedChildrenIndices = Pool.CreateArray<FChildrenIndices>(NumMerged);

			auto ChildrenIndicesA = Pool.FromCPU(LevelA.ChildrenIndices);
			auto ChildrenIndicesB = Pool.FromCPU(LevelB.ChildrenIndices);
			LevelA.ChildrenIndices.Free();
			LevelB.ChildrenIndices.Free();

			ScatterWithTransform(ChildrenIndicesA, TransformA, IndicesAToMergedIndices, MergedChildrenIndices);
			ScatterWithTransform(ChildrenIndicesB, TransformB, IndicesBToMergedIndices, MergedChildrenIndices);
			
			Pool.FreeArray(ChildrenIndicesA);
			Pool.FreeArray(ChildrenIndicesB);

			MergedLevel.ChildrenIndices = MergedChildrenIndices.CreateCPU();
			Pool.FreeArray(MergedChildrenIndices);
		}

		MergedCpuDag.Levels[LevelIndex] = MergedLevel;

		Pool.FreeArray(PreviousIndicesAToMergedIndices);
		Pool.FreeArray(PreviousIndicesBToMergedIndices);
		PreviousIndicesAToMergedIndices = IndicesAToMergedIndices;
		PreviousIndicesBToMergedIndices = IndicesBToMergedIndices;
		IndicesAToMergedIndices.Reset();
		IndicesBToMergedIndices.Reset();
	}

	Pool.FreeArray(PreviousIndicesAToMergedIndices);
	Pool.FreeArray(PreviousIndicesBToMergedIndices);

	return MergedCpuDag;
}

template<class T>
class TSafeQueue
{
public:
	TSafeQueue() = default;
	~TSafeQueue() = default;

	void Enqueue(T Element)
	{
		std::lock_guard<std::mutex> Lock(Mutex);
		Queue.push(Element);
	}
	bool Dequeue(T& Element)
	{
		std::unique_lock<std::mutex> Lock(Mutex);
		if (Queue.empty()) return false;
		Element = Queue.front();
		Queue.pop();
		return true;
	}
	
private:
	std::queue<T> Queue;
	mutable std::mutex Mutex;
};

class FThreadPool
{
public:
	using FTask = std::function<void(FPool&)>;
	
private:
	std::vector<std::thread> Threads;
	TSafeQueue<FTask> Tasks;
	std::atomic<uint32> Stop{ 0 };

	std::mutex OnEnqueueMutex;
	std::condition_variable OnEnqueue;
	
	std::atomic<uint32> WaitingThreads{ 0 };
	std::mutex OnAllWaitingMutex;
	std::condition_variable OnAllWaiting;
	
public:
	explicit FThreadPool(int32 NumThreads)
	{
		PROFILE_FUNCTION();
		
		Threads.reserve(NumThreads);
		for (int32 Index = 0; Index < NumThreads; ++Index)
		{
			Threads.emplace_back([this]() { RunThread(); });
		}
	}
	~FThreadPool()
	{
		check(Stop);
	}

	void WaitForCompletion()
	{
		PROFILE_FUNCTION();
		
		if (Threads.empty())
		{
			FTask Task;
			while (Tasks.Dequeue(Task))
			{
				Task(GetGlobalPool());
			}
		}
		else
		{
			std::unique_lock<std::mutex> Lock(OnAllWaitingMutex);
			OnAllWaiting.wait(Lock);
		}
	}
	void Enqueue(FTask Task)
	{
		PROFILE_FUNCTION();

		Tasks.Enqueue(std::move(Task));
		OnEnqueue.notify_all();
	}
	void Destroy()
	{
		PROFILE_FUNCTION();

		Stop = 1;
		OnEnqueue.notify_all();
		for (auto& Thread : Threads)
		{
			Thread.join();
		}
	}

private:
	void RunThread()
	{
		PROFILE_FUNCTION();
		NAME_THREAD("ThreadPool");
		
		auto Pool = FPool::Create();
		while (!Stop)
		{
			FTask Task;
			if (Tasks.Dequeue(Task))
			{
				Task(*Pool);
			}
			else
			{
				std::unique_lock<std::mutex> Lock(OnEnqueueMutex);
				if (1 + WaitingThreads.fetch_add(1) == Threads.size())
				{
					OnAllWaiting.notify_all();
				}
				OnEnqueue.wait(Lock);
				WaitingThreads--;
			}
		}
		Pool->FreeAll();
	}
};

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
		for (uint32 Index = 0; Index < ColorsToMerge.size(); ++Index)
		{
			NAME_THREAD("MergeColorThreadPool");
			Threads.emplace_back([Index, &ColorsToMerge, &Offsets, &MergedColors]()
			{
				std::memcpy(&MergedColors[Offsets[Index]], ColorsToMerge[Index].GetData(), ColorsToMerge[Index].SizeInBytes());
				ColorsToMerge[Index].Free();
			});
		}
		for(auto& Thread : Threads)
		{
			Thread.join();
		}
	});
#endif

	GetGlobalPool().FreeAll();
	
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
			ThreadPool.Enqueue([&, SubDagIndex](FPool& Pool) 
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
					MergedDag = MergeDAGs(Pool, std::move(MergedDag), std::move(GetDag(Index)));
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

FFinalDag DAGCompression::CreateFinalDAG(FCpuDag&& CpuDag)
{
	PROFILE_FUNCTION();
	
	FPool& Pool = GetGlobalPool();
	
	std::vector<FGpuLevel> Levels;
	for (auto& Level : CpuDag.Levels)
	{
		Levels.push_back({});
		Levels.back().FromCPU(Pool, Level, false);
		Level.Free();
	}
	auto Leaves = Pool.FromCPU(CpuDag.Leaves);
	CpuDag.Leaves.Free();
	
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
		checkInfEqual(Offset, Num);
		auto DagWithOffset = TStaticArray<uint32, EMemoryType::GPU>(Dag.GetData() + Offset, Num - Offset);
		if (LevelIndex == Levels.size() - 1)
		{
			const auto GetChildIndex = [=] GPU_LAMBDA (uint32 Index) { return NextLevelStart + 2 * Index; };
			Level.WriteTo(DagWithOffset, GetChildIndex);
		}
		else
		{
			const auto NextLevelChildPositions = Levels[LevelIndex + 1].ChildPositions;
			const auto GetChildIndex = [=] GPU_LAMBDA (uint32 Index) { return NextLevelStart + NextLevelChildPositions[Index]; };
			Level.WriteTo(DagWithOffset, GetChildIndex);
		}
		Offset += Levels[LevelIndex].NumWords;
	}
	checkEqual(Offset + 2 * Leaves.Num(), Num);
	CUDA_CHECKED_CALL cudaMemcpyAsync(Dag.GetData() + Offset, Leaves.GetData(), Leaves.Num() * sizeof(uint64), cudaMemcpyDeviceToDevice);
	CUDA_CHECKED_CALL cudaStreamSynchronize(0);
		
	for (auto& Level : Levels)
	{
		Level.Free(Pool, false);
	}
	Pool.FreeArray(Leaves);

	Pool.FreeAll();

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

void DAGCompression::FreeAll()
{
	GetGlobalPool().FreeAll();
}

void DAGCompression::Test()
{
	TStaticArray<uint64, EMemoryType::GPU> Array("Name", 100000000);
	Array.MemSet(255);

	const auto Lambda = [=](TStaticArray<uint64, EMemoryType::GPU> Array2)
	{
		PROFILE_FUNCTION();
		Transform(Array, Array2, [] GPU_LAMBDA (uint64 Value) { return Value - 1; });
		auto CpuArray = Array2.CreateCPU();
		for (uint64 Value : CpuArray)
		{
			check(Value == uint64(-2));
		}
	};
	TStaticArray<uint64, EMemoryType::GPU> ArrayA("Name", Array.Num());
	TStaticArray<uint64, EMemoryType::GPU> ArrayB("Name", Array.Num());
	TStaticArray<uint64, EMemoryType::GPU> ArrayC("Name", Array.Num());
	TStaticArray<uint64, EMemoryType::GPU> ArrayD("Name", Array.Num());
	
	const auto Lambda2 = [=](TStaticArray<uint64, EMemoryType::GPU> Array2) { return [=]() { Lambda(Array2); }; };
	std::thread thread0(Lambda2(ArrayA));
	std::thread thread1(Lambda2(ArrayB));
	std::thread thread2(Lambda2(ArrayC));
	std::thread thread3(Lambda2(ArrayD));
	thread0.join();
	thread1.join();
	thread2.join();
	thread3.join();
	while (true);
}
