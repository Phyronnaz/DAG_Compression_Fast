#pragma once

#include "Core.h"
#include "Utils.h"
#include "device_atomic_functions.h"

#pragma push_macro("check")
#undef check
#include <thrust/adjacent_difference.h>
#include <thrust/sequence.h>
#include <thrust/scatter.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#pragma pop_macro("check")

#include "cub/iterator/transform_input_iterator.cuh"
#include "cub/iterator/counting_input_iterator.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_reduce.cuh"
#include "cub/device/device_scan.cuh"
#include "cub/device/device_select.cuh"

#include "moderngpu/kernel_merge.hxx"

#ifdef TRACY_ENABLE
struct FScopeBandwidthReporter
{
	explicit FScopeBandwidthReporter(uint64 Num, tracy::ScopedZone& Zone)
		: Num(Num)
		, StartTime(Utils::Seconds())
		, Zone(Zone)
	{
	}
	~FScopeBandwidthReporter()
	{
		CUDA_SYNCHRONIZE_STREAM();
		CUDA_CHECK_LAST_ERROR();
		char String[1024];
		const int32 Size = sprintf(String, "%" PRIu64 " elements | %f G/s", Num, Num / (Utils::Seconds() - StartTime) / 1000000000.);
		checkInfEqual(Size, 1024);
		Zone.Text(String, Size);
	}

	const uint64 Num;
	const double StartTime;
	tracy::ScopedZone& Zone;
};
#define SCOPED_BANDWIDTH_TRACY(Num) FScopeBandwidthReporter __ScopeBandwidthReporter(Num, ___tracy_scoped_zone);
#else
#define SCOPED_BANDWIDTH_TRACY(Num)
#endif

template<EMemoryType MemoryType>
auto GetExecutionPolicy();

struct my_policy : thrust::system::cpp::execution_policy<my_policy> {};

template<>
inline auto GetExecutionPolicy<EMemoryType::CPU>()
{
	static auto ExecutionPolicy = my_policy();
	return ExecutionPolicy;
}

template<>
inline auto GetExecutionPolicy<EMemoryType::GPU>()
{
#if DEBUG_THRUST
	return GetExecutionPolicy<EMemoryType::CPU>();
#else
	thread_local const auto ExecutionPolicy = thrust::cuda::par.on(DEFAULT_STREAM);
	return ExecutionPolicy;
#endif
}

template<typename T, typename F, EMemoryType MemoryType>
inline void Foreach(
	const TFixedArray<T, MemoryType>& In,
	F Lambda)
{
	CUDA_CHECK_LAST_ERROR();
	SCOPED_BANDWIDTH_TRACY(In.Num());
	
	thrust::for_each(
		GetExecutionPolicy<EMemoryType::GPU>(),
		In.GetData(),
		In.GetData() + In.Num(),
		Lambda);
}

template<typename T, typename F>
inline auto TransformIterator(const T* Ptr, F Lambda)
{
	using TReturnType = decltype(Lambda(T()));
	return cub::TransformInputIterator<const TReturnType, F, const T*>(Ptr, Lambda);
}

template<typename T, EMemoryType MemoryType>
inline void CheckEqual(const TFixedArray<T, MemoryType>& ArrayA, const TFixedArray<T, MemoryType>& ArrayB)
{
	(void)ArrayA;
	(void)ArrayB;
#if ENABLE_CHECKS
	CUDA_CHECK_LAST_ERROR();
	checkEqual(ArrayA.Num(), ArrayB.Num());
	const auto Check = [=] GPU_LAMBDA (uint64 Index)
	{
		checkf(ArrayA[Index] == ArrayB[Index], "%" PRIu64 " == %" PRIu64 " (Index %" PRIu64 ")", uint64(ArrayA[Index]), uint64(ArrayB[Index]), Index);
	};
	const auto It = thrust::make_counting_iterator<uint64>(0);
	thrust::for_each(GetExecutionPolicy<MemoryType>(), It, It + ArrayA.Num(), Check);
	CUDA_CHECK_LAST_ERROR();
#endif
}

template<typename T, EMemoryType MemoryType>
inline void CheckIsSorted(const TFixedArray<T, MemoryType>& Array)
{
	(void)Array;
#if ENABLE_CHECKS
	CUDA_CHECK_LAST_ERROR();
	const auto Check = [=] GPU_LAMBDA (uint64 Index)
	{
		checkf(Array[Index] <= Array[Index + 1], "%" PRIx64 " <= %" PRIx64 "; Index %" PRIu64, uint64(Array[Index]), uint64(Array[Index + 1]), Index);
	};
	const auto It = thrust::make_counting_iterator<uint64>(0);
	thrust::for_each(GetExecutionPolicy<MemoryType>(), It, It + Array.Num() - 1, Check);
	CUDA_CHECK_LAST_ERROR();
#endif
}

template<EMemoryType MemoryType, typename TIn, typename TOut, typename F, typename FCondition>
inline void CheckFunctionBoundsIf(
	F Function,
	FCondition Condition,
	TIn MinInput, 
	TIn MaxInput,
	TOut MinOutput, 
	TOut MaxOutput)
{
	(void)Function;
	(void)Condition;
	(void)MinInput;
	(void)MaxInput;
	(void)MinOutput;
	(void)MaxOutput;
#if ENABLE_CHECKS
	CUDA_CHECK_LAST_ERROR();
	const auto Check = [=] GPU_LAMBDA (TIn Value)
	{
		if (!Condition(Value)) return;
		const TOut Output = Function(Value);
		checkf(MinOutput <= Output, "%" PRIi64 " <= %" PRIi64, int64(MinOutput), int64(Output));
		checkf(Output <= MaxOutput, "%" PRIi64 " <= %" PRIi64, int64(Output), int64(MaxOutput));
	};
	const auto It = thrust::make_counting_iterator<TIn>(MinInput);
	thrust::for_each(GetExecutionPolicy<MemoryType>(), It, It + MaxInput - MinInput, Check);
	CUDA_CHECK_LAST_ERROR();
#endif
}

template<EMemoryType MemoryType, typename TIn, typename TOut, typename F>
inline void CheckFunctionBounds(
	F Function, 
	TIn MinInput, 
	TIn MaxInput,
	TOut MinOutput, 
	TOut MaxOutput)
{
	CheckFunctionBoundsIf<MemoryType>(Function, [] GPU_LAMBDA (TIn) { return true; }, MinInput, MaxInput, MinOutput, MaxOutput);
}

template<EMemoryType MemoryType, typename T>
inline void CheckArrayBounds(const TFixedArray<T, MemoryType>& Array, T Min, T Max)
{
	CheckFunctionBounds<MemoryType>(Array, uint64(0), Array.Num() - 1, Min, Max);
}

template<EMemoryType MemoryType, typename TDataPred, typename TIn, typename F, typename FCondition>
inline void CheckFunctionIsInjectiveIf(
	TDataPred Data,
	F Function,
	FCondition Condition,
	TIn MinInput,
	TIn MaxInput,
	uint64 NumOutputs)
{
	(void)Data;
	(void)Function;
	(void)Condition;
	(void)MinInput;
	(void)MaxInput;
	(void)NumOutputs;
#if ENABLE_CHECKS
	CUDA_CHECK_LAST_ERROR();
#if 0 // Strict injectivity is probably not needed: we just need to write the same values
	auto Counters = TGpuArray<uint64>("Counters", NumOutputs);
	Counters.MemSet(0);

	const auto Add = [=] GPU_ONLY_LAMBDA (TIn Value)
	{
		if (!Condition(Value)) return;
		const uint64 Output = Function(Value);
		atomicAdd(const_cast<uint64*>(&Counters[Output]), uint64(1));
	};
	const auto It = thrust::make_counting_iterator<TIn>(MinInput);
	thrust::for_each(GetExecutionPolicy<MemoryType>(), It, It + MaxInput - MinInput, Add);
	CheckArrayBounds(Counters, uint64(0), uint64(1));
	CUDA_CHECK_LAST_ERROR();

	Counters.Free();
#else
	using TData = typename std::remove_reference<decltype(Data({}))>::type;
	auto DataTestArray = TFixedArray<TData, MemoryType>("DataTest", NumOutputs);
	auto PreviousIndicesArray = TFixedArray<uint64, MemoryType>("PreviousIndicesArray", NumOutputs);
	DataTestArray.MemSet(0);
	PreviousIndicesArray.MemSet(0xFF);

	if (MemoryType == EMemoryType::GPU)
	{
		const auto Check = [=] GPU_ONLY_LAMBDA (TIn Input)
		{
			if (!Condition(Input)) return;
			const uint64 Output = Function(Input);
			checkInf(Output, NumOutputs);
			const auto DataValue = Data(Input);
			auto& DataTest = const_cast<TData&>(DataTestArray[Output]);

			const uint32* RESTRICT const DataValueWords = reinterpret_cast<const uint32*>(&DataValue);
			uint32* RESTRICT const DataTestWords = reinterpret_cast<uint32*>(&DataTest);

			static_assert(sizeof(Data) % 4 == 0, "");
			for (int32 WordIndex = 0; WordIndex < sizeof(TData) / 4; WordIndex++)
			{
				const uint32 ExistingWord = atomicExch(&DataTestWords[WordIndex], DataValueWords[WordIndex]);
				const uint64 PreviousIndex = atomicExch(
                        const_cast<unsigned long long int*>(reinterpret_cast<const unsigned long long int*>(&PreviousIndicesArray[Output])), uint64(Input));
				if (ExistingWord == 0) continue;
				if (ExistingWord == DataValueWords[WordIndex]) continue;
				printf("Different words: existing 0x%08x, writing 0x%08x. Index: %9" PRIu64 ". Mapping: %9" PRIu64 ". Word Index: %4u. Previous Write Index: %" PRIu64 "\n",
					ExistingWord,
					DataValueWords[WordIndex],
					uint64(Input),
					Output,
					WordIndex,
					PreviousIndex);
				DEBUG_BREAK();
			}
		};

		const auto It = thrust::make_counting_iterator<TIn>(MinInput);
		thrust::for_each(GetExecutionPolicy<EMemoryType::GPU>(), It, It + MaxInput - MinInput, Check);
		CUDA_CHECK_LAST_ERROR();
	}
	else
	{
		for (TIn Input = MinInput; Input <= MaxInput; Input++)
		{
			if (!Condition(Input)) continue;
			const uint64 Output = Function(Input);
			checkInf(Output, NumOutputs);
			const auto DataValue = Data(Input);
			auto& DataTest = const_cast<TData&>(DataTestArray[Output]);

			const uint32* RESTRICT const DataValueWords = reinterpret_cast<const uint32*>(&DataValue);
			uint32* RESTRICT const DataTestWords = reinterpret_cast<uint32*>(&DataTest);

			static_assert(sizeof(Data) % 4 == 0, "");
			for (int32 WordIndex = 0; WordIndex < sizeof(TData) / 4; WordIndex++)
			{
				const uint32 ExistingWord = DataTestWords[WordIndex];
				DataTestWords[WordIndex] = DataValueWords[WordIndex];
				const uint64 PreviousIndex = PreviousIndicesArray[Output];
				PreviousIndicesArray[Output] = uint64(Input);
				if (ExistingWord == 0) continue;
				if (ExistingWord == DataValueWords[WordIndex]) continue;
				printf("Different words: existing 0x%08x, writing 0x%08x. Index: %9" PRIu64 ". Mapping: %9" PRIu64 ". Word Index: %4u. Previous Write Index: %" PRIu64 "\n",
					ExistingWord,
					DataValueWords[WordIndex],
					uint64(Input),
					Output,
					WordIndex,
					PreviousIndex);
				DEBUG_BREAK();
			}
		}
	}

	DataTestArray.Free();
	PreviousIndicesArray.Free();
#endif
#endif
}

template<EMemoryType MemoryType, typename TData, typename TIn, typename F>
inline void CheckFunctionIsInjective(
	TData Data, 
	F Function, 
	TIn MinInput, 
	TIn MaxInput,
	uint64 NumOutputs)
{
	CheckFunctionIsInjectiveIf<MemoryType>(Data, Function, [] GPU_LAMBDA (TIn) { return true; }, MinInput, MaxInput, NumOutputs);
}

template<EMemoryType MemoryType, typename TData, typename TB>
inline void CheckArrayIsInjective(
	TData Data, 
	const TFixedArray<TB, MemoryType>& Array, 
	uint64 NumOutputs)
{
	CheckFunctionIsInjective<MemoryType>(Data, Array, uint64(0), Array.Num() - 1, NumOutputs);
}

namespace cub
{
	struct DeviceSelect2
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
			CUDA_CHECK_LAST_ERROR();
			PROFILE_FUNCTION_TRACY();
			SCOPED_BANDWIDTH_TRACY(num_items);

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
	};
}

template<typename T>
inline void SetElement(TGpuArray<T>& Array, uint64 Index, T Value)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	
	FMemory::SetGPUValue(&Array[Index], Value);
}
template<typename T>
inline T GetElement(const TGpuArray<T>& Array, uint64 Index)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	
	return FMemory::ReadGPUValue(&Array[Index]);
}

template<typename InType, typename OutType, typename F1>
inline void AdjacentDifference(
	const TGpuArray<InType>& In,
	TGpuArray<OutType>& Out,
	F1 BinaryOp,
	OutType FirstElement)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(In.Num());
	
	checkEqual(In.Num(), Out.Num())

	thrust::transform(
		GetExecutionPolicy<EMemoryType::GPU>(),
		In.GetData() + 1,
		In.GetData() + In.Num(),
		In.GetData(),
		Out.GetData() + 1,
		BinaryOp);
	
	SetElement(Out, 0, FirstElement);
	
	CUDA_CHECK_LAST_ERROR();
}

template<typename InType, typename OutType, typename F1, typename F2>
inline void AdjacentDifferenceWithTransform(
	const TGpuArray<InType>& In, F1 Transform,
	TGpuArray<OutType>& Out,
	F2 BinaryOp,
	OutType FirstElement)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(In.Num());
	
	checkEqual(In.Num(), Out.Num())

	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);
	thrust::transform(
		GetExecutionPolicy<EMemoryType::GPU>(),
		It + 1,
		It + In.Num(),
		It,
		Out.GetData() + 1,
		BinaryOp);
	
	SetElement(Out, 0, FirstElement);
	
	CUDA_CHECK_LAST_ERROR();
}

template<typename TypeIn, typename TypeOut, typename MapFunction, EMemoryType MemoryType>
inline void ScatterPred(
	const TFixedArray<TypeIn, MemoryType>& In,
	MapFunction Map,
	TFixedArray<TypeOut, MemoryType>& Out)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(In.Num());
	
	CheckFunctionBounds<MemoryType>(Map, uint64(0), In.Num() - 1, uint64(0), Out.Num() - 1);
	CheckFunctionIsInjective<MemoryType>(In, Map, uint64(0), In.Num() - 1, Out.Num());
	
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	thrust::scatter(
		GetExecutionPolicy<MemoryType>(),
		In.GetData(),
		In.GetData() + In.Num(),
		MapTransform,
		Out.GetData());
	
	CUDA_CHECK_LAST_ERROR();
}

template<typename TypeIn, typename TypeOut, typename IndexType, EMemoryType MemoryType>
inline void Scatter(
	const TFixedArray<TypeIn, MemoryType>& In,
	const TFixedArray<IndexType, MemoryType>& Map,
	TFixedArray<TypeOut, MemoryType>& Out)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(In.Num());
	
	CheckArrayBounds<MemoryType>(Map, IndexType(0), IndexType(Out.Num() - 1));
	CheckArrayIsInjective<MemoryType>(In, Map, Out.Num());
	
	thrust::scatter(
		GetExecutionPolicy<MemoryType>(),
		In.GetData(),
		In.GetData() + In.Num(),
		Map.GetData(),
		Out.GetData());
	
	CUDA_CHECK_LAST_ERROR();
}

template<typename TypeIn, typename TypeOut, typename InF, typename IndexType, EMemoryType MemoryType>
inline void ScatterWithTransform(
	const TFixedArray<TypeIn, MemoryType>& In, InF Transform,
	const TFixedArray<IndexType, MemoryType>& Map,
	TFixedArray<TypeOut, MemoryType>& Out)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(In.Num());
	
	CheckArrayBounds<MemoryType, IndexType>(Map, 0, IndexType(Out.Num()) - 1);
	
	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);
	
	CheckArrayIsInjective<MemoryType>([=] GPU_LAMBDA (uint64 Index) { return It[Index]; }, Map, Out.Num());
	
	thrust::scatter(
		GetExecutionPolicy<MemoryType>(),
		It,
		It + In.Num(),
		Map.GetData(),
		Out.GetData());
	
	CUDA_CHECK_LAST_ERROR();
}

template<typename TypeIn, typename TypeOut, typename MapFunction, typename F, EMemoryType MemoryType>
inline void ScatterIf(
	const TFixedArray<TypeIn, MemoryType>& In,
	MapFunction Map,
	TFixedArray<TypeOut, MemoryType>& Out,
	F Condition)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(In.Num());
	
	CheckFunctionBoundsIf<MemoryType>(Map, Condition, uint64(0), In.Num() - 1, uint64(0), Out.Num() - 1);
	CheckFunctionIsInjectiveIf<MemoryType>(In, Map, Condition, uint64(0), In.Num() - 1, Out.Num());
	
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	thrust::scatter_if(
		GetExecutionPolicy<MemoryType>(),
		In.GetData(),
		In.GetData() + In.Num(),
		MapTransform,
		thrust::make_counting_iterator<uint64>(0),
		Out.GetData(),
		Condition);
	
	CUDA_CHECK_LAST_ERROR();
}

template<typename TypeIn, typename TypeOut, typename InF, typename MapFunction, typename F, EMemoryType MemoryType>
inline void ScatterIfWithTransform(
	const TFixedArray<TypeIn, MemoryType>& In, InF Transform,
	MapFunction Map,
	TFixedArray<TypeOut, MemoryType>& Out,
	F Condition)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(In.Num());
	
	CheckFunctionBoundsIf<MemoryType>(Map, Condition, uint64(0), In.Num() - 1, uint64(0), Out.Num() - 1);
	CheckFunctionIsInjectiveIf<MemoryType>(In, Map, Condition, uint64(0), In.Num() - 1, Out.Num());
	
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);
	thrust::scatter_if(
		GetExecutionPolicy<MemoryType>(),
		It,
		It + In.Num(),
		MapTransform,
		thrust::make_counting_iterator<uint64>(0),
		Out.GetData(),
		Condition);
	
	CUDA_CHECK_LAST_ERROR();
}

template<typename InType, typename OutType, typename F, EMemoryType MemoryType>
inline void Transform(
	const TFixedArray<InType, MemoryType>& In,
	TFixedArray<OutType, MemoryType>& Out,
	F Lambda)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(In.Num());
	
	checkEqual(In.Num(), Out.Num())
	
	thrust::transform(
		GetExecutionPolicy<MemoryType>(),
		In.GetData(),
		In.GetData() + In.Num(),
		Out.GetData(),
		Lambda);
	
	CUDA_CHECK_LAST_ERROR();
}

template<typename InType, typename OutType, typename F1, typename F2>
inline void TransformIf(
	InType In,
	OutType Out,
	const uint64 Num,
	F1 Lambda,
	F2 Condition)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(Num);
	
	thrust::transform_if(
		GetExecutionPolicy<EMemoryType::GPU>(),
		In,
		In + Num,
		Out,
		Lambda,
		Condition);
	
	CUDA_CHECK_LAST_ERROR();
}

template<typename TValueIt, typename TKey, typename TValue, typename F>
inline void MergePairs(
	const TGpuArray<TKey>& InKeysA,
	TValueIt InValuesA,
	const TGpuArray<TKey>& InKeysB,
	TValueIt InValuesB,
	TGpuArray<TKey>& OutKeys,
	TGpuArray<TValue>& OutValues,
	F Comp)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(InKeysA.Num() + InKeysB.Num());
	
	checkEqual(OutKeys.Num(), OutValues.Num());
	checkEqual(InKeysA.Num() + InKeysB.Num(), OutValues.Num());

	struct mgpu_context_t : mgpu::standard_context_t
	{
		mgpu_context_t()
			: mgpu::standard_context_t(false, DEFAULT_STREAM)
		{
		}

		void* alloc(size_t size, mgpu::memory_space_t space) override
		{
			(void)space;
			check(space == mgpu::memory_space_device);
			return FMemory::Malloc<uint8>("mgpu", size, EMemoryType::GPU);
		}
		void free(void* p, mgpu::memory_space_t space) override
		{
			(void)space;
			check(space == mgpu::memory_space_device);
			FMemory::Free(p);
		}
		void synchronize() override
		{
			CUDA_CHECK_LAST_ERROR();
			CUDA_SYNCHRONIZE_STREAM();
			CUDA_CHECK_LAST_ERROR();
		}
	};
	thread_local mgpu_context_t* mgpu_context = nullptr;
	if (!mgpu_context)
	{
		mgpu_context = new mgpu_context_t();
	}
	
	mgpu::merge(
		InKeysA.GetData(), InValuesA, Cast<int32>(InKeysA.Num()),
		InKeysB.GetData(), InValuesB, Cast<int32>(InKeysB.Num()),
		OutKeys.GetData(), OutValues.GetData(),
		Comp,
		*mgpu_context);
	
	CUDA_CHECK_LAST_ERROR();
}

template<typename T, EMemoryType MemoryType>
inline void MakeSequence(TFixedArray<T, MemoryType>& Array, T Init = 0)
{
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(Array.Num());
	
	thrust::sequence(GetExecutionPolicy<MemoryType>(), Array.GetData(), Array.GetData() + Array.Num(), Init);
	
	CUDA_CHECK_LAST_ERROR();
}

template<typename TKeys, typename TValues, EMemoryType MemoryType>
inline void SortByKey(
	TFixedArray<TKeys, MemoryType>& Keys,
	TFixedArray<TValues, MemoryType>& Values)
{
	checkEqual(Keys.Num(), Values.Num());
	
	CUDA_CHECK_LAST_ERROR();
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(Keys.Num());
	
	thrust::sort_by_key(GetExecutionPolicy<MemoryType>(), Keys.GetData(), Keys.GetData() + Keys.Num(), Values.GetData());
	
	CUDA_CHECK_LAST_ERROR();
}