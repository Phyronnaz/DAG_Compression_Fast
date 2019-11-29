#pragma once

#include "Core.h"

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

#if DEBUG_THRUST
struct my_policy : thrust::system::cpp::execution_policy<my_policy> {};
static auto ExecutionPolicy = my_policy();
#else
static auto ExecutionPolicy = thrust::cuda::par.on(0);
#endif

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
inline void SetElement(TStaticArray<T, EMemoryType::GPU>& Array, uint64 Index, T Value)
{
	FMemory::SetGPUValue(&Array[Index], Value);
}
template<typename T>
inline T GetElement(const TStaticArray<T, EMemoryType::GPU>& Array, uint64 Index)
{
	return FMemory::ReadGPUValue(&Array[Index]);
}

template<typename InType, typename OutType, typename F1>
inline void AdjacentDifference(
	const TStaticArray<InType, EMemoryType::GPU>& In,
	TStaticArray<OutType, EMemoryType::GPU>& Out,
	F1 BinaryOp,
	OutType FirstElement)
{
	checkEqual(In.Num(), Out.Num())

	thrust::transform(
		ExecutionPolicy,
		In.GetData() + 1,
		In.GetData() + In.Num(),
		In.GetData(),
		Out.GetData() + 1,
		BinaryOp);
	SetElement(Out, 0, FirstElement);
	
	CUDA_CHECK_ERROR();
}

template<typename InType, typename OutType, typename F1, typename F2>
inline void AdjacentDifferenceWithTransform(
	const TStaticArray<InType, EMemoryType::GPU>& In, F1 Transform,
	TStaticArray<OutType, EMemoryType::GPU>& Out,
	F2 BinaryOp,
	OutType FirstElement)
{
	checkEqual(In.Num(), Out.Num())

	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);
	thrust::transform(
		ExecutionPolicy,
		It + 1,
		It + In.Num(),
		It,
		Out.GetData() + 1,
		BinaryOp);
	SetElement(Out, 0, FirstElement);
	
	CUDA_CHECK_ERROR();
}

template<typename TypeIn, typename TypeOut, typename MapFunction>
inline void ScatterPred(
	const TStaticArray<TypeIn, EMemoryType::GPU>& In,
	MapFunction Map,
	TStaticArray<TypeOut, EMemoryType::GPU>& Out)
{
	CheckFunctionBounds<uint64, uint64>(Map, 0, In.Num() - 1, 0, Out.Num() - 1);
	
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	thrust::scatter(
		ExecutionPolicy,
		In.GetData(),
		In.GetData() + In.Num(),
		MapTransform,
		Out.GetData());
	
	CUDA_CHECK_ERROR();
}

template<typename TypeIn, typename TypeOut, typename IndexType>
inline void Scatter(
	const TStaticArray<TypeIn, EMemoryType::GPU>& In,
	const TStaticArray<IndexType, EMemoryType::GPU>& Map,
	TStaticArray<TypeOut, EMemoryType::GPU>& Out)
{
	CheckArrayBounds<IndexType>(Map, 0, IndexType(Out.Num()) - 1);
	
	thrust::scatter(
		ExecutionPolicy,
		In.GetData(),
		In.GetData() + In.Num(),
		Map.GetData(),
		Out.GetData());
	
	CUDA_CHECK_ERROR();
}

template<typename TypeIn, typename TypeOut, typename InF, typename IndexType>
inline void ScatterWithTransform(
	const TStaticArray<TypeIn, EMemoryType::GPU>& In, InF Transform,
	const TStaticArray<IndexType, EMemoryType::GPU>& Map,
	TStaticArray<TypeOut, EMemoryType::GPU>& Out)
{
	CheckArrayBounds<IndexType>(Map, 0, IndexType(Out.Num()) - 1);
	
	const auto It = thrust::make_transform_iterator(In.GetData(), Transform);
	thrust::scatter(
		ExecutionPolicy,
		It,
		It + In.Num(),
		Map.GetData(),
		Out.GetData());
	
	CUDA_CHECK_ERROR();
}

template<typename TypeIn, typename TypeOut, typename MapFunction, typename F>
inline void ScatterIf(
	const TStaticArray<TypeIn, EMemoryType::GPU>& In,
	MapFunction Map,
	TStaticArray<TypeOut, EMemoryType::GPU>& Out,
	F Condition)
{
	CheckFunctionBoundsIf<uint64, uint64>(Map, Condition, 0, In.Num() - 1, 0, Out.Num() - 1);
	
	const auto MapTransform = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64>(0), Map);
	thrust::scatter_if(
		ExecutionPolicy,
		In.GetData(),
		In.GetData() + In.Num(),
		MapTransform,
		thrust::make_counting_iterator<uint64>(0),
		Out.GetData(),
		Condition);
	
	CUDA_CHECK_ERROR();
}

template<typename TypeIn, typename TypeOut, typename InF, typename MapFunction, typename F>
inline void ScatterIfWithTransform(
	const TStaticArray<TypeIn, EMemoryType::GPU>& In, InF Transform,
	MapFunction Map,
	TStaticArray<TypeOut, EMemoryType::GPU>& Out,
	F Condition)
{
	CheckFunctionBoundsIf<uint64, uint64>(Map, Condition, 0, In.Num() - 1, 0, Out.Num() - 1);
	
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
	
	CUDA_CHECK_ERROR();
}

template<typename InType, typename OutType, typename F>
inline void Transform(
	const TStaticArray<InType, EMemoryType::GPU>& In,
	TStaticArray<OutType, EMemoryType::GPU>& Out,
	F Lambda)
{
	checkEqual(In.Num(), Out.Num())
	
	thrust::transform(
		ExecutionPolicy,
		In.GetData(),
		In.GetData() + In.Num(),
		Out.GetData(),
		Lambda);
	
	CUDA_CHECK_ERROR();
}

template<typename InType, typename OutType, typename F1, typename F2>
inline void TransformIf(
	InType In,
	OutType Out,
	const uint64 Num,
	F1 Lambda,
	F2 Condition)
{
	thrust::transform_if(
		ExecutionPolicy,
		In,
		In + Num,
		Out,
		Lambda,
		Condition);
	
	CUDA_CHECK_ERROR();
}

template<typename TValueIt, typename TKey, typename TValue, typename F>
inline void MergePairs(
	const TStaticArray<TKey, EMemoryType::GPU>& InKeysA,
	TValueIt InValuesA,
	const TStaticArray<TKey, EMemoryType::GPU>& InKeysB,
	TValueIt InValuesB,
	TStaticArray<TKey, EMemoryType::GPU>& OutKeys,
	TStaticArray<TValue, EMemoryType::GPU>& OutValues,
	F Comp)
{
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
			check(space == mgpu::memory_space_device);
			return FMemory::Malloc<uint8>("mgpu", size, EMemoryType::GPU);
		}
		void free(void* p, mgpu::memory_space_t space) override
		{
			check(space == mgpu::memory_space_device);
			FMemory::Free(p);
		}
		void synchronize() override
		{
			CUDA_CHECK_ERROR();
		}
	};
	static mgpu_context_t* mgpu_context = nullptr;
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
	CUDA_CHECK_ERROR();
}

template<typename T>
inline void MakeSequence(TStaticArray<T, EMemoryType::GPU>& Array, T Init = 0)
{
	thrust::sequence(ExecutionPolicy, Array.GetData(), Array.GetData() + Array.Num(), Init);
	
	CUDA_CHECK_ERROR();
}