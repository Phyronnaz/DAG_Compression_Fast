#pragma once

#include "Core.h"
#include "Utils.h"
#include "Memory.h"

template<typename ElementType, uint64 ArraySize>
struct TStaticArray
{
	using TElementType = ElementType;

	TStaticArray() = default;
	explicit TStaticArray(ElementType Value)
	{
		for (auto& It : *this)
		{
			It = Value;
		}
	}
	TStaticArray(std::initializer_list<TElementType> List)
	{
		check(List.size() == ArraySize);
		std::memcpy(ArrayData, List.begin(), SizeInBytes());
	}

	void MemSet(uint8 Value)
	{
		PROFILE_FUNCTION();
		std::memset(GetData(), Value, SizeInBytes());
	}
	
	static uint64 Num()
	{
		return ArraySize;
	}

	const TElementType* GetData() const
	{
		return ArrayData;
	}
	TElementType* GetData()
	{
		return ArrayData;
	}
	
	static bool IsValidIndex(uint64 Index)
	{
		return Index < ArraySize;
	}
	static uint64 SizeInBytes()
	{
		return ArraySize * sizeof(TElementType);
	}
	static double SizeInMB()
	{
		return Utils::ToMB(SizeInBytes());
	}

	TElementType& operator[](uint64 Index)
	{
		checkf(Index < ArraySize, "invalid index: %" PRIu64 " for size %" PRIu64, Index, ArraySize);
		return ArrayData[Index];
	}
	const TElementType& operator[](uint64 Index) const
	{
		checkf(Index < ArraySize, "invalid index: %" PRIu64 " for size %" PRIu64, Index, ArraySize);
		return ArrayData[Index];
	}

	TElementType& operator()(uint64 Index)
	{
		return (*this)[Index];
	}
	const TElementType& operator()(uint64 Index) const
	{
		return (*this)[Index];
	}

	const TElementType* begin() const
	{
		return ArrayData;
	}
	const TElementType* end() const
	{
		return ArrayData + ArraySize;
	}
	
	TElementType* begin()
	{
		return ArrayData;
	}
	TElementType* end()
	{
		return ArrayData + ArraySize;
	}

	uint64 Find(const TElementType& Element) const
	{
		for (uint64 Index = 0; Index < Num(); Index++)
		{
			if ((*this)[Index] == Element)
			{
				return Index;
			}
		}
		return uint64(-1);
	}
	uint64 FindChecked(const TElementType& Element) const
	{
		const uint64 Index = Find(Element);
		check(Index != uint64(-1));
		return Index;
	}

private:
	TElementType ArrayData[ArraySize];
};

template<typename ElementType, EMemoryType MemoryType, typename TSize = uint64>
struct TFixedArray
{
	using TElementType = ElementType;
	
	TFixedArray() = default;
	HOST_DEVICE TFixedArray(TElementType* RESTRICT Data, TSize Size)
		: ArrayData(Data)
		, ArraySize(Size)
	{
	}
	TFixedArray(const char* Name, TSize Size)
		: ArrayData(FMemory::Malloc<TElementType>(Name, Size * sizeof(TElementType), MemoryType))
		, ArraySize(Size)
	{
#if ENABLE_CHECKS
		MemSet(0xFF);
#endif
	}
	TFixedArray(const char* Name, std::initializer_list<TElementType> List)
		: TFixedArray(Name, List.size())
	{
		static_assert(MemoryType == EMemoryType::CPU, "");
		std::memcpy(ArrayData, List.begin(), SizeInBytes());
	}
	HOST_DEVICE explicit TFixedArray(decltype(nullptr))
		: TFixedArray(nullptr, 0)
	{
	}

	HOST_DEVICE TFixedArray(const TFixedArray& Other)
		: ArrayData(Other.ArrayData)
		, ArraySize(Other.ArraySize)
	{
	}
	HOST_DEVICE TFixedArray(TFixedArray&& Other) noexcept
		: ArrayData(Other.ArrayData)
		, ArraySize(Other.ArraySize)
	{
		Other.Reset();
	}

	HOST_DEVICE TFixedArray& operator=(const TFixedArray& Other)
	{
		check(!IsValid());
		ArrayData = Other.ArrayData;
		ArraySize = Other.ArraySize;
		return *this;
	}
	HOST_DEVICE TFixedArray& operator=(TFixedArray&& Other) noexcept
	{
		check(!IsValid());
		ArrayData = Other.ArrayData;
		ArraySize = Other.ArraySize;
		Other.Reset();
		return *this;
	}

	template<typename TOther>
	HOST TFixedArray<TOther, MemoryType, TSize> CastTo() const
	{
		checkAlways((ArraySize * sizeof(TElementType)) % sizeof(TOther) == 0);

		TFixedArray<TOther, MemoryType, TSize> Result;
		Result.ArrayData = reinterpret_cast<TOther*>(ArrayData);
		Result.ArraySize = (ArraySize * sizeof(TElementType)) / sizeof(TOther);
		return Result;
	}

	HOST TFixedArray<TElementType, EMemoryType::GPU, TSize> CreateGPU() const
	{
		static_assert(MemoryType == EMemoryType::CPU, "");
		if (!IsValid())
		{
			return {};
		}
		auto Result = TFixedArray<TElementType, EMemoryType::GPU, TSize>(FMemory::GetAllocInfo(GetData()).Name, Num());
		CopyTo(Result);
		return Result;
	}
	HOST TFixedArray<TElementType, EMemoryType::CPU, TSize> CreateCPU() const
	{
		static_assert(MemoryType == EMemoryType::GPU, "");
		if (!IsValid())
		{
			return {};
		}
		auto Result = TFixedArray<TElementType, EMemoryType::CPU, TSize>(FMemory::GetAllocInfo(GetData()).Name, Num());
		CopyTo(Result);
		return Result;
	}
	HOST TFixedArray<TElementType, MemoryType, TSize> Clone() const
	{
		if (!IsValid())
		{
			return {};
		}
		auto Result = TFixedArray<TElementType, MemoryType, TSize>(FMemory::GetAllocInfo(GetData()).Name, Num());
		CopyTo(Result);
		return Result;
	}

	template<EMemoryType OtherMemoryType>
	HOST void CopyTo(TFixedArray<TElementType, OtherMemoryType, TSize>& OtherArray) const
	{
		check(OtherArray.Num() == Num());
		FMemory::CudaMemcpy(OtherArray.GetData(), GetData(), SizeInBytes(),
			MemoryType == EMemoryType::CPU
			? OtherMemoryType == EMemoryType::CPU
			? cudaMemcpyHostToHost
			: cudaMemcpyHostToDevice
			: OtherMemoryType == EMemoryType::CPU
			? cudaMemcpyDeviceToHost
			: cudaMemcpyDeviceToDevice);
	}

	HOST void MemSet(uint8 Value)
	{
		PROFILE_FUNCTION_TRACY();
		
		if (MemoryType == EMemoryType::GPU)
		{
			CUDA_CHECKED_CALL cudaMemset(GetData(), Value, SizeInBytes());
		}
		else
		{
			std::memset(GetData(), Value, SizeInBytes());
		}
	}

	HOST void Free()
	{
		FMemory::Free(ArrayData);
		ArrayData = nullptr;
		ArraySize = 0;
	}
	HOST_DEVICE void Reset()
	{
		ArrayData = nullptr;
		ArraySize = 0;
	}

	HOST_DEVICE TSize Num() const
	{
		return ArraySize;
	}
	HOST_DEVICE const TElementType* GetData() const
	{
		return ArrayData;
	}
	HOST_DEVICE TElementType* GetData()
	{
		return ArrayData;
	}

	HOST_DEVICE static constexpr EMemoryType GetMemoryType()
	{
		return MemoryType;
	}
	
	HOST_DEVICE bool IsValid() const
	{
		return ArrayData != nullptr;
	}

	HOST_DEVICE bool IsValidIndex(TSize Index) const
	{
		return Index < ArraySize;
	}
	
	HOST_DEVICE uint64 SizeInBytes() const
	{
		return ArraySize * sizeof(TElementType);
	}
	HOST_DEVICE double SizeInMB() const
	{
		return Utils::ToMB(SizeInBytes());
	}

	HOST_DEVICE TElementType& operator[](TSize Index)
	{
		checkf(Index < ArraySize, "invalid index: %" PRIu64 " for size %" PRIu64, Index, ArraySize);
		return ArrayData[Index];
	}
	HOST_DEVICE const TElementType& operator[](TSize Index) const
	{
		checkf(Index < ArraySize, "invalid index: %" PRIu64 " for size %" PRIu64, Index, ArraySize);
		return ArrayData[Index];
	}

	HOST TElementType& CheckedAt(TSize Index)
	{
		checkfAlways(Index < ArraySize, "invalid index: %" PRIu64 " for size %" PRIu64, Index, ArraySize);
		return ArrayData[Index];
	}
	HOST const TElementType& CheckedAt(TSize Index) const
	{
		checkfAlways(Index < ArraySize, "invalid index: %" PRIu64 " for size %" PRIu64, Index, ArraySize);
		return ArrayData[Index];
	}

	HOST_DEVICE TElementType& operator()(TSize Index)
	{
		return (*this)[Index];
	}
	HOST_DEVICE const TElementType& operator()(TSize Index) const
	{
		return (*this)[Index];
	}
	
	HOST_DEVICE bool operator==(const TFixedArray& Other) const
	{
		return Other.ArrayData == ArrayData && Other.ArraySize == ArraySize;
	}
	HOST_DEVICE bool operator!=(const TFixedArray& Other) const
	{
		return Other.ArrayData != ArrayData || Other.ArraySize != ArraySize;
	}

	HOST_DEVICE const TElementType* begin() const
	{
		return ArrayData;
	}
	HOST_DEVICE const TElementType* end() const
	{
		return ArrayData + ArraySize;
	}
	
	HOST_DEVICE TElementType* begin()
	{
		return ArrayData;
	}
	HOST_DEVICE TElementType* end()
	{
		return ArrayData + ArraySize;
	}

	HOST_DEVICE TSize Find(const TElementType& Element) const
	{
		for (TSize Index = 0; Index < Num(); Index++)
		{
			if ((*this)[Index] == Element)
			{
				return Index;
			}
		}
		return TSize(-1);
	}
	HOST_DEVICE TSize FindChecked(const TElementType& Element) const
	{
		const TSize Index = Find(Element);
		check(Index != TSize(-1));
		return Index;
	}

protected:
	TElementType* RESTRICT ArrayData = nullptr;
	TSize ArraySize = 0;

	template<typename, EMemoryType, typename>
	friend struct TFixedArray;
};

template<typename TElementType>
using TGpuArray = TFixedArray<TElementType, EMemoryType::GPU>;

template<typename TElementType>
using TCpuArray = TFixedArray<TElementType, EMemoryType::CPU>;

template<EMemoryType MemoryType, typename TSize = uint64>
struct TBitArray
{
	TBitArray() = default;
	TBitArray(const char* Name, TSize Size)
		: Array(Name, Utils::DivideCeil<TSize>(Size, 32))
	{
	}
	
	HOST_DEVICE bool operator[](TSize Index) const
	{
		return Array[Index / 32] & (1u << (Index % 32));
	}
	HOST_DEVICE void Set(TSize Index, bool Value)
	{
		if (Value)
		{
			Array[Index / 32] |= (1u << (Index % 32));
		}
		else
		{
			Array[Index / 32] &= ~(1u << (Index % 32));
		}
	}

	HOST void MemSet(bool Value)
	{
		Array.MemSet(Value ? 0xFF : 0x00);
	}

private:
	TFixedArray<uint32, MemoryType, TSize> Array;
};