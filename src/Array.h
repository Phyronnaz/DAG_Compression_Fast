#pragma once

#include "Core.h"
#include "Utils.h"
#include "Memory.h"

template<typename TElementType, EMemoryType MemoryType, typename TSize = uint64>
struct TStaticArray
{
	TStaticArray() = default;
	HOST_DEVICE TStaticArray(TElementType* RESTRICT Data, TSize Size)
		: ArrayData(Data)
		, ArraySize(Size)
	{
	}
	TStaticArray(const char* Name, TSize Size)
		: ArrayData(FMemory::Malloc<TElementType>(Name, Size * sizeof(TElementType), MemoryType))
		, ArraySize(Size)
	{
	}
	TStaticArray(const char* Name, std::initializer_list<TElementType> List)
		: TStaticArray(Name, List.size())
	{
		static_assert(MemoryType == EMemoryType::CPU, "");
		std::memcpy(ArrayData, List.begin(), SizeInBytes());
	}
	HOST_DEVICE explicit TStaticArray(decltype(nullptr))
		: TStaticArray(nullptr, 0)
	{
	}

	HOST_DEVICE TStaticArray(const TStaticArray& Other)
		: ArrayData(Other.ArrayData),
		  ArraySize(Other.ArraySize)
	{
	}
	HOST_DEVICE TStaticArray(TStaticArray&& Other) noexcept
		: ArrayData(Other.ArrayData),
		  ArraySize(Other.ArraySize)
	{
		Other.Reset();
	}

	HOST_DEVICE TStaticArray& operator=(const TStaticArray& Other)
	{
		check(!IsValid());
		ArrayData = Other.ArrayData;
		ArraySize = Other.ArraySize;
		return *this;
	}
	HOST_DEVICE TStaticArray& operator=(TStaticArray&& Other) noexcept
	{
		check(!IsValid());
		ArrayData = Other.ArrayData;
		ArraySize = Other.ArraySize;
		Other.Reset();
		return *this;
	}

	template<typename TOther>
	HOST TStaticArray<TOther, MemoryType, TSize> CastTo() const
	{
		checkAlways((ArraySize * sizeof(TElementType)) % sizeof(TOther) == 0);

		TStaticArray<TOther, MemoryType, TSize> Result;
		Result.ArrayData = reinterpret_cast<TOther*>(ArrayData);
		Result.ArraySize = (ArraySize * sizeof(TElementType)) / sizeof(TOther);
		return Result;
	}

	HOST TStaticArray<TElementType, EMemoryType::GPU, TSize> CreateGPU() const
	{
		ZoneScoped;
		
		static_assert(MemoryType == EMemoryType::CPU, "");
	    check(IsValid());
        auto Result = TStaticArray<TElementType, EMemoryType::GPU, TSize>(FMemory::GetAllocInfo(GetData()).Name, Num());
        CopyToGPU(Result);
        return Result;
    }
	HOST TStaticArray<TElementType, EMemoryType::CPU, TSize> CreateCPU() const
	{
		ZoneScoped;
		
		static_assert(MemoryType == EMemoryType::GPU, "");
	    check(IsValid());
        auto Result = TStaticArray<TElementType, EMemoryType::CPU, TSize>(FMemory::GetAllocInfo(GetData()).Name, Num());
        CopyToCPU(Result);
        return Result;
    }

    HOST void CopyToGPU(TStaticArray<TElementType, EMemoryType::GPU, TSize>& GpuArray) const
    {
		ZoneScoped;
		
		static_assert(MemoryType == EMemoryType::CPU, "");
        check(GpuArray.Num() == Num());
        CUDA_CHECKED_CALL cudaMemcpy(GpuArray.GetData(), GetData(), SizeInBytes(), cudaMemcpyHostToDevice);
    }
    HOST void CopyToCPU(TStaticArray<TElementType, EMemoryType::CPU, TSize>& CpuArray) const
    {
		ZoneScoped;
		
		static_assert(MemoryType == EMemoryType::GPU, "");
        check(CpuArray.Num() == Num());
        CUDA_CHECKED_CALL cudaMemcpy(CpuArray.GetData(), GetData(), SizeInBytes(), cudaMemcpyDeviceToHost);
	}

	HOST void MemSet(uint8 Value)
	{
		ZoneScoped;
		
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
	HOST_DEVICE bool operator==(const TStaticArray& Other) const
	{
		return Other.ArrayData == ArrayData && Other.ArraySize == ArraySize;
	}
	HOST_DEVICE bool operator!=(const TStaticArray& Other) const
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
	friend struct TStaticArray;
};

template<typename TElementType, EMemoryType MemoryType, typename TSize = uint64>
struct TDynamicArray : TStaticArray<TElementType, MemoryType, TSize>
{
public:
	TDynamicArray() = default;
	TDynamicArray(TElementType* RESTRICT Data, TSize Size)
		: TStaticArray<TElementType, MemoryType, TSize>(Data, Size)
		, AllocatedSize(Size)
	{
	}

	explicit TDynamicArray(decltype(nullptr))
		: TStaticArray<TElementType, MemoryType, TSize>(nullptr, 0)
		, AllocatedSize(0)
	{
	}
	explicit TDynamicArray(const TStaticArray<TElementType, MemoryType, TSize>& Array)
		: TStaticArray<TElementType, MemoryType, TSize>(Array)
		, AllocatedSize(Array.Num())
	{
	}
	TDynamicArray(const char* Name, TSize Size)
		: TStaticArray<TElementType, MemoryType, TSize>(Name, Size)
		, AllocatedSize(Size)
	{
	}

    HOST void CopyToGPU(TDynamicArray<TElementType, EMemoryType::GPU, TSize>& GPUArray) const
    {
		ZoneScoped;
		
		static_assert(MemoryType == EMemoryType::CPU, "");
        if (!GPUArray.is_valid())
        {
            GPUArray = this->Allocate(FMemory::GetAllocInfo(this->GetData()).Name, this->GetAllocatedSize(), EMemoryType::GPU);
        }
        if (GPUArray.GetAllocatedSize() < this->size())
        {
            GPUArray.Reserve(this->GetAllocatedSize() - GPUArray.GetAllocatedSize());
        }
        GPUArray.ArraySize = this->Num();
        CUDA_CHECKED_CALL cudaMemcpy(GPUArray.GetData(), this->GetData(), this->SizeInBytes(), cudaMemcpyHostToDevice);
    }

	HOST_DEVICE TSize GetAllocatedSize() const
	{
		return AllocatedSize;
	}
	
	HOST_DEVICE uint64 AllocatedSizeInBytes() const
	{
		return AllocatedSize * sizeof(TElementType);
	}
	HOST_DEVICE double AllocatedSizeInMB() const
	{
		return Utils::ToMB(AllocatedSizeInBytes());
	}

	HOST TSize Add(TElementType Element)
	{
		check(this->ArraySize <= AllocatedSize);
		if(this->ArraySize == AllocatedSize)
		{
			Reserve(AllocatedSize); // Double the storage
		}

		const TSize OldSize = this->ArraySize;
		this->ArraySize++;
		check(this->ArraySize <= AllocatedSize);

		(*this)[OldSize] = Element;

		return OldSize;
	}
	template<typename... TArgs>
	HOST TSize Emplace(TArgs... Args)
	{
		return Add(TElementType{ std::forward<TArgs>(Args)... });
	}
	
	HOST void Resize(TSize NewSize)
	{
		ZoneScoped;
		
		checkInfEqual(this->ArraySize, AllocatedSize);
		check(this->ArrayData); // For the name
		this->ArraySize = NewSize;
		if (AllocatedSize < this->ArraySize)
		{
			AllocatedSize = this->ArraySize;
			FMemory::Realloc(this->ArrayData, AllocatedSizeInBytes());
		}
		checkInfEqual(this->ArraySize, AllocatedSize);
	}
	HOST void ResizeUninitialized(TSize NewSize)
	{
		ZoneScoped;
		
		checkInfEqual(this->ArraySize, AllocatedSize);
		check(this->ArrayData); // For the name
		this->ArraySize = NewSize;
		if (AllocatedSize < this->ArraySize)
		{
			AllocatedSize = this->ArraySize;
			FMemory::Realloc(this->ArrayData, AllocatedSizeInBytes(), false);
		}
		checkInfEqual(this->ArraySize, AllocatedSize);
	}
	
	HOST void Shrink()
	{
		ZoneScoped;
		
		check(this->ArrayData);
		check(this->ArraySize != 0);
		checkInfEqual(this->ArraySize, AllocatedSize);
		if (AllocatedSize != this->ArraySize)
		{
			AllocatedSize = this->ArraySize;
			FMemory::Realloc(this->ArrayData, AllocatedSizeInBytes());
		}
		checkInfEqual(this->ArraySize, AllocatedSize);
	}

protected:
	TSize AllocatedSize = 0;
};