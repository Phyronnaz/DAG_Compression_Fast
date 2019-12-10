#pragma once

#include "Core.h"
#include <unordered_map>
#include <mutex>
#include <functional>

enum class EMemoryType
{
	GPU,
	CPU
};

class FMemory
{
public:
	struct FAlloc
	{
		const char* Name = nullptr;
		uint64 Size = uint64(-1);
		EMemoryType Type = EMemoryType::CPU;
	};

	template<typename T>
	static T* Malloc(const char* Name, uint64 Size, EMemoryType Type)
	{
		checkAlways(Size % sizeof(T) == 0);
		return reinterpret_cast<T*>(Singleton.MallocImpl(Name, Size, Type));
	}
	template<typename T>
	static void Free(T* Ptr)
	{
		CUDA_CHECK_LAST_ERROR();
		Singleton.FreeImpl(reinterpret_cast<void*>(Ptr));
	}
	template<typename T>
	static void RegisterCustomAlloc(T* Ptr, const char* Name, uint64 Size, EMemoryType Type)
	{
		checkAlways(Size % sizeof(T) == 0);
		Singleton.RegisterCustomAllocImpl(reinterpret_cast<void*>(Ptr), Name, Size, Type);
	}
	template<typename T>
	static void UnregisterCustomAlloc(T* Ptr)
	{
		Singleton.UnregisterCustomAllocImpl(reinterpret_cast<void*>(Ptr));
	}
	template<typename T>
	static FAlloc GetAllocInfo(const T* Ptr)
	{
		return Singleton.GetAllocInfoImpl(reinterpret_cast<void*>(const_cast<T*>(Ptr)));
	}
	static std::string GetStatsString()
	{
		return Singleton.GetStatsStringImpl();
	}

public:
	inline static uint64 GetCpuAllocatedMemory()
	{
		FScopeLock Lock(Singleton.MemoryMutex);
		return Singleton.TotalAllocatedCpuMemory;
	}
	inline static uint64 GetGpuAllocatedMemory()
	{
		FScopeLock Lock(Singleton.MemoryMutex);
		return Singleton.TotalAllocatedGpuMemory;
	}
	inline static uint64 GetCpuMaxAllocatedMemory()
	{
		FScopeLock Lock(Singleton.MemoryMutex);
		return Singleton.MaxTotalAllocatedCpuMemory;
	}
	inline static uint64 GetGpuMaxAllocatedMemory()
	{
		FScopeLock Lock(Singleton.MemoryMutex);
		return Singleton.MaxTotalAllocatedGpuMemory;
	}

public:
	template<typename T>
	static T ReadGPUValue(const T* GPUPtr)
	{
		PROFILE_FUNCTION_TRACY();
		CUDA_CHECK_LAST_ERROR();

		thread_local T* CPU = nullptr;
		if (!CPU)
		{
			CUDA_CHECKED_CALL cudaMallocHost(reinterpret_cast<void**>(&CPU), sizeof(T));
		}
		CudaMemcpy(CPU, GPUPtr, sizeof(T), cudaMemcpyDeviceToHost);

		const T Value = *CPU;
		return Value;
	}
	template<typename T>
	static void SetGPUValue(T* GPUPtr, T Value)
	{
		PROFILE_FUNCTION_TRACY();
		CUDA_CHECK_LAST_ERROR();

		thread_local T* CPU = nullptr;
		if (!CPU)
		{
			CUDA_CHECKED_CALL cudaMallocHost(reinterpret_cast<void**>(&CPU), sizeof(T));
		}
		*CPU = Value;
		CudaMemcpy(GPUPtr, CPU, sizeof(T), cudaMemcpyHostToDevice);
	}

	template<typename T>
	static void CudaMemcpy(T* Dst, const T* Src, uint64 Size, cudaMemcpyKind MemcpyKind)
	{
		CUDA_CHECK_LAST_ERROR();
		check(Size % sizeof(T) == 0);
		Singleton.CudaMemcpyImpl(reinterpret_cast<uint8*>(Dst), reinterpret_cast<const uint8*>(Src), Size, MemcpyKind);
	}

private:
	FMemory() = default;
	~FMemory();
	
	void CudaMemcpyImpl(uint8* Dst, const uint8* Src, uint64 Size, cudaMemcpyKind MemcpyKind);
    void* MallocImpl(const char* Name, uint64 Size, EMemoryType Type);
    void FreeImpl(void* Ptr);
	void RegisterCustomAllocImpl(void* Ptr, const char* Name, uint64 Size, EMemoryType Type);
	void UnregisterCustomAllocImpl(void* Ptr);
	FAlloc GetAllocInfoImpl(void* Ptr) const;
    std::string GetStatsStringImpl() const;

    mutable TracyLockable(std::mutex, MemoryMutex)
    size_t TotalAllocatedGpuMemory = 0;
    size_t TotalAllocatedCpuMemory = 0;
    size_t MaxTotalAllocatedGpuMemory = 0;
    size_t MaxTotalAllocatedCpuMemory = 0;
    std::unordered_map<void*, FAlloc> Allocations;

    static FMemory Singleton;
};

struct FTempStorage
{
	void* Ptr = nullptr;
	size_t Bytes = 0;

	FTempStorage()
	{
		CUDA_CHECK_LAST_ERROR();
	}
	~FTempStorage()
	{
		check(!Ptr);
		check(Bytes == 0);
	}
	void Allocate();
	void Free();
};

template<typename T>
class TSingleGPUElement
{
public:
	TSingleGPUElement()
	{
		Storage.Bytes = sizeof(T);
		Storage.Allocate();
	}
	~TSingleGPUElement()
	{
		Storage.Free();
	}

	inline T* GetPtr()
	{
		return reinterpret_cast<T*>(Storage.Ptr);
	}
	inline T GetValue() const
	{
		return FMemory::ReadGPUValue(reinterpret_cast<T*>(Storage.Ptr));
	}

private:
	FTempStorage Storage;
};

// TODO async copy