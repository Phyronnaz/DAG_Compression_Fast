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
        Singleton.FreeImpl(reinterpret_cast<void*>(Ptr));
    }
    template<typename T>
    static void Realloc(T*& Ptr, uint64 NewSize, bool bCopyData = true)
    {
		void* Copy = reinterpret_cast<void*>(Ptr);
        Singleton.ReallocImpl(Copy, NewSize, bCopyData);
		Ptr = reinterpret_cast<T*>(Copy);
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
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        return Singleton.TotalAllocatedCpuMemory;
    }
    inline static uint64 GetGpuAllocatedMemory()
    {
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        return Singleton.TotalAllocatedGpuMemory;
    }
    inline static uint64 GetCpuMaxAllocatedMemory()
    {
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        return Singleton.MaxTotalAllocatedCpuMemory;
    }
    inline static uint64 GetGpuMaxAllocatedMemory()
    {
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        return Singleton.MaxTotalAllocatedGpuMemory;
    }

public:
	template<typename T>
	static T ReadGPUValue(const T* GPUPtr)
	{
		PROFILE_FUNCTION();

		CUDA_CHECK_ERROR();
		
		thread_local T* CPU = nullptr;
		if (!CPU)
		{
			CUDA_CHECKED_CALL cudaMallocHost(reinterpret_cast<void**>(&CPU), sizeof(T));
		}
		CUDA_CHECKED_CALL cudaMemcpyAsync(CPU, GPUPtr, sizeof(T), cudaMemcpyDeviceToHost);

		CUDA_CHECK_ERROR();
		
		const T Value = *CPU;
		return Value;
	}
	template<typename T>
	static void SetGPUValue(T* GPUPtr, T Value)
	{
		PROFILE_FUNCTION();

		CUDA_CHECK_ERROR();
		
		thread_local T* CPU = nullptr;
		if (!CPU)
		{
			CUDA_CHECKED_CALL cudaMallocHost(reinterpret_cast<void**>(&CPU), sizeof(T));
		}
		*CPU = Value;
		CUDA_CHECKED_CALL cudaMemcpyAsync(GPUPtr, CPU, sizeof(T), cudaMemcpyHostToDevice);

		CUDA_CHECK_ERROR();
	}

private:
    FMemory() = default;
    ~FMemory();

    void* MallocImpl(const char* Name, uint64 Size, EMemoryType Type);
    void FreeImpl(void* Ptr);
	void ReallocImpl(void*& Ptr, uint64 NewSize, bool bCopyData);
	void RegisterCustomAllocImpl(void* Ptr, const char* Name, uint64 Size, EMemoryType Type);
	void UnregisterCustomAllocImpl(void* Ptr);
	FAlloc GetAllocInfoImpl(void* Ptr) const;
    std::string GetStatsStringImpl() const;

    mutable std::mutex Mutex;
    size_t TotalAllocatedGpuMemory = 0;
    size_t TotalAllocatedCpuMemory = 0;
    size_t MaxTotalAllocatedGpuMemory = 0;
    size_t MaxTotalAllocatedCpuMemory = 0;
    std::unordered_map<void*, FAlloc> Allocations;

    static FMemory Singleton;
};

template<typename T>
class TSingleGPUElement
{
public:
	TSingleGPUElement()
	{
		CUDA_CHECKED_CALL cnmemMalloc(reinterpret_cast<void**>(&Ptr), sizeof(T), DEFAULT_STREAM);
	}
	~TSingleGPUElement()
	{
		CUDA_CHECKED_CALL cnmemFree(Ptr, DEFAULT_STREAM);
	}

	inline T* GetPtr()
	{
		return Ptr;
	}
	inline T GetValue() const
	{
		return FMemory::ReadGPUValue(Ptr);
	}

private:
	T* Ptr = nullptr;
};