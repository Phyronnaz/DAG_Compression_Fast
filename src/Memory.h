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
    struct Element
    {
        const char* Name = nullptr;
        uint64 Size = uint64(-1);
        EMemoryType Type = EMemoryType::CPU;
    };

    template<typename T>
    static T* Malloc(const char* Name, uint64 Size, EMemoryType Type)
	{
		ZoneScoped;
        checkAlways(Size % sizeof(T) == 0);
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        return reinterpret_cast<T*>(Singleton.MallocImpl(Name, Size, Type));
    }
    template<typename T>
    static void Free(T* Ptr)
    {
		ZoneScoped;
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        Singleton.FreeImpl(reinterpret_cast<void*>(Ptr));
    }
    template<typename T>
    static void Realloc(T*& Ptr, uint64 NewSize, bool bCopyData = true)
    {
		ZoneScoped;
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
		void* Copy = reinterpret_cast<void*>(Ptr);
        Singleton.ReallocImpl(Copy, NewSize, bCopyData);
		Ptr = reinterpret_cast<T*>(Copy);
    }
    template<typename T>
    static void RegisterCustomAlloc(T* Ptr, const char* Name, uint64 Size, EMemoryType Type)
    {
        checkAlways(Size % sizeof(T) == 0);
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        Singleton.RegisterCustomAllocImpl(reinterpret_cast<void*>(Ptr), Name, Size, Type);
    }
    template<typename T>
    static void UnregisterCustomAlloc(T* Ptr)
    {
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        Singleton.UnregisterCustomAllocImpl(reinterpret_cast<void*>(Ptr));
    }
    template<typename T>
    static Element GetAllocInfo(const T* Ptr)
    {
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        return Singleton.GetAllocInfoImpl(reinterpret_cast<void*>(const_cast<T*>(Ptr)));
    }
    static std::string GetStatsString()
    {
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        return Singleton.GetStatsStringImpl();
    }
	static void RegisterOutOfMemoryEvent(std::function<void()>&& Callback)
    {
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
		Singleton.OnOutOfMemoryEvents.push_back(std::move(Callback));
    }
	static void RegisterOutOfMemoryFallback(std::function<void()>&& Callback)
    {
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
		Singleton.OnOutOfMemoryFallbacks.push_back(std::move(Callback));
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

private:
    FMemory() = default;
    ~FMemory();

    void* MallocImpl(const char* Name, uint64 Size, EMemoryType Type);
    void FreeImpl(void* Ptr);
	void ReallocImpl(void*& Ptr, uint64 NewSize, bool bCopyData);
	void RegisterCustomAllocImpl(void* Ptr, const char* Name, uint64 Size, EMemoryType Type);
	void UnregisterCustomAllocImpl(void* Ptr);
	Element GetAllocInfoImpl(void* Ptr) const;
    std::string GetStatsStringImpl() const;

    std::mutex Mutex;
    size_t TotalAllocatedGpuMemory = 0;
    size_t TotalAllocatedCpuMemory = 0;
    size_t MaxTotalAllocatedGpuMemory = 0;
    size_t MaxTotalAllocatedCpuMemory = 0;
    std::unordered_map<void*, Element> Allocations;
	std::vector<std::function<void()>> OnOutOfMemoryFallbacks;
	std::vector<std::function<void()>> OnOutOfMemoryEvents;

    static FMemory Singleton;
};

template<typename T, typename TPool>
class TGPUSingleElementAlloc
{
public:
	TGPUSingleElementAlloc(TPool& Pool)
		: Ptr(Pool.template Malloc<T>(sizeof(T)))
		, Pool(Pool)
	{
	}
	~TGPUSingleElementAlloc()
	{
		Pool.Free(Ptr);
	}

	inline T* ToGPU() const
	{
		return Ptr;
	}
	inline T ToCPU() const
	{
		T CPU;
		CUDA_CHECKED_CALL cudaMemcpy(&CPU, Ptr, sizeof(T), cudaMemcpyDeviceToHost);
		return CPU;
	}

private:
	T* const Ptr;
	TPool& Pool;
};

#define SINGLE_GPU_ELEMENT(Type, Name, Pool) const TGPUSingleElementAlloc<Type, decltype(Pool)> Name(Pool)