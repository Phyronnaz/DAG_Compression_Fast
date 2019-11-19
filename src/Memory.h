#pragma once

#include "Core.h"
#include <unordered_map>
#include <mutex>

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
        checkAlways(Size % sizeof(T) == 0);
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        return reinterpret_cast<T*>(Singleton.MallocImpl(Name, Size, Type));
    }
    template<typename T>
    static void Free(T* Ptr)
    {
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        Singleton.FreeImpl(reinterpret_cast<void*>(Ptr));
    }
    template<typename T>
    static void Realloc(T*& Ptr, uint64 NewSize, bool bCopyData = true)
    {
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

public:
    inline static uint64 GetCpuAllocatedMemory()
    {
        return Singleton.TotalAllocatedCpuMemory;
    }
    inline static uint64 GetGpuAllocatedMemory()
    {
        return Singleton.TotalAllocatedGpuMemory;
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
    std::unordered_map<void*, Element> Allocations;

    static FMemory Singleton;
};

template<typename T>
class TGPUSingleElementAlloc
{
public:
	TGPUSingleElementAlloc(const char* Name)
		: Ptr(FMemory::Malloc<T>(Name, sizeof(T), EMemoryType::GPU))
	{
	}
	~TGPUSingleElementAlloc()
	{
		FMemory::Free(Ptr);
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
};

#define SINGLE_GPU_ELEMENT(Type, Name) const TGPUSingleElementAlloc<Type> Name(__FILE__ ":" STR(__LINE__))