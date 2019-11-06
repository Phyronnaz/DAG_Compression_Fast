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
    template<typename T>
    static T* Malloc(const char* Name, uint64 Size, EMemoryType Type)
    {
        checkAlways(Size % sizeof(T) == 0);
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        return reinterpret_cast<T*>(Singleton.MallocImpl(Size, Name, Type));
    }
    template<typename T>
    static void Free(T* Ptr)
    {
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        Singleton.FreeImpl(reinterpret_cast<void*>(Ptr));
    }
    template<typename T>
    static void Realloc(T*& Ptr, uint64 NewSize)
    {
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
		void* Copy = reinterpret_cast<void*>(Ptr);
        Singleton.ReallocImpl(Copy, NewSize);
		Ptr = reinterpret_cast<T*>(Copy);
    }
    template<typename T>
    static const char* GetAllocName(const T* Ptr)
    {
        std::lock_guard<std::mutex> Guard(Singleton.Mutex);
        return Singleton.GetAllocNameImpl(reinterpret_cast<void*>(const_cast<T*>(Ptr)));
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

    void* MallocImpl(uint64 Size, const char* Name, EMemoryType Type);
    void FreeImpl(void* Ptr);
	void ReallocImpl(void*& Ptr, uint64 NewSize);
	const char* GetAllocNameImpl(void* Ptr) const;
    std::string GetStatsStringImpl() const;

    struct Element
    {
        const char* Name = nullptr;
        EMemoryType Type = EMemoryType::CPU;
        uint64 Size = uint64(-1);
    };

    std::mutex Mutex;
    size_t TotalAllocatedGpuMemory = 0;
    size_t TotalAllocatedCpuMemory = 0;
    std::unordered_map<void*, Element> Allocations;

    static FMemory Singleton;
};