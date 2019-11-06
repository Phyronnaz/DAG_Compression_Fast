#include "Memory.h"
#include "Utils.h"
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream>

FMemory FMemory::Singleton;

inline const char* TypeToString(EMemoryType Type)
{
    switch (Type)
    {
        case EMemoryType::GPU:
            return "GPU";
        case EMemoryType::CPU:
            return "CPU";
        default:
            check(false);
            return "ERROR";
    }
}

FMemory::~FMemory()
{
    check(this == &Singleton);
    for (auto& [_, Alloc] : Allocations)
    {
        LOG("%s (%s) leaked %fMB", Alloc.Name, TypeToString(Alloc.Type), Utils::ToMB(Alloc.Size));
    }
    checkAlways(Allocations.empty());
    if (Allocations.empty())
    {
        LOG("No leaks!");
    }
}

void* FMemory::MallocImpl(uint64 Size, const char* Name, EMemoryType Type)
{
	checkAlways(Size != 0);
    void* Ptr = nullptr;
    if (Type == EMemoryType::GPU)
    {
        const cudaError_t Error = cudaMalloc(&Ptr, Size);
        if (Error != cudaSuccess)
		{
			LOG("\n\n\n");
			LOG("Fatal error when allocating %fMB of %s memory!", Utils::ToMB(Size), TypeToString(Type));
			std::cout << GetStatsStringImpl() << std::endl;
		}
		CUDA_CHECKED_CALL Error;
		TotalAllocatedGpuMemory += Size;
	}
	else
    {
        check(Type == EMemoryType::CPU);
        Ptr = std::malloc(Size);
        TotalAllocatedCpuMemory += Size;
    }
    checkAlways(Allocations.find(Ptr) == Allocations.end());
    Allocations[Ptr] = { Name, Type, Size };

    return Ptr;
}

void FMemory::FreeImpl(void* Ptr)
{
    if (!Ptr) return;

	checkAlways(Allocations.find(Ptr) != Allocations.end());
	auto& Alloc = Allocations[Ptr];
	if (Alloc.Type == EMemoryType::GPU)
	{
		CUDA_CHECKED_CALL cudaFree(Ptr);
		TotalAllocatedGpuMemory -= Alloc.Size;
	}
	else
	{
		check(Alloc.Type == EMemoryType::CPU);
		std::free(Ptr);
		TotalAllocatedCpuMemory -= Alloc.Size;
	}
	Allocations.erase(Ptr);
}

void FMemory::ReallocImpl(void*& Ptr, uint64 NewSize)
{
	checkAlways(Ptr);
    checkAlways(Allocations.find(Ptr) != Allocations.end());
    const auto OldPtr = Ptr;
	const auto OldAlloc = Allocations[Ptr];

    LOG("reallocating %s (%s)", OldAlloc.Name, TypeToString(OldAlloc.Type));

    if (OldAlloc.Type == EMemoryType::GPU)
    {
        Ptr = MallocImpl(NewSize, OldAlloc.Name, OldAlloc.Type);
        if (Ptr) CUDA_CHECKED_CALL cudaMemcpy(Ptr, OldPtr, OldAlloc.Size, cudaMemcpyDeviceToDevice);
        FreeImpl(OldPtr);
    }
    else
    {
        check(OldAlloc.Type == EMemoryType::CPU);
        Allocations.erase(Ptr);
        Ptr = std::realloc(Ptr, NewSize);
        Allocations[Ptr] = { OldAlloc.Name, OldAlloc.Type, OldAlloc.Size };
    }
}

const char* FMemory::GetAllocNameImpl(void* Ptr) const
{
	checkAlways(Ptr);
    checkAlways(Allocations.find(Ptr) != Allocations.end());
    return Allocations.at(Ptr).Name;
}

std::string FMemory::GetStatsStringImpl() const
{
    struct AllocCount
    {
        uint64 Size = 0;
        uint64 Count = 0;

    };
    std::unordered_map<std::string, AllocCount> Map;
    for (auto& [_, Alloc] : Allocations)
	{
		auto Name = std::string(Alloc.Name) + " (" + TypeToString(Alloc.Type) + ")";
		Map[Name].Size += Alloc.Size;
        Map[Name].Count++;
    }

    std::vector<std::pair<std::string, AllocCount>> List(Map.begin(), Map.end());
    std::sort(List.begin(), List.end(), [](auto& A, auto& B) { return A.second.Size > B.second.Size; });

    std::stringstream StringStream;
	StringStream << std::endl;
	StringStream << "GPU memory: " << Utils::ToMB(TotalAllocatedGpuMemory) << "MB" << std::endl;
	StringStream << "CPU memory: " << Utils::ToMB(TotalAllocatedCpuMemory) << "MB" << std::endl;
	StringStream << std::endl;
	StringStream << "Allocations:" << std::endl;
    for (auto& [Name, Alloc] : List)
    {
        StringStream << Name << ": " << Utils::ToMB(Alloc.Size) << "MB (" << Alloc.Count << " allocs)" << std::endl;
    }
    return StringStream.str();
}