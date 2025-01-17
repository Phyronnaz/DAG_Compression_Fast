#include "Memory.h"
#include "Utils.h"
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

FMemory FMemory::Singleton;

inline const char* MemoryTypeToString(EMemoryType Type)
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
	LOG("");
	LOG("");
	LOG("");
    for (auto& [_, Alloc] : Allocations)
    {
        LOG("%s (%s) leaked %fMB", Alloc.Name, MemoryTypeToString(Alloc.Type), Utils::ToMB(Alloc.Size));
    }
    checkAlways(Allocations.empty());
    if (Allocations.empty())
    {
		LOG("No leaks!");
	}
}

void* FMemory::MallocImpl(const char* Name, uint64 Size, EMemoryType Type)
{
	ZoneScopedf("Alloc %" PRIu64 "B on %s", Size, MemoryTypeToString(Type));

	checkAlways(Size != 0);
	void* Ptr = nullptr;
	if (Type == EMemoryType::GPU)
	{
#if DEBUG_GPU_ARRAYS
		cudaError_t Error = cudaMallocManaged(&Ptr, Size);
#else
		cudaError_t Error = cudaMalloc(&Ptr, Size);
#endif

		if (Error == cudaErrorMemoryAllocation)
		{
			LOG("Out of memory, freeing memory and retrying...");
			checkAlways(Error == cudaGetLastError());
			for (auto& Callback : OnOutOfMemoryFallbacks)
			{
				Callback([this](void* InPtr) { FreeImpl(InPtr); });
			}

#if DEBUG_GPU_ARRAYS
			Error = cudaMallocManaged(&Ptr, Size);
#else
			Error = cudaMalloc(&Ptr, Size);
#endif
		}
		
		if (Error != cudaSuccess)
		{
			LOG("\n\n\n");
			LOG("Fatal error when allocating %fMB of GPU memory!", Utils::ToMB(Size));
			std::cout << GetStatsStringImpl() << std::endl;

			for (auto& Callback : OnOutOfMemoryEvents)
			{
				Callback();
			}
		}
		CUDA_CHECKED_CALL Error;
		TotalAllocatedGpuMemory += Size;
		MaxTotalAllocatedGpuMemory = std::max(MaxTotalAllocatedGpuMemory, TotalAllocatedGpuMemory);
	}
	else
	{
		check(Type == EMemoryType::CPU);
		Ptr = std::malloc(Size);
		TotalAllocatedCpuMemory += Size;
		MaxTotalAllocatedCpuMemory = std::max(MaxTotalAllocatedCpuMemory, TotalAllocatedCpuMemory);
	}
	checkAlways(Allocations.find(Ptr) == Allocations.end());
	Allocations[Ptr] = { Name, Size, Type };

	return Ptr;
}

void FMemory::FreeImpl(void* Ptr)
{
    if (!Ptr) return;

	checkAlways(Allocations.find(Ptr) != Allocations.end());
	auto& Alloc = Allocations[Ptr];
	
	ZoneScopedf("Free %" PRIu64 "B on %s", Alloc.Size, MemoryTypeToString(Alloc.Type));

	if (Alloc.Type == EMemoryType::GPU)
	{
		CUDA_CHECK_ERROR();
		CUDA_CHECKED_CALL cudaFree(Ptr);
		TotalAllocatedGpuMemory -= Alloc.Size;
		MaxTotalAllocatedGpuMemory = std::max(MaxTotalAllocatedGpuMemory, TotalAllocatedGpuMemory);
	}
	else
	{
		check(Alloc.Type == EMemoryType::CPU);
		std::free(Ptr);
		TotalAllocatedCpuMemory -= Alloc.Size;
		MaxTotalAllocatedCpuMemory = std::max(MaxTotalAllocatedCpuMemory, TotalAllocatedCpuMemory);
	}
	Allocations.erase(Ptr);
}

void FMemory::ReallocImpl(void*& Ptr, uint64 NewSize, bool bCopyData)
{
	checkAlways(Ptr);
	checkAlways(Allocations.find(Ptr) != Allocations.end());
	const auto OldPtr = Ptr;
	const auto OldAlloc = Allocations[Ptr];

	//LOG("reallocating %s (%s)", OldAlloc.Name, TypeToString(OldAlloc.Type));

	if (OldAlloc.Type == EMemoryType::GPU)
	{
		CUDA_CHECK_ERROR();
		if (!bCopyData)
		{
			// Free ASAP
			FreeImpl(OldPtr);
		}
		Ptr = MallocImpl(OldAlloc.Name, NewSize, OldAlloc.Type);
		if (Ptr && bCopyData)
		{
			const cudaError_t Error = cudaMemcpy(Ptr, OldPtr, std::min(NewSize, OldAlloc.Size), cudaMemcpyDeviceToDevice);
			if (Error != cudaSuccess)
			{
				LOG("\n\n\n");
				LOG("Fatal error when copying %fMB of GPU memory!", Utils::ToMB(OldAlloc.Size));
				std::cout << GetStatsStringImpl() << std::endl;
				CUDA_CHECKED_CALL Error;
			}
		}
		if (bCopyData)
		{
			FreeImpl(OldPtr);
		}
	}
	else
	{
		check(OldAlloc.Type == EMemoryType::CPU);
		Allocations.erase(Ptr);
		if (NewSize == 0)
		{
			std::free(Ptr);
			Ptr = nullptr;
		}
		else
		{
			Ptr = std::realloc(Ptr, NewSize);
			checkAlways(Ptr);
		}
		Allocations[Ptr] = { OldAlloc.Name, OldAlloc.Size, OldAlloc.Type };
	}
}

void FMemory::RegisterCustomAllocImpl(void* Ptr, const char* Name, uint64 Size, EMemoryType Type)
{
	checkAlways(Ptr);
	checkAlways(Allocations.find(Ptr) == Allocations.end());
	checkAlways(Size != 0);
	if (Type == EMemoryType::GPU)
	{
		TotalAllocatedGpuMemory += Size;
		MaxTotalAllocatedGpuMemory = std::max(MaxTotalAllocatedGpuMemory, TotalAllocatedGpuMemory);
	}
	else
	{
		check(Type == EMemoryType::CPU);
		TotalAllocatedCpuMemory += Size;
		MaxTotalAllocatedCpuMemory = std::max(MaxTotalAllocatedCpuMemory, TotalAllocatedCpuMemory);
	}
	checkAlways(Allocations.find(Ptr) == Allocations.end());
	Allocations[Ptr] = { Name, Size, Type };
}

void FMemory::UnregisterCustomAllocImpl(void* Ptr)
{
	checkAlways(Allocations.find(Ptr) != Allocations.end());
	auto& Alloc = Allocations[Ptr];
	if (Alloc.Type == EMemoryType::GPU)
	{
		CUDA_CHECK_ERROR();
		TotalAllocatedGpuMemory -= Alloc.Size;
	}
	else
	{
		check(Alloc.Type == EMemoryType::CPU);
		TotalAllocatedCpuMemory -= Alloc.Size;
	}
	Allocations.erase(Ptr);
}

FMemory::Element FMemory::GetAllocInfoImpl(void* Ptr) const
{
	checkAlways(Ptr);
    checkAlways(Allocations.find(Ptr) != Allocations.end());
    return Allocations.at(Ptr);
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
		std::stringstream StringStream;
		StringStream << std::setw(15) << Alloc.Name << " (" << MemoryTypeToString(Alloc.Type) << ")";
		auto& MapElement = Map[StringStream.str()];
		MapElement.Size += Alloc.Size;
        MapElement.Count++;
    }

    std::vector<std::pair<std::string, AllocCount>> List(Map.begin(), Map.end());
    std::sort(List.begin(), List.end(), [](auto& A, auto& B) { return A.second.Size > B.second.Size; });

    std::stringstream StringStream;
	StringStream << std::endl;
	StringStream << "GPU memory: " << Utils::ToMB(TotalAllocatedGpuMemory) << "MB" << std::endl;
	StringStream << "CPU memory: " << Utils::ToMB(TotalAllocatedCpuMemory) << "MB" << std::endl;
	StringStream << std::endl;
	StringStream << "Allocations overview:" << std::endl;
    for (auto& [Name, Alloc] : List)
    {
        StringStream << "\t" << Name << ": " << std::setw(10) << std::fixed << Utils::ToMB(Alloc.Size) << "MB (" << Alloc.Count << " allocs)" << std::endl;
    }
	StringStream << std::endl;
	StringStream << "Detailed allocations:" << std::endl;
	
    std::vector<std::pair<void*, Element>> AllocationsList(Allocations.begin(), Allocations.end());
    std::sort(AllocationsList.begin(), AllocationsList.end(), [](auto& A, auto& B) { return A.second.Size > B.second.Size; });
    for (auto& [Ptr, Alloc] : AllocationsList)
	{
		StringStream << "\t" << MemoryTypeToString(Alloc.Type) << "; 0x" << std::hex << Ptr << "; " << std::dec << std::setw(10) << Alloc.Size << " B; " << Alloc.Name << std::endl;
    }
    return StringStream.str();
}