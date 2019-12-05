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
	PROFILE_FUNCTION();
	CUDA_CHECK_ERROR();
	
	checkAlways(Size != 0);
	void* Ptr = nullptr;
	{
		PROFILE_SCOPE("Alloc %" PRIu64 "B on %s", Size, MemoryTypeToString(Type));
		if (Type == EMemoryType::GPU)
		{
			const auto Error = cnmemMalloc(&Ptr, Size, DEFAULT_STREAM);
			if (Error != CNMEM_STATUS_SUCCESS)
			{
				LOG("\n\n\n");
				LOG("Fatal error when allocating %fMB of GPU memory!", Utils::ToMB(Size));
				std::cout << GetStatsStringImpl() << std::endl;
				FILE* File = fopen("memory_state.log", "w");
				CUDA_CHECKED_CALL cnmemPrintMemoryState(File, DEFAULT_STREAM);
			}
			CUDA_CHECKED_CALL Error;
		}
		else
		{
			check(Type == EMemoryType::CPU);
			const auto Error = cudaMallocHost(&Ptr, Size);
			if (Error != cudaSuccess)
			{
				LOG("\n\n\n");
				LOG("Fatal error when allocating %fMB of CPU memory!", Utils::ToMB(Size));
				std::cout << GetStatsStringImpl() << std::endl;
			}
			CUDA_CHECKED_CALL Error;
		}
	}

	{
		std::lock_guard<std::mutex> Guard(Mutex);
		if (Type == EMemoryType::GPU)
		{
			TotalAllocatedGpuMemory += Size;
			MaxTotalAllocatedGpuMemory = std::max(MaxTotalAllocatedGpuMemory, TotalAllocatedGpuMemory);
		}
		else
		{
			TotalAllocatedCpuMemory += Size;
			MaxTotalAllocatedCpuMemory = std::max(MaxTotalAllocatedCpuMemory, TotalAllocatedCpuMemory);
		}
		checkAlways(Allocations.find(Ptr) == Allocations.end());
		Allocations[Ptr] = { Name, Size, Type };
	}

	return Ptr;
}

void FMemory::FreeImpl(void* Ptr)
{
	PROFILE_FUNCTION();
	CUDA_CHECK_ERROR();
	
	if (!Ptr) return;

	FAlloc Alloc;
	{
		std::lock_guard<std::mutex> Guard(Mutex);
		checkAlways(Allocations.find(Ptr) != Allocations.end());
		Alloc = Allocations[Ptr];
		Allocations.erase(Ptr);
	}

	{
		PROFILE_SCOPE("Free %" PRIu64 "B on %s", Alloc.Size, MemoryTypeToString(Alloc.Type));
		if (Alloc.Type == EMemoryType::GPU)
		{
			CUDA_CHECK_ERROR();
			CUDA_CHECKED_CALL cnmemFree(Ptr, DEFAULT_STREAM);
		}
		else
		{
			check(Alloc.Type == EMemoryType::CPU);
			CUDA_CHECKED_CALL cudaFreeHost(Ptr);
		}
	}

	{
		std::lock_guard<std::mutex> Guard(Mutex);
		if (Alloc.Type == EMemoryType::GPU)
		{
			TotalAllocatedGpuMemory -= Alloc.Size;
			MaxTotalAllocatedGpuMemory = std::max(MaxTotalAllocatedGpuMemory, TotalAllocatedGpuMemory);
		}
		else 
		{
			TotalAllocatedCpuMemory -= Alloc.Size;
			MaxTotalAllocatedCpuMemory = std::max(MaxTotalAllocatedCpuMemory, TotalAllocatedCpuMemory);
		}
	}
}

void FMemory::ReallocImpl(void*& Ptr, uint64 NewSize, bool bCopyData)
{
	PROFILE_FUNCTION();
	CUDA_CHECK_ERROR();

	const auto OldPtr = Ptr;
	FAlloc OldAlloc;
	{
		std::lock_guard<std::mutex> Guard(Mutex);
		checkAlways(Ptr);
		checkAlways(Allocations.find(Ptr) != Allocations.end());
		OldAlloc = Allocations[Ptr];
	}

	if (!bCopyData)
	{
		// Free ASAP
		FreeImpl(OldPtr);
	}
	Ptr = MallocImpl(OldAlloc.Name, NewSize, OldAlloc.Type);
	if (Ptr && bCopyData)
	{
		if (OldAlloc.Type == EMemoryType::GPU)
		{
			cudaError_t Error = cudaMemcpyAsync(Ptr, OldPtr, std::min(NewSize, OldAlloc.Size), cudaMemcpyDeviceToDevice);
			if (Error == cudaSuccess) Error = cudaStreamSynchronize(DEFAULT_STREAM);
			if (Error != cudaSuccess)
			{
				LOG("\n\n\n");
				LOG("Fatal error when copying %fMB of GPU memory!", Utils::ToMB(OldAlloc.Size));
				std::cout << GetStatsStringImpl() << std::endl;
				CUDA_CHECKED_CALL Error;
			}
		}
		else
		{
			std::memcpy(Ptr, OldPtr, std::min(NewSize, OldAlloc.Size));
		}
	}
	if (bCopyData)
	{
		FreeImpl(OldPtr);
	}
}

void FMemory::RegisterCustomAllocImpl(void* Ptr, const char* Name, uint64 Size, EMemoryType Type)
{
	PROFILE_FUNCTION();

	std::lock_guard<std::mutex> Guard(Mutex);
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
	PROFILE_FUNCTION();

	std::lock_guard<std::mutex> Guard(Mutex);
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

FMemory::FAlloc FMemory::GetAllocInfoImpl(void* Ptr) const
{
	std::lock_guard<std::mutex> Guard(Mutex);
	checkAlways(Ptr);
	checkAlways(Allocations.find(Ptr) != Allocations.end());
	return Allocations.at(Ptr);
}

std::string FMemory::GetStatsStringImpl() const
{
	std::lock_guard<std::mutex> Guard(Mutex);

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

	std::vector<std::pair<void*, FAlloc>> AllocationsList(Allocations.begin(), Allocations.end());
	std::sort(AllocationsList.begin(), AllocationsList.end(), [](auto& A, auto& B) { return A.second.Size > B.second.Size; });
	for (auto& [Ptr, Alloc] : AllocationsList)
	{
		StringStream << "\t" << MemoryTypeToString(Alloc.Type) << "; 0x" << std::hex << Ptr << "; " << std::dec << std::setw(10) << Alloc.Size << " B; " << Alloc.Name << std::endl;
	}
	return StringStream.str();
}