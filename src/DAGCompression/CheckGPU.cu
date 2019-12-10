#include "DAGCompression/DAGCompression.h"
#include "GpuPrimitives.h"

void DAGCompression::CheckGPU()
{
	CUDA_CHECKED_CALL cudaDeviceSynchronize();

	TGpuArray<uint8> A = { "Test", size_t(double(GPU_MEMORY_ALLOCATED_IN_GB) * (uint64(1) << 30)) };
	for (uint8 Byte = 0; Byte < 256; Byte += 0xFF)
	{
		LOG("Byte %d", Byte);

		A.MemSet(Byte);
		CUDA_CHECKED_CALL cudaDeviceSynchronize();
		
		{
			const auto Check = [=] GPU_ONLY_LAMBDA (uint64 Index)
			{
				uint32* WordPtr = reinterpret_cast<uint32*>(const_cast<uint8*>(&A[4 * (Index % 4)]));
				const uint32 ByteValue =
					(uint32(Byte) << 0) |
					(uint32(Byte) << 8) |
					(uint32(Byte) << 16) |
					(uint32(Byte) << 24);
				const uint32 Value = atomicExch(WordPtr, ByteValue);
				if (Value != ByteValue)
				{
					LOG("ERROR: %x for Index %" PRIu64, Value, Index);
				}
			};
			const auto It = thrust::make_counting_iterator<uint64>(0);
			thrust::for_each(GetExecutionPolicy<EMemoryType::GPU>(), It, It + A.Num(), Check);
			CUDA_CHECKED_CALL cudaDeviceSynchronize();
		}

		{
			const auto Check = [=] GPU_LAMBDA(uint64 Index)
			{
				if (A[Index] != Byte)
				{
					LOG("ERROR: %x at %p", A[Index], &A[Index]);
				}
			};
			const auto It = thrust::make_counting_iterator<uint64>(0);
			thrust::for_each(GetExecutionPolicy<EMemoryType::GPU>(), It, It + A.Num(), Check);
			CUDA_CHECKED_CALL cudaDeviceSynchronize();
		}
	}
}
