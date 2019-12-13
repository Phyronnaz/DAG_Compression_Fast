#include "DAGCompression/DAGCompression.h"
#include "GpuPrimitives.h"

void DAGCompression::CheckGPU()
{
	CUDA_CHECKED_CALL cudaDeviceSynchronize();

	const uint64 MemorySize = size_t(double(2.0) * (uint64(1) << 30));

	for (int32 Pass = 0;; Pass++)
	{
		LOG("Pass %d", Pass);
		
		const uint32 NumHashes = MemorySize / 8 / 4;
		
		TGpuArray<uint64> Hashes = { "Hashes", NumHashes };
		MakeSequence(Hashes);
		Transform(Hashes, Hashes, [] GPU_LAMBDA (uint64 Value) { return Utils::MurmurHash64(Value) & 0xF0F0F0F0F0F0F0F0ul; });
		CUDA_CHECKED_CALL cudaDeviceSynchronize();

		auto SortedHashes = TGpuArray<uint64>("SortedHashes", NumHashes);
		auto SortedHashesToHashes = TGpuArray<uint32>("SortedHashesToHashes", NumHashes);

		auto Sequence = TGpuArray<uint32>("Sequence", NumHashes);
		MakeSequence(Sequence);

		cub::DoubleBuffer<uint64> Keys(Hashes.GetData(), SortedHashes.GetData());
		cub::DoubleBuffer<uint32> Values(Sequence.GetData(), SortedHashesToHashes.GetData());
		
		FTempStorage Storage;
		CUDA_CHECKED_CALL cub::DeviceRadixSort::SortPairs(Storage.Ptr, Storage.Bytes, Keys, Values, NumHashes);
		Storage.Allocate();
		CUDA_CHECKED_CALL cub::DeviceRadixSort::SortPairs(Storage.Ptr, Storage.Bytes, Keys, Values, NumHashes);
		Storage.Free();

		if (Keys.Current() != SortedHashes.GetData())
		{
			check(Keys.Current() == Hashes.GetData());
			std::swap(SortedHashes, Hashes);
		}
		if (Values.Current() != SortedHashesToHashes.GetData())
		{
			check(Values.Current() == Sequence.GetData());
			std::swap(SortedHashesToHashes, Sequence);
		}

		CheckIsSorted(SortedHashes);
		Hashes.Free();
		Sequence.Free();
		SortedHashes.Free();
		SortedHashesToHashes.Free();
	}
#if 0
	TGpuArray<uint8> A = { "Test", MemorySize };
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
#endif
}
