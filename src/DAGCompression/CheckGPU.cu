#include "DAGCompression/DAGCompression.h"
#include "GpuPrimitives.h"

void DAGCompression::CheckGPU()
{
	CUDA_CHECKED_CALL cudaDeviceSynchronize();

	TGpuArray<uint8> A = { "Test", 5 * (1u << 30) };
	A.MemSet(0x00);

	for (int32 Pass = 1; Pass < 2048; Pass++)
	{
		LOG("Pass %d", Pass);
		Transform(A, A, [] GPU_LAMBDA (uint8 Index) { return Index + 1; });
		Foreach(A, [=] GPU_LAMBDA (uint8 Index) { if (Index != uint8(Pass)) LOG("ERROR: %x != %x", Index, uint8(Pass)); });
		CUDA_CHECKED_CALL cudaDeviceSynchronize();
	}
}