#include "DAGCompression/DAGCompression.h"
#include "Voxelizer.h"
#include "GpuPrimitives.h"
#include "Threadpool.h"

void DAGCompression::AddParentToFragments(const TGpuArray<uint64>& Src, TGpuArray<uint64>& Dst, uint32 Parent, uint32 ParentLevel)
{
	PROFILE_FUNCTION_TRACY();
	SCOPED_BANDWIDTH_TRACY(Src.Num());

	if (Parent == 0)
	{
		FMemory::CudaMemcpy(Dst.GetData(), Src.GetData(), Src.SizeInBytes(), cudaMemcpyDeviceToDevice);
	}
	else
	{
		Transform(Src, Dst, [=] GPU_LAMBDA(uint64 Value) { return Value | (uint64(Parent) << (3 * ParentLevel)); });
	}
}
