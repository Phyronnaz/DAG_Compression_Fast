#include "DAGCompression.h"
#include "GpuPrimitives.h"

template<typename T>
inline void DAGCompression::CheckLevelIndices(const T& Level)
{
	(void)Level;
#if ENABLE_CHECKS
	const auto ChildrenIndices = Level.ChildrenIndices.template CastTo<uint32>();
	const auto ChildMasks = Level.ChildMasks;
	const auto Check = [=] GPU_LAMBDA (uint64 Index)
	{
		const uint64 NodeIndex = Index / 8;
		const uint64 ChildIndex = Index % 8;
		const uint8 ChildMask = ChildMasks[NodeIndex];
		if ((ChildrenIndices[Index] == 0xFFFFFFFF) != !(ChildMask & (1 << ChildIndex)))
		{
			printf("Invalid Children Indices:\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n"
				"0x%08x : %i\n",
				ChildrenIndices[8 * NodeIndex + 0], bool(ChildMask & 0x01),
				ChildrenIndices[8 * NodeIndex + 1], bool(ChildMask & 0x02),
				ChildrenIndices[8 * NodeIndex + 2], bool(ChildMask & 0x04),
				ChildrenIndices[8 * NodeIndex + 3], bool(ChildMask & 0x08),
				ChildrenIndices[8 * NodeIndex + 4], bool(ChildMask & 0x10),
				ChildrenIndices[8 * NodeIndex + 5], bool(ChildMask & 0x20),
				ChildrenIndices[8 * NodeIndex + 6], bool(ChildMask & 0x40),
				ChildrenIndices[8 * NodeIndex + 7], bool(ChildMask & 0x80));
			DEBUG_BREAK();
		}
	};
	const auto It = thrust::make_counting_iterator<uint64>(0);
	thrust::for_each(GetExecutionPolicy<ChildrenIndices.GetMemoryType()>(), It, It + ChildrenIndices.Num(), Check);
	CUDA_CHECK_LAST_ERROR();
#endif	
}
template void DAGCompression::CheckLevelIndices<FCpuLevel>(const FCpuLevel& Level);
template void DAGCompression::CheckLevelIndices<FGpuLevel>(const FGpuLevel& Level);

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

uint64 CheckDagImpl(const TCpuArray<uint32>& Dag, const TCpuArray<uint64>& EnclosedLeaves, uint32 Index, uint32 Level)
{
	if (Level == LEVELS - 2)
	{
		return Utils::Popc(Dag.CheckedAt(Index + 0)) + Utils::Popc(Dag.CheckedAt(Index + 1));
	}
	else
	{
		const uint8 ChildMask = Utils::ChildMask(Dag.CheckedAt(Index));
		uint64 Count = 0;
		for (uint32 ChildIndex = 0; ChildIndex < Utils::Popc(ChildMask); ChildIndex++)
		{
			Count += CheckDagImpl(Dag, EnclosedLeaves, Dag.CheckedAt(Index + 1 + ChildIndex), Level + 1);
		}
#if ENABLE_COLORS
		if (Level < TOP_LEVEL)
		{
			checkAlways(Count == EnclosedLeaves.CheckedAt(Dag.CheckedAt(Index) >> 8));
		}
		else
		{
			checkAlways(Count == (Dag.CheckedAt(Index) >> 8));
		}
#endif
		return Count;
	}
}

void DAGCompression::CheckDag(const FFinalDag& FinalDag)
{
	PROFILE_FUNCTION();

	auto CpuDagCopy = FinalDag.Dag.CreateCPU();
	auto CpuEnclosedLeavesCopy = FinalDag.EnclosedLeaves.CreateCPU();
	const uint64 NumLeaves = CheckDag(CpuDagCopy, CpuEnclosedLeavesCopy);
	(void)NumLeaves;
#if ENABLE_COLORS
	checkAlways(NumLeaves == FinalDag.Colors.Num());
#endif
	CpuDagCopy.Free();
	CpuEnclosedLeavesCopy.Free();
}

uint64 DAGCompression::CheckDag(const TCpuArray<uint32>& Dag, const TCpuArray<uint64>& EnclosedLeaves)
{
	return CheckDagImpl(Dag, EnclosedLeaves, 0, 0);
}