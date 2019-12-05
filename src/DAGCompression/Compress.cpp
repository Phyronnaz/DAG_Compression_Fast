#include "DAGCompression/DAGCompression.h"
#include <cuda_profiler_api.h>
#include "Utils/Aabb.h"
#include "Voxelizer.h"
#include "Thread.h"

FFinalDag DAGCompression::Compress_SingleThreaded(const FScene& Scene, const FAABB& AABB)
{
	const int32 NumberOfSplits = LEVELS - SUBDAG_LEVELS;
	std::vector<FAABB> AABBs = { AABB };
	for (int32 Index = 0; Index < NumberOfSplits; Index++) AABBs = SplitAABB(AABBs);
	checkEqual(AABBs.size(), (1 << (3 * NumberOfSplits)));

	const double GlobalStartTime = Utils::Seconds();
	FrameMark;

	cudaProfilerStart();

	NAME_THREAD("Main");

	double TotalGenerateFragmentsTime = 0;
	double TotalCreateDAGTime = 0;
	uint64 TotalFragments = 0;

	std::vector<FCpuDag> DagsToMerge;
	{
		PROFILE_SCOPE("Create Dags");

		FVoxelizer Voxelizer(1 << SUBDAG_LEVELS, Scene);
		for (uint32 SubDagIndex = 0; SubDagIndex < AABBs.size(); ++SubDagIndex)
		{
			PROFILE_SCOPE("Sub Dag %d", SubDagIndex);

			const double GenerateFragmentsStartTime = Utils::Seconds();
			auto Fragments = Voxelizer.GenerateFragments(AABBs[SubDagIndex]);
			const double GenerateFragmentsElapsed = Utils::Seconds() - GenerateFragmentsStartTime;
			TotalGenerateFragmentsTime += GenerateFragmentsElapsed;
			TotalFragments += Fragments.Num();
			LOG_DEBUG("GenerateFragments took %fs", GenerateFragmentsElapsed);

			const double CreateDAGStartTime = Utils::Seconds();
			auto CpuDag = DAGCompression::CreateSubDAG(Fragments);
			const double CreateDAGElapsed = Utils::Seconds() - CreateDAGStartTime;
			TotalCreateDAGTime += CreateDAGElapsed;
			LOG_DEBUG("CreateDAG took %fs", CreateDAGElapsed);

			LOG("Sub Dag %u/%u %" PRIu64 " fragments", SubDagIndex + 1, uint32(AABBs.size()), Fragments.Num());
			DagsToMerge.push_back(CpuDag);
		}}

	FrameMark;

	LOG("");
	LOG("############################################################");
	LOG("Merging");
	LOG("############################################################");
	LOG("");

	const double MergeDagsStartTime = Utils::Seconds();
	FCpuDag MergedDag;
	if (DagsToMerge.size() == 1)
	{
		MergedDag = DagsToMerge[0];
	}
	else
	{
		MergedDag = DAGCompression::MergeDAGs(std::move(DagsToMerge));
	}
	const double MergeDagsElapsed = Utils::Seconds() - MergeDagsStartTime;
	LOG_DEBUG("MergeDags took %fs", MergeDagsElapsed);

	FrameMark;

	const double CreateFinalDAGStartTime = Utils::Seconds();
	const auto FinalDag = DAGCompression::CreateFinalDAG(std::move(MergedDag));
	const double CreateFinalDAGElapsed = Utils::Seconds() - CreateFinalDAGStartTime;
	LOG_DEBUG("CreateFinalDAG took %fs", CreateFinalDAGElapsed);

	const double TotalElapsed = Utils::Seconds() - GlobalStartTime;
	FrameMark;

	cudaProfilerStop();

	LOG("");
	LOG("############################################################");
	LOG("Total elapsed: %fs", TotalElapsed);
	LOG("Total GenerateFragments: %fs", TotalGenerateFragmentsTime);
	LOG("Total CreateDAG: %fs", TotalCreateDAGTime);
	LOG("MergeDags: %fs", MergeDagsElapsed);
	LOG("CreateFinalDAG: %fs", CreateFinalDAGElapsed);
	LOG("Total Fragments: %" PRIu64, TotalFragments);
	LOG("Throughput: %f BV/s", TotalFragments / TotalElapsed / 1.e9);
	LOG("############################################################");
	LOG("");

	LOG("Peak CPU memory usage: %f MB", Utils::ToMB(FMemory::GetCpuMaxAllocatedMemory()));
	LOG("Peak GPU memory usage: %f MB", Utils::ToMB(FMemory::GetGpuMaxAllocatedMemory()));

	CheckDag(FinalDag);
	return FinalDag;
}