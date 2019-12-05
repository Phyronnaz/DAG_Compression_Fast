#include "Core.h"
#include <iostream>
#include <chrono>
#include <cuda_profiler_api.h>
#include "GLTFLoader.h"
#include "Voxelizer.h"
#include "Serializer.h"
#include "Thread.h"
#include "Engine.h"
#include "DAGCompression/DAGCompression.h"

int main()
{
	struct FScopedCNMEM
	{
		FScopedCNMEM()
		{
			PROFILE_FUNCTION();
			
			cnmemDevice_t Device;
			std::memset(&Device, 0, sizeof(Device));
			Device.size = size_t(3) * (size_t(1) << 30);
			uint32 Flags = CNMEM_FLAGS_DEFAULT | CNMEM_FLAGS_CANNOT_GROW | CNMEM_FLAGS_CANNOT_STEAL;
#if DEBUG_GPU_ARRAYS
			Flags |= CNMEM_FLAGS_MANAGED;
#endif
			CUDA_CHECKED_CALL cnmemInit(1, &Device, Flags);
		}
		~FScopedCNMEM()
		{
			PROFILE_FUNCTION();
			CUDA_CHECKED_CALL cnmemFinalize();
		}
	};
	FScopedCNMEM CNMEM;

	FEngine Engine;
	
	Engine.Init();

	FAABB AABB;
	FFinalDag Dag;

#if 1
	{
		PROFILE_SCOPE("Main");

		const double LoadSceneStartTime = Utils::Seconds();
		FGLTFLoader Loader("GLTF/Sponza/glTF/", "Sponza.gltf");
		const auto Scene = Loader.GetScene();
		LOG("Loading Scene took %fs", Utils::Seconds() - LoadSceneStartTime);
		AABB = MakeSquareAABB(Scene.AABB);
		
		LOG("");
		LOG("############################################################");
		LOG("Depth = %d (%d)", LEVELS, 1 << LEVELS);
		LOG("ENABLE_CHECKS = %d", ENABLE_CHECKS);
		LOG("DEBUG_GPU_ARRAYS = %d", DEBUG_GPU_ARRAYS);
		LOG("############################################################");
		LOG("");

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
		Dag = DAGCompression::CreateFinalDAG(std::move(MergedDag));
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

		std::cout << FMemory::GetStatsString();
		
		FreeScene(Scene);

#if 0
		Dag.Dag.Free();
		Dag.Colors.Free();
		Dag.EnclosedLeaves.Free();
		return 0;
#endif
	}
#else
	FFileReader Reader("C:/ROOT/DAG_Compression/cache/result.basic_dag.dag.bin");
	const auto ReadDouble = [&]() { double X; Reader.Read(X); return float(X); };
	AABB.Min.x = ReadDouble();
	AABB.Min.y = ReadDouble();
	AABB.Min.z = ReadDouble();
	AABB.Max.x = ReadDouble();
	AABB.Max.y = ReadDouble();
	AABB.Max.z = ReadDouble();

	uint32 Levels;
	Reader.Read(Levels);
	check(Levels == LEVELS);

	TStaticArray<uint32, EMemoryType::CPU> DagCpu;
	Reader.Read(DagCpu, "DagCpu");

	Dag = DagCpu.CreateGPU();
#endif

	Engine.Loop(AABB, Dag);
	
	Dag.Dag.Free();
	Dag.Colors.Free();
	Dag.EnclosedLeaves.Free();

	return 0;
}
