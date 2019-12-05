#include "DAGCompression/DAGCompression.h"
#include <cuda_profiler_api.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Utils/Aabb.h"
#include "Voxelizer.h"
#include "Thread.h"

inline std::vector<FAABB> GetAABBs(const FAABB& AABB)
{
	const int32 NumberOfSplits = LEVELS - SUBDAG_LEVELS;
	std::vector<FAABB> AABBs = { AABB };
	for (int32 Index = 0; Index < NumberOfSplits; Index++) AABBs = SplitAABB(AABBs);
	checkEqual(AABBs.size(), (1 << (3 * NumberOfSplits)));
	return AABBs;
}

FFinalDag DAGCompression::Compress_SingleThreaded(const FScene& Scene, const FAABB& AABB)
{
	const auto AABBs = GetAABBs(AABB);
	
	const double GlobalStartTime = Utils::Seconds();
	MARK("Start");

	cudaProfilerStart();

	NAME_THREAD("Main");

	double TotalGenerateFragmentsTime = 0;
	double TotalCreateDAGTime = 0;
	uint64 TotalFragments = 0;

	std::vector<FCpuDag> DagsToMerge;
	{
		PROFILE_SCOPE("Create Dags");

		FVoxelizer Voxelizer(1u << SUBDAG_LEVELS, Scene);
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
			auto CpuDag = CreateSubDAG(Fragments);
			const double CreateDAGElapsed = Utils::Seconds() - CreateDAGStartTime;
			TotalCreateDAGTime += CreateDAGElapsed;
			LOG_DEBUG("CreateDAG took %fs", CreateDAGElapsed);

			LOG("Sub Dag %u/%u %" PRIu64 " fragments", SubDagIndex + 1, uint32(AABBs.size()), Fragments.Num());
			DagsToMerge.push_back(CpuDag);
		}
	}

	MARK("Merge");

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

	MARK("CreateFinalDag");

	const double CreateFinalDAGStartTime = Utils::Seconds();
	const auto FinalDag = DAGCompression::CreateFinalDAG(std::move(MergedDag));
	const double CreateFinalDAGElapsed = Utils::Seconds() - CreateFinalDAGStartTime;
	LOG_DEBUG("CreateFinalDAG took %fs", CreateFinalDAGElapsed);

	const double TotalElapsed = Utils::Seconds() - GlobalStartTime;
	MARK("Done");

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

FFinalDag DAGCompression::Compress_MultiThreaded(const FScene& Scene, const FAABB& AABB)
{
	FVoxelizer Voxelizer(1 << SUBDAG_LEVELS, Scene);
	auto* const MainWindow = glfwGetCurrentContext();
	glfwMakeContextCurrent(nullptr);

	const auto AABBs = GetAABBs(AABB);

	std::atomic<uint32> SubDagIndex = 0;

#if ENABLE_COLORS
	std::vector<TCpuArray<uint32>> SubDagColors(AABBs.size());
	std::condition_variable SubDagColorsFence;
	std::atomic<bool> SubDagColorsFinished{ false };
#endif
	
	using FFragments = TGpuArray<uint64>;
	
	const auto VoxelizerThreadInput = [&]() -> TOptional<TIndexedData<FEmpty>>
	{
		const uint32 Index = SubDagIndex.fetch_add(1);
		if (Index < AABBs.size())
		{
			return TIndexedData<FEmpty>{ Index, {} };
		}
		else
		{
			return {};
		}
	};
	const auto VoxelizerThreadOutput = [&](uint32 Index, FEmpty)
	{
		glfwMakeContextCurrent(MainWindow);
		const auto GlFragments = Voxelizer.GenerateFragments(AABBs[Index]);
		if (GlFragments.Num() == 0) return FFragments();
		
		auto CudaFragments = TGpuArray<uint64>("Fragments", GlFragments.Num());
		CUDA_CHECKED_CALL cudaMemcpyAsync(CudaFragments.GetData(), GlFragments.GetData(), GlFragments.SizeInBytes(), cudaMemcpyDeviceToDevice);
		CUDA_SYNCHRONIZE_STREAM();
		return CudaFragments;
	};
	TIndexedThread<FEmpty, FFragments> VoxelizerThread(
		"0 - VoxelizerThread",
		VoxelizerThreadInput, 
		VoxelizerThreadOutput, 
		1); // Must be 1

	const auto SortFragmentsThreadOutput = [](uint32 Index, FFragments Fragments)
	{
		if (Fragments.Num() == 0) return Fragments;
		return SortFragmentsAndRemoveDuplicates(std::move(Fragments));
	};
	TIndexedThread<FFragments, FFragments> SortFragmentsThread(
		"1 - SortFragmentsThread",
		GetResultLambda(VoxelizerThread),
		SortFragmentsThreadOutput,
		1);

	const auto ExtractColorsThreadOutput = [&](uint32 Index, FFragments Fragments)
	{
#if ENABLE_COLORS
		if (Fragments.Num() > 0)
		{
			auto Colors = ExtractColorsAndFixFragments(Fragments);
			SubDagColors[Index] = Colors.CreateCPU(); // TODO async copy
			Colors.Free();
		}
		
		if (Index == AABBs.size() - 1)
		{
			SubDagColorsFinished = true;
			SubDagColorsFence.notify_all();
		}
#endif
		return Fragments;
	};
	TIndexedThread<FFragments, FFragments> ExtractColorsThread(
		"2 - ExtractColorsThread",
		GetResultLambda(SortFragmentsThread),
		ExtractColorsThreadOutput,
		1);

	struct FLeavesFragments
	{
		TGpuArray<uint64> Leaves;
		FFragments Fragments;
		TGpuArray<uint32> FragmentIndicesToChildrenIndices;
	};
	const auto ExtractAndSortLeavesThreadOutput = [](uint32 Index, FFragments Fragments)
	{
		if (Fragments.Num() == 0) return FLeavesFragments();

		auto Leaves = ExtractLeavesAndShiftReduceFragments(Fragments);
		
		TGpuArray<uint32> FragmentIndicesToChildrenIndices;
		SortLeaves(Leaves, FragmentIndicesToChildrenIndices);
		return FLeavesFragments{ Leaves, Fragments, FragmentIndicesToChildrenIndices };
	};
	TIndexedThread<FFragments, FLeavesFragments> ExtractAndSortLeavesThread(
		"3 - ExtractAndSortLeavesThread",
		GetResultLambda(ExtractColorsThread),
		ExtractAndSortLeavesThreadOutput,
		1);
	
	const auto ExtractSubDagThreadOutput = [&AABBs](uint32 Index, FLeavesFragments LeavesFragments)
	{
		auto& Fragments = LeavesFragments.Fragments;
		const auto& Leaves = LeavesFragments.Leaves;
		auto& FragmentIndicesToChildrenIndices = LeavesFragments.FragmentIndicesToChildrenIndices;

		FCpuDag CpuDag;
		CpuDag.Levels.resize(SUBDAG_LEVELS - 2);
		if (Fragments.Num() == 0) return CpuDag;
		
		CpuDag.Leaves = Leaves.CreateCPU();
		
		auto PreviousLevelHashes = Leaves;
		for (int32 LevelDepth = SUBDAG_LEVELS - 3; LevelDepth >= 0; LevelDepth--)
		{
			PROFILE_SCOPE("Level %d", LevelDepth);

			FGpuLevel Level = ExtractLevelAndShiftReduceFragments(Fragments, FragmentIndicesToChildrenIndices);
			FragmentIndicesToChildrenIndices.Free();

			ComputeHashes(Level, PreviousLevelHashes);
			PreviousLevelHashes.Free();

			SortLevel(Level, FragmentIndicesToChildrenIndices);

			CpuDag.Levels[LevelDepth] = Level.ToCPU();
			Level.Free(false);
			PreviousLevelHashes = Level.Hashes;
		}
		check(PreviousLevelHashes.Num() == 1);
		PreviousLevelHashes.Free();

		FragmentIndicesToChildrenIndices.Free();
		Fragments.Free();

		LOG("Subdag %u/%u done", Index + 1, uint32(AABBs.size()));
		
		return CpuDag;
	};
	TIndexedThread<FLeavesFragments, FCpuDag> ExtractSubDagThread(
		"4 - ExtractSubDagThread",
		GetResultLambda(ExtractAndSortLeavesThread),
		ExtractSubDagThreadOutput,
		8);

	using FDagChildren = TStaticArray<TIndexedData<FCpuDag>, 8>;
	std::vector<std::unique_ptr<TThread<FDagChildren, TIndexedData<FCpuDag>>>> MergeThreads(LEVELS - SUBDAG_LEVELS);

	for (int32 MergePassIndex = 0; MergePassIndex < LEVELS - SUBDAG_LEVELS; MergePassIndex++)
	{
		const auto MergeDagsThreadOutput = [](FDagChildren Dags)
		{
			check(Dags[0].Index == Dags[1].Index - 1);
			check(Dags[1].Index == Dags[2].Index - 1);
			check(Dags[2].Index == Dags[3].Index - 1);
			check(Dags[3].Index == Dags[4].Index - 1);
			check(Dags[4].Index == Dags[5].Index - 1);
			check(Dags[5].Index == Dags[6].Index - 1);
			check(Dags[6].Index == Dags[7].Index - 1);
			check(Dags[0].Index % 8 == 0);
			const uint32 SubDagIndex = Dags[0].Index / 8;

			uint64 Hashes[8];
			const auto CopyHash = [&](int32 Index)
			{
				auto& TopLevelHashes = Dags[Index].Data.Levels[0].Hashes;
				checkInfEqual(TopLevelHashes.Num(), 1);
				if (TopLevelHashes.Num() > 0)
				{
					check(TopLevelHashes[0] != 0);
					Hashes[Index] = TopLevelHashes[0];
				}
				else
				{
					Hashes[Index] = 0;
				}
			};

			CopyHash(0);
			FCpuDag MergedDag = Dags[0].Data;
			for (int32 Index = 1; Index < 8; Index++)
			{
				CopyHash(Index);
				MergedDag = MergeDAGs(std::move(MergedDag), std::move(Dags[Index].Data));
			}

			// MergedHashes contains all the root subdags hashes
			// Use the stored hashes to find back the roots indices
			// The new children will be in the correct order as dags are in morton order (see AABB split)

			uint8 ChildMask = 0;
			FChildrenIndices ChildrenIndices{};
			const auto& MergedHashes = MergedDag.Levels[0].Hashes;
			for (int32 Index = 0; Index < 8; Index++)
			{
				const uint64 Hash = Hashes[Index];
				if (Hash == 0)
				{
					ChildrenIndices.Indices[Index] = 0xFFFFFFFF;
				}
				else
				{
					ChildMask |= 1 << Index;
					ChildrenIndices.Indices[Index] = uint32(MergedHashes.FindChecked(Hash));
				}
			}
			const uint64 Hash = HashChildrenHashes(Hashes);

			FCpuLevel TopLevel;
			if (ChildMask != 0)
			{
				TopLevel.ChildMasks = TCpuArray<uint8>("ChildMasks", { ChildMask });
				TopLevel.ChildrenIndices = TCpuArray<FChildrenIndices>("ChildrenIndices", { ChildrenIndices });
				TopLevel.Hashes = TCpuArray<uint64>("Hashes", { Hash });
			}
			CheckLevelIndices(TopLevel);

			MergedDag.Levels.insert(MergedDag.Levels.begin(), TopLevel);

			return TIndexedData<FCpuDag>{ SubDagIndex, MergedDag };
		};
		const auto MergeThreadInput = [MergePassIndex, &ExtractSubDagThread, &MergeThreads]()
		{
			if (MergePassIndex == 0)
			{
				return ExtractSubDagThread.GetResult<8>();
			}
			else
			{
				return MergeThreads[MergePassIndex - 1]->GetResult<8>();
			}
		};
		char* Buffer = new char[1024];
		sprintf(Buffer, "%d - MergeThread Pass %d", 5 + MergePassIndex, MergePassIndex);
		MergeThreads[MergePassIndex] = std::make_unique<TThread<FDagChildren, TIndexedData<FCpuDag>>>(
			Buffer,
			MergeThreadInput,
			MergeDagsThreadOutput,
			8);
	}

	MARK("Threads started");
	
#if ENABLE_COLORS
	{
		PROFILE_SCOPE_COLOR(COLOR_YELLOW, "Wait for sub dag colors");
		std::mutex SubDagColorsFenceMutex;
		std::unique_lock<std::mutex> Lock(SubDagColorsFenceMutex);
		SubDagColorsFence.wait(Lock, [&]() { return SubDagColorsFinished.load(); });
	}
	TCpuArray<uint32> MergedColors;
	std::thread MergeColorThread = MergeColors(std::move(SubDagColors), MergedColors);
#endif
	
	auto& MainThread = *MergeThreads.back();
	MainThread.Join();
	const auto IndexedFinalDag = MainThread.GetSingleResult().GetValue();
	check(IndexedFinalDag.Index == 0);
	FCpuDag FinalDagData = IndexedFinalDag.Data;

	MARK("Threads done");
	
#if ENABLE_COLORS
	{
		PROFILE_SCOPE_COLOR(COLOR_YELLOW, "Wait for merge colors");
		MergeColorThread.join();
		FinalDagData.Colors = MergedColors;
	}
#endif

	const auto FinalDag = CreateFinalDAG(std::move(FinalDagData));

	glfwMakeContextCurrent(MainWindow);

	return FinalDag;
}