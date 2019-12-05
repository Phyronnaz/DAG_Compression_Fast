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

	return FinalDag;
}

template<uint32 Prefetch, typename T1, typename T2>
inline auto GetResultLambda_Prefetch(T1& Thread, T2 MaxSize)
{
	return [&Thread, MaxSize](uint32 Query)
	{
		TStaticArray<uint32, Prefetch + 1> Queries;
		for (uint32 Index = 0; Index < Prefetch + 1; Index++)
		{
			if (Query + Index < MaxSize)
			{
				Queries[Index] = Query + Index;
			}
			else
			{
				Queries[Index] = Query;
			}
		}
		Thread.StartQueries(Queries);
		return Thread.GetResult_DoNotStartQuery(Query);
	};
}

inline void MakeContextCurrent(GLFWwindow* Window)
{
	PROFILE_FUNCTION();
	glfwMakeContextCurrent(Window);
}

FFinalDag DAGCompression::Compress_MultiThreaded(const FScene& Scene, const FAABB& AABB)
{
	cudaProfilerStart();
	MARK("Start");
	const double GlobalStartTime = Utils::Seconds();
	std::atomic<uint64> TotalFragments{ 0 };
	
	FVoxelizer Voxelizer(1 << SUBDAG_LEVELS, Scene);
	auto* const MainWindow = glfwGetCurrentContext();
	MakeContextCurrent(nullptr);

	FFinalDag FinalDag;
	// Scope so that the threads are destroyed before reaching the next glfwMakeContextCurrent
	{
		const auto AABBs = GetAABBs(AABB);
#if ENABLE_COLORS
		std::vector<TCpuArray<uint32>> SubDagColors(AABBs.size());
		std::atomic<uint32> NumSubDagColorsFinished{ 0 };

		TCpuArray<uint32> MergedColors;
		std::thread MergeColorThread;
#endif

		using FFragments = TGpuArray<uint64>;

		const auto VoxelizerThreadInput = [&](uint32) { return FEmpty(); };
		const auto VoxelizerThreadOutput = [&](uint32 Index, FEmpty)
		{
			const auto GlFragments = Voxelizer.GenerateFragments(AABBs[Index]);
			TotalFragments.fetch_add(GlFragments.Num());
			if (GlFragments.Num() == 0) return FFragments();

			auto CudaFragments = TGpuArray<uint64>("Fragments", GlFragments.Num());
			FMemory::CudaMemcpy(CudaFragments.GetData(), GlFragments.GetData(), GlFragments.SizeInBytes(), cudaMemcpyDeviceToDevice);
			return CudaFragments;
		};
		TIndexedThread<FEmpty, FFragments> VoxelizerThread(
			"0 - VoxelizerThread",
			VoxelizerThreadInput,
			VoxelizerThreadOutput,
			1, // Must be 1
			[=]() { MakeContextCurrent(MainWindow); },
			[=]() { MakeContextCurrent(nullptr); });

		const auto SortFragmentsThreadOutput = [](uint32 Index, FFragments Fragments)
		{
			if (Fragments.Num() == 0) return Fragments;
			return SortFragmentsAndRemoveDuplicates(std::move(Fragments));
		};
		TIndexedThread<FFragments, FFragments> SortFragmentsThread(
			"1 - SortFragmentsThread",
			GetResultLambda_Prefetch<4>(VoxelizerThread, AABBs.size()),
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

			if (NumSubDagColorsFinished.fetch_add(1) + 1 == AABBs.size())
			{
				MergeColorThread = MergeColors(std::move(SubDagColors), MergedColors);
			}
#endif
			return Fragments;
		};
		TIndexedThread<FFragments, FFragments> ExtractColorsThread(
			"2 - ExtractColorsThread",
			GetResultLambda_Prefetch<4>(SortFragmentsThread, AABBs.size()),
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
			GetResultLambda_Prefetch<4>(ExtractColorsThread, AABBs.size()),
			ExtractAndSortLeavesThreadOutput,
			1);

		const auto ExtractSubDagThreadOutput = [](uint32 Index, FLeavesFragments LeavesFragments)
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

			return CpuDag;
		};
		TIndexedThread<FLeavesFragments, FCpuDag> ExtractSubDagThread(
			"4 - ExtractSubDagThread",
			GetResultLambda_Prefetch<4>(ExtractAndSortLeavesThread, AABBs.size()),
			ExtractSubDagThreadOutput,
			4);

		using FDagChildren = TStaticArray<FCpuDag, 8>;
		std::vector<std::unique_ptr<TIndexedThread<FDagChildren, FCpuDag>>> MergeThreads(LEVELS - SUBDAG_LEVELS);

		for (int32 MergePassIndex = 0; MergePassIndex < LEVELS - SUBDAG_LEVELS; MergePassIndex++)
		{
			const auto MergeDagsThreadOutput = [](uint32 DagIndex, FDagChildren Dags)
			{
				uint64 Hashes[8];
				const auto CopyHash = [&](int32 Index)
				{
					auto& TopLevelHashes = Dags[Index].Levels[0].Hashes;
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
				FCpuDag MergedDag = Dags[0];
				for (int32 Index = 1; Index < 8; Index++)
				{
					CopyHash(Index);
					MergedDag = MergeDAGs(std::move(MergedDag), std::move(Dags[Index]));
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

				return MergedDag;
			};
			const auto MergeThreadInput = [&, MergePassIndex](uint32 DagIndex)
			{
				FDagChildren Dags;
				TStaticArray<uint32, 8> ChildrenIndices;
				for (int32 Index = 0; Index < 8; Index++)
				{
					ChildrenIndices[Index] = 8 * DagIndex + Index;
				}
				if (MergePassIndex == 0)
				{
					Dags = ExtractSubDagThread.GetResults(ChildrenIndices);
					LOG("Subdags %u-%u/%u done", 8 * DagIndex + 1, 8 * DagIndex + 8, uint32(AABBs.size()));
				}
				else
				{
					Dags = MergeThreads[MergePassIndex - 1]->GetResults(ChildrenIndices);
				}
				return Dags;
			};
			char* Buffer = new char[1024];
			sprintf(Buffer, "%d - MergeThread Pass %d", 5 + MergePassIndex, MergePassIndex);
			MergeThreads[MergePassIndex] = std::make_unique<TIndexedThread<FDagChildren, FCpuDag>>(
				Buffer,
				MergeThreadInput,
				MergeDagsThreadOutput,
				4);
		}

		MARK("Threads started");

		auto& MainThread = *MergeThreads.back();
		const FCpuDag FinalDagData = MainThread.GetResult(0);

		MARK("Threads done");

#if ENABLE_COLORS
		{
			PROFILE_SCOPE_COLOR(COLOR_YELLOW, "Wait for merge colors");
			MergeColorThread.join();
			FinalDagData.Colors = MergedColors;
		}
#endif
		FinalDag = CreateFinalDAG(std::move(FinalDagData));
	}
	
	const double TotalElapsed = Utils::Seconds() - GlobalStartTime;
	MARK("Done");

	cudaProfilerStop();

	LOG("");
	LOG("############################################################");
	LOG("Total elapsed: %fs", TotalElapsed);
	LOG("Total Fragments: %" PRIu64, TotalFragments.load());
	LOG("Throughput: %f BV/s", TotalFragments.load() / TotalElapsed / 1.e9);
	LOG("############################################################");
	LOG("");

	LOG("Peak CPU memory usage: %f MB", Utils::ToMB(FMemory::GetCpuMaxAllocatedMemory()));
	LOG("Peak GPU memory usage: %f MB", Utils::ToMB(FMemory::GetGpuMaxAllocatedMemory()));

	MakeContextCurrent(MainWindow);
	return FinalDag;
}