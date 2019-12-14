#include "DAGCompression/DAGCompression.h"
#include <cuda_profiler_api.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Utils/Aabb.h"
#include "Voxelizer.h"
#include "Thread.h"
#include "Threadpool.h"
#include "CudaMath.h"

inline void MakeContextCurrent(GLFWwindow* Window)
{
	PROFILE_FUNCTION();
	glfwMakeContextCurrent(Window);
}

std::vector<FAABB> DAGCompression::GetAABBs(const FAABB& AABB, const int32 NumberOfSplits)
{
	std::vector<FAABB> AABBs = { AABB };
	for (int32 Index = 0; Index < NumberOfSplits; Index++) AABBs = SplitAABB(AABBs);
	checkEqual(AABBs.size(), uint64(1) << (3 * NumberOfSplits));
	return AABBs;
}

FFinalDag DAGCompression::Compress_SingleThreaded(const FScene& Scene, const FAABB& SceneAABB)
{
	const auto AABBs = GetAABBs(SceneAABB, LEVELS - SUBDAG_LEVELS);

	DECLARE_TIME(TotalTime);
	DECLARE_TIME(IsEmptyTime);
	DECLARE_TIME(CreateDagsTime);
	DECLARE_TIME(GenerateFragmentsTime);
	DECLARE_TIME(AddParentToFragmentsTime);
	DECLARE_TIME(CreateSubDagTime);
	DECLARE_TIME(SortFragmentsTime);
	DECLARE_TIME(WaitingForSubDagThreadTime);
	DECLARE_TIME(MergeTime);
	DECLARE_TIME(CreateFinalDagTime);

	uint64 TotalFragments = 0;
	uint64 EmptySubDags = 0;

	FFinalDag FinalDag;
	{
		SCOPE_TIME(TotalTime);

		MARK("Start");
		cudaProfilerStart();
		NAME_THREAD("Main");
		TCpuArray<bool> IsEmpty;
		{
			PROFILE_SCOPE("IsEmpty");
			SCOPE_TIME(IsEmptyTime);
			
			const int32 Size = 1u << (LEVELS - VOXELIZER_LEVELS);
			TCpuArray<uint64> Fragments;
			{
				FVoxelizer Voxelizer(1 << 20, Size, Scene);
				auto OpenGlFragments = Voxelizer.GenerateFragments(SceneAABB);
				auto GpuFragments = OpenGlFragments.Clone();
				GpuFragments = SortFragmentsAndRemoveDuplicates(std::move(GpuFragments));
				Fragments = GpuFragments.CreateCPU();
				GpuFragments.Free();
			}
			IsEmpty = { "IsEmpty", Size * Size * Size };
			IsEmpty.MemSet(true);
			for (uint64 Fragment : Fragments)
			{
				const uint64 MortonCode = Fragment & ((uint64(1) << 40) - 1); // Remove colors
				IsEmpty[MortonCode] = false;
#if 0
				const uint3 Position = Utils::MortonDecode64<uint3>(MortonCode);
				// Add neighbors to avoid false positives
				for (int32 X = -1; X < 2; X++)
				{
					for (int32 Y = -1; Y < 2; Y++)
					{
						for (int32 Z = -1; Z < 2; Z++)
						{
							const int3 P = make_int3(Position) + make_int3(X, Y, Z);
							if (make_int3(0) <= P && P < make_int3(Size))
							{
								IsEmpty[Utils::MortonEncode64(P.x, P.y, P.z)] = false;
							}
						}
					}
				}
#endif
			}
		}
		std::vector<std::shared_ptr<FCpuDag>> DagsToMerge;
		{
			PROFILE_SCOPE("Create Dags");
			SCOPE_TIME(CreateDagsTime);

			FThreadPool AddParentToFragmentsPool(1, "AddParentToFragments");
			FThreadPool CreateCpuDagPool(6, "CreateCpuDag");
			FVoxelizer Voxelizer(VOXELIZER_MAX_NUM_FRAGMENTS, 1u << VOXELIZER_LEVELS, Scene);
			for (uint32 SubDagIndex = 0; SubDagIndex < AABBs.size(); ++SubDagIndex)
			{
				PROFILE_SCOPE("Sub Dag %d", SubDagIndex);
				TGpuArray<uint64> Fragments;
				const auto SubDagAABB = AABBs[SubDagIndex];
				if (true)
				{
					Fragments = { "Fragments", SUBDAG_MAX_NUM_FRAGMENTS };
					uint64 Num = 0;
					const auto SubDagAABBs = GetAABBs(SubDagAABB, SUBDAG_LEVELS - VOXELIZER_LEVELS);
					for (uint32 Index = 0; Index < SubDagAABBs.size(); Index++)
					{
						const uint32 GlobalIndex = SubDagIndex * uint32(SubDagAABBs.size()) + Index;
						if (IsEmpty[GlobalIndex] && !ENABLE_CHECKS) continue;

						TGpuArray<uint64> VoxelizerFragments;
						{
							SCOPE_TIME(GenerateFragmentsTime);
							VoxelizerFragments = Voxelizer.GenerateFragments(SubDagAABBs[Index]);
						}

						{
							SCOPE_TIME(AddParentToFragmentsTime);
							AddParentToFragmentsPool.WaitForCompletion();
							Voxelizer.SwapBuffers();
						}
						
						if (VoxelizerFragments.Num() > 0)
						{
							check(!IsEmpty[GlobalIndex]);
							const uint64 Offset = Num;
							Num += VoxelizerFragments.Num();

							AddParentToFragmentsPool.Enqueue([&Fragments, Offset, VoxelizerFragments, Index]()
							{
								NAME_THREAD("AddParentToFragments Thread");
								TGpuArray<uint64> CudaFragments(Fragments.GetData() + Offset, VoxelizerFragments.Num());
								AddParentToFragments(VoxelizerFragments, CudaFragments, Index, VOXELIZER_LEVELS);
								CUDA_SYNCHRONIZE_STREAM();
							});
						}
						else
						{
							EmptySubDags++;
						}
					}
					AddParentToFragmentsPool.WaitForCompletion();
					checkfAlways(Num < Fragments.Num(), "SUBDAG_MAX_NUM_FRAGMENTS is %u, needs to be >= to %" PRIu64, SUBDAG_MAX_NUM_FRAGMENTS, Num);
					if (Num == 0)
					{
						Fragments.Free();
					}
					auto Copy = Fragments;
					Fragments.Reset();
					Fragments = { Copy.GetData(), Num };
				}
				else
				{
					checkAlways(VOXELIZER_LEVELS == SUBDAG_LEVELS);
					auto VoxelizerFragments = Voxelizer.GenerateFragments(SubDagAABB);
					if (VoxelizerFragments.Num() > 0)
					{
						Fragments = { "Fragments", VoxelizerFragments.Num() };
						FMemory::CudaMemcpy(Fragments.GetData(), VoxelizerFragments.GetData(), VoxelizerFragments.SizeInBytes(), cudaMemcpyDeviceToDevice);
						CUDA_SYNCHRONIZE_STREAM();
					}
				}
				TotalFragments += Fragments.Num();
				const uint32 NumFragments = Cast<uint32>(Fragments.Num());

				{
					SCOPE_TIME(CreateSubDagTime);
//                    CreateCpuDagPool.WaitForCompletion();
					auto CpuDag = CreateSubDAG(std::move(Fragments), SortFragmentsTime, CreateCpuDagPool);
					CUDA_SYNCHRONIZE_STREAM();
					DagsToMerge.push_back(CpuDag);
				}

				LOG("Sub Dag %u/%u %u fragments", SubDagIndex + 1, uint32(AABBs.size()), NumFragments);
			}
			AddParentToFragmentsPool.Destroy();
			{
				SCOPE_TIME(WaitingForSubDagThreadTime);
				CreateCpuDagPool.WaitForCompletion();
			}
			CreateCpuDagPool.Destroy();
			IsEmpty.Free();
		}
		
		MARK("Merge");
		LOG("");
		LOG("############################################################");
		LOG("Merging");
		LOG("############################################################");
		LOG("");
		FCpuDag MergedDag;
		{
			SCOPE_TIME(MergeTime);
			if (DagsToMerge.size() == 1)
			{
				MergedDag = *DagsToMerge[0];
			}
			else
			{
				std::vector<FCpuDag> Dags;
				Dags.reserve(DagsToMerge.size());
				for (auto& Dag : DagsToMerge)
				{
					Dags.push_back(*Dag);
				}
				DagsToMerge.clear();
				MergedDag = DAGCompression::MergeDAGs(std::move(Dags));
				CUDA_SYNCHRONIZE_STREAM();
			}
		}

		MARK("CreateFinalDag");
		{
			SCOPE_TIME(CreateFinalDagTime);
			FinalDag = DAGCompression::CreateFinalDAG(std::move(MergedDag));
			CUDA_SYNCHRONIZE_STREAM();
		}
		
		MARK("Done");
		cudaProfilerStop();
	}
	
	LOG("");
	LOG("############################################################");;
	TotalTime.Print();
	IsEmptyTime.Print();
	CreateDagsTime.Print();
	GenerateFragmentsTime.Print();
	AddParentToFragmentsTime.Print();
	CreateSubDagTime.Print();
	SortFragmentsTime.Print();
	WaitingForSubDagThreadTime.Print();
	MergeTime.Print();
	CreateFinalDagTime.Print();
	LOG("############################################################");
	LOG("Empty Sub Dags still processed: %" PRIu64, EmptySubDags);
	LOG("############################################################");
	LOG("Total Fragments: %" PRIu64, TotalFragments);
	LOG("Throughput: %f BV/s", double(TotalFragments) / TotalTime.Time / 1.e9);
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

#if 0
FFinalDag DAGCompression::Compress_MultiThreaded(const FScene& Scene, const FAABB& SceneAABB)
{
	cudaProfilerStart();
	MARK("Start");
	const double GlobalStartTime = Utils::Seconds();
	std::atomic<uint64> TotalFragments{ 0 };
	
	FVoxelizer Voxelizer(SUBDAG_MAX_NUM_FRAGMENTS, 1 << SUBDAG_LEVELS, Scene);
	auto* const MainWindow = glfwGetCurrentContext();
	MakeContextCurrent(nullptr);

	FFinalDag FinalDag;
	// Scope so that the threads are destroyed before reaching the next glfwMakeContextCurrent
	{
		const auto AABBs = GetAABBs(SceneAABB, LEVELS - SUBDAG_LEVELS);
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
			GetResultLambda_Prefetch<1>(VoxelizerThread, AABBs.size()),
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
			GetResultLambda_Prefetch<1>(SortFragmentsThread, AABBs.size()),
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
			GetResultLambda_Prefetch<1>(ExtractColorsThread, AABBs.size()),
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
			GetResultLambda_Prefetch<1>(ExtractAndSortLeavesThread, AABBs.size()),
			ExtractSubDagThreadOutput,
			2);

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
#endif