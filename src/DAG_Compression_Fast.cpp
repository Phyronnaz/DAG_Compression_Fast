#include "Core.h"
#include <iostream>
#include "GLTFLoader.h"
#include "Serializer.h"
#include "Engine.h"
#include "DAGCompression/DAGCompression.h"

int main()
{
	cudaDeviceProp Props{};
	CUDA_CHECKED_CALL cudaGetDeviceProperties(&Props, 0);
	LOG("CUDA: %d Async Engines", Props.asyncEngineCount);
	
	FEngine Engine;
	Engine.Init();
	
	struct FScopedCNMEM
	{
		FScopedCNMEM()
		{
			PROFILE_FUNCTION();
			
			cnmemDevice_t Device;
			std::memset(&Device, 0, sizeof(Device));
			Device.size = size_t(double(GPU_MEMORY_ALLOCATED_IN_GB) * (uint64(1) << 30));
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
	
	//DAGCompression::CheckGPU();

	FAABB AABB;
	FFinalDag Dag;

	NAME_THREAD("Main");

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

		MARK("Compression Start");
#if 1
		Dag = DAGCompression::Compress_SingleThreaded(Scene, AABB);
#else
		Dag = DAGCompression::Compress_MultiThreaded(Scene, AABB);
#endif
		MARK("Compression End");

		DAGCompression::CheckDag(Dag);

		std::cout << FMemory::GetStatsString();
		
		FreeScene(Scene);
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

	TCpuArray<uint32> DagCpu;
	Reader.Read(DagCpu, "DagCpu");

	Dag.Dag = DagCpu.CreateGPU();
#endif

#if 1
	Engine.Loop(AABB, Dag);
#endif
	
	Dag.Dag.Free();
	Dag.Colors.Free();
	Dag.EnclosedLeaves.Free();

	return 0;
}
