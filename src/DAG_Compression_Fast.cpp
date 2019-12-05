#include "Core.h"
#include <iostream>
#include "GLTFLoader.h"
#include "Serializer.h"
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

	FEngine Engine;
	Engine.Init(); // Need to be initialized to generate the fragments

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

		Dag = DAGCompression::Compress_SingleThreaded(Scene, AABB);
		
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
