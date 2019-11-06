#include "cub/cub.cuh"

#include "Core.h"
#include "Array.h"
#include "CudaPerfRecorder.h"

int main()
{
	srand(0);
	
	FCudaPerfRecorder CudaPerfRecorder;
	TStaticArray<uint64, EMemoryType::CPU> RandomArray("random", 4e8);
	for (auto& Element : RandomArray)
	{
		Element = uint64(rand()) + uint64(rand()) << 16 + uint64(rand()) << 32 + uint64(rand()) << 48;
	}
	
	const auto Cub = [&CudaPerfRecorder, &RandomArray](auto TypeInst, int32 NumBits, auto List)
	{
		using Type = decltype(TypeInst);
		
		for (uint64 Size : List)
		{
			TStaticArray<Type, EMemoryType::CPU> Array("array", Size);
			for (uint64 Index = 0; Index < Array.Num(); Index++)
			{
				Array[Index] = Type(RandomArray[Index]);
			}

			auto KeysIn = Array.CreateGPU();

			{
				TStaticArray<Type, EMemoryType::GPU> KeysOut("gpu", Array.Num());

				TStaticArray<uint8, EMemoryType::GPU> TempStorage;
				size_t TempStorageBytes = 0;
				cub::DeviceRadixSort::SortKeys(TempStorage.GetData(), TempStorageBytes, KeysIn.GetData(), KeysOut.GetData(), Array.Num(), 0, NumBits);
				TempStorage = TStaticArray<uint8, EMemoryType::GPU>("temp storage", TempStorageBytes);

				CudaPerfRecorder.Start();
				cub::DeviceRadixSort::SortKeys(TempStorage.GetData(), TempStorageBytes, KeysIn.GetData(), KeysOut.GetData(), Array.Num(), 0, NumBits);
				const float Elapsed = CudaPerfRecorder.End();
				LOG("%d bits; Num bits: %d; Elapsed: %fs for %" PRIu64 " elements; %f B/s; Memory used: %fMB (tmp: %fMB)",
					int32(sizeof(Type) * 8),
					NumBits,
					Elapsed,
					Array.Num(),
					Array.Num() / Elapsed / 1.e9,
					KeysIn.SizeInMB() + KeysOut.SizeInMB() + TempStorage.SizeInMB(),
					TempStorage.SizeInMB());
				TempStorage.Free();

				KeysOut.Free();
			}

			KeysIn.Free();
			Array.Free();
		}
	};
	{
		const auto List32 = { 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 2e8, 3e8, 4e8 };
		const auto List64 = { 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 2e8 };
		Cub(uint32(), 32, List32);
		LOG("");
		Cub(uint64(), 64, List64);
		LOG("");
	}

	{
		const auto List = { 1e8 };
		for (int32 NumBits = 1; NumBits <= 64; NumBits++)
		{
			if (NumBits <= 32) Cub(uint32(), NumBits, List);
			Cub(uint64(), NumBits, List);
		}
	}
	
	RandomArray.Free();
}
