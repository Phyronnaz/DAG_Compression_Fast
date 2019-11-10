#include "cub/cub.cuh"

#include "Core.h"
#include "Array.h"
#include "CudaPerfRecorder.h"

int main()
{
	srand(0);
	
	FCudaPerfRecorder CudaPerfRecorder;
	TStaticArray<uint32, EMemoryType::CPU> RandomArray("random", 4e8);
	for (auto& Element : RandomArray)
	{
		Element = rand();
	}
	
	const auto Cub = [&CudaPerfRecorder, &RandomArray](auto TypeInst, int32 NumBits, auto List)
	{
		using Type = decltype(TypeInst);
		
		for (uint64 Size : List)
		{
			TStaticArray<Type, EMemoryType::CPU> Array("array", Size);
			for (uint64 Index = 0; Index < Array.Num(); Index++)
			{
				Type* Ptr = &Array[Index];
				uint32* Ptr32 = reinterpret_cast<uint32*>(Ptr);
				constexpr uint32 NumWords = sizeof(Type) / 4;
				for (int32 Word = 0; Word < NumWords; Word++)
				{
					Ptr32[Word] = RandomArray[Index * NumWords + Word];
				}
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
#if 0
				LOG("%d bits; Num bits: %d; Elapsed: %fs for %" PRIu64 " elements; %f B/s; Memory used: %fMB (tmp: %fMB)",
					int32(sizeof(Type) * 8),
					NumBits,
					Elapsed,
					Array.Num(),
					Array.Num() / Elapsed / 1.e9,
					KeysIn.SizeInMB() + KeysOut.SizeInMB() + TempStorage.SizeInMB(),
					TempStorage.SizeInMB());
#else
				LOG("%d;%" PRIu64 ";%f;%f",
					NumBits,
					Array.Num(),
					Array.Num() / Elapsed / 1.e9,
					KeysIn.SizeInMB() + KeysOut.SizeInMB() + TempStorage.SizeInMB());
#endif
				TempStorage.Free();

				KeysOut.Free();
			}

			KeysIn.Free();
			Array.Free();
		}
	};
	{
		const auto List32  = { 1e2, 2e2, 1e3, 2e3, 1e4, 2e4, 1e5, 2e5, 1e6, 2e6, 4e6, 1e7, 2e7, 4e7, 1e8, 2e8, 3e8, 4e8 };
		const auto List64  = { 1e2, 2e2, 1e3, 2e3, 1e4, 2e4, 1e5, 2e5, 1e6, 2e6, 4e6, 1e7, 2e7, 4e7, 1e8, 2e8 };
		Cub(uint32 (), 32,  List32);
		LOG("");
		Cub(uint64 (), 64,  List64);
		LOG("");
	}

	{
		const auto List = { 1e8 };
		for (int32 NumBits = 1; NumBits <= 32; NumBits++)
		{
			Cub(uint32(), NumBits, List);
		}
		LOG("");
		for (int32 NumBits = 1; NumBits <= 64; NumBits++)
		{
			Cub(uint64(), NumBits, List);
		}
	}
	
	RandomArray.Free();
}
