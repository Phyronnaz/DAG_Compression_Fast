#pragma once

#include "Core.h"
#include <chrono>

namespace Utils
{
	HOST_DEVICE uint32 Popc(uint32 A)
	{
#if defined(__CUDA_ARCH__)
		return __popc(A);
#else
#if USE_POPC_INTRINSICS
		return __builtin_popcount(a);
#else
		// Source: http://graphics.stanford.edu/~seander/bithacks.html
		A = A - ((A >> 1) & 0x55555555);
		A = (A & 0x33333333) + ((A >> 2) & 0x33333333);
		return ((A + (A >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif
#endif
	}
	HOST_DEVICE uint32 Popcll(uint64 A)
	{
#if defined(__CUDA_ARCH__)
		return __popcll(A);
#else
#if USE_POPC_INTRINSICS
		return __builtin_popcountl(A);
#else
		return Popc(uint32(A >> 32)) + Popc(uint32(A & 0xFFFFFFFF));
#endif
#endif
	}
	template<uint32 Bit = 31>
	HOST_DEVICE bool HasFlag(uint32 Index)
	{
		return Index & (1u << Bit);
	}
	template<uint32 Bit = 31>
	HOST_DEVICE uint32 SetFlag(uint32 Index)
	{
		return Index | (1u << Bit);
	}
	template<uint32 Bit = 31>
	HOST_DEVICE uint32 ClearFlag(uint32 Index)
	{
		return Index & ~(1u << Bit);
	}
	HOST_DEVICE uint32 LevelMaxSize(uint32 Level)
	{
		return 1u << (3 * (12 - Level));
	}
	HOST_DEVICE uint8 ChildMask(uint32 Node)
	{
		return Node & 0xFF;
	}
	HOST_DEVICE uint32 TotalSize(uint32 Node)
	{
		return Popc(ChildMask(Node)) + 1;;
	}
	HOST_DEVICE uint32 ChildOffset(uint8 ChildMask, uint8 Child)
	{
		return Popc(ChildMask & ((1u << Child) - 1u)) + 1;
	}
	HOST_DEVICE uint32 MurmurHash32(uint32 H)
	{
		H ^= H >> 16;
		H *= 0x85ebca6b;
		H ^= H >> 13;
		H *= 0xc2b2ae35;
		H ^= H >> 16;
		return H;
	}
	HOST_DEVICE uint64 MurmurHash64(uint64 H)
	{
		H ^= H >> 33;
		H *= 0xff51afd7ed558ccd;
		H ^= H >> 33;
		H *= 0xc4ceb9fe1a85ec53;
		H ^= H >> 33;
		return H;
	}

	HOST_DEVICE uint32 MurmurHash32xN(uint32 const* Hash, uint64 Size, uint32 Seed = 0)
	{
		uint32 H = Seed;
		for (uint64 Index = 0; Index < Size; ++Index)
		{
			uint32 K = Hash[Index];
			K *= 0xcc9e2d51;
			K = (K << 15) | (K >> 17);
			K *= 0x1b873593;
			H ^= K;
			H = (H << 13) | (H >> 19);
			H = H * 5 + 0xe6546b64;
		}

		H ^= uint32(Size);
		H ^= H >> 16;
		H *= 0x85ebca6b;
		H ^= H >> 13;
		H *= 0xc2b2ae35;
		H ^= H >> 16;
		return H;
	}

	template<typename T>
	HOST_DEVICE double ToMB(T Bytes)
	{
		return double(Bytes) / double(1 << 20);
	}

	HOST_DEVICE constexpr uint32 SubtractMod(uint32 Value, uint32 Max /* exclusive */)
	{
		if (Value == 0)
		{
			return Max - 1;
		}
		else
		{
			return Value - 1;
		}
	}

	template<typename T>
	HOST_DEVICE constexpr T DivideCeil(T Dividend, T Divisor)
	{
		return 1 + (Dividend - 1) / Divisor;
	}

    HOST double Seconds()
    {
        static auto Start = std::chrono::high_resolution_clock::now();
        return double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - Start).count()) / 1.e9;
    }
}
