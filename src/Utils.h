#pragma once

#include "Core.h"
#include <chrono>

struct uint128
{
	uint64 Data[2];
};
static_assert(sizeof(uint128) == 2 * sizeof(uint64), "");

namespace Utils
{
	HOST_DEVICE uint32 Popc(uint32 A)
	{
#if defined(__CUDA_ARCH__)
		return __popc(A);
#else
#if USE_POPC_INTRINSICS
		return __builtin_popcount(A);
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
	HOST_DEVICE uint64 Rotl64(uint64 X, int8 R)
	{
		return (X << R) | (X >> (64 - R));
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

	HOST_DEVICE uint32 MurmurHash32xN(const uint32* Data, const uint32 Size, const uint32 Seed = 0)
	{
		uint32 H = Seed;
		for (uint32 Index = 0; Index < Size; ++Index)
		{
			uint32 K = Data[Index];
			K *= 0xcc9e2d51;
			K = (K << 15) | (K >> 17);
			K *= 0x1b873593;
			H ^= K;
			H = (H << 13) | (H >> 19);
			H = H * 5 + 0xe6546b64;
		}

		H ^= Size;
		H ^= H >> 16;
		H *= 0x85ebca6b;
		H ^= H >> 13;
		H *= 0xc2b2ae35;
		H ^= H >> 16;
		return H;
	}

	HOST_DEVICE uint128 MurmurHash128xN(const uint128* Data, const uint32 Size, const uint32 Seed = 0)
	{
		uint64 H1 = Seed;
		uint64 H2 = Seed;

		constexpr uint64 C1 = 0x87c37b91114253d5llu;
		constexpr uint64 C2 = 0x4cf5ad432745937fllu;

		for (uint32 Index = 0; Index < Size; Index++)
		{
			uint64 K1 = Data[Index].Data[0];
			uint64 K2 = Data[Index].Data[1];

			K1 *= C1; K1 = Rotl64(K1, 31); K1 *= C2; H1 ^= K1;

			H1 = Rotl64(H1, 27); H1 += H2; H1 = H1 * 5 + 0x52dce729;

			K2 *= C2; K2 = Rotl64(K2, 33); K2 *= C1; H2 ^= K2;

			H2 = Rotl64(H2, 31); H2 += H1; H2 = H2 * 5 + 0x38495ab5;
		}

		H1 ^= Size;
		H2 ^= Size;

		H1 += H2;
		H2 += H1;

		H1 = MurmurHash64(H1);
		H2 = MurmurHash64(H2);

		H1 += H2;
		H2 += H1;

		return { H1, H2 };
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

	template<typename T>
	HOST_DEVICE constexpr bool IsPowerOf2(T Value)
	{
		return (Value & (Value - 1)) == 0;
	}

	HOST_DEVICE uint64 MortonEncode64(uint32 X, uint32 Y, uint32 Z)
	{
		const auto SplitBy3 = [] GPU_LAMBDA (uint32 A)
		{
			uint64 X = A & 0x1fffff; // we only look at the first 21 bits
			X = (X | X << 32) & 0x1f00000000fffful; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
			X = (X | X << 16) & 0x1f0000ff0000fful; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
			X = (X | X << 8) & 0x100f00f00f00f00ful; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
			X = (X | X << 4) & 0x10c30c30c30c30c3ul; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
			X = (X | X << 2) & 0x1249249249249249ul;
			return X;
		};
		return SplitBy3(X) << 2 | SplitBy3(Y) << 1 | SplitBy3(Z);
	}
	template<typename T>
	HOST_DEVICE T MortonDecode64(uint64 A)
	{
		const auto Compact3 = [] GPU_LAMBDA (uint64 X)
		{
			X = X & 0x1249249249249249ul;
			X = (X | (X >> 2)) & 0x10c30c30c30c30c3ul;
			X = (X | (X >> 4)) & 0x100f00f00f00f00ful;
			X = (X | (X >> 8)) & 0x001f0000ff0000fful;
			X = (X | (X >> 16)) & 0x001f00000000fffful;
			X = (X | (X >> 32)) & 0x00000000001ffffful;
			return uint32(X);
		};
		return T{ Compact3(A >> 2), Compact3(A >> 1), Compact3(A) };
	}

    HOST double Seconds()
    {
        static auto Start = std::chrono::high_resolution_clock::now();
        return double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - Start).count()) / 1.e9;
    }
}

struct FTimeCounter
{
	const char* const Name;
	double Time = 0;
	int32 Indent = 0;

	explicit FTimeCounter(const char* Name)
		: Name(Name)
	{
	}
	inline void Print() const
	{
		for (int32 Index = 0; Index < Indent; Index++)
		{
			printf("\t");
		}
		printf("%s: %fs\n", Name, Time);
	}
};

struct FScopedTime
{
	explicit FScopedTime(FTimeCounter& OutTime)
		: OutTime(OutTime)
		, StartTime(Utils::Seconds())
		, ScopeDepth(StaticScopeDepth++)
	{
		checkAlways(OutTime.Indent == 0 || OutTime.Indent == ScopeDepth);
		OutTime.Indent = ScopeDepth;
	}
	~FScopedTime()
	{
		OutTime.Time += Utils::Seconds() - StartTime;
		checkAlways(--StaticScopeDepth == ScopeDepth);
	}

private:
	FTimeCounter& OutTime;
	const double StartTime;
	const int32 ScopeDepth;
	static int32 StaticScopeDepth;
};

#define DECLARE_TIME(Name) FTimeCounter Name{ #Name }
#define SCOPE_TIME(Time) FScopedTime __ScopedTime ## __LINE__ (Time)