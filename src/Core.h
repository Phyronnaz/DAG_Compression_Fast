#pragma once

#define ENABLE_CHECKS 1
#define ENABLE_FORCEINLINE 0
#define ENABLE_FLATTEN 0
#define LEVELS 12

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER

#if defined(__CUDA_ARCH__)
#define DEBUG_BREAK() __threadfence(); asm("trap;")
#else
#define DEBUG_BREAK() __debugbreak()
#endif
#define ASSUME(expr) check(expr); __assume(expr);

#define _CRT_SECURE_NO_WARNINGS

#else

#include <csignal>

#if defined(__CUDA_ARCH__)
#define DEBUG_BREAK() __threadfence(); asm("trap;")
#else
#define DEBUG_BREAK() raise(SIGTRAP)
#endif
#define ASSUME(expr)

#endif

#define RESTRICT __restrict__

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include <cinttypes> // PRIu64
#include <cstdint>
#include <cstdio> // fprintf
#include <cstdlib> // abort

using int8 = std::int8_t;
using int16 = std::int16_t;
using int32 = std::int32_t;
using int64 = std::int64_t;

using uint8 = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;
using uint64 = std::uint64_t;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#if defined(__INTELLISENSE__) || defined(__RSCPP_VERSION) || defined(__CLION_IDE__) || defined(__JETBRAINS_IDE__)
#define __PARSER__
#endif

#ifdef __PARSER__
#error "compiler detected as parser"

#define __CUDACC__
#define __CUDA_ARCH__
#include "crt/math_functions.h"
#include "crt/device_functions.h"
#include "vector_functions.h"

#define __host__
#define __device__
#define __global__
#define __device_builtin__

struct TmpIntVector
{
	int x, y, z;
};
TmpIntVector blockIdx;
TmpIntVector blockDim;
TmpIntVector threadIdx;
#endif

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime.h>

#if ENABLE_CHECKS
#define checkf(expr, msg, ...) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s: " msg "\n", __LINE__, #expr, ##__VA_ARGS__); DEBUG_BREAK(); }
#define check(expr) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s:\n", __LINE__, #expr); DEBUG_BREAK(); }
#define checkAlways(expr) check(expr)
#define checkfAlways(expr, msg, ...) checkf(expr,msg,##__VA_ARGS__)

#define checkEqual(a, b) checkf((a) == (b), "As uint64: a = %" PRIu64 "; b = %" PRIu64, uint64(a), uint64(b))
#define checkInf(a, b) checkf((a) < (b), "As uint64: a = %" PRIu64 "; b = %" PRIu64, uint64(a), uint64(b))
#define checkInfEqual(a, b) checkf((a) <= (b), "As uint64: a = %" PRIu64 "; b = %" PRIu64, uint64(a), uint64(b))
#else
#define checkf(expr, msg, ...)
#define check(...)
#define checkAlways(expr) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s\n", __LINE__, #expr); DEBUG_BREAK(); std::abort(); }
#define checkfAlways(expr, msg, ...) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s: " msg "\n", __LINE__, #expr,##__VA_ARGS__); DEBUG_BREAK(); std::abort(); }
#define checkEqual(a, b)
#define checkInf(a, b)
#define checkInfEqual(a, b)
#endif

#define LOG(msg, ...) printf(msg "\n",##__VA_ARGS__)

#if ENABLE_FORCEINLINE
#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#else
#define FORCEINLINE __attribute__((always_inline))
#endif
#else
#define FORCEINLINE
#endif

#if ENABLE_FLATTEN
#ifdef _MSC_VER
#define FLATTEN
#else
#define FLATTEN __attribute__((flatten))
#endif
#else
#define FLATTEN
#endif

#define HOST           __host__ inline FORCEINLINE FLATTEN
#define HOST_RECURSIVE __host__ inline

#define DEVICE           __device__ inline FORCEINLINE FLATTEN
#define DEVICE_RECURSIVE __device__ inline

#define HOST_DEVICE           __host__ __device__ inline FORCEINLINE FLATTEN
#define HOST_DEVICE_RECURSIVE __host__ __device__ inline

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#define CUDA_CHECKED_CALL ::detail::CudaErrorChecker(__LINE__,__FILE__) =
#define CUDA_CHECK_ERROR() \
	{ \
		cudaDeviceSynchronize(); \
		CUDA_CHECKED_CALL cudaGetLastError(); \
	}

namespace detail
{
	struct CudaErrorChecker
	{
		const int32 Line;
		const char* File;

		inline CudaErrorChecker(int32 Line, const char* File) noexcept
			: Line(Line)
			, File(File)
		{
		}

		FORCEINLINE cudaError_t operator=(cudaError_t Error) const
		{
			if (Error != cudaSuccess)
			{
				std::fprintf(stderr, "ERROR %s:%d: cuda error \"%s\"\n", File, Line, cudaGetErrorString(Error));
				std::abort();
			}
			return Error;
		}

		FORCEINLINE CUresult operator=(CUresult Error) const
		{
			if (Error != CUDA_SUCCESS) 
			{
				const char* Message = nullptr;
				cuGetErrorString(Error, &Message);
				std::fprintf(stderr, "ERROR [driver] %s:%d: cuda error \"%s\"\n", File, Line, Message);
				std::abort();
			}

			return Error;
		}
	};
}