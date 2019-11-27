#pragma once

#define ENABLE_CHECKS 0
#define DEBUG_GPU_ARRAYS 0

#define LEVELS 13
#define TOP_LEVEL (LEVELS - 8)
#define SUBDAG_LEVELS 12
#define FRAGMENTS_MEMORY_IN_MILLIONS 160

#define NUM_MERGE_THREADS 1
#define PRINT_DEBUG_INFO 0
#define DEBUG_THRUST 0

#define ENABLE_FORCEINLINE !ENABLE_CHECKS
#define ENABLE_FLATTEN 0

#define ENABLE_COLORS 1

static_assert(SUBDAG_LEVELS <= LEVELS, "");
static_assert(!DEBUG_GPU_ARRAYS || NUM_MERGE_THREADS == 0 || LEVELS == SUBDAG_LEVELS, "");

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

#if ENABLE_CHECKS
#undef NDEBUG
#endif

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
#define check(expr) if(!(expr)) { printf("Assertion failed " __FILE__ ":%d: %s\n", __LINE__, #expr); DEBUG_BREAK(); }
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

#if PRINT_DEBUG_INFO
#define LOG_DEBUG(msg, ...) LOG(msg, __VA_ARGS__)
#else
#define LOG_DEBUG(msg, ...)
#endif

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

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

#define GPU_LAMBDA __host__ __device__

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#define CUDA_CHECKED_CALL ::detail::CudaErrorChecker(__LINE__,__FILE__) =
#define CUDA_CHECK_ERROR() \
	{ \
		CUDA_CHECKED_CALL cudaStreamSynchronize(0); \
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

#include <limits>

template<typename T, typename U>
HOST_DEVICE T Cast(U Value)
{
	static_assert(std::numeric_limits<T>::max() <= std::numeric_limits<int64>::max(), "invalid cast");
	checkf(
		int64(std::numeric_limits<T>::min()) <= int64(Value) && int64(Value) <= int64(std::numeric_limits<T>::max()),
		"overflow: %" PRIi64 " not in [%" PRIi64 ", %" PRIi64 "]",
		int64(Value),
		int64(std::numeric_limits<T>::min()), int64(std::numeric_limits<T>::max()));
	return static_cast<T>(Value);
}

#pragma warning ( push )
#pragma warning ( disable: 4099 4456 4505 )
#include "tracy/Tracy.hpp"
#pragma warning ( pop )

#ifndef TRACY_ENABLE
#define ZoneScopedf(Format, ...) if(0) { printf(Format, ##__VA_ARGS__); } // Avoid unused variable warnings
#else
#define ZoneScopedf(Format, ...) \
	ZoneScoped; \
	{ \
		char __String[1024]; \
		const int32 __Size = sprintf(__String, Format, ##__VA_ARGS__); \
		checkInfEqual(__Size, 1024); \
		ZoneName(__String, __Size); \
	}
#endif

#if 0
#define PROFILE_SCOPE(Format, ...) ZoneScopedf(Format, ##__VA_ARGS__)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#else

#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"

struct FNVExtScope
{
	template<typename... TArgs>
	FNVExtScope(const char* Format, TArgs... Args)
	{
		const int32 Size = sprintf(String, Format, Args...);
		checkInfEqual(Size, 1024);
		
		eventAttrib.version = NVTX_VERSION;
		eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		eventAttrib.colorType = NVTX_COLOR_ARGB;
		eventAttrib.color = 0xFF0000;
		eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
		eventAttrib.message.ascii = String;
		nvtxRangePushEx(&eventAttrib);
	}
	~FNVExtScope()
	{
		nvtxRangePop();
	}
	
    nvtxEventAttributes_t eventAttrib = {0};
	char String[1024];
};

#define JOIN_IMPL(A, B) A ## B
#define JOIN(A, B) JOIN_IMPL(A, B)

#define PROFILE_SCOPE(Format, ...) FNVExtScope JOIN(ScopePerf, __LINE__)(Format, ##__VA_ARGS__);
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__);

#define MARK(Name) nvtxMarkA(Name)
#define NAME_THREAD(Name) nvtxNameOsThreadA(GetCurrentThreadId(), Name);

#endif

using FMortonCode = uint64;

static_assert(3 * SUBDAG_LEVELS <= sizeof(FMortonCode) * 8, "");